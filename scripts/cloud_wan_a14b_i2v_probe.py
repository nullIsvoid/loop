#!/usr/bin/env python3
"""Phase A: Wan2.2 I2V-A14B *native* latent probe (architecture control).

Official WanI2V only:
  - NO Symmetric Circular Temporal RoPE / Mode B
  - NO conditioning edits
  - reference enters as y = cat(mask, vae(img+zeros)) channel condition
  - generated latent stays pure noise denoise (no per-step latent[0]=ref)

Question: does late full-ring still show TI2V-5B-style index-0 double spike
(F-1→0 AND 0→1 both anomalous)?

Cloud ckpt default: /model/HuggingFace/Wan-AI/Wan2.2-I2V-A14B
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud_mode_b_generate import (  # noqa: E402
    LOOP_SRC,
    WAN_ROOT,
    _bootstrap_wan_namespace,
    _install_official_attention_backend,
    _tensor_to_uint8_frames,
    _write_mp4,
    _write_png,
    _x3,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

CKPT = Path(
    os.environ.get("WAN_I2V_A14B_CKPT", "/model/HuggingFace/Wan-AI/Wan2.2-I2V-A14B")
)
OUT_ROOT = Path(
    os.environ.get(
        "WAN_A14B_PROBE_OUT",
        "/root/latent-loop/artifacts/wan_a14b_i2v_probe",
    )
)

PERSON_PROMPT = (
    "Subtle seamless looping live wallpaper, fixed camera, preserve the "
    "woman identity pose and clothing exactly, only gentle hair and "
    "clothing hem sway, no cuts, no camera move"
)


def _full_ring_adjacent_gaps(latent: torch.Tensor) -> dict:
    x = latent.detach().float().cpu()
    _c, f, _h, _w = x.shape
    diffs: dict[str, float] = {}
    vals: list[float] = []
    for i in range(f):
        j = (i + 1) % f
        d = torch.norm(x[:, i] - x[:, j]).item()
        diffs[f"{i}->{j}"] = d
        vals.append(d)
    max_i = int(np.argmax(vals))
    return {
        "F": f,
        "diffs_l2": diffs,
        "max_gap_l2": float(vals[max_i]),
        "max_gap_edge": f"{max_i}->{(max_i + 1) % f}",
        "median_gap_l2": float(np.median(vals)),
        "mean_gap_l2": float(np.mean(vals)),
        "max_vs_median": float(vals[max_i]) / max(float(np.median(vals)), 1e-6),
        "seam_F-1->0": diffs[f"{f - 1}->0"],
        "seam_0->1": diffs["0->1"],
    }


def _save_probe(
    latent: torch.Tensor,
    out_dir: Path,
    step_i: int,
    tag: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"step{step_i:02d}_full_latent.pt"
    torch.save(
        {
            "step": step_i,
            "tag": tag,
            "latent": latent.detach().float().cpu().contiguous(),
            "latent_shape": list(latent.shape),
        },
        pt_path,
    )
    meta = {
        "step": step_i,
        "tag": tag,
        "full_latent_pt": str(pt_path),
        **_full_ring_adjacent_gaps(latent),
    }
    (out_dir / f"step{step_i:02d}_{tag}_full_ring_gaps.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def _default_probe_steps(sampling_steps: int) -> tuple[int, ...]:
    # early / middle / late indices for arbitrary step counts
    e = min(2, sampling_steps - 1)
    m = max(e, sampling_steps // 2 - 1)
    late = sampling_steps - 1
    return tuple(sorted({e, m, late}))


def generate_a14b_native_with_probes(
    pipe,
    *,
    prompt: str,
    img: Image.Image,
    n_prompt: str,
    seed: int,
    max_area: int,
    frame_num: int,
    shift: float,
    sample_solver: str,
    sampling_steps: int,
    guide_scale: float | tuple[float, float],
    probe_dir: Path,
    probe_steps: tuple[int, ...],
    offload_model: bool = True,
) -> tuple[torch.Tensor, list[dict]]:
    """Mirror official ``WanI2V.generate`` with full-ring latent probes."""
    import torchvision.transforms.functional as TF
    from tqdm import tqdm
    from wan.utils.fm_solvers import (
        FlowDPMSolverMultistepScheduler,
        get_sampling_sigmas,
        retrieve_timesteps,
    )
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    if isinstance(guide_scale, float):
        guide_scale = (guide_scale, guide_scale)

    img_t = TF.to_tensor(img).sub_(0.5).div_(0.5).to(pipe.device)
    f_pix = frame_num
    h0, w0 = img_t.shape[1:]
    aspect_ratio = h0 / w0
    lat_h = round(
        np.sqrt(max_area * aspect_ratio)
        // pipe.vae_stride[1]
        // pipe.patch_size[1]
        * pipe.patch_size[1]
    )
    lat_w = round(
        np.sqrt(max_area / aspect_ratio)
        // pipe.vae_stride[2]
        // pipe.patch_size[2]
        * pipe.patch_size[2]
    )
    h = lat_h * pipe.vae_stride[1]
    w = lat_w * pipe.vae_stride[2]

    max_seq_len = ((f_pix - 1) // pipe.vae_stride[0] + 1) * lat_h * lat_w // (
        pipe.patch_size[1] * pipe.patch_size[2]
    )
    max_seq_len = int(math.ceil(max_seq_len / pipe.sp_size)) * pipe.sp_size

    seed_g = torch.Generator(device=pipe.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        16,
        (f_pix - 1) // pipe.vae_stride[0] + 1,
        lat_h,
        lat_w,
        dtype=torch.float32,
        generator=seed_g,
        device=pipe.device,
    )

    msk = torch.ones(1, f_pix, lat_h, lat_w, device=pipe.device)
    msk[:, 1:] = 0
    msk = torch.concat(
        [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
        dim=1,
    )
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    if not pipe.t5_cpu:
        pipe.text_encoder.model.to(pipe.device)
        context = pipe.text_encoder([prompt], pipe.device)
        context_null = pipe.text_encoder([n_prompt], pipe.device)
        if offload_model:
            pipe.text_encoder.model.cpu()
    else:
        context = pipe.text_encoder([prompt], torch.device("cpu"))
        context_null = pipe.text_encoder([n_prompt], torch.device("cpu"))
        context = [t.to(pipe.device) for t in context]
        context_null = [t.to(pipe.device) for t in context_null]

    y = pipe.vae.encode(
        [
            torch.concat(
                [
                    torch.nn.functional.interpolate(
                        img_t[None].cpu(), size=(h, w), mode="bicubic"
                    ).transpose(0, 1),
                    torch.zeros(3, f_pix - 1, h, w),
                ],
                dim=1,
            ).to(pipe.device)
        ]
    )[0]
    y = torch.concat([msk, y])

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_low = getattr(pipe.low_noise_model, "no_sync", noop_no_sync)
    no_sync_high = getattr(pipe.high_noise_model, "no_sync", noop_no_sync)
    probe_metas: list[dict] = []

    with (
        torch.amp.autocast("cuda", dtype=pipe.param_dtype),
        torch.no_grad(),
        no_sync_low(),
        no_sync_high(),
    ):
        boundary = pipe.boundary * pipe.num_train_timesteps
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(
                sampling_steps, device=pipe.device, shift=shift
            )
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler, device=pipe.device, sigmas=sampling_sigmas
            )
        else:
            raise NotImplementedError(sample_solver)

        latent = noise
        arg_c = {"context": [context[0]], "seq_len": max_seq_len, "y": [y]}
        arg_null = {"context": context_null, "seq_len": max_seq_len, "y": [y]}

        if offload_model:
            torch.cuda.empty_cache()

        for step_i, t in enumerate(tqdm(timesteps)):
            latent_model_input = [latent.to(pipe.device)]
            timestep = torch.stack([t]).to(pipe.device)
            model = pipe._prepare_model_for_timestep(t, boundary, offload_model)
            sample_guide_scale = (
                guide_scale[1] if t.item() >= boundary else guide_scale[0]
            )

            noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
            if offload_model:
                torch.cuda.empty_cache()
            noise_pred_uncond = model(
                latent_model_input, t=timestep, **arg_null
            )[0]
            if offload_model:
                torch.cuda.empty_cache()
            noise_pred = noise_pred_uncond + sample_guide_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latent.unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latent = temp_x0.squeeze(0)
            # Official A14B: NO latent[:,0] overwrite here.

            if step_i in probe_steps:
                tag = {
                    probe_steps[0]: "early",
                    probe_steps[len(probe_steps) // 2]: "middle",
                    probe_steps[-1]: "late",
                }.get(step_i, f"step{step_i}")
                logging.info(
                    "probe step=%s tag=%s F=%s", step_i, tag, latent.shape[1]
                )
                probe_metas.append(
                    _save_probe(latent, probe_dir / "latent", step_i, tag)
                )

        if offload_model:
            pipe.low_noise_model.cpu()
            pipe.high_noise_model.cpu()
            torch.cuda.empty_cache()

        videos = pipe.vae.decode([latent])

    return videos[0], probe_metas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--image",
        type=str,
        default="/root/latent-loop/artifacts/mode_b_i2v/_inputs/person.png",
    )
    p.add_argument("--prompt", type=str, default=PERSON_PROMPT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-area",
        type=int,
        default=480 * 832,
        help="Default 480*832 for 24GB; use 720*1280 if VRAM allows.",
    )
    p.add_argument("--frame-num", type=int, default=81)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--sample-solver", type=str, default="unipc")
    p.add_argument("--guide-scale", type=float, default=3.5)
    p.add_argument("--ckpt", type=str, default=str(CKPT))
    p.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    p.add_argument(
        "--probe-steps",
        type=int,
        nargs="*",
        default=None,
        help="Defaults to early/mid/late for --steps",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"missing image {image_path}")

    probe_steps = (
        tuple(args.probe_steps)
        if args.probe_steps
        else _default_probe_steps(args.steps)
    )
    assert max(probe_steps) < args.steps

    attn_info = _install_official_attention_backend()
    _ = attn_info.pop("flash_fn", None)
    _bootstrap_wan_namespace()

    from wan.configs import WAN_CONFIGS
    from wan.image2video import WanI2V

    cfg = WAN_CONFIGS["i2v-A14B"]
    n_prompt = cfg.sample_neg_prompt
    fps = int(cfg.sample_fps)

    logging.info("Building WanI2V A14B from %s ...", args.ckpt)
    pipe = WanI2V(
        config=cfg,
        checkpoint_dir=str(args.ckpt),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        init_on_cpu=True,
        convert_model_dtype=True,
    )

    img = Image.open(image_path).convert("RGB")
    img.save(out_root / "source.png")
    t0 = time.time()
    video, probes = generate_a14b_native_with_probes(
        pipe,
        prompt=args.prompt,
        img=img,
        n_prompt=n_prompt,
        seed=args.seed,
        max_area=args.max_area,
        frame_num=args.frame_num,
        shift=args.shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.steps,
        guide_scale=args.guide_scale,
        probe_dir=out_root,
        probe_steps=probe_steps,
        offload_model=True,
    )
    elapsed = time.time() - t0
    frames = _tensor_to_uint8_frames(video)
    _write_mp4(frames, out_root / "out.mp4", fps)
    _write_mp4(_x3(frames), out_root / "out_x3.mp4", fps)
    _write_png(frames[0], out_root / "first.png")
    _write_png(frames[-1], out_root / "last.png")

    late = next((p for p in probes if p["tag"] == "late"), probes[-1] if probes else None)
    experiment = {
        "name": "wan_a14b_i2v_native_probe",
        "question": (
            "Does native A14B I2V late latent still show TI2V-5B-style "
            "index-0 double spike (F-1→0 and 0→1 both high)?"
        ),
        "attention_backend": attn_info,
        "mode_b_rope": False,
        "conditioning_edits": False,
        "ckpt": args.ckpt,
        "shared": {
            "seed": args.seed,
            "steps": args.steps,
            "frame_num": args.frame_num,
            "max_area": args.max_area,
            "guide_scale": args.guide_scale,
            "shift": args.shift,
            "sample_solver": args.sample_solver,
            "probe_steps": list(probe_steps),
        },
        "elapsed_sec": round(elapsed, 2),
        "probes": probes,
        "late": None
        if late is None
        else {
            "max_gap_edge": late["max_gap_edge"],
            "max_gap_l2": late["max_gap_l2"],
            "median_gap_l2": late["median_gap_l2"],
            "max_vs_median": late["max_vs_median"],
            "seam_F-1->0": late["seam_F-1->0"],
            "seam_0->1": late["seam_0->1"],
        },
        "note": (
            "Architecture control only. Compare late seam_0->1 to median; "
            "not a product score."
        ),
    }
    (out_root / "experiment.json").write_text(
        json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
    )
    (out_root / "run.json").write_text(
        json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, "late": experiment["late"]}, indent=2))
    print("A14B_NATIVE_PROBE_OK")


if __name__ == "__main__":
    sys.path.insert(0, str(LOOP_SRC))
    main()
