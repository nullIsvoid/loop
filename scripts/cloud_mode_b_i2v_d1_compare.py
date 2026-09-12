#!/usr/bin/env python3
"""B vs D1 I2V compare: hard frame-0 anchor vs Late Anchor Release.

Keeps Symmetric Circular Temporal RoPE unchanged.
Only changes post-step frame-0 reference clamp strength.

B  = HardAnchorSchedule (official a=1 every step)
D1 = LateAnchorReleaseSchedule (release late)

Probes steps 2/10/19. Primary question: does late F-1→0 / 0→1 double spike fall?
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

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud_mode_b_generate import (  # noqa: E402
    CKPT,
    LOOP_SRC,
    WAN_ROOT,
    _install_official_attention_backend,
    _resolve_schedule,
    _tensor_to_uint8_frames,
    _write_mp4,
    _write_png,
    _x3,
)
from cloud_mode_b_i2v_latent_probe import PROBE_STEPS, _save_latent_probe  # noqa: E402

sys.path.insert(0, str(WAN_ROOT))
sys.path.insert(0, str(LOOP_SRC))

from latent_loop.i2v_conditioning import (  # noqa: E402
    HardAnchorSchedule,
    LateAnchorReleaseSchedule,
    apply_frame0_anchor,
    list_anchor_strengths,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

OUT_ROOT = Path(
    os.environ.get(
        "MODE_B_I2V_D1_OUT",
        "/root/latent-loop/artifacts/mode_b_i2v_d1",
    )
)

CASES = [
    (
        "person",
        "Subtle seamless looping live wallpaper, fixed camera, preserve the "
        "woman identity pose and clothing exactly, only gentle hair and "
        "clothing hem sway, no cuts, no camera move",
    ),
    (
        "environment",
        "Subtle seamless looping live wallpaper, fixed camera, preserve the "
        "cat identity and composition exactly, gentle ocean water ripples "
        "and soft ambient motion, no cuts, no camera move",
    ),
]


def _i2v_with_anchor(
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
    guide_scale: float,
    anchor_schedule,
    probe_dir: Path | None,
    probe_steps: tuple[int, ...],
) -> tuple[torch.Tensor, list[dict], list[float]]:
    import torchvision.transforms.functional as TF
    from tqdm import tqdm
    from wan.utils.fm_solvers import (
        FlowDPMSolverMultistepScheduler,
        get_sampling_sigmas,
        retrieve_timesteps,
    )
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from wan.utils.utils import best_output_size, masks_like

    ih, iw = img.height, img.width
    dh = pipe.patch_size[1] * pipe.vae_stride[1]
    dw = pipe.patch_size[2] * pipe.vae_stride[2]
    ow, oh = best_output_size(iw, ih, dw, dh, max_area)
    scale = max(ow / iw, oh / ih)
    img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
    x1 = (img.width - ow) // 2
    y1 = (img.height - oh) // 2
    img = img.crop((x1, y1, x1 + ow, y1 + oh))
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(pipe.device).unsqueeze(1)

    f_pix = frame_num
    seq_len = ((f_pix - 1) // pipe.vae_stride[0] + 1) * (
        oh // pipe.vae_stride[1]
    ) * (ow // pipe.vae_stride[2]) // (pipe.patch_size[1] * pipe.patch_size[2])
    seq_len = int(math.ceil(seq_len / pipe.sp_size)) * pipe.sp_size

    seed_g = torch.Generator(device=pipe.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        pipe.vae.model.z_dim,
        (f_pix - 1) // pipe.vae_stride[0] + 1,
        oh // pipe.vae_stride[1],
        ow // pipe.vae_stride[2],
        dtype=torch.float32,
        generator=seed_g,
        device=pipe.device,
    )

    if not pipe.t5_cpu:
        pipe.text_encoder.model.to(pipe.device)
        context = pipe.text_encoder([prompt], pipe.device)
        context_null = pipe.text_encoder([n_prompt], pipe.device)
        pipe.text_encoder.model.cpu()
    else:
        context = pipe.text_encoder([prompt], torch.device("cpu"))
        context_null = pipe.text_encoder([n_prompt], torch.device("cpu"))
        context = [t.to(pipe.device) for t in context]
        context_null = [t.to(pipe.device) for t in context_null]

    z = pipe.vae.encode([img])
    strengths = list_anchor_strengths(sampling_steps, anchor_schedule)

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(pipe.model, "no_sync", noop_no_sync)
    probe_metas: list[dict] = []

    with (
        torch.amp.autocast("cuda", dtype=pipe.param_dtype),
        torch.no_grad(),
        no_sync(),
    ):
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
        _mask1, mask2 = masks_like([noise], zero=True)
        # Initial: use step-0 strength (usually 1.0)
        latent = apply_frame0_anchor(latent, z[0], strengths[0])

        arg_c = {"context": [context[0]], "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        pipe.model.to(pipe.device)
        torch.cuda.empty_cache()

        for step_i, t in enumerate(tqdm(timesteps)):
            latent_model_input = [latent.to(pipe.device)]
            timestep = torch.stack([t]).to(pipe.device)
            # Keep official timestep mask path (unchanged); only clamp blend changes.
            temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
            temp_ts = torch.cat(
                [
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep,
                ]
            )
            timestep = temp_ts.unsqueeze(0)

            noise_pred_cond = pipe.model(latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = pipe.model(
                latent_model_input, t=timestep, **arg_null
            )[0]
            noise_pred = noise_pred_uncond + guide_scale * (
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
            a = strengths[step_i]
            latent = apply_frame0_anchor(latent, z[0], a)

            if probe_dir is not None and step_i in probe_steps:
                tag = {2: "early", 10: "middle", 19: "late"}.get(
                    step_i, f"step{step_i}"
                )
                logging.info(
                    "probe step=%s tag=%s anchor_strength=%.3f", step_i, tag, a
                )
                meta = _save_latent_probe(latent, probe_dir / "latent", step_i, tag)
                meta["anchor_strength"] = a
                probe_metas.append(meta)

        pipe.model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        videos = pipe.vae.decode([latent])

    return videos[0], probe_metas, strengths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs-dir",
        type=str,
        default="/root/latent-loop/artifacts/mode_b_i2v/_inputs",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-area", type=int, default=704 * 1280)
    p.add_argument("--frame-num", type=int, default=81)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guide-scale", type=float, default=5.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--sample-solver", type=str, default="unipc")
    p.add_argument("--rope-schedule", type=str, default="symmetric")
    p.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    p.add_argument(
        "--cases",
        nargs="+",
        default=["person", "environment"],
        help="subset of person environment",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert args.steps == 20, "D1 first table is defined for 20 steps"
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    inputs = Path(args.inputs_dir)

    attn_info = _install_official_attention_backend()
    flash_fn = attn_info.pop("flash_fn")

    from wan.configs import WAN_CONFIGS
    from wan.textimage2video import WanTI2V
    from latent_loop.adapters.wan.attention import (
        disable_mode_b_on_wan_model,
        enable_mode_b_on_wan_model,
    )

    rope_sched, rope_name = _resolve_schedule(args.rope_schedule)
    cfg = WAN_CONFIGS["ti2v-5B"]
    n_prompt = cfg.sample_neg_prompt
    fps = int(cfg.sample_fps)

    variants = [
        ("B_hard_anchor", HardAnchorSchedule()),
        ("D1_late_release", LateAnchorReleaseSchedule()),
    ]

    logging.info("Building WanTI2V once ...")
    pipe = WanTI2V(
        config=cfg,
        checkpoint_dir=str(CKPT),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        init_on_cpu=True,
        convert_model_dtype=True,
    )
    enable_mode_b_on_wan_model(
        pipe.model, schedule=rope_sched, flash_attention_fn=flash_fn, enabled=True
    )

    case_map = {cid: prompt for cid, prompt in CASES}
    experiment_runs = []

    for case_id in args.cases:
        if case_id not in case_map:
            raise SystemExit(f"unknown case {case_id}")
        prompt = case_map[case_id]
        image_path = inputs / f"{case_id}.png"
        if not image_path.is_file():
            raise SystemExit(f"missing input {image_path}")
        img = Image.open(image_path).convert("RGB")

        for variant_id, anchor_sched in variants:
            run_dir = out_root / case_id / variant_id
            run_dir.mkdir(parents=True, exist_ok=True)
            img.save(run_dir / "source.png")
            logging.info("=== %s / %s ===", case_id, variant_id)
            t0 = time.time()
            video, probes, strengths = _i2v_with_anchor(
                pipe,
                prompt=prompt,
                img=img,
                n_prompt=n_prompt,
                seed=args.seed,
                max_area=args.max_area,
                frame_num=args.frame_num,
                shift=args.shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.steps,
                guide_scale=args.guide_scale,
                anchor_schedule=anchor_sched,
                probe_dir=run_dir,
                probe_steps=PROBE_STEPS,
            )
            elapsed = time.time() - t0
            frames = _tensor_to_uint8_frames(video)
            _write_mp4(frames, run_dir / "out.mp4", fps)
            _write_mp4(_x3(frames), run_dir / "out_x3.mp4", fps)
            _write_png(frames[0], run_dir / "first.png")
            _write_png(frames[-1], run_dir / "last.png")

            late = next((p for p in probes if p["step"] == 19), None)
            summary = {
                "case_id": case_id,
                "variant": variant_id,
                "anchor_schedule": type(anchor_sched).__name__,
                "anchor_strengths": strengths,
                "rope_schedule": rope_name,
                "seed": args.seed,
                "steps": args.steps,
                "elapsed_sec": round(elapsed, 2),
                "video_shape": list(video.shape),
                "probes": probes,
                "late_diffs_l2": None if late is None else late["diffs_l2"],
                "late_seam_vs_adj": None if late is None else late["seam_vs_adj"],
            }
            (run_dir / "run.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            (run_dir / "latent_probe_summary.json").write_text(
                json.dumps(
                    {
                        "variant": variant_id,
                        "probe_steps": list(PROBE_STEPS),
                        "probes": probes,
                        "anchor_strengths": strengths,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            experiment_runs.append(
                {
                    "case_id": case_id,
                    "variant": variant_id,
                    "late_seam_vs_adj": summary["late_seam_vs_adj"],
                    "late_F-1->0": None
                    if late is None
                    else late["diffs_l2"]["F-1->0"],
                    "late_0->1": None if late is None else late["diffs_l2"]["0->1"],
                    "late_1->2": None if late is None else late["diffs_l2"]["1->2"],
                    "dir": str(run_dir),
                }
            )
            logging.info(
                "%s/%s done late seam_vs_adj=%s F-1->0=%s 0->1=%s",
                case_id,
                variant_id,
                summary["late_seam_vs_adj"],
                experiment_runs[-1]["late_F-1->0"],
                experiment_runs[-1]["late_0->1"],
            )

    disable_mode_b_on_wan_model(pipe.model)
    experiment = {
        "name": "mode_b_i2v_d1_late_anchor_release",
        "question": (
            "Does Late Anchor Release remove the late F-1→0 / 0→1 double spike "
            "vs hard-anchor Symmetric Mode B?"
        ),
        "attention_backend": attn_info,
        "rope_schedule": rope_name,
        "shared": {
            "seed": args.seed,
            "steps": args.steps,
            "frame_num": args.frame_num,
            "max_area": args.max_area,
            "guide_scale": args.guide_scale,
            "shift": args.shift,
            "sample_solver": args.sample_solver,
        },
        "variants": {
            "B_hard_anchor": list_anchor_strengths(20, HardAnchorSchedule()),
            "D1_late_release": list_anchor_strengths(20, LateAnchorReleaseSchedule()),
        },
        "runs": experiment_runs,
        "note": "No F-1 conditioning. No circular temporal context. No new RoPE schedule.",
    }
    (out_root / "experiment.json").write_text(
        json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, "runs": experiment_runs}, indent=2))
    print("D1_COMPARE_OK")


if __name__ == "__main__":
    main()
