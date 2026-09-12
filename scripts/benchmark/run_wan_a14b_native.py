#!/usr/bin/env python3
"""WA-0: Wan2.2 I2V-A14B native I2V on fixed loop benchmark.

No Mode-B RoPE. No conditioning edits. Independent ``y`` channel conditioning.
"""
from __future__ import annotations

import argparse
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

_SCRIPTS = Path(__file__).resolve().parents[1]
_BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_SCRIPTS))

from common import (  # noqa: E402
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_BENCHMARK_DIR,
    default_probe_steps,
    finalize_media,
    load_benchmark,
    prepare_run_dir,
    probe_tag_map,
    save_latent_probe,
    write_json,
    write_result_stub,
)
from cloud_mode_b_generate import (  # noqa: E402
    LOOP_SRC,
    WAN_ROOT,
    _bootstrap_wan_namespace,
    _install_official_attention_backend,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

CKPT = Path(
    os.environ.get("WAN_I2V_A14B_CKPT", "/model/HuggingFace/Wan-AI/Wan2.2-I2V-A14B")
)


def _generate_a14b_native_with_probes(
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

    tags = probe_tag_map(probe_steps)
    probe_metas: list[dict] = []

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_low = getattr(pipe.low_noise_model, "no_sync", noop_no_sync)
    no_sync_high = getattr(pipe.high_noise_model, "no_sync", noop_no_sync)

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

            if step_i in tags:
                tag = tags[step_i]
                logging.info("probe step=%s tag=%s F=%s", step_i, tag, latent.shape[1])
                probe_metas.append(
                    save_latent_probe(
                        latent, probe_dir / "latent", tag=tag, step_i=step_i
                    )
                )

        if offload_model:
            pipe.low_noise_model.cpu()
            pipe.high_noise_model.cpu()
            torch.cuda.empty_cache()

        videos = pipe.vae.decode([latent])

    return videos[0], probe_metas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WA-0 Wan A14B native benchmark")
    p.add_argument("--benchmark-dir", type=str, default=str(DEFAULT_BENCHMARK_DIR))
    p.add_argument(
        "--out-root",
        type=str,
        default=str(
            Path(os.environ.get("LOOP_ARTIFACTS_ROOT", DEFAULT_ARTIFACTS_ROOT))
        ),
    )
    p.add_argument("--run-name", type=str, default="wan_a14b_native")
    p.add_argument("--ckpt", type=str, default=str(CKPT))
    p.add_argument("--max-area", type=int, default=480 * 832)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guide-scale", type=float, default=3.5)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--sample-solver", type=str, default="unipc")
    return p.parse_args()


def main() -> None:
    import json

    args = parse_args()
    bench = load_benchmark(args.benchmark_dir)
    assert bench.frame_num % 4 == 1, "Wan frame_num must be 4n+1"

    out_dir = prepare_run_dir(args.out_root, args.run_name, bench)
    probe_steps = default_probe_steps(args.steps)

    sys.path.insert(0, str(WAN_ROOT))
    sys.path.insert(0, str(LOOP_SRC))
    attn_info = _install_official_attention_backend()
    attn_info.pop("flash_fn", None)
    _bootstrap_wan_namespace()

    from wan.configs import WAN_CONFIGS
    from wan.image2video import WanI2V

    cfg = WAN_CONFIGS["i2v-A14B"]
    n_prompt = cfg.sample_neg_prompt
    fps = int(cfg.sample_fps)

    logging.info("WA-0 building WanI2V A14B from %s", args.ckpt)
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

    img = Image.open(bench.source_path).convert("RGB")
    t0 = time.time()
    video, probes = _generate_a14b_native_with_probes(
        pipe,
        prompt=bench.prompt,
        img=img,
        n_prompt=n_prompt,
        seed=bench.seed,
        max_area=args.max_area,
        frame_num=bench.frame_num,
        shift=args.shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.steps,
        guide_scale=args.guide_scale,
        probe_dir=out_dir,
        probe_steps=probe_steps,
        offload_model=True,
    )
    elapsed = time.time() - t0
    media = finalize_media(video, out_dir, fps=fps)
    late = next((p for p in probes if p["tag"] == "late"), None)

    run = {
        "experiment_id": "WA-0",
        "name": "wan_a14b_native",
        "mode_b_rope": False,
        "conditioning_edits": False,
        "benchmark": bench.as_metadata(),
        "model": {
            "family": "Wan2.2",
            "variant": "I2V-A14B",
            "ckpt": args.ckpt,
        },
        "model_specific": {
            "steps": args.steps,
            "guide_scale": args.guide_scale,
            "shift": args.shift,
            "sample_solver": args.sample_solver,
            "max_area": args.max_area,
            "probe_steps": list(probe_steps),
            "attention_backend": attn_info,
        },
        "elapsed_sec": round(elapsed, 2),
        "media": media,
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
        "question": (
            "On person_loop_v1, does native A14B late latent keep only F-1→0 elevated "
            "while 0→1 ≈ median?"
        ),
        "note": (
            "artifacts/wan_a14b_i2v_probe is OLD RESEARCH EVIDENCE (substitute image); "
            "this run is the strict benchmark."
        ),
    }
    write_json(out_dir / "run.json", run)
    write_result_stub(
        out_dir,
        experiment_id="WA-0",
        question=run["question"],
        late=run["late"],
        notes=[
            "Native independent y conditioning.",
            "Diagnosis only — not a scoring Gate.",
            "Previous substitute-image Phase A is architecture evidence only.",
        ],
    )
    print(json.dumps({"ok": True, "late": run["late"]}, indent=2))
    print("WA_0_OK")


if __name__ == "__main__":
    main()
