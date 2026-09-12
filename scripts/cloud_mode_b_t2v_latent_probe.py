#!/usr/bin/env python3
"""Mode B Symmetric T2V latent seam probe — control vs I2V hard first-frame anchor.

No image conditioning. No first-frame mask/clamp.
Same probes as I2V: steps 2/10/19, slices F-3..F-1 | 0..2.

Decisive question: does late seam spike exist without I2V anchoring?
If T2V late stays normal → investigate Circular I2V Conditioning next.
If T2V late also spikes → green-light Circular Temporal Context.
Do NOT implement D in this script.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud_mode_b_generate import (  # noqa: E402
    CKPT,
    LOOP_SRC,
    WAN_ROOT,
    _install_official_attention_backend,
    _resolve_schedule,
)
from cloud_mode_b_i2v_latent_probe import (  # noqa: E402
    PROBE_STEPS,
    _save_latent_probe,
)

sys.path.insert(0, str(WAN_ROOT))
sys.path.insert(0, str(LOOP_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

OUT_ROOT = Path(
    os.environ.get(
        "MODE_B_T2V_LATENT_PROBE_OUT",
        "/root/latent-loop/artifacts/mode_b_t2v_seam_diag",
    )
)

DEFAULT_PROMPT = (
    "A seamless looping live wallpaper of a candle flame gently flickering "
    "and swaying in a continuous periodic cycle, fixed camera, no cuts, "
    "smooth looping motion, cinematic lighting"
)


def _t2v_mode_b_with_probes(
    pipe,
    *,
    prompt: str,
    n_prompt: str,
    seed: int,
    size: tuple[int, int],
    frame_num: int,
    shift: float,
    sample_solver: str,
    sampling_steps: int,
    guide_scale: float,
    probe_dir: Path,
    probe_steps: tuple[int, ...],
) -> torch.Tensor:
    """Instrumented WanTI2V.t2v loop — no I2V first-frame clamp."""
    from tqdm import tqdm
    from wan.utils.fm_solvers import (
        FlowDPMSolverMultistepScheduler,
        get_sampling_sigmas,
        retrieve_timesteps,
    )
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from wan.utils.utils import masks_like

    f_pix = frame_num
    target_shape = (
        pipe.vae.model.z_dim,
        (f_pix - 1) // pipe.vae_stride[0] + 1,
        size[1] // pipe.vae_stride[1],
        size[0] // pipe.vae_stride[2],
    )
    seq_len = math.ceil(
        (target_shape[2] * target_shape[3])
        / (pipe.patch_size[1] * pipe.patch_size[2])
        * target_shape[1]
        / pipe.sp_size
    ) * pipe.sp_size

    seed_g = torch.Generator(device=pipe.device)
    seed_g.manual_seed(seed)

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

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=pipe.device,
            generator=seed_g,
        )
    ]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(pipe.model, "no_sync", noop_no_sync)
    probe_metas = []

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

        latents = noise
        # Official T2V: masks_like(..., zero=False) — frame 0 is NOT hard-clamped.
        _mask1, mask2 = masks_like(noise, zero=False)

        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        pipe.model.to(pipe.device)
        torch.cuda.empty_cache()

        for step_i, t in enumerate(tqdm(timesteps)):
            latent_model_input = latents
            timestep = torch.stack([t])
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
                latents[0].unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latents = [temp_x0.squeeze(0)]
            # Intentionally NO: latent = (1-mask)*z0 + mask*latent

            if step_i in probe_steps:
                tag = {2: "early", 10: "middle", 19: "late"}.get(
                    step_i, f"step{step_i}"
                )
                logging.info("Saving T2V latent probe step=%s tag=%s", step_i, tag)
                probe_metas.append(
                    _save_latent_probe(latents[0], probe_dir / "latent", step_i, tag)
                )

        pipe.model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        videos = pipe.vae.decode(latents)

    (probe_dir / "latent_probe_summary.json").write_text(
        json.dumps(
            {
                "path": "t2v",
                "i2v_first_frame_hard_anchor": False,
                "probe_steps": list(probe_steps),
                "sampling_steps": sampling_steps,
                "probes": probe_metas,
                "readout": (
                    "Compare to I2V probes. If T2V late seam_vs_adj stays ~1 "
                    "while I2V late spikes with both F-1→0 and 0→1 high → "
                    "blame I2V conditioning boundary. If T2V late also spikes "
                    "on F-1→0 → Circular Temporal Context is more justified."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return videos[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=str, default="1280*704", help="W*H")
    p.add_argument("--frame-num", type=int, default=81)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guide-scale", type=float, default=5.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--sample-solver", type=str, default="unipc")
    p.add_argument("--schedule", type=str, default="symmetric")
    p.add_argument("--case-id", type=str, default="t2v_candle")
    p.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert args.steps > max(PROBE_STEPS)
    w_str, h_str = args.size.split("*")
    size = (int(w_str), int(h_str))

    out_dir = Path(args.out_root) / args.case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    attn_info = _install_official_attention_backend()
    flash_fn = attn_info.pop("flash_fn")

    from wan.configs import WAN_CONFIGS
    from wan.textimage2video import WanTI2V
    from latent_loop.adapters.wan.attention import (
        disable_mode_b_on_wan_model,
        enable_mode_b_on_wan_model,
    )

    schedule, schedule_name = _resolve_schedule(args.schedule)
    cfg = WAN_CONFIGS["ti2v-5B"]
    n_prompt = cfg.sample_neg_prompt

    logging.info("Building WanTI2V for T2V latent probe (no I2V anchor) ...")
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
        pipe.model, schedule=schedule, flash_attention_fn=flash_fn, enabled=True
    )

    t0 = time.time()
    video = _t2v_mode_b_with_probes(
        pipe,
        prompt=args.prompt,
        n_prompt=n_prompt,
        seed=args.seed,
        size=size,
        frame_num=args.frame_num,
        shift=args.shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.steps,
        guide_scale=args.guide_scale,
        probe_dir=out_dir,
        probe_steps=PROBE_STEPS,
    )
    elapsed = time.time() - t0
    disable_mode_b_on_wan_model(pipe.model)

    meta = {
        "case_id": args.case_id,
        "path": "t2v",
        "i2v_first_frame_hard_anchor": False,
        "schedule": schedule_name,
        "seed": args.seed,
        "size_wh": list(size),
        "frame_num": args.frame_num,
        "latent_frames_F": (args.frame_num - 1) // 4 + 1,
        "steps": args.steps,
        "probe_steps": list(PROBE_STEPS),
        "prompt": args.prompt,
        "elapsed_sec": round(elapsed, 2),
        "video_shape": list(video.shape),
        "attention_backend": attn_info,
        "note": "Control: Mode B Symmetric T2V latent probe. Do not implement D here.",
    }
    (out_dir / "probe_run.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, **{k: meta[k] for k in (
        "case_id", "path", "schedule", "elapsed_sec", "video_shape"
    )}}, indent=2))
    print("T2V_LATENT_PROBE_OK")


if __name__ == "__main__":
    main()
