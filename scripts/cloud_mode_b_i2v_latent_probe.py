#!/usr/bin/env python3
"""Mode B I2V only — latent seam probes at early/middle/late denoise steps.

No new schedule. Saves temporal latent slices around the ring seam:
  F-3, F-2, F-1 | 0, 1, 2
at denoise step indices 2 / 10 / 19 (0-based) for sampling_steps=20.

Auxiliary diagnosis only — not a score/gate.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud_mode_b_generate import (  # noqa: E402
    CKPT,
    LOOP_SRC,
    WAN_ROOT,
    _install_official_attention_backend,
    _resolve_schedule,
    _write_png,
)

sys.path.insert(0, str(WAN_ROOT))
sys.path.insert(0, str(LOOP_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

PROBE_STEPS = (2, 10, 19)  # early / middle / late for 20-step runs
OUT_ROOT = Path(
    os.environ.get(
        "MODE_B_I2V_LATENT_PROBE_OUT",
        "/root/latent-loop/artifacts/mode_b_i2v_seam_diag",
    )
)


def _probe_labels(half_window: int) -> list[str]:
    """Labels F-k..F-1 | 0..k for half_window=k."""
    if half_window < 1:
        raise ValueError("half_window must be >= 1")
    return [f"F-{k}" for k in range(half_window, 0, -1)] + [
        str(k) for k in range(half_window + 1)
    ]


def _temporal_indices(f: int, half_window: int = 3) -> list[int]:
    # F-k..F-1, 0..k on a ring of length F
    labels_n = 2 * half_window + 1
    if f < labels_n:
        raise ValueError(f"latent F={f} too short for half_window={half_window}")
    return [(f - k) % f for k in range(half_window, 0, -1)] + list(
        range(half_window + 1)
    )


def _save_latent_probe(
    latent: torch.Tensor,
    out_dir: Path,
    step_i: int,
    tag: str,
    *,
    half_window: int = 3,
) -> dict:
    """latent: [C, F, H, W] on device/cpu.

    ``half_window=3`` → F-3..2 (legacy). ``half_window=4`` → F-4..4 (D2).
    """
    x = latent.detach().float().cpu()
    _c, f, _h, _w = x.shape
    labels = _probe_labels(half_window)
    idxs = _temporal_indices(f, half_window)
    slice_ = x[:, idxs, :, :].contiguous()
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / f"step{step_i:02d}_{tag}_slices.pt"
    torch.save(
        {
            "step": step_i,
            "tag": tag,
            "latent_shape": list(x.shape),
            "indices": idxs,
            "labels": labels,
            "half_window": half_window,
            "slices": slice_,
        },
        pt_path,
    )

    panels = []
    for j, _lab in enumerate(labels):
        m = slice_[:, j].mean(0).numpy()
        m = m - m.min()
        m = m / max(float(m.max()), 1e-6)
        img = (m * 255.0).astype(np.uint8)
        img = np.stack([img, img, img], axis=-1)
        panels.append(img)
    montage = np.concatenate(panels, axis=1)
    _write_png(montage, out_dir / f"step{step_i:02d}_{tag}_montage.png")

    diffs = {}
    for a, b in zip(labels, labels[1:]):
        ia, ib = labels.index(a), labels.index(b)
        d = torch.norm(slice_[:, ia] - slice_[:, ib]).item()
        diffs[f"{a}->{b}"] = d

    seam = diffs["F-1->0"]
    seam_pair_mean = float(np.mean([diffs["F-1->0"], diffs["0->1"]]))
    non_seam_keys = [k for k in diffs if k not in ("F-1->0", "0->1")]
    adj = float(np.mean([diffs[k] for k in non_seam_keys])) if non_seam_keys else 0.0
    all_vals = list(diffs.values())
    max_key = max(diffs, key=diffs.get)
    meta = {
        "step": step_i,
        "tag": tag,
        "half_window": half_window,
        "indices": idxs,
        "labels": labels,
        "diffs_l2": diffs,
        "seam_l2": seam,
        "seam_pair_mean_l2": seam_pair_mean,
        "adj_mean_l2": adj,
        "seam_vs_adj": seam / max(adj, 1e-6),
        "max_adj_l2": float(max(all_vals)),
        "max_adj_edge": max_key,
        "median_adj_l2": float(np.median(all_vals)),
        "max_vs_median": float(max(all_vals)) / max(float(np.median(all_vals)), 1e-6),
        "pt": str(pt_path),
    }
    (out_dir / f"step{step_i:02d}_{tag}_diffs.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def _i2v_mode_b_with_probes(
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
    probe_dir: Path,
    probe_steps: tuple[int, ...],
) -> torch.Tensor:
    """Instrumented copy of WanTI2V.i2v sampling loop (Mode B already enabled)."""
    import random
    import torchvision.transforms.functional as TF
    from contextlib import contextmanager

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

        latent = noise
        _mask1, mask2 = masks_like([noise], zero=True)
        latent = (1.0 - mask2[0]) * z[0] + mask2[0] * latent

        arg_c = {"context": [context[0]], "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        pipe.model.to(pipe.device)
        torch.cuda.empty_cache()

        for step_i, t in enumerate(tqdm(timesteps)):
            latent_model_input = [latent.to(pipe.device)]
            timestep = torch.stack([t]).to(pipe.device)
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
            latent = (1.0 - mask2[0]) * z[0] + mask2[0] * latent

            if step_i in probe_steps:
                tag = {2: "early", 10: "middle", 19: "late"}.get(
                    step_i, f"step{step_i}"
                )
                logging.info("Saving latent probe step=%s tag=%s", step_i, tag)
                probe_metas.append(
                    _save_latent_probe(latent, probe_dir / "latent", step_i, tag)
                )

        pipe.model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        videos = pipe.vae.decode([latent])

    (probe_dir / "latent_probe_summary.json").write_text(
        json.dumps(
            {
                "probe_steps": list(probe_steps),
                "sampling_steps": sampling_steps,
                "probes": probe_metas,
                "readout": (
                    "If seam_vs_adj is already high at early → topology/RoPE; "
                    "if only late is high → late denoise detail convergence."
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
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--case-id", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-area", type=int, default=704 * 1280)
    p.add_argument("--frame-num", type=int, default=81)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guide-scale", type=float, default=5.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--sample-solver", type=str, default="unipc")
    p.add_argument("--schedule", type=str, default="symmetric")
    p.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert args.steps > max(PROBE_STEPS), "need sampling_steps covering probe indices"

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
    img = Image.open(args.image).convert("RGB")
    img.save(out_dir / "source.png")

    logging.info("Building WanTI2V for latent probe ...")
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
    video = _i2v_mode_b_with_probes(
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
        probe_dir=out_dir,
        probe_steps=PROBE_STEPS,
    )
    elapsed = time.time() - t0
    disable_mode_b_on_wan_model(pipe.model)

    meta = {
        "case_id": args.case_id,
        "schedule": schedule_name,
        "seed": args.seed,
        "steps": args.steps,
        "probe_steps": list(PROBE_STEPS),
        "elapsed_sec": round(elapsed, 2),
        "video_shape": list(video.shape),
        "attention_backend": attn_info,
        "note": "Mode B only. No new schedule. Latent probes auxiliary.",
    }
    (out_dir / "probe_run.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, **meta}, indent=2))
    print("LATENT_PROBE_OK")


if __name__ == "__main__":
    main()
