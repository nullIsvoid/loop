#!/usr/bin/env python3
"""Real Mode A/B generation on official Wan2.2 TI2V-5B (full denoise + VAE).

A = untouched Wan baseline
B = same pipeline + enable_mode_b_on_wan_model (default SymmetricShiftSchedule)

Shared: prompt, negative, seed, noise (via same seed), scheduler, CFG,
frame count, resolution, steps.

Attention: prefer official wan flash_attention. If flash_attn is missing,
use Wan's own attention() SDPA fallback (not the smoke-test stub).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

WAN_ROOT = Path(os.environ.get("WAN_ROOT", "/root/Wan2.2"))
CKPT = Path(
    os.environ.get("WAN_TI2V_CKPT", "/model/HuggingFace/Wan-AI/Wan2.2-TI2V-5B")
)
LOOP_SRC = Path(os.environ.get("LOOP_SRC", "/root/latent-loop/src"))
OUT_DIR = Path(
    os.environ.get("MODE_B_OUT", str(Path(__file__).resolve().parents[1] / "artifacts" / "mode_b_real"))
)

sys.path.insert(0, str(WAN_ROOT))
sys.path.insert(0, str(LOOP_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)


def _bootstrap_wan_namespace() -> None:
    """Register ``wan`` as a namespace package without executing wan/__init__.py.

    Official ``wan/__init__.py`` eagerly imports S2V/Animate (decord/librosa/…).
    TI2V generation only needs configs + textimage2video + modules.
    """
    import types

    wan_dir = WAN_ROOT / "wan"
    if "wan" not in sys.modules:
        pkg = types.ModuleType("wan")
        pkg.__path__ = [str(wan_dir)]  # type: ignore[attr-defined]
        pkg.__file__ = str(wan_dir / "__init__.py")
        sys.modules["wan"] = pkg


def _install_official_attention_backend() -> dict:
    """Prefer flash_attn; otherwise wire Wan attention() SDPA fallback into flash_attention."""
    _bootstrap_wan_namespace()
    import wan.modules.attention as attn_mod
    import wan.modules.model as model_mod

    info = {
        "flash_attn_2": bool(getattr(attn_mod, "FLASH_ATTN_2_AVAILABLE", False)),
        "flash_attn_3": bool(getattr(attn_mod, "FLASH_ATTN_3_AVAILABLE", False)),
        "backend": "flash_attention",
    }
    if info["flash_attn_2"] or info["flash_attn_3"]:
        flash_fn = attn_mod.flash_attention
        info["backend"] = "official_flash_attention"
        return info | {"flash_fn": flash_fn}

    def flash_via_wan_attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
        **kwargs,
    ):
        # Official wan.modules.attention.attention SDPA branch (flash unavailable).
        return attn_mod.attention(
            q,
            k,
            v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            fa_version=version,
        )

    attn_mod.flash_attention = flash_via_wan_attention
    model_mod.flash_attention = flash_via_wan_attention
    info["backend"] = "official_attention_sdpa_fallback"
    info["flash_fn"] = flash_via_wan_attention
    logging.warning(
        "flash_attn not installed; using Wan attention() SDPA fallback for full generate"
    )
    return info


def _tensor_to_uint8_frames(video: torch.Tensor) -> list:
    """video: [C, T, H, W] in [-1, 1] -> list of HxWx3 uint8."""
    x = video.detach().float().cpu().clamp(-1, 1)
    x = ((x + 1.0) * 0.5 * 255.0).byte()  # C,T,H,W
    x = x.permute(1, 2, 3, 0).numpy()  # T,H,W,C
    return [x[i] for i in range(x.shape[0])]


def _write_mp4(frames: list, path: Path, fps: int) -> None:
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    for fr in frames:
        writer.append_data(fr)
    writer.close()


def _write_png(frame, path: Path) -> None:
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), frame)


def _x3(frames: list) -> list:
    return frames + frames + frames


def _resolve_schedule(name: str):
    from latent_loop.rope.schedule import (
        FixedShiftSchedule,
        IdentityShiftSchedule,
        LoopyShiftSchedule,
        SymmetricShiftSchedule,
    )

    key = name.strip().lower()
    mapping = {
        "symmetric": SymmetricShiftSchedule(),
        "s3": SymmetricShiftSchedule(),
        "loopy": LoopyShiftSchedule(),
        "s1": LoopyShiftSchedule(),
        "fixed1": FixedShiftSchedule(1),
        "s2": FixedShiftSchedule(1),
        "identity": IdentityShiftSchedule(),
        "s0": IdentityShiftSchedule(),
    }
    if key not in mapping:
        raise SystemExit(
            f"unknown --schedule {name!r}; "
            "use symmetric|loopy|fixed1|identity (or s0|s1|s2|s3)"
        )
    return mapping[key], type(mapping[key]).__name__


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default=(
        "A seamless looping live wallpaper of a candle flame gently flickering "
        "and swaying in a continuous periodic cycle, fixed camera, no cuts, "
        "smooth looping motion, cinematic lighting"
    ))
    p.add_argument("--n-prompt", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=str, default="1280*704", help="W*H official ti2v size")
    p.add_argument("--frame-num", type=int, default=81, help="pixel frames; must be 4n+1")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guide-scale", type=float, default=5.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--sample-solver", type=str, default="unipc")
    p.add_argument(
        "--schedule",
        type=str,
        default="symmetric",
        help="Mode B schedule: symmetric|loopy|fixed1|identity (default: symmetric)",
    )
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    w_str, h_str = args.size.split("*")
    size = (int(w_str), int(h_str))  # Wan size = (width, height)
    assert args.frame_num % 4 == 1, "frame_num must be 4n+1"

    attn_info = _install_official_attention_backend()
    flash_fn = attn_info.pop("flash_fn")

    from wan.configs import WAN_CONFIGS
    from wan.textimage2video import WanTI2V
    from latent_loop.adapters.wan.attention import (
        disable_mode_b_on_wan_model,
        enable_mode_b_on_wan_model,
    )
    from latent_loop.rope.schedule import list_layer_time_shifts

    schedule, schedule_name = _resolve_schedule(args.schedule)

    cfg = WAN_CONFIGS["ti2v-5B"]
    n_prompt = args.n_prompt if args.n_prompt else cfg.sample_neg_prompt
    latent_f = (args.frame_num - 1) // 4 + 1

    logging.info("Building WanTI2V ...")
    t_build = time.time()
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
    logging.info("WanTI2V ready in %.1fs", time.time() - t_build)

    gen_kwargs = dict(
        input_prompt=args.prompt,
        img=None,
        size=size,
        frame_num=args.frame_num,
        shift=args.shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.steps,
        guide_scale=args.guide_scale,
        n_prompt=n_prompt,
        seed=args.seed,
        offload_model=True,
    )

    # ---- Mode A ----
    disable_mode_b_on_wan_model(pipe.model)
    logging.info("Generating Mode A (baseline) ...")
    t0 = time.time()
    video_a = pipe.generate(**gen_kwargs)
    time_a = time.time() - t0
    logging.info("Mode A done in %.1fs shape=%s", time_a, tuple(video_a.shape))

    # ---- Mode B ----
    enable_mode_b_on_wan_model(
        pipe.model,
        schedule=schedule,
        flash_attention_fn=flash_fn,
        enabled=True,
    )
    shift_table = list_layer_time_shifts(
        len(pipe.model.blocks), latent_f, schedule
    )
    logging.info(
        "Generating Mode B (%s) shifts[:8]=%s ...",
        schedule_name,
        shift_table[:8],
    )
    t0 = time.time()
    video_b = pipe.generate(**gen_kwargs)
    time_b = time.time() - t0
    logging.info("Mode B done in %.1fs shape=%s", time_b, tuple(video_b.shape))
    disable_mode_b_on_wan_model(pipe.model)

    frames_a = _tensor_to_uint8_frames(video_a)
    frames_b = _tensor_to_uint8_frames(video_b)
    fps = int(cfg.sample_fps)

    _write_mp4(frames_a, out_dir / "A_baseline.mp4", fps)
    _write_mp4(frames_b, out_dir / "B_rope_roll.mp4", fps)
    _write_mp4(_x3(frames_a), out_dir / "A_x3.mp4", fps)
    _write_mp4(_x3(frames_b), out_dir / "B_x3.mp4", fps)
    _write_png(frames_a[0], out_dir / "A_first.png")
    _write_png(frames_a[-1], out_dir / "A_last.png")
    _write_png(frames_b[0], out_dir / "B_first.png")
    _write_png(frames_b[-1], out_dir / "B_last.png")

    run = {
        "task": "ti2v-5B",
        "ckpt": str(CKPT),
        "wan_root": str(WAN_ROOT),
        "prompt": args.prompt,
        "n_prompt": n_prompt,
        "seed": args.seed,
        "size_wh": list(size),
        "frame_num": args.frame_num,
        "latent_frames_F": latent_f,
        "steps": args.steps,
        "guide_scale": args.guide_scale,
        "shift": args.shift,
        "sample_solver": args.sample_solver,
        "fps": fps,
        "attention_backend": attn_info,
        "mode_b_schedule": schedule_name,
        "mode_b_schedule_cli": args.schedule,
        "mode_b_time_shifts": shift_table,
        "time_a_sec": round(time_a, 2),
        "time_b_sec": round(time_b, 2),
        "video_a_shape": list(video_a.shape),
        "video_b_shape": list(video_b.shape),
        "outputs": [
            "A_baseline.mp4",
            "B_rope_roll.mp4",
            "A_x3.mp4",
            "B_x3.mp4",
            "A_first.png",
            "A_last.png",
            "B_first.png",
            "B_last.png",
        ],
        "note": (
            "Visual check: play A_x3.mp4 and B_x3.mp4; focus on last→first seam. "
            "Default Mode B schedule is SymmetricShiftSchedule."
        ),
    }
    (out_dir / "run.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logging.info("Wrote artifacts to %s", out_dir)
    print(json.dumps({"ok": True, "out_dir": str(out_dir), **{k: run[k] for k in ("latent_frames_F", "steps", "attention_backend", "mode_b_schedule", "time_a_sec", "time_b_sec")}}, indent=2))
    print("GENERATE_OK")


if __name__ == "__main__":
    main()
