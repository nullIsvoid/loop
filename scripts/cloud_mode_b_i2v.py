#!/usr/bin/env python3
"""Mode A/B Image-to-Video on official Wan2.2 TI2V-5B.

Same TI2V-5B checkpoint as T2V: pass ``img=PIL.Image`` → official I2V path.
A = baseline Wan I2V
B = Wan I2V + Circular Temporal RoPE (default SymmetricShiftSchedule)

First product question: keep wallpaper identity/composition while adding
gentle motion and a loopable last→first seam.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

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

sys.path.insert(0, str(WAN_ROOT))
sys.path.insert(0, str(LOOP_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

OUT_ROOT = Path(
    os.environ.get(
        "MODE_B_I2V_OUT",
        "/root/latent-loop/artifacts/mode_b_i2v",
    )
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wan TI2V-5B Mode A/B I2V")
    p.add_argument("--image", type=str, required=True, help="input wallpaper path")
    p.add_argument("--case-id", type=str, required=True, help="output subfolder name")
    p.add_argument(
        "--prompt",
        type=str,
        default=(
            "Subtle seamless looping live wallpaper motion, fixed camera, "
            "preserve identity and composition, gentle continuous movement, no cuts"
        ),
    )
    p.add_argument("--n-prompt", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-area",
        type=int,
        default=704 * 1280,
        help="official I2V max_area (resolution from image aspect)",
    )
    p.add_argument("--frame-num", type=int, default=81, help="must be 4n+1")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guide-scale", type=float, default=5.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--sample-solver", type=str, default="unipc")
    p.add_argument(
        "--schedule",
        type=str,
        default="symmetric",
        help="Mode B schedule: symmetric|loopy|fixed1|identity",
    )
    p.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert args.frame_num % 4 == 1, "frame_num must be 4n+1"

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"image not found: {image_path}")

    out_dir = Path(args.out_root) / args.case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    input_image = Image.open(image_path).convert("RGB")
    source_path = out_dir / "source.png"
    input_image.save(source_path)
    # Keep original filename meta if different
    if image_path.resolve() != source_path.resolve():
        shutil.copy2(image_path, out_dir / f"source_original{image_path.suffix.lower()}")

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

    logging.info("Building WanTI2V once for I2V ...")
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
        img=input_image,
        max_area=args.max_area,
        frame_num=args.frame_num,
        shift=args.shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.steps,
        guide_scale=args.guide_scale,
        n_prompt=n_prompt,
        seed=args.seed,
        offload_model=True,
    )

    disable_mode_b_on_wan_model(pipe.model)
    logging.info("Generating Mode A baseline I2V ...")
    t0 = time.time()
    video_a = pipe.generate(**gen_kwargs)
    time_a = time.time() - t0
    logging.info("Mode A done in %.1fs shape=%s", time_a, tuple(video_a.shape))

    enable_mode_b_on_wan_model(
        pipe.model,
        schedule=schedule,
        flash_attention_fn=flash_fn,
        enabled=True,
    )
    shift_table = list_layer_time_shifts(len(pipe.model.blocks), latent_f, schedule)
    logging.info(
        "Generating Mode B I2V (%s) shifts[:8]=%s ...",
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
    _write_mp4(frames_b, out_dir / "B_symmetric.mp4", fps)
    _write_mp4(_x3(frames_a), out_dir / "A_x3.mp4", fps)
    _write_mp4(_x3(frames_b), out_dir / "B_x3.mp4", fps)
    _write_png(frames_a[0], out_dir / "A_first.png")
    _write_png(frames_a[-1], out_dir / "A_last.png")
    _write_png(frames_b[0], out_dir / "B_first.png")
    _write_png(frames_b[-1], out_dir / "B_last.png")

    run = {
        "task": "ti2v-5B-i2v",
        "path": "image_to_video",
        "ckpt": str(CKPT),
        "case_id": args.case_id,
        "source_image": str(image_path),
        "source_size_wh": [input_image.width, input_image.height],
        "prompt": args.prompt,
        "n_prompt": n_prompt,
        "seed": args.seed,
        "max_area": args.max_area,
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
            "source.png",
            "A_baseline.mp4",
            "B_symmetric.mp4",
            "A_x3.mp4",
            "B_x3.mp4",
            "A_first.png",
            "A_last.png",
            "B_first.png",
            "B_last.png",
        ],
        "human_check": (
            "Compare A_x3 vs B_x3: last→first seam AND identity/composition "
            "vs source.png (hair/cloth or env motion only)."
        ),
    }
    (out_dir / "run.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, "out_dir": str(out_dir), **{k: run[k] for k in (
        "case_id", "mode_b_schedule", "time_a_sec", "time_b_sec", "video_a_shape"
    )}}, indent=2))
    print("I2V_OK")


if __name__ == "__main__":
    main()
