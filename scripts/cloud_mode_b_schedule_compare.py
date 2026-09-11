#!/usr/bin/env python3
"""Compare four temporal shift geometries on official Wan2.2 TI2V-5B.

S0 Identity | S1 Loopy | S2 FixedShift(1) | S3 Symmetric

Same prompt / seed / F / steps / resolution. Full videos stay on cloud;
repo should only keep metadata + RESULT (+ optional small previews).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

# Reuse helpers from the real generate script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud_mode_b_generate import (  # noqa: E402
    CKPT,
    LOOP_SRC,
    WAN_ROOT,
    _bootstrap_wan_namespace,
    _install_official_attention_backend,
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

OUT_DIR = Path(
    os.environ.get(
        "MODE_B_SCHED_OUT",
        "/root/latent-loop/artifacts/mode_b_schedules",
    )
)

PROMPT = (
    "A seamless looping live wallpaper of a candle flame gently flickering "
    "and swaying in a continuous periodic cycle, fixed camera, no cuts, "
    "smooth looping motion, cinematic lighting"
)
SEED = 42
SIZE = (1280, 704)
FRAME_NUM = 81
STEPS = 20
GUIDE = 5.0
SHIFT = 5.0
SOLVER = "unipc"


def _schedules():
    from latent_loop.rope.schedule import (
        FixedShiftSchedule,
        IdentityShiftSchedule,
        LoopyShiftSchedule,
        SymmetricShiftSchedule,
    )

    return [
        ("S0_identity", IdentityShiftSchedule(), "Identity — all zeros (baseline)"),
        ("S1_loopy", LoopyShiftSchedule(), "Loopy monotonic — current winner"),
        ("S2_fixed1", FixedShiftSchedule(1), "FixedShift(1) — constant +1 after anchor"),
        (
            "S3_symmetric",
            SymmetricShiftSchedule(),
            "Symmetric bidirectional — 0,+1,-1,+2,-2,... (mod F)",
        ),
    ]


def main() -> None:
    attn_info = _install_official_attention_backend()
    flash_fn = attn_info.pop("flash_fn")

    from wan.configs import WAN_CONFIGS
    from wan.textimage2video import WanTI2V
    from latent_loop.adapters.wan.attention import (
        disable_mode_b_on_wan_model,
        enable_mode_b_on_wan_model,
    )
    from latent_loop.rope.schedule import list_layer_time_shifts

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    latent_f = (FRAME_NUM - 1) // 4 + 1
    num_layers = 30

    cfg = WAN_CONFIGS["ti2v-5B"]
    n_prompt = cfg.sample_neg_prompt

    schedule_specs = []
    for key, sched, desc in _schedules():
        table = list_layer_time_shifts(num_layers, latent_f, sched)
        schedule_specs.append(
            {
                "id": key,
                "description": desc,
                "class": type(sched).__name__,
                "time_shifts": table,
            }
        )
    (out_dir / "schedule_tables.json").write_text(
        json.dumps(
            {"latent_frames_F": latent_f, "num_layers": num_layers, "schedules": schedule_specs},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    logging.info("Building WanTI2V once ...")
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
    logging.info("ready in %.1fs", time.time() - t_build)

    gen_kwargs = dict(
        input_prompt=PROMPT,
        img=None,
        size=SIZE,
        frame_num=FRAME_NUM,
        shift=SHIFT,
        sample_solver=SOLVER,
        sampling_steps=STEPS,
        guide_scale=GUIDE,
        n_prompt=n_prompt,
        seed=SEED,
        offload_model=True,
    )

    runs = []
    fps = int(cfg.sample_fps)
    for key, sched, desc in _schedules():
        disable_mode_b_on_wan_model(pipe.model)
        # Identity: leave baseline attention (no Mode B patch).
        if type(sched).__name__ != "IdentityShiftSchedule":
            enable_mode_b_on_wan_model(
                pipe.model,
                schedule=sched,
                flash_attention_fn=flash_fn,
                enabled=True,
            )
        logging.info("Generating %s (%s) ...", key, desc)
        t0 = time.time()
        video = pipe.generate(**gen_kwargs)
        elapsed = time.time() - t0
        disable_mode_b_on_wan_model(pipe.model)

        frames = _tensor_to_uint8_frames(video)
        # Full videos on cloud only (not for git).
        _write_mp4(frames, out_dir / f"{key}.mp4", fps)
        _write_mp4(_x3(frames), out_dir / f"{key}_x3.mp4", fps)
        _write_png(frames[0], out_dir / f"{key}_first.png")
        _write_png(frames[-1], out_dir / f"{key}_last.png")

        entry = {
            "id": key,
            "description": desc,
            "schedule_class": type(sched).__name__,
            "time_sec": round(elapsed, 2),
            "shape": list(video.shape),
            "cloud_mp4": str(out_dir / f"{key}.mp4"),
            "cloud_x3": str(out_dir / f"{key}_x3.mp4"),
        }
        runs.append(entry)
        logging.info("%s done in %.1fs", key, elapsed)

    experiment = {
        "name": "mode_b_schedule_geometry_s0_s3",
        "goal": (
            "Compare shift schedule geometries; ask whether Loopy monotonic "
            "layer shift is optimal vs Fixed(1) vs Symmetric bidirectional."
        ),
        "shared": {
            "prompt": PROMPT,
            "n_prompt": n_prompt,
            "seed": SEED,
            "size_wh": list(SIZE),
            "frame_num": FRAME_NUM,
            "latent_frames_F": latent_f,
            "steps": STEPS,
            "guide_scale": GUIDE,
            "shift": SHIFT,
            "sample_solver": SOLVER,
            "ckpt": str(CKPT),
            "attention_backend": attn_info,
        },
        "schedules": schedule_specs,
        "runs": runs,
        "git_policy": (
            "Do not commit full mp4s from this experiment. Commit experiment.json, "
            "schedule_tables.json, RESULT.md, and selected first/last PNG previews only."
        ),
        "human_check": (
            "Watch S0_x3 / S1_x3 / S2_x3 / S3_x3 on cloud; pick winner for last→first seam."
        ),
    }
    (out_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, "out_dir": str(out_dir), "runs": [r["id"] for r in runs]}, indent=2))
    print("SCHEDULE_COMPARE_OK")


if __name__ == "__main__":
    main()
