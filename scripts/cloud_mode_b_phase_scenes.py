#!/usr/bin/env python3
"""Phase-clear scenes × S0–S3 schedule geometry (seed 42).

Scenes: pendulum, rotating_fan, human_sway
Schedules: Identity / Loopy / Fixed(1) / Symmetric

Same size/frames/steps/CFG as candle round. Full mp4s on cloud; x3 for review.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud_mode_b_generate import (  # noqa: E402
    CKPT,
    LOOP_SRC,
    WAN_ROOT,
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
        "MODE_B_PHASE_OUT",
        "/root/latent-loop/artifacts/mode_b_phase_scenes",
    )
)

SEED = 42
SIZE = (1280, 704)
FRAME_NUM = 81
STEPS = 20
GUIDE = 5.0
SHIFT = 5.0
SOLVER = "unipc"

SCENES = [
    (
        "pendulum",
        "A classic pendulum clock bob swinging left and right in a continuous "
        "periodic arc, fixed camera, steady period, clear left-right motion and "
        "velocity reversal at each end, seamless looping live wallpaper, no cuts",
    ),
    (
        "rotating_fan",
        "A desk fan with three blades spinning at constant speed, continuous "
        "rotation with unbroken rotational phase, fixed camera, seamless looping "
        "live wallpaper, no cuts, no speed changes",
    ),
    (
        "human_sway",
        "A young woman standing in place, fixed camera, gently swaying her upper "
        "body left and right in a clear periodic rhythm, long hair swinging with "
        "the same cycle, seamless looping live wallpaper, no walking, no cuts",
    ),
]


def _schedules():
    from latent_loop.rope.schedule import (
        FixedShiftSchedule,
        IdentityShiftSchedule,
        LoopyShiftSchedule,
        SymmetricShiftSchedule,
    )

    return [
        ("S0_identity", IdentityShiftSchedule()),
        ("S1_loopy", LoopyShiftSchedule()),
        ("S2_fixed1", FixedShiftSchedule(1)),
        ("S3_symmetric", SymmetricShiftSchedule()),
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
    cfg = WAN_CONFIGS["ti2v-5B"]
    n_prompt = cfg.sample_neg_prompt
    fps = int(cfg.sample_fps)

    schedule_tables = {
        key: list_layer_time_shifts(30, latent_f, sched)
        for key, sched in _schedules()
    }
    (out_dir / "schedule_tables.json").write_text(
        json.dumps({"latent_frames_F": latent_f, "tables": schedule_tables}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    logging.info("Building WanTI2V once ...")
    t0 = time.time()
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
    logging.info("ready in %.1fs", time.time() - t0)

    runs = []
    for scene_id, prompt in SCENES:
        for sched_key, sched in _schedules():
            run_id = f"{scene_id}__{sched_key}"
            disable_mode_b_on_wan_model(pipe.model)
            if type(sched).__name__ != "IdentityShiftSchedule":
                enable_mode_b_on_wan_model(
                    pipe.model,
                    schedule=sched,
                    flash_attention_fn=flash_fn,
                    enabled=True,
                )
            logging.info("Generating %s ...", run_id)
            t1 = time.time()
            video = pipe.generate(
                input_prompt=prompt,
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
            elapsed = time.time() - t1
            disable_mode_b_on_wan_model(pipe.model)

            frames = _tensor_to_uint8_frames(video)
            _write_mp4(frames, out_dir / f"{run_id}.mp4", fps)
            _write_mp4(_x3(frames), out_dir / f"{run_id}_x3.mp4", fps)
            _write_png(frames[0], out_dir / f"{run_id}_first.png")
            _write_png(frames[-1], out_dir / f"{run_id}_last.png")

            runs.append(
                {
                    "id": run_id,
                    "scene": scene_id,
                    "schedule": sched_key,
                    "prompt": prompt,
                    "time_sec": round(elapsed, 2),
                    "shape": list(video.shape),
                    "cloud_x3": str(out_dir / f"{run_id}_x3.mp4"),
                }
            )
            logging.info("%s done in %.1fs", run_id, elapsed)

    experiment = {
        "name": "mode_b_phase_scenes_s0_s3",
        "research_question": (
            "Is Loopy layer-wise shift necessary, or does any non-zero temporal "
            "RoPE roll close the loop on phase-clear motion?"
        ),
        "shared": {
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
            "n_prompt": n_prompt,
        },
        "scenes": [{"id": s, "prompt": p} for s, p in SCENES],
        "schedule_tables": schedule_tables,
        "runs": runs,
        "git_policy": "Commit x3 previews + metadata only; full mp4s stay on cloud.",
        "human_check": (
            "For each scene, watch S0–S3 x3; score seam on phase continuity "
            "(pendulum direction, fan rotation phase, human sway)."
        ),
    }
    (out_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ok": True, "n_runs": len(runs), "out_dir": str(out_dir)}, indent=2))
    print("PHASE_SCENES_OK")


if __name__ == "__main__":
    main()
