#!/usr/bin/env python3
"""H0: HunyuanVideo-1.5 native I2V on fixed loop benchmark.

No Circular RoPE. No conditioning surgery.
Uses official ``HunyuanVideo_1_5_Pipeline``; probes via scheduler.step wrapper.

Cloud defaults:
  HY_ROOT=/root/HunyuanVideo-1.5
  HY_MODEL_PATH=/model/ModelScope/Tencent-Hunyuan/HunyuanVideo-1.5

Requires text_encoder + vision_encoder layout under model_path (see
``notes/hunyuan_h0_runbook.md``). Prompt rewrite is forced OFF so the
benchmark prompt stays identical.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from types import SimpleNamespace
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH))

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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

HY_ROOT = Path(os.environ.get("HY_ROOT", "/root/HunyuanVideo-1.5"))
HY_MODEL_PATH = Path(
    os.environ.get(
        "HY_MODEL_PATH",
        "/model/ModelScope/Tencent-Hunyuan/HunyuanVideo-1.5",
    )
)


def _aspect_ratio_label(width: int, height: int) -> str:
    """Pick nearest common Hunyuan aspect label from image geometry."""
    r = width / max(height, 1)
    candidates = {
        "16:9": 16 / 9,
        "4:3": 4 / 3,
        "1:1": 1.0,
        "3:4": 3 / 4,
        "9:16": 9 / 16,
    }
    best = min(candidates.items(), key=lambda kv: abs(kv[1] - r))
    return best[0]


def _required_paths(model_path: Path) -> list[Path]:
    return [
        model_path / "transformer" / "480p_i2v",
        model_path / "vae",
        model_path / "text_encoder",
        model_path / "vision_encoder",
    ]


def check_prerequisites(model_path: Path) -> list[str]:
    missing = []
    for p in _required_paths(model_path):
        if not p.exists():
            missing.append(str(p))
    return missing


def _install_latent_probes(pipe, probe_dir: Path, probe_steps: tuple[int, ...]):
    """Wrap scheduler.step to dump early/middle/late full-ring gaps."""
    tags = probe_tag_map(probe_steps)
    state = {"i": 0, "metas": []}
    orig_step = pipe.scheduler.step

    def stepped(*args, **kwargs):
        out = orig_step(*args, **kwargs)
        latents = out[0] if isinstance(out, (tuple, list)) else out
        step_i = state["i"]
        if step_i in tags:
            tag = tags[step_i]
            # Hunyuan latents: [B, C, F, H, W]
            logging.info(
                "H0 probe step=%s tag=%s shape=%s",
                step_i,
                tag,
                tuple(latents.shape),
            )
            state["metas"].append(
                save_latent_probe(
                    latents, probe_dir / "latent", tag=tag, step_i=step_i
                )
            )
        state["i"] += 1
        return out

    pipe.scheduler.step = stepped  # type: ignore[method-assign]
    return state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H0 HunyuanVideo-1.5 native benchmark")
    p.add_argument("--benchmark-dir", type=str, default=str(DEFAULT_BENCHMARK_DIR))
    p.add_argument(
        "--out-root",
        type=str,
        default=str(
            Path(os.environ.get("LOOP_ARTIFACTS_ROOT", DEFAULT_ARTIFACTS_ROOT))
        ),
    )
    p.add_argument("--run-name", type=str, default="hunyuan15_native")
    p.add_argument("--hy-root", type=str, default=str(HY_ROOT))
    p.add_argument("--model-path", type=str, default=str(HY_MODEL_PATH))
    p.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    p.add_argument(
        "--aspect-ratio",
        type=str,
        default="",
        help="Default: infer from person.png (usually 9:16).",
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument(
        "--video-length",
        type=int,
        default=0,
        help="0 = use benchmark frame_num (81). Official sweet spot is 121.",
    )
    p.add_argument("--sr", action="store_true", help="Enable SR (default off for H0).")
    p.add_argument(
        "--rewrite",
        action="store_true",
        help="Enable prompt rewrite (default OFF — keep benchmark prompt).",
    )
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Validate benchmark + model layout; do not load GPU weights.",
    )
    return p.parse_args()


def main() -> None:
    import json

    args = parse_args()
    bench = load_benchmark(args.benchmark_dir)
    out_dir = prepare_run_dir(args.out_root, args.run_name, bench)

    from PIL import Image

    img = Image.open(bench.source_path).convert("RGB")
    aspect = args.aspect_ratio or _aspect_ratio_label(img.width, img.height)
    video_length = args.video_length or bench.frame_num
    probe_steps = default_probe_steps(args.steps)

    missing = check_prerequisites(Path(args.model_path))
    prereq = {
        "model_path": args.model_path,
        "missing": missing,
        "ok": len(missing) == 0,
    }
    write_json(out_dir / "prerequisites.json", prereq)

    if args.check_only:
        print(json.dumps({"check_only": True, **prereq, "aspect_ratio": aspect}, indent=2))
        if missing:
            raise SystemExit("H0_PREREQ_MISSING")
        print("H0_CHECK_OK")
        return

    if missing:
        write_result_stub(
            out_dir,
            experiment_id="H0",
            question="Native Hunyuan I2V late seam structure on person_loop_v1?",
            late=None,
            notes=[
                "Blocked: missing text_encoder / vision_encoder (or other) under model_path.",
                f"Missing: {missing}",
                "See notes/hunyuan_h0_runbook.md",
            ],
        )
        raise SystemExit(
            "H0 missing prerequisites:\n  - " + "\n  - ".join(missing)
        )

    hy_root = Path(args.hy_root)
    if not (hy_root / "hyvideo").is_dir():
        raise SystemExit(f"HY_ROOT invalid: {hy_root}")
    sys.path.insert(0, str(hy_root))

    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    from hyvideo.commons.infer_state import initialize_infer_state
    from hyvideo.commons.parallel_states import initialize_parallel_state
    from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline

    initialize_parallel_state(sp=int(os.environ.get("WORLD_SIZE", "1")))
    # Mirror generate.py defaults: no sageattn/cache/compile/fp8 for clean H0.
    infer_args = SimpleNamespace(
        use_sageattn=False,
        sage_blocks_range="0-53",
        enable_torch_compile=False,
        enable_cache=False,
        cache_type="deepcache",
        no_cache_block_id="53",
        cache_start_step=11,
        cache_end_step=45,
        total_steps=args.steps,
        cache_step_interval=4,
        use_fp8_gemm=False,
        quant_type="fp8-per-token-sgl",
        include_patterns="double_blocks",
    )
    infer_state = initialize_infer_state(infer_args)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    task = "i2v"
    transformer_version = HunyuanVideo_1_5_Pipeline.get_transformer_version(
        args.resolution, task, False, False, False
    )
    transformer_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    enable_offloading = True
    offloading_config = HunyuanVideo_1_5_Pipeline.get_offloading_config()
    enable_group_offloading = offloading_config["enable_group_offloading"]
    device = torch.device("cpu")
    transformer_init_device = (
        torch.device("cpu") if enable_group_offloading else device
    )

    logging.info(
        "H0 create_pipeline version=%s aspect=%s frames=%s steps=%s",
        transformer_version,
        aspect,
        video_length,
        args.steps,
    )
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=str(args.model_path),
        transformer_version=transformer_version,
        create_sr_pipeline=bool(args.sr),
        transformer_dtype=transformer_dtype,
        device=device,
        transformer_init_device=transformer_init_device,
    )
    pipe.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=enable_offloading,
        enable_group_offloading=enable_group_offloading,
        overlap_group_offloading=True,
    )

    probe_state = _install_latent_probes(pipe, out_dir, probe_steps)

    t0 = time.time()
    out = pipe(
        enable_sr=bool(args.sr),
        prompt=bench.prompt,
        aspect_ratio=aspect,
        num_inference_steps=args.steps,
        sr_num_inference_steps=None,
        video_length=video_length,
        negative_prompt="",
        seed=bench.seed,
        output_type="pt",
        prompt_rewrite=bool(args.rewrite),
        return_pre_sr_video=False,
        reference_image=str(bench.source_path),
    )
    elapsed = time.time() - t0

    video = out.videos
    if hasattr(video, "ndim") and video.ndim == 5:
        # [B, C, F, H, W] in [0,1]
        video_c = video[0]
    else:
        video_c = video

    media = finalize_media(video_c, out_dir, fps=24)
    probes = probe_state["metas"]
    late = next((p for p in probes if p["tag"] == "late"), None)

    run = {
        "experiment_id": "H0",
        "name": "hunyuan15_native",
        "mode_b_rope": False,
        "conditioning_edits": False,
        "benchmark": bench.as_metadata(),
        "model": {
            "family": "HunyuanVideo-1.5",
            "variant": transformer_version,
            "model_path": args.model_path,
            "hy_root": str(hy_root),
        },
        "model_specific": {
            "resolution": args.resolution,
            "aspect_ratio": aspect,
            "steps": args.steps,
            "video_length": video_length,
            "sr": bool(args.sr),
            "prompt_rewrite": bool(args.rewrite),
            "dtype": args.dtype,
            "probe_steps": list(probe_steps),
            "note_frames": (
                "Official README prefers 121 frames; benchmark uses 81 unless "
                "--video-length overrides."
            ),
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
            "On person_loop_v1, what is native Hunyuan I2V late temporal seam "
            "structure (F-1→0 vs 0→1 vs median)?"
        ),
    }
    write_json(out_dir / "run.json", run)
    write_result_stub(
        out_dir,
        experiment_id="H0",
        question=run["question"],
        late=run["late"],
        notes=[
            "Native non-destructive I2V conditioning only.",
            "No Circular RoPE in this run.",
            "Diagnosis only — not a scoring Gate.",
        ],
    )
    print(json.dumps({"ok": True, "late": run["late"]}, indent=2))
    print("H0_OK")


if __name__ == "__main__":
    main()
