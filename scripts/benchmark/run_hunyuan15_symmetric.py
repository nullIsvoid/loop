#!/usr/bin/env python3
"""H1: HunyuanVideo-1.5 + Symmetric Circular Temporal RoPE on person_loop_v1.

Same inputs as H0. Only change: Mode B temporal freqs roll on img Q/K.
See ``notes/hunyuan_h1_rope_call_chain.md``.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_BENCH = Path(__file__).resolve().parent
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_REPO / "src"))

from common import (  # noqa: E402
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_BENCHMARK_DIR,
    default_probe_steps,
    finalize_media,
    load_benchmark,
    prepare_run_dir,
    write_json,
    write_result_stub,
)
from run_hunyuan15_native import (  # noqa: E402
    HY_MODEL_PATH,
    HY_ROOT,
    _aspect_ratio_label,
    _install_latent_probes,
    check_prerequisites,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H1 Hunyuan + Symmetric Circular RoPE")
    p.add_argument("--benchmark-dir", type=str, default=str(DEFAULT_BENCHMARK_DIR))
    p.add_argument(
        "--out-root",
        type=str,
        default=str(
            Path(os.environ.get("LOOP_ARTIFACTS_ROOT", DEFAULT_ARTIFACTS_ROOT))
        ),
    )
    p.add_argument("--run-name", type=str, default="hunyuan15_symmetric")
    p.add_argument("--hy-root", type=str, default=str(HY_ROOT))
    p.add_argument("--model-path", type=str, default=str(HY_MODEL_PATH))
    p.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    p.add_argument("--aspect-ratio", type=str, default="")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--video-length", type=int, default=0)
    p.add_argument("--sr", action="store_true")
    p.add_argument("--rewrite", action="store_true")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--check-only", action="store_true")
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
    write_json(
        out_dir / "prerequisites.json",
        {"model_path": args.model_path, "missing": missing, "ok": len(missing) == 0},
    )
    if args.check_only:
        print(json.dumps({"check_only": True, "missing": missing, "ok": not missing}, indent=2))
        if missing:
            raise SystemExit("H1_PREREQ_MISSING")
        print("H1_CHECK_OK")
        return
    if missing:
        raise SystemExit("H1 missing prerequisites:\n  - " + "\n  - ".join(missing))

    hy_root = Path(args.hy_root)
    sys.path.insert(0, str(hy_root))
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    from hyvideo.commons.infer_state import initialize_infer_state
    from hyvideo.commons.parallel_states import initialize_parallel_state
    from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
    from latent_loop.adapters.hunyuan.attention import (
        disable_mode_b_on_hunyuan_transformer,
        enable_mode_b_on_hunyuan_transformer,
    )
    from latent_loop.rope.schedule import SymmetricShiftSchedule, list_layer_time_shifts

    initialize_parallel_state(sp=int(os.environ.get("WORLD_SIZE", "1")))
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
        "H1 create_pipeline version=%s aspect=%s frames=%s steps=%s",
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

    schedule = SymmetricShiftSchedule()
    # Approximate F from pixel frames (Wan-style 4n+1 → latent); Hunyuan VAE t-factor=4.
    approx_tt = (video_length - 1) // 4 + 1
    pipe.transformer._latent_loop_num_latent_frames = approx_tt  # type: ignore[attr-defined]
    shifts = enable_mode_b_on_hunyuan_transformer(
        pipe.transformer, schedule=schedule, enabled=True
    )
    n_double = len(pipe.transformer.double_blocks)
    n_single = len(pipe.transformer.single_blocks)
    logging.info(
        "H1 Mode B on: double=%s single=%s preview_shifts_head=%s",
        n_double,
        n_single,
        shifts[:8] if shifts else list_layer_time_shifts(8, approx_tt, schedule),
    )

    probe_state = _install_latent_probes(pipe, out_dir, probe_steps)

    t0 = time.time()
    try:
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
    finally:
        disable_mode_b_on_hunyuan_transformer(pipe.transformer)

    elapsed = time.time() - t0
    video = out.videos
    video_c = video[0] if hasattr(video, "ndim") and video.ndim == 5 else video
    media = finalize_media(video_c, out_dir, fps=24)
    probes = probe_state["metas"]
    late = next((p for p in probes if p["tag"] == "late"), None)

    # H0 baseline numbers for in-run comparison note
    h0_late = {
        "median_gap_l2": 54.711273193359375,
        "seam_0->1": 87.00177001953125,
        "seam_F-1->0": 130.13558959960938,
        "max_vs_median": 2.3785882141635972,
    }

    run = {
        "experiment_id": "H1",
        "name": "hunyuan15_symmetric",
        "mode_b_rope": True,
        "rope_schedule": "SymmetricShiftSchedule",
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
            "n_double_blocks": n_double,
            "n_single_blocks": n_single,
            "approx_tt": approx_tt,
            "matched_to_h0": True,
        },
        "h0_baseline_late": h0_late,
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
            "On person_loop_v1, does Symmetric Circular Temporal RoPE reduce "
            "Hunyuan F-1→0 without worsening 0→1 vs H0?"
        ),
    }
    write_json(out_dir / "run.json", run)
    write_result_stub(
        out_dir,
        experiment_id="H1",
        question=run["question"],
        late=run["late"],
        notes=[
            "Same inputs as H0; only Symmetric Circular Temporal RoPE added.",
            "Judge vs H0: F-1→0↓, 0→1 not worse, no new ring walls.",
            "Visual: out_x3 jump/stall/reverse/speed-pop only.",
        ],
    )
    print(json.dumps({"ok": True, "late": run["late"], "h0_baseline_late": h0_late}, indent=2))
    print("H1_OK")


if __name__ == "__main__":
    main()
