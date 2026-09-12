#!/usr/bin/env python3
"""Loopy-style HunyuanVideo-1.5 anchor-layer probe.

Implements the layer sensitivity measurement described by Loopy:
for one target transformer block l, roll only that block's temporal RoPE by
floor(T_latent/2), compare the denoiser prediction against the native baseline
on the *same transformer input* at every diffusion step, and average MSE:

    alpha_l = mean_n ||v'_n - v_n||^2

The public Loopy inference repository hard-codes the already-measured anchor;
this script reconstructs the measurement for our actual Hunyuan 480p_i2v
runtime instead of assuming the paper's layer index transfers unchanged.

Important:
- Native Hunyuan I2V conditioning is unchanged.
- No scheduler/latent surgery.
- Baseline transformer inputs/outputs are cached once.
- During a perturbed run, the transformer input at every step is replaced by
  the matching cached baseline input. This prevents trajectory divergence from
  contaminating alpha_l.
- Default benchmark is person_loop_v1.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_BENCH = Path(__file__).resolve().parent
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_REPO / "src"))

from common import (  # noqa: E402
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_BENCHMARK_DIR,
    load_benchmark,
    prepare_run_dir,
    write_json,
)
from run_hunyuan15_native import (  # noqa: E402
    HY_MODEL_PATH,
    HY_ROOT,
    _aspect_ratio_label,
    check_prerequisites,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)


@dataclass(frozen=True)
class SingleLayerHalfShiftSchedule:
    """Shift only target block by floor(T_latent/2); all other blocks stay native."""

    target_block: int

    def time_shift(self, block_idx: int, num_latent_frames: int) -> int:
        if num_latent_frames < 2:
            raise ValueError("num_latent_frames must be >= 2")
        if block_idx != self.target_block:
            return 0
        return int(num_latent_frames // 2)


def _parse_blocks(spec: str, n_blocks: int) -> list[int]:
    text = spec.strip().lower()
    if text in ("", "all"):
        return list(range(n_blocks))
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_s, b_s = part.split("-", 1)
            a, b = int(a_s), int(b_s)
            if b < a:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    out = sorted(set(out))
    bad = [x for x in out if x < 0 or x >= n_blocks]
    if bad:
        raise ValueError(f"block ids out of range 0..{n_blocks - 1}: {bad}")
    return out


def _tensor_from_output(output: Any):
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "sample"):
        return output.sample
    return output


def _save_tensor(path: Path, tensor) -> dict:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve exact dtype (typically bf16). Casting the native reference would
    # itself create a small artificial prediction error.
    x = tensor.detach().to(device="cpu")
    torch.save(x, path)
    return {
        "path": str(path),
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "numel": int(x.numel()),
    }


def _install_baseline_cache_hooks(transformer, cache_dir: Path):
    """Capture exact transformer input/output for every native diffusion step."""
    state = {"step": 0, "inputs": [], "outputs": []}

    def pre_hook(_module, args, kwargs):
        step = int(state["step"])
        if not args:
            raise RuntimeError("anchor probe expected transformer first positional arg")
        meta = _save_tensor(cache_dir / "inputs" / f"step_{step:03d}.pt", args[0])
        state["inputs"].append(meta)
        return args, kwargs

    def post_hook(_module, args, kwargs, output):
        step = int(state["step"])
        pred = _tensor_from_output(output)
        meta = _save_tensor(cache_dir / "outputs" / f"step_{step:03d}.pt", pred)
        state["outputs"].append(meta)
        state["step"] = step + 1
        return output

    h_pre = transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h_post = transformer.register_forward_hook(post_hook, with_kwargs=True)
    return state, (h_pre, h_post)


def _install_frozen_input_alpha_hooks(transformer, cache_dir: Path):
    """Feed cached native input each step and measure prediction MSE vs native output."""
    import torch

    state: dict[str, Any] = {"step": 0, "mse_by_step": []}

    def pre_hook(_module, args, kwargs):
        step = int(state["step"])
        input_path = cache_dir / "inputs" / f"step_{step:03d}.pt"
        if not input_path.is_file():
            raise RuntimeError(f"missing baseline transformer input: {input_path}")
        if not args:
            raise RuntimeError("anchor probe expected transformer first positional arg")
        native_input = torch.load(input_path, map_location="cpu")
        current = args[0]
        native_input = native_input.to(device=current.device, dtype=current.dtype)
        args = (native_input, *args[1:])
        return args, kwargs

    def post_hook(_module, args, kwargs, output):
        step = int(state["step"])
        output_path = cache_dir / "outputs" / f"step_{step:03d}.pt"
        if not output_path.is_file():
            raise RuntimeError(f"missing baseline transformer output: {output_path}")
        pred = _tensor_from_output(output)
        native_pred = torch.load(output_path, map_location="cpu")
        native_pred = native_pred.to(device=pred.device, dtype=pred.dtype)
        mse = (pred.float() - native_pred.float()).square().mean().item()
        state["mse_by_step"].append(float(mse))
        state["step"] = step + 1
        return output

    h_pre = transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h_post = transformer.register_forward_hook(post_hook, with_kwargs=True)
    return state, (h_pre, h_post)


def _remove_handles(handles) -> None:
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass


def _run_pipeline_latent(
    pipe,
    *,
    bench,
    aspect: str,
    steps: int,
    video_length: int,
    sr: bool,
    rewrite: bool,
):
    return pipe(
        enable_sr=bool(sr),
        prompt=bench.prompt,
        aspect_ratio=aspect,
        num_inference_steps=steps,
        sr_num_inference_steps=None,
        video_length=video_length,
        negative_prompt="",
        seed=bench.seed,
        output_type="latent",
        prompt_rewrite=bool(rewrite),
        return_pre_sr_video=False,
        reference_image=str(bench.source_path),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Loopy-style HunyuanVideo-1.5 per-layer anchor probe"
    )
    p.add_argument("--benchmark-dir", type=str, default=str(DEFAULT_BENCHMARK_DIR))
    p.add_argument(
        "--out-root",
        type=str,
        default=str(Path(os.environ.get("LOOP_ARTIFACTS_ROOT", DEFAULT_ARTIFACTS_ROOT))),
    )
    p.add_argument("--run-name", type=str, default="hunyuan15_anchor_probe")
    p.add_argument("--hy-root", type=str, default=str(HY_ROOT))
    p.add_argument("--model-path", type=str, default=str(HY_MODEL_PATH))
    p.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    p.add_argument("--aspect-ratio", type=str, default="")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--video-length", type=int, default=0)
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    p.add_argument(
        "--blocks",
        type=str,
        default="all",
        help='Blocks to measure: "all", "0,1,2", or ranges such as "0-10,20".',
    )
    p.add_argument("--sr", action="store_true")
    p.add_argument(
        "--rewrite",
        action="store_true",
        help="Keep OFF for person_loop_v1 so prompt is identical to H0/H1.",
    )
    p.add_argument(
        "--capture-only",
        action="store_true",
        help="Only create/recreate native baseline transformer cache.",
    )
    p.add_argument(
        "--force-baseline",
        action="store_true",
        help="Delete and recapture native transformer input/output cache.",
    )
    p.add_argument(
        "--force-blocks",
        action="store_true",
        help="Re-run block probes even if block_XXX.json already exists.",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Validate model/benchmark and print runtime block topology only.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bench = load_benchmark(args.benchmark_dir)
    out_dir = prepare_run_dir(args.out_root, args.run_name, bench)
    cache_dir = out_dir / "native_cache"
    block_dir = out_dir / "blocks"
    block_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    img = Image.open(bench.source_path).convert("RGB")
    aspect = args.aspect_ratio or _aspect_ratio_label(img.width, img.height)
    video_length = args.video_length or bench.frame_num

    missing = check_prerequisites(Path(args.model_path))
    write_json(
        out_dir / "prerequisites.json",
        {"model_path": args.model_path, "missing": missing, "ok": not missing},
    )
    if missing:
        raise SystemExit("anchor probe missing prerequisites:\n  - " + "\n  - ".join(missing))

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
    from latent_loop.adapters.hunyuan.attention import (
        disable_mode_b_on_hunyuan_transformer,
        enable_mode_b_on_hunyuan_transformer,
    )

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
    transformer_init_device = torch.device("cpu") if enable_group_offloading else device

    logging.info(
        "anchor probe create_pipeline version=%s aspect=%s frames=%s steps=%s",
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

    n_double = len(getattr(pipe.transformer, "double_blocks", []))
    n_single = len(getattr(pipe.transformer, "single_blocks", []))
    n_blocks = n_double + n_single
    blocks = _parse_blocks(args.blocks, n_blocks)
    topology = {
        "transformer_version": transformer_version,
        "n_double_blocks": n_double,
        "n_single_blocks": n_single,
        "n_blocks": n_blocks,
        "requested_blocks": blocks,
        "benchmark": bench.as_metadata(),
        "steps": args.steps,
        "video_length": video_length,
        "aspect_ratio": aspect,
        "method": (
            "Loopy-style: single-layer temporal RoPE shift=floor(T_latent/2); "
            "alpha_l=mean stepwise denoiser-output MSE on frozen native inputs"
        ),
    }
    signature = {
        "transformer_version": transformer_version,
        "benchmark": bench.as_metadata(),
        "steps": args.steps,
        "video_length": video_length,
        "aspect_ratio": aspect,
        "dtype": args.dtype,
        "sr": bool(args.sr),
        "prompt_rewrite": bool(args.rewrite),
    }
    topology["signature"] = signature
    write_json(out_dir / "topology.json", topology)

    if args.check_only:
        print(json.dumps(topology, indent=2, ensure_ascii=False))
        print("ANCHOR_PROBE_CHECK_OK")
        return

    cache_meta_path = cache_dir / "meta.json"
    if args.force_baseline and cache_dir.exists():
        shutil.rmtree(cache_dir)

    expected_cache_files = [
        cache_dir / "inputs" / f"step_{i:03d}.pt" for i in range(args.steps)
    ] + [
        cache_dir / "outputs" / f"step_{i:03d}.pt" for i in range(args.steps)
    ]
    cache_ready = False
    if cache_meta_path.is_file() and all(p.is_file() for p in expected_cache_files):
        try:
            with cache_meta_path.open("r", encoding="utf-8") as f:
                old_cache_meta = json.load(f)
            cache_ready = old_cache_meta.get("signature") == signature
        except Exception:
            cache_ready = False

    if not cache_ready:
        logging.info("capturing native transformer cache for %s diffusion steps", args.steps)
        cache_dir.mkdir(parents=True, exist_ok=True)
        state, handles = _install_baseline_cache_hooks(pipe.transformer, cache_dir)
        t0 = time.time()
        try:
            _run_pipeline_latent(
                pipe,
                bench=bench,
                aspect=aspect,
                steps=args.steps,
                video_length=video_length,
                sr=args.sr,
                rewrite=args.rewrite,
            )
        finally:
            _remove_handles(handles)
        if int(state["step"]) != args.steps:
            raise RuntimeError(
                f"native cache expected {args.steps} transformer calls, got {state['step']}"
            )
        cache_meta = {
            **topology,
            "captured_steps": int(state["step"]),
            "elapsed_sec": round(time.time() - t0, 2),
            "input_files": state["inputs"],
            "output_files": state["outputs"],
        }
        write_json(cache_meta_path, cache_meta)
    else:
        logging.info("reusing native cache: %s", cache_dir)

    if args.capture_only:
        print(json.dumps({"ok": True, "cache": str(cache_dir)}, indent=2))
        print("ANCHOR_PROBE_CAPTURE_OK")
        return

    results: list[dict[str, Any]] = []
    for block_idx in blocks:
        result_path = block_dir / f"block_{block_idx:03d}.json"
        if result_path.is_file() and not args.force_blocks:
            with result_path.open("r", encoding="utf-8") as f:
                result = json.load(f)
            if result.get("signature") == signature:
                results.append(result)
                logging.info(
                    "skip block=%s existing alpha=%s",
                    block_idx,
                    result.get("alpha"),
                )
                continue
            logging.info("block=%s cached result signature changed; re-running", block_idx)

        logging.info(
            "probe block=%s/%s with single-layer half-cycle temporal RoPE shift",
            block_idx,
            n_blocks - 1,
        )
        schedule = SingleLayerHalfShiftSchedule(target_block=block_idx)
        enable_mode_b_on_hunyuan_transformer(
            pipe.transformer,
            schedule=schedule,
            enabled=True,
        )
        alpha_state, alpha_handles = _install_frozen_input_alpha_hooks(
            pipe.transformer, cache_dir
        )
        t0 = time.time()
        try:
            _run_pipeline_latent(
                pipe,
                bench=bench,
                aspect=aspect,
                steps=args.steps,
                video_length=video_length,
                sr=args.sr,
                rewrite=args.rewrite,
            )
        finally:
            _remove_handles(alpha_handles)
            disable_mode_b_on_hunyuan_transformer(pipe.transformer)

        mse_by_step = [float(x) for x in alpha_state["mse_by_step"]]
        if len(mse_by_step) != args.steps:
            raise RuntimeError(
                f"block {block_idx}: expected {args.steps} MSE samples, got {len(mse_by_step)}"
            )
        alpha = float(sum(mse_by_step) / len(mse_by_step))
        result = {
            "block_idx": block_idx,
            "alpha": alpha,
            "signature": signature,
            "half_shift_rule": "floor(T_latent/2)",
            "mse_by_step": mse_by_step,
            "n_steps": len(mse_by_step),
            "elapsed_sec": round(time.time() - t0, 2),
        }
        write_json(result_path, result)
        results.append(result)
        logging.info("block=%s alpha=%.8g", block_idx, alpha)

    results.sort(key=lambda x: int(x["block_idx"]))
    ranked = sorted(results, key=lambda x: float(x["alpha"]), reverse=True)
    summary = {
        **topology,
        "completed_blocks": [int(r["block_idx"]) for r in results],
        "anchor_layer": None if not ranked else int(ranked[0]["block_idx"]),
        "anchor_alpha": None if not ranked else float(ranked[0]["alpha"]),
        "ranked": [
            {"block_idx": int(r["block_idx"]), "alpha": float(r["alpha"])}
            for r in ranked
        ],
        "alpha_by_block": {
            str(int(r["block_idx"])): float(r["alpha"]) for r in results
        },
        "warning": (
            "This run uses person_loop_v1 only. Loopy paper averages over many prompts; "
            "repeat across prompts before treating the anchor index as universal."
        ),
    }
    write_json(out_dir / "alpha_summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("ANCHOR_PROBE_OK")


if __name__ == "__main__":
    main()
