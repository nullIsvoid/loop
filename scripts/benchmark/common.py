#!/usr/bin/env python3
"""Shared loop benchmark loader + latent full-ring probe helpers.

All native baseline scripts must load source/prompt/seed/frame_num from
``assets/loop_benchmark/`` — never hardcode a parallel prompt/source.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "assets" / "loop_benchmark"
DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "loop_benchmark_v1"

PROBE_TAGS = ("early", "middle", "late")


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    root: Path
    source_path: Path
    prompt: str
    seed: int
    frame_num: int
    motion_intent: str
    camera: str
    loop_goal: str
    source_sha256: str | None
    raw: dict[str, Any]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "source": str(self.source_path),
            "source_sha256": self.source_sha256,
            "prompt": self.prompt,
            "seed": self.seed,
            "frame_num": self.frame_num,
            "motion_intent": self.motion_intent,
            "camera": self.camera,
            "loop_goal": self.loop_goal,
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_benchmark(benchmark_dir: Path | str | None = None) -> BenchmarkSpec:
    root = Path(benchmark_dir) if benchmark_dir else DEFAULT_BENCHMARK_DIR
    meta_path = root / "benchmark.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing benchmark.json: {meta_path}")
    raw = json.loads(meta_path.read_text(encoding="utf-8"))

    source_name = raw.get("source", "person.png")
    prompt_name = raw.get("prompt_file", "person_prompt.txt")
    source_path = root / source_name
    prompt_path = root / prompt_name
    if not source_path.is_file():
        raise FileNotFoundError(f"missing source image: {source_path}")
    if not prompt_path.is_file():
        raise FileNotFoundError(f"missing prompt file: {prompt_path}")

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"empty prompt: {prompt_path}")

    seed = int(raw["seed"])
    frame_num = int(raw["frame_num"])
    if frame_num < 1:
        raise ValueError(f"invalid frame_num={frame_num}")

    digest = sha256_file(source_path)
    expected = raw.get("source_sha256")
    if expected and digest != expected:
        raise ValueError(
            f"person.png sha256 mismatch: got {digest}, expected {expected}. "
            "Bump benchmark_id if the image was intentionally replaced."
        )

    return BenchmarkSpec(
        benchmark_id=str(raw["benchmark_id"]),
        root=root.resolve(),
        source_path=source_path.resolve(),
        prompt=prompt,
        seed=seed,
        frame_num=frame_num,
        motion_intent=str(raw.get("motion_intent", "")),
        camera=str(raw.get("camera", "")),
        loop_goal=str(raw.get("loop_goal", "")),
        source_sha256=digest,
        raw=raw,
    )


def default_probe_steps(sampling_steps: int) -> tuple[int, ...]:
    if sampling_steps < 3:
        raise ValueError(f"need >=3 steps for early/middle/late, got {sampling_steps}")
    early = min(2, sampling_steps - 1)
    middle = max(early, sampling_steps // 2 - 1)
    late = sampling_steps - 1
    return tuple(sorted({early, middle, late}))


def probe_tag_map(probe_steps: tuple[int, ...]) -> dict[int, str]:
    steps = sorted(probe_steps)
    if len(steps) == 1:
        return {steps[0]: "late"}
    if len(steps) == 2:
        return {steps[0]: "early", steps[1]: "late"}
    # first / middle / last among selected
    mid = steps[len(steps) // 2]
    return {steps[0]: "early", mid: "middle", steps[-1]: "late"}


def _as_cfhw(latent: Any) -> Any:
    """Normalize to [C, F, H, W] for ring gaps."""
    import torch

    x = latent.detach().float().cpu()
    if x.ndim == 5:
        # [B, C, F, H, W] or [B, C, H, W, F] — Hunyuan uses B,C,F,H,W
        if x.shape[0] != 1:
            raise ValueError(f"expected batch=1 latent, got {tuple(x.shape)}")
        x = x[0]
    if x.ndim != 4:
        raise ValueError(f"expected 4D/5D latent, got {tuple(x.shape)}")
    # Prefer [C, F, H, W]: F is usually much smaller than H/W
    c0, a, b, c3 = x.shape
    # If looks like [C, H, W, F] with F small on last dim and H≈W large — rare; keep Wan/Hunyuan convention C,F,H,W
    _ = (c0, a, b, c3)
    return x


def full_ring_adjacent_gaps(latent: Any) -> dict[str, Any]:
    import torch

    x = _as_cfhw(latent)
    _c, f, _h, _w = x.shape
    diffs: dict[str, float] = {}
    vals: list[float] = []
    for i in range(f):
        j = (i + 1) % f
        d = float(torch.norm(x[:, i] - x[:, j]).item())
        diffs[f"{i}->{j}"] = d
        vals.append(d)
    max_i = int(np.argmax(vals))
    med = float(np.median(vals))
    return {
        "F": f,
        "diffs_l2": diffs,
        "max_gap_l2": float(vals[max_i]),
        "max_gap_edge": f"{max_i}->{(max_i + 1) % f}",
        "median_gap_l2": med,
        "mean_gap_l2": float(np.mean(vals)),
        "max_vs_median": float(vals[max_i]) / max(med, 1e-6),
        "seam_F-1->0": diffs[f"{f - 1}->0"],
        "seam_0->1": diffs["0->1"],
    }


def save_latent_probe(
    latent: Any,
    out_dir: Path,
    *,
    tag: str,
    step_i: int,
) -> dict[str, Any]:
    """Write ``{tag}_full_latent.pt`` + ``{tag}_full_ring_gaps.json``."""
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x = _as_cfhw(latent).contiguous()
    pt_path = out_dir / f"{tag}_full_latent.pt"
    torch.save(
        {
            "step": step_i,
            "tag": tag,
            "latent": x,
            "latent_shape": list(x.shape),
        },
        pt_path,
    )
    meta = {
        "step": step_i,
        "tag": tag,
        "full_latent_pt": str(pt_path),
        **full_ring_adjacent_gaps(x),
    }
    (out_dir / f"{tag}_full_ring_gaps.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def prepare_run_dir(
    artifacts_root: Path | str,
    run_name: str,
    bench: BenchmarkSpec,
) -> Path:
    out_dir = Path(artifacts_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latent").mkdir(parents=True, exist_ok=True)
    dest = out_dir / "source.png"
    shutil.copy2(bench.source_path, dest)
    return out_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_result_stub(
    out_dir: Path,
    *,
    experiment_id: str,
    question: str,
    late: dict[str, Any] | None,
    notes: list[str],
) -> None:
    lines = [
        f"# {experiment_id} — RESULT",
        "",
        "## Question",
        "",
        question,
        "",
        "## Late seam (diagnosis only)",
        "",
    ]
    if late is None:
        lines.append("_No late probe yet._")
    else:
        lines.extend(
            [
                "| metric | value |",
                "|--------|------:|",
                f"| median | {late.get('median_gap_l2')} |",
                f"| `0→1` | {late.get('seam_0->1')} |",
                f"| `F-1→0` | {late.get('seam_F-1->0')} |",
                f"| max edge | {late.get('max_gap_edge')} |",
                f"| max/median | {late.get('max_vs_median')} |",
                "",
            ]
        )
    lines.append("## Notes")
    lines.append("")
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")
    (out_dir / "RESULT.md").write_text("\n".join(lines), encoding="utf-8")


def tensor_to_uint8_frames(video: Any) -> list:
    """video [C,T,H,W] in [-1,1] or [0,1] → list HxWx3 uint8."""
    import torch

    x = video.detach().float().cpu()
    if x.ndim == 5:
        x = x[0]
    if x.min() < -0.1:
        x = ((x + 1.0) * 0.5).clamp(0, 1)
    else:
        x = x.clamp(0, 1)
    x = (x * 255.0).byte().permute(1, 2, 3, 0).numpy()
    return [x[i] for i in range(x.shape[0])]


def write_mp4(frames: list, path: Path, fps: int) -> None:
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8)
    for fr in frames:
        writer.append_data(fr)
    writer.close()


def write_png(frame, path: Path) -> None:
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), frame)


def x3(frames: list) -> list:
    return frames + frames + frames


def finalize_media(
    video: Any,
    out_dir: Path,
    *,
    fps: int,
) -> dict[str, str]:
    frames = tensor_to_uint8_frames(video)
    paths = {
        "out": str(out_dir / "out.mp4"),
        "out_x3": str(out_dir / "out_x3.mp4"),
        "first": str(out_dir / "first.png"),
        "last": str(out_dir / "last.png"),
    }
    write_mp4(frames, out_dir / "out.mp4", fps)
    write_mp4(x3(frames), out_dir / "out_x3.mp4", fps)
    write_png(frames[0], out_dir / "first.png")
    write_png(frames[-1], out_dir / "last.png")
    return paths


__all__ = [
    "BenchmarkSpec",
    "DEFAULT_ARTIFACTS_ROOT",
    "DEFAULT_BENCHMARK_DIR",
    "REPO_ROOT",
    "asdict",
    "default_probe_steps",
    "finalize_media",
    "full_ring_adjacent_gaps",
    "load_benchmark",
    "prepare_run_dir",
    "probe_tag_map",
    "save_latent_probe",
    "sha256_file",
    "write_json",
    "write_result_stub",
]
