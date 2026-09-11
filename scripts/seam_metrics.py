#!/usr/bin/env python3
"""Multi-detector loop-seam metrics for *_x3.mp4 previews.

Detectors (independent; none is ground truth alone):
  D1 pixel_mae_seam     — |last - first| mean abs error on cycle
  D2 pixel_rmse_seam    — RMSE last vs first
  D3 flow_mag_seam      — Farneback mean flow magnitude last→next (x3 join)
  D4 flow_mag_adj_mean  — mean adjacent flow mag inside first cycle
  D5 flow_ratio         — D3 / max(D4, eps); >1 means seam jumps harder than motion
  D6 flow_mag_seam_b    — second x3 join (sanity; should ≈ D3)
  D7 adj_pixel_mae_mean — mean |frame_i - frame_{i+1}| inside cycle
  D8 pixel_ratio        — D1 / max(D7, eps)

Lower is better for D1/D2/D3/D5/D8. Rank schedules per scene by majority vote
across normalized ranks of D1, D3, D5, D8.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read_frames(path: Path, max_side: int = 640) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            bgr = cv2.resize(
                bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        frames.append(bgr)
    cap.release()
    if len(frames) < 6:
        raise RuntimeError(f"too few frames in {path}: {len(frames)}")
    return frames


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _pixel_mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def _pixel_rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.sqrt(np.mean(d * d)))


def _flow_mag(prev_bgr: np.ndarray, next_bgr: np.ndarray) -> float:
    prev = _to_gray(prev_bgr)
    nxt = _to_gray(next_bgr)
    flow = cv2.calcOpticalFlowFarneback(
        prev,
        nxt,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.mean(mag))


def analyze_x3(path: Path, max_side: int = 640) -> dict[str, Any]:
    frames = _read_frames(path, max_side=max_side)
    n_total = len(frames)
    if n_total % 3 != 0:
        # tolerate encode pad: use floor
        cycle = n_total // 3
    else:
        cycle = n_total // 3
    if cycle < 2:
        raise RuntimeError(f"cycle too short: {path} frames={n_total}")

    first = frames[0]
    last = frames[cycle - 1]
    # x3 joins: last of copy0 → first of copy1 (== frames[cycle])
    seam_a = _flow_mag(frames[cycle - 1], frames[cycle])
    seam_b = (
        _flow_mag(frames[2 * cycle - 1], frames[2 * cycle])
        if 2 * cycle < n_total
        else seam_a
    )

    adj_flows: list[float] = []
    adj_maes: list[float] = []
    for i in range(cycle - 1):
        adj_flows.append(_flow_mag(frames[i], frames[i + 1]))
        adj_maes.append(_pixel_mae(frames[i], frames[i + 1]))

    adj_flow_mean = float(np.mean(adj_flows)) if adj_flows else 0.0
    adj_mae_mean = float(np.mean(adj_maes)) if adj_maes else 0.0
    pixel_mae = _pixel_mae(first, last)
    pixel_rmse = _pixel_rmse(first, last)
    eps = 1e-6

    return {
        "path": str(path).replace("\\", "/"),
        "frames_total": n_total,
        "cycle_frames": cycle,
        "max_side": max_side,
        "D1_pixel_mae_seam": pixel_mae,
        "D2_pixel_rmse_seam": pixel_rmse,
        "D3_flow_mag_seam": seam_a,
        "D4_flow_mag_adj_mean": adj_flow_mean,
        "D5_flow_ratio": seam_a / max(adj_flow_mean, eps),
        "D6_flow_mag_seam_b": seam_b,
        "D7_adj_pixel_mae_mean": adj_mae_mean,
        "D8_pixel_ratio": pixel_mae / max(adj_mae_mean, eps),
        "flow_seam_consistency": abs(seam_a - seam_b),
    }


def _parse_stem(stem: str) -> tuple[str, str]:
    # e.g. pendulum__S1_loopy_x3 or S1_loopy_x3
    name = stem[: -len("_x3")] if stem.endswith("_x3") else stem
    if "__" in name:
        scene, sched = name.split("__", 1)
        return scene, sched
    return "candle", name


def rank_matrix(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-scene majority rank over D1, D3, D5, D8 (lower better)."""
    keys = [
        "D1_pixel_mae_seam",
        "D3_flow_mag_seam",
        "D5_flow_ratio",
        "D8_pixel_ratio",
    ]
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_scene.setdefault(r["scene"], []).append(r)

    out: dict[str, Any] = {}
    for scene, items in sorted(by_scene.items()):
        scores: dict[str, float] = {it["schedule"]: 0.0 for it in items}
        detail: dict[str, dict[str, int]] = {it["schedule"]: {} for it in items}
        for k in keys:
            ordered = sorted(items, key=lambda x: x[k])
            for rank, it in enumerate(ordered):
                scores[it["schedule"]] += rank
                detail[it["schedule"]][k] = rank
        winner = min(scores.items(), key=lambda kv: kv[1])[0]
        out[scene] = {
            "rank_sum_lower_better": scores,
            "per_detector_rank": detail,
            "majority_winner": winner,
            "detectors": keys,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "dirs",
        nargs="+",
        type=Path,
        help="artifact dirs containing *_x3.mp4",
    )
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="write metrics.json here (default: first dir / seam_metrics.json)",
    )
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for d in args.dirs:
        for path in sorted(d.glob("*_x3.mp4")):
            scene, sched = _parse_stem(path.stem)
            print(f"analyze {path.name} ...", flush=True)
            m = analyze_x3(path, max_side=args.max_side)
            m["scene"] = scene
            m["schedule"] = sched
            rows.append(m)

    ranking = rank_matrix(rows)
    payload = {
        "version": 1,
        "note": (
            "Auxiliary multi-detector seam metrics. Not a substitute for "
            "human / ChatGPT visual review. Lower D1/D3/D5/D8 better."
        ),
        "count": len(rows),
        "rows": rows,
        "ranking": ranking,
    }

    out = args.out or (args.dirs[0] / "seam_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # compact markdown table
    md_lines = [
        "# Seam metrics (multi-detector)",
        "",
        payload["note"],
        "",
        "| scene | schedule | D1 mae | D3 flow_seam | D5 flow_ratio | D8 pixel_ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['scene']} | {r['schedule']} | "
            f"{r['D1_pixel_mae_seam']:.2f} | {r['D3_flow_mag_seam']:.3f} | "
            f"{r['D5_flow_ratio']:.3f} | {r['D8_pixel_ratio']:.3f} |"
        )
    md_lines.append("")
    md_lines.append("## Majority ranking (D1+D3+D5+D8 ranks)")
    md_lines.append("")
    for scene, info in ranking.items():
        md_lines.append(
            f"- **{scene}**: winner `{info['majority_winner']}` — "
            f"rank_sum={info['rank_sum_lower_better']}"
        )
    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {md_path}")
    for scene, info in ranking.items():
        print(f"  {scene}: {info['majority_winner']} {info['rank_sum_lower_better']}")


if __name__ == "__main__":
    main()
