#!/usr/bin/env python3
"""Decoded-frame seam diagnostics for Mode B I2V only (no auto score).

Exports:
  - pixel frames 75..80 and 0..5 (from single loop; x3 uses join at N)
  - Farneback optical-flow visualizations for
    78→79, 79→80, 80→0, 0→1, 1→2

Goal: help humans classify residual seam as position jump / velocity jump /
direction reverse / phase misalign / local desync — not Gate/Repair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


FLOW_PAIRS = [(78, 79), (79, 80), (80, 0), (0, 1), (1, 2)]
FRAME_IDS = list(range(75, 81)) + list(range(0, 6))


def _read_all_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    if len(frames) < 6:
        raise RuntimeError(f"too few frames: {path} n={len(frames)}")
    return frames


def _cycle_frames(frames: list[np.ndarray], cycle_hint: int | None) -> tuple[list[np.ndarray], int]:
    """Prefer single-loop length; if x3 (3*N), take first cycle + next-frame joins via pad."""
    n = len(frames)
    if cycle_hint is not None:
        cycle = cycle_hint
    elif n % 3 == 0 and n // 3 >= 20:
        cycle = n // 3
    else:
        cycle = n
    if cycle > n:
        raise RuntimeError(f"cycle {cycle} > frames {n}")
    # Build a ring view: frames[0:cycle] then frames[0].. for wrap indexing
    ring = frames[:cycle]
    return ring, cycle


def _frame(ring: list[np.ndarray], idx: int) -> np.ndarray:
    return ring[idx % len(ring)]


def _flow_viz(prev: np.ndarray, nxt: np.ndarray) -> tuple[np.ndarray, dict]:
    g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*g0.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    stats = {
        "mean_mag": float(np.mean(mag)),
        "median_mag": float(np.median(mag)),
        "p95_mag": float(np.percentile(mag, 95)),
        "mean_dx": float(np.mean(flow[..., 0])),
        "mean_dy": float(np.mean(flow[..., 1])),
    }
    return bgr, stats


def diagnose_video(video: Path, out_dir: Path, cycle_hint: int | None = 81) -> dict:
    frames = _read_all_frames(video)
    ring, cycle = _cycle_frames(frames, cycle_hint)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    flow_dir = out_dir / "flow"
    frames_dir.mkdir(exist_ok=True)
    flow_dir.mkdir(exist_ok=True)

    exported = []
    for i in FRAME_IDS:
        path = frames_dir / f"f{i:03d}.png"
        cv2.imwrite(str(path), _frame(ring, i))
        exported.append(str(path.name))

    flow_rows = []
    for a, b in FLOW_PAIRS:
        viz, stats = _flow_viz(_frame(ring, a), _frame(ring, b))
        name = f"flow_{a:03d}_to_{b:03d}.png"
        cv2.imwrite(str(flow_dir / name), viz)
        # side-by-side: prev | next | flow
        left = _frame(ring, a)
        right = _frame(ring, b)
        h = min(left.shape[0], right.shape[0], viz.shape[0])
        w = min(left.shape[1], right.shape[1], viz.shape[1])
        strip = np.concatenate(
            [left[:h, :w], right[:h, :w], viz[:h, :w]], axis=1
        )
        strip_name = f"strip_{a:03d}_to_{b:03d}.png"
        cv2.imwrite(str(flow_dir / strip_name), strip)
        row = {"pair": f"{a}->{b}", "flow_png": name, "strip_png": strip_name, **stats}
        # Mark seam pair
        row["is_seam"] = a == cycle - 1 or (a == 80 and b == 0) or (a > b)
        if cycle == 81:
            row["is_seam"] = (a, b) == (80, 0)
        flow_rows.append(row)

    # Simple continuity ratios for human notes (not a score gate)
    by_pair = {r["pair"]: r for r in flow_rows}
    adj = np.mean(
        [by_pair["78->79"]["mean_mag"], by_pair["79->80"]["mean_mag"], by_pair["0->1"]["mean_mag"], by_pair["1->2"]["mean_mag"]]
    )
    seam = by_pair["80->0"]["mean_mag"]
    meta = {
        "video": str(video).replace("\\", "/"),
        "cycle_frames": cycle,
        "total_frames_read": len(frames),
        "exported_frame_ids": FRAME_IDS,
        "flow_pairs": FLOW_PAIRS,
        "flow_stats": flow_rows,
        "aux_seam_vs_adj_flow_ratio": float(seam / max(adj, 1e-6)),
        "note": (
            "Auxiliary only. Compare seam strip 80→0 vs neighbors: "
            "position jump (big RGB change), velocity jump (mag spike), "
            "direction reverse (hue flip in flow), local desync (regional)."
        ),
    }
    (out_dir / "diagnostics.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        f"# Seam frame diagnostics — `{video.name}`",
        "",
        meta["note"],
        "",
        f"cycle_frames={cycle}, seam_vs_adj_flow_ratio={meta['aux_seam_vs_adj_flow_ratio']:.3f}",
        "",
        "| pair | seam? | mean_mag | median | p95 | mean_dx | mean_dy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in flow_rows:
        md.append(
            f"| {r['pair']} | {r['is_seam']} | {r['mean_mag']:.3f} | "
            f"{r['median_mag']:.3f} | {r['p95_mag']:.3f} | "
            f"{r['mean_dx']:.3f} | {r['mean_dy']:.3f} |"
        )
    md.append("")
    md.append("See `frames/` and `flow/strip_*.png`.")
    (out_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--videos",
        nargs="+",
        type=Path,
        help="Mode B videos (prefer single-loop; x3 also ok if length%%3==0)",
    )
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--cycle", type=int, default=81)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    summary = []
    for video in args.videos:
        case = video.parent.name
        out = args.out_root / case
        print(f"diagnose {video} -> {out}")
        summary.append(diagnose_video(video, out, cycle_hint=args.cycle))
    (args.out_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out_root}")


if __name__ == "__main__":
    main()
