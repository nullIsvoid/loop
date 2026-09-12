from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))

from common import (  # noqa: E402
    default_probe_steps,
    full_ring_adjacent_gaps,
    load_benchmark,
    probe_tag_map,
    save_latent_probe,
)


def test_load_person_loop_v1():
    bench = load_benchmark(ROOT / "assets" / "loop_benchmark")
    assert bench.benchmark_id == "person_loop_v1"
    assert bench.seed == 42
    assert bench.frame_num == 81
    assert "seamless looping" in bench.prompt.lower()
    assert bench.source_path.is_file()
    assert bench.source_sha256 == (
        "6807cba271b86885329d64b75ad7c3bb4b8d860374c03f414b0cdfbf4014ad9f"
    )


def test_probe_steps_and_tags():
    steps = default_probe_steps(40)
    assert steps[0] <= 2
    assert steps[-1] == 39
    tags = probe_tag_map(steps)
    assert set(tags.values()) >= {"early", "late"}


def test_full_ring_gaps_and_save(tmp_path: Path):
    # Frames 0..7 = identity ramp → only wrap edge 7→0 is large.
    x = torch.zeros(4, 8, 2, 2)
    for i in range(8):
        x[:, i] = float(i)
    meta = full_ring_adjacent_gaps(x)
    assert meta["F"] == 8
    assert meta["max_gap_edge"] == "7->0"
    assert meta["seam_F-1->0"] == meta["max_gap_l2"]
    assert meta["seam_0->1"] < meta["seam_F-1->0"]

    saved = save_latent_probe(x, tmp_path, tag="late", step_i=39)
    assert (tmp_path / "late_full_latent.pt").is_file()
    gaps = json.loads((tmp_path / "late_full_ring_gaps.json").read_text(encoding="utf-8"))
    assert gaps["tag"] == "late"
    assert gaps["seam_0->1"] == pytest.approx(saved["seam_0->1"])


def test_hunyuan_bcfhw_normalized():
    x = torch.randn(1, 16, 21, 4, 4)
    meta = full_ring_adjacent_gaps(x)
    assert meta["F"] == 21
