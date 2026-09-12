"""Unit tests for I2V frame-0 anchor schedules (D1)."""

from __future__ import annotations

import torch

from latent_loop.i2v_conditioning import (
    HardAnchorSchedule,
    LateAnchorReleaseSchedule,
    apply_frame0_anchor,
    list_anchor_strengths,
)


def test_hard_anchor_always_one():
    assert list_anchor_strengths(20, HardAnchorSchedule()) == [1.0] * 20


def test_late_release_table_20():
    s = LateAnchorReleaseSchedule()
    got = list_anchor_strengths(20, s)
    assert got[:13] == [1.0] * 13
    assert got[13:] == [0.90, 0.75, 0.60, 0.45, 0.30, 0.15, 0.00]


def test_apply_frame0_anchor_blends_only_index_zero():
    torch.manual_seed(0)
    gen = torch.randn(4, 5, 2, 2)
    ref = torch.ones(4, 5, 2, 2)
    out = apply_frame0_anchor(gen, ref, 0.5)
    assert torch.allclose(out[:, 0], 0.5 * ref[:, 0] + 0.5 * gen[:, 0])
    assert torch.equal(out[:, 1:], gen[:, 1:])


def test_apply_frame0_accepts_wan_i2v_ref_f1():
    gen = torch.randn(4, 5, 2, 2)
    ref = torch.ones(4, 1, 2, 2)
    out = apply_frame0_anchor(gen, ref, 1.0)
    assert torch.equal(out[:, 0], ref[:, 0])
    assert torch.equal(out[:, 1:], gen[:, 1:])


def test_apply_frame0_hard_and_free():
    gen = torch.randn(2, 3, 1, 1)
    ref = torch.ones(2, 3, 1, 1)
    hard = apply_frame0_anchor(gen, ref, 1.0)
    free = apply_frame0_anchor(gen, ref, 0.0)
    assert torch.equal(hard[:, 0], ref[:, 0])
    assert torch.equal(free, gen)
