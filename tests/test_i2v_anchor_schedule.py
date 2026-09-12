"""Unit tests for I2V frame-0 anchor schedules (D1)."""

from __future__ import annotations

import torch

from latent_loop.i2v_conditioning import (
    HardAnchorSchedule,
    LateAnchorReleaseSchedule,
    apply_frame0_anchor,
    apply_frame0_timestep_mask,
    frame0_timestep_factor,
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


def test_coupled_timestep_factor():
    assert frame0_timestep_factor(1.0, couple_timestep=False) == 0.0
    assert frame0_timestep_factor(0.5, couple_timestep=False) == 0.0
    assert frame0_timestep_factor(1.0, couple_timestep=True) == 0.0
    assert frame0_timestep_factor(0.0, couple_timestep=True) == 1.0
    assert abs(frame0_timestep_factor(0.25, couple_timestep=True) - 0.75) < 1e-9


def test_apply_frame0_timestep_mask_coupled():
    mask = torch.ones(2, 4, 3, 3)
    mask[:, 0] = 0.0
    soft = apply_frame0_timestep_mask(mask, 0.4, couple_timestep=True)
    assert torch.allclose(soft[:, 0], torch.full_like(soft[:, 0], 0.6))
    assert torch.allclose(soft[:, 1:], torch.ones_like(soft[:, 1:]))
    hard = apply_frame0_timestep_mask(mask, 0.4, couple_timestep=False)
    assert torch.equal(hard[:, 0], torch.zeros_like(hard[:, 0]))
