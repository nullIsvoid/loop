"""Unit tests for D2 circular soft I2V conditioning."""

from __future__ import annotations

import torch

from latent_loop.i2v_conditioning import (
    Radius2CosineProfile,
    apply_circular_latent_conditioning,
    apply_circular_timestep_mask,
    ring_distance,
)


def test_ring_distance():
    assert ring_distance(0, 21) == 0
    assert ring_distance(1, 21) == 1
    assert ring_distance(20, 21) == 1
    assert ring_distance(2, 21) == 2
    assert ring_distance(19, 21) == 2
    assert ring_distance(10, 21) == 10


def test_radius2_profile_symmetric():
    w = Radius2CosineProfile().weights(21)
    assert len(w) == 21
    assert w[0] == 1.0
    assert w[1] == w[20] == 0.75
    assert w[2] == w[19] == 0.25
    assert w[3] == w[18] == 0.0
    assert all(x == 0.0 for x in w[3:19])


def test_apply_circular_latent_blends_neighbors():
    torch.manual_seed(0)
    gen = torch.randn(4, 7, 2, 2)
    ref = torch.ones(4, 1, 2, 2)
    weights = Radius2CosineProfile().weights(7)
    out = apply_circular_latent_conditioning(gen, ref, weights)
    assert torch.allclose(out[:, 0], ref[:, 0])
    assert torch.allclose(out[:, 1], 0.75 * ref[:, 0] + 0.25 * gen[:, 1])
    assert torch.allclose(out[:, 6], 0.75 * ref[:, 0] + 0.25 * gen[:, 6])
    assert torch.allclose(out[:, 2], 0.25 * ref[:, 0] + 0.75 * gen[:, 2])
    assert torch.equal(out[:, 3], gen[:, 3])


def test_apply_circular_timestep_mask_coupled():
    mask = torch.ones(2, 7, 3, 3)
    mask[:, 0] = 0.0
    weights = Radius2CosineProfile().weights(7)
    ts = apply_circular_timestep_mask(mask, weights)
    assert torch.allclose(ts[:, 0], torch.zeros_like(ts[:, 0]))
    assert torch.allclose(ts[:, 1], torch.full_like(ts[:, 1], 0.25))
    assert torch.allclose(ts[:, 6], torch.full_like(ts[:, 6], 0.25))
    assert torch.allclose(ts[:, 2], torch.full_like(ts[:, 2], 0.75))
    assert torch.allclose(ts[:, 3], torch.ones_like(ts[:, 3]))
