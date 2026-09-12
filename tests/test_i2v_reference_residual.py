"""Unit tests for D3 circular reference residual conditioning."""

from __future__ import annotations

import math

import torch

from latent_loop.i2v_conditioning import (
    FullRingCosineProfile,
    apply_circular_reference_residual,
)


def test_full_ring_cosine_endpoints():
    w = FullRingCosineProfile().weights(21)
    assert len(w) == 21
    assert abs(w[0] - 1.0) < 1e-9
    assert abs(w[10] - 0.0) < 1e-9
    assert abs(w[1] - w[20]) < 1e-9
    # ±1 ≈ 0.5*(1+cos(pi/10))
    expected = 0.5 * (1.0 + math.cos(math.pi / 10))
    assert abs(w[1] - expected) < 1e-9
    assert all(w[i] >= w[i + 1] - 1e-9 for i in range(10))


def test_residual_forces_frame0_to_ref_keeps_neighbors_relative():
    torch.manual_seed(0)
    gen = torch.randn(4, 7, 2, 2)
    ref = torch.ones(4, 1, 2, 2) * 2.0
    weights = FullRingCosineProfile().weights(7)
    assert abs(weights[0] - 1.0) < 1e-9
    out = apply_circular_reference_residual(gen, ref, weights)
    assert torch.allclose(out[:, 0], ref[:, 0])
    delta = ref[:, 0] - gen[:, 0]
    # Neighbor keeps own content + scaled delta (not pulled toward ref0 as state)
    assert torch.allclose(out[:, 1], gen[:, 1] + weights[1] * delta)
    assert not torch.allclose(
        out[:, 1], weights[1] * ref[:, 0] + (1.0 - weights[1]) * gen[:, 1]
    )
