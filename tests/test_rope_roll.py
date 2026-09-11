"""Unit tests for Loopy-style temporal RoPE roll (Mode B core)."""

from __future__ import annotations

import torch

from latent_loop.rope.core import (
    expand_wan_freqs_3d,
    roll_temporal_freqs_3d,
    rope_apply_baseline,
    rope_apply_loopy_roll,
)
from latent_loop.rope.schedule import (
    FixedShiftSchedule,
    IdentityShiftSchedule,
    LoopyShiftSchedule,
    SymmetricShiftSchedule,
    list_layer_time_shifts,
)


def _rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len, dtype=torch.float64),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def _wan_freqs(head_dim: int, max_seq: int = 128) -> torch.Tensor:
    d = head_dim
    return torch.cat(
        [
            _rope_params(max_seq, d - 4 * (d // 6)),
            _rope_params(max_seq, 2 * (d // 6)),
            _rope_params(max_seq, 2 * (d // 6)),
        ],
        dim=1,
    )


def _loopy_reference_rope_apply_loop(x, grid_sizes, freqs, time_shift=0):
    """Inlined WeChatCV/Loopy ``rope_apply_loop`` for numerical parity checks."""
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(s, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        freqs_3d = freqs_i.view(f, h, w, 1, -1)
        if time_shift != 0:
            freqs_3d = torch.roll(freqs_3d, shifts=time_shift, dims=0)
        freqs_i = freqs_3d.reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])
        output.append(x_i)
    return torch.stack(output).float()


def test_loopy_schedule_anchor_and_cycle():
    sched = LoopyShiftSchedule()
    assert sched.time_shift(0, 16) == 0
    assert list_layer_time_shifts(5, 16, sched) == [0, 1, 2, 3, 4]
    # wrap: block 16 with F=16 -> (15) % 15 + 1 = 1
    assert sched.time_shift(16, 16) == 1


def test_identity_and_fixed_schedules_keep_block0_anchor():
    assert IdentityShiftSchedule().time_shift(3, 8) == 0
    assert FixedShiftSchedule(7).time_shift(0, 8) == 0
    assert FixedShiftSchedule(7).time_shift(2, 8) == 7


def test_symmetric_schedule_bidirectional_then_mod_f():
    sched = SymmetricShiftSchedule()
    assert sched.time_shift(0, 21) == 0
    # signed intent: +1,-1,+2,-2 ; applied as % F  (-1%21==20, -2%21==19)
    assert list_layer_time_shifts(7, 21, sched) == [0, 1, 20, 2, 19, 3, 18]
    assert sched.time_shift(2, 21) == 20


def test_roll_only_on_temporal_axis():
    f, h, w, c = 4, 3, 2, 6
    base = torch.arange(f * h * w * c, dtype=torch.float64).reshape(f, h, w, 1, c)
    rolled = roll_temporal_freqs_3d(base, time_shift=1)
    # Frame 0 of rolled equals former last frame; H/W neighbourhood intact.
    assert torch.equal(rolled[0], base[-1])
    assert torch.equal(rolled[1], base[0])
    # Same spatial slice pattern: rolling F must not scramble H,W order inside a frame.
    assert torch.equal(rolled[2, 1, 0], base[1, 1, 0])


def test_roll_zero_is_identity():
    x = torch.randn(5, 2, 2, 1, 4)
    assert roll_temporal_freqs_3d(x, 0) is x or torch.equal(roll_temporal_freqs_3d(x, 0), x)


def test_rope_apply_loopy_roll_matches_loopy_reference_equal_length():
    """Equal unpadded L == F*H*W: matches historical Loopy rope_apply_loop."""
    torch.manual_seed(0)
    b, f, h, w, n, head_dim = 2, 4, 2, 2, 4, 48
    l = f * h * w
    x = torch.randn(b, l, n, head_dim)
    grid = torch.tensor([[f, h, w], [f, h, w]], dtype=torch.long)
    freqs = _wan_freqs(head_dim)

    for shift in (0, 1, 2, 3):
        ours = rope_apply_loopy_roll(x, grid, freqs, time_shift=shift)
        ref = _loopy_reference_rope_apply_loop(x, grid, freqs, time_shift=shift)
        assert ours.shape == x.shape == ref.shape
        assert ours.dtype == ref.dtype == torch.float32
        assert ours.device == x.device
        assert torch.allclose(ours, ref, rtol=0, atol=0)


def _wan_official_style_rope_apply(x, grid_sizes, freqs):
    """Official Wan rope_apply: per-sample seq_len, leave padding untouched."""
    n, c = x.size(2), x.size(3) // 2
    freqs_f, freqs_h, freqs_w = freqs.split(
        [c - 2 * (c // 3), c // 3, c // 3], dim=1
    )
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, c)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


def test_padded_heterogeneous_batch_shift0_matches_wan_official():
    """P1: batch items with different FHW + padding must follow Wan seq_len."""
    torch.manual_seed(7)
    n, head_dim = 2, 48
    grid = torch.tensor([[4, 2, 2], [3, 2, 2]], dtype=torch.long)  # 16 and 12
    padded_l = 16
    x = torch.randn(2, padded_l, n, head_dim)
    freqs = _wan_freqs(head_dim)

    ours = rope_apply_loopy_roll(x, grid, freqs, time_shift=0)
    ref = _wan_official_style_rope_apply(x, grid, freqs)
    assert ours.shape == (2, 16, 2, 48)
    assert torch.allclose(ours, ref, rtol=0, atol=0)
    # Padding of shorter sample must be untouched.
    assert torch.equal(ours[1, 12:], x[1, 12:])


def test_padded_heterogeneous_batch_with_nonzero_shift_runs():
    torch.manual_seed(8)
    n, head_dim = 2, 48
    grid = torch.tensor([[4, 2, 2], [3, 2, 2]], dtype=torch.long)
    x = torch.randn(2, 16, n, head_dim)
    freqs = _wan_freqs(head_dim)
    out = rope_apply_loopy_roll(x, grid, freqs, time_shift=2)
    assert out.shape == (2, 16, 2, 48)
    assert torch.equal(out[1, 12:], x[1, 12:])


def test_list_layer_time_shifts_default_is_symmetric():
    f = 21
    # schedule=None must freeze to Symmetric: 0,+1,-1,+2,-2,...
    assert list_layer_time_shifts(7, f, None) == [0, 1, 20, 2, 19, 3, 18]
    assert list_layer_time_shifts(7, f) == [0, 1, 20, 2, 19, 3, 18]



def test_shift0_matches_baseline():
    torch.manual_seed(1)
    b, f, h, w, n, head_dim = 1, 3, 2, 2, 2, 48
    l = f * h * w
    x = torch.randn(b, l, n, head_dim)
    grid = torch.tensor([[f, h, w]], dtype=torch.long)
    freqs = _wan_freqs(head_dim)
    assert torch.equal(
        rope_apply_loopy_roll(x, grid, freqs, time_shift=0),
        rope_apply_baseline(x, grid, freqs),
    )


def test_expand_freqs_shape():
    head_dim = 48
    freqs = _wan_freqs(head_dim)
    grid = expand_wan_freqs_3d(freqs, 5, 3, 2, head_half=head_dim // 2)
    assert grid.shape == (5, 3, 2, 1, head_dim // 2)
