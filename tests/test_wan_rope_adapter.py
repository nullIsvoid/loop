"""Tests for official Wan Mode B adapter (no Comfy, no full Wan install required)."""

from __future__ import annotations

import torch
from torch import nn

from latent_loop.adapters.wan.attention import (
    disable_mode_b_on_wan_model,
    enable_mode_b_on_wan_model,
)
from latent_loop.adapters.wan.rope import wan_rope_apply
from latent_loop.rope.core import rope_apply_baseline, rope_apply_loopy_roll
from latent_loop.rope.schedule import FixedShiftSchedule, LoopyShiftSchedule


def _rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = torch.outer(
        torch.arange(max_seq_len, dtype=torch.float64),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def _wan_freqs(head_dim: int, max_seq: int = 64) -> torch.Tensor:
    d = head_dim
    return torch.cat(
        [
            _rope_params(max_seq, d - 4 * (d // 6)),
            _rope_params(max_seq, 2 * (d // 6)),
            _rope_params(max_seq, 2 * (d // 6)),
        ],
        dim=1,
    )


def test_wan_rope_apply_enabled_matches_core():
    torch.manual_seed(2)
    b, f, h, w, n, head_dim = 1, 4, 2, 2, 2, 48
    l = f * h * w
    x = torch.randn(b, l, n, head_dim)
    grid = torch.tensor([[f, h, w]], dtype=torch.long)
    freqs = _wan_freqs(head_dim)
    assert torch.equal(
        wan_rope_apply(x, grid, freqs, time_shift=2, enabled=True),
        rope_apply_loopy_roll(x, grid, freqs, time_shift=2),
    )


def test_wan_rope_apply_disabled_is_baseline():
    torch.manual_seed(3)
    b, f, h, w, n, head_dim = 1, 3, 2, 2, 2, 48
    l = f * h * w
    x = torch.randn(b, l, n, head_dim)
    grid = torch.tensor([[f, h, w]], dtype=torch.long)
    freqs = _wan_freqs(head_dim)
    assert torch.equal(
        wan_rope_apply(x, grid, freqs, time_shift=5, enabled=False),
        rope_apply_baseline(x, grid, freqs),
    )


class _FakeSelfAttn(nn.Module):
    def __init__(self, dim: int = 48, num_heads: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = (-1, -1)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        self.calls = []

    def forward(self, x, seq_lens, grid_sizes, freqs):
        # Baseline stub: identity projection path without RoPE.
        b, s, _ = x.shape
        q = self.q(x).view(b, s, self.num_heads, self.head_dim)
        return self.o(q.flatten(2))


class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeSelfAttn()


class _FakeWan(nn.Module):
    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlock() for _ in range(n_layers)])


def test_enable_mode_b_uses_injectable_schedule_and_preserves_v_path():
    torch.manual_seed(4)
    model = _FakeWan(n_layers=3)
    f, h, w = 4, 2, 2
    l = f * h * w
    dim = 48
    x = torch.randn(1, l, dim)
    seq_lens = torch.tensor([l])
    grid = torch.tensor([[f, h, w]], dtype=torch.long)
    head_dim = model.blocks[0].self_attn.head_dim
    freqs = _wan_freqs(head_dim)

    captured = {}

    def flash_capture(*, q, k, v, k_lens, window_size=(-1, -1)):
        captured["q"] = q.detach().clone()
        captured["k"] = k.detach().clone()
        captured["v"] = v.detach().clone()
        _ = k_lens, window_size, k
        return v

    shifts = enable_mode_b_on_wan_model(
        model,
        schedule=LoopyShiftSchedule(),
        flash_attention_fn=flash_capture,
        enabled=True,
    )
    assert shifts == []  # F unknown until forward; ok

    out = model.blocks[1].self_attn(x, seq_lens, grid, freqs)
    assert out.shape == (1, l, dim)
    assert "v" in captured
    attn = model.blocks[1].self_attn
    v_raw = attn.v(x).view(1, l, attn.num_heads, attn.head_dim)
    assert torch.allclose(captured["v"], v_raw)

    # Expected shift for block 1 with F=4 is 1
    q_raw = attn.norm_q(attn.q(x)).view(1, l, attn.num_heads, attn.head_dim)
    expected_q = rope_apply_loopy_roll(q_raw, grid, freqs, time_shift=1)
    assert torch.allclose(captured["q"], expected_q)

    n = disable_mode_b_on_wan_model(model)
    assert n == 3


def test_mode_b_disabled_uses_baseline_rope_multipliers():
    torch.manual_seed(5)
    model = _FakeWan(n_layers=2)
    f, h, w = 3, 2, 2
    l = f * h * w
    dim = 48
    x = torch.randn(1, l, dim)
    seq_lens = torch.tensor([l])
    grid = torch.tensor([[f, h, w]], dtype=torch.long)
    head_dim = model.blocks[0].self_attn.head_dim
    freqs = _wan_freqs(head_dim)

    captured = {}

    def flash_capture(*, q, k, v, k_lens, window_size=(-1, -1)):
        captured["q"] = q.detach().clone()
        return v

    enable_mode_b_on_wan_model(
        model,
        schedule=FixedShiftSchedule(3),
        flash_attention_fn=flash_capture,
        enabled=False,
    )
    model.blocks[1].self_attn(x, seq_lens, grid, freqs)
    attn = model.blocks[1].self_attn
    q_raw = attn.norm_q(attn.q(x)).view(1, l, attn.num_heads, attn.head_dim)
    expected = rope_apply_baseline(q_raw, grid, freqs)
    assert torch.allclose(captured["q"], expected)
