"""Unit tests for Hunyuan Symmetric Circular Temporal RoPE helpers."""

from __future__ import annotations

import torch
from torch import nn

from latent_loop.adapters.hunyuan.attention import (
    disable_mode_b_on_hunyuan_transformer,
    enable_mode_b_on_hunyuan_transformer,
)
from latent_loop.adapters.hunyuan.rope import (
    describe_official_rope_layout,
    roll_hunyuan_freqs_cis,
)
from latent_loop.rope.schedule import SymmetricShiftSchedule, list_layer_time_shifts


def test_rope_layout_marks_mode_b_ready():
    meta = describe_official_rope_layout()
    assert meta["rope_dim_list_default"] == [16, 56, 56]
    assert meta["circular_mode_b"] == "symmetric_temporal_freqs_roll"


def test_roll_hunyuan_freqs_identity_and_shift():
    tt, th, tw, d = 4, 2, 2, 8
    cos = torch.zeros(tt * th * tw, d)
    sin = torch.zeros(tt * th * tw, d)
    for t in range(tt):
        cos.view(tt, th, tw, d)[t, :, :, 0] = float(t)
    out0 = roll_hunyuan_freqs_cis((cos, sin), tt=tt, th=th, tw=tw, time_shift=0)
    assert torch.equal(out0[0], cos)

    rolled, _ = roll_hunyuan_freqs_cis((cos, sin), tt=tt, th=th, tw=tw, time_shift=1)
    assert float(rolled.view(tt, th, tw, d)[0, 0, 0, 0]) == 3.0
    assert float(rolled.view(tt, th, tw, d)[1, 0, 0, 0]) == 0.0


def test_symmetric_schedule_preview_60_layers():
    shifts = list_layer_time_shifts(60, 21, SymmetricShiftSchedule())
    assert shifts[0] == 0
    assert shifts[1] == 1
    assert shifts[2] == (-1 % 21)


class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_freqs = None

    def forward(self, freqs_cis=None, **kwargs):
        self.last_freqs = freqs_cis
        return None


class _FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = nn.ModuleList([_FakeBlock() for _ in range(2)])
        self.single_blocks = nn.ModuleList([_FakeBlock() for _ in range(3)])

    def get_rotary_pos_embed(self, rope_sizes):
        tt, th, tw = rope_sizes
        L = tt * th * tw
        cos = torch.arange(L, dtype=torch.float32).unsqueeze(1).repeat(1, 4)
        sin = torch.zeros_like(cos)
        return cos, sin


def test_enable_disable_mode_b_rolls_per_block():
    tr = _FakeTransformer()
    enable_mode_b_on_hunyuan_transformer(tr, schedule=SymmetricShiftSchedule(), enabled=True)
    cos, sin = tr.get_rotary_pos_embed((4, 1, 1))
    assert tr._latent_loop_rope_sizes == (4, 1, 1)

    tr.double_blocks[0](freqs_cis=(cos.clone(), sin.clone()))
    assert float(tr.double_blocks[0].last_freqs[0][0, 0]) == 0.0

    tr.double_blocks[1](freqs_cis=(cos.clone(), sin.clone()))
    assert float(tr.double_blocks[1].last_freqs[0][0, 0]) == 3.0

    n = disable_mode_b_on_hunyuan_transformer(tr)
    assert n == 5
    assert not hasattr(tr, "_latent_loop_mode_b")
