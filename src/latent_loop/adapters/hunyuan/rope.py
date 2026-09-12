"""Hunyuan ND RoPE helpers + Circular Temporal freqs roll.

Official layout: ``rope_dim_list=[16,56,56]`` (t, h, w), real cos/sin,
token order t-major from ``meshgrid(..., indexing='ij')``.

Symmetric Circular Temporal RoPE = ``torch.roll`` on the **temporal** axis of
the expanded freqs grid only — same responsibility as Wan Mode B, different
grid packing. Do not import Wan rope helpers here.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from latent_loop.rope.core import roll_temporal_freqs_3d


def describe_official_rope_layout() -> dict:
    """Static metadata for docs/tests; no model import."""
    return {
        "rope_dim_list_default": [16, 56, 56],
        "axes": ["temporal", "height", "width"],
        "apply_sites": [
            "MMDoubleStreamBlock.forward → apply_rotary_emb(img_q, img_k)",
            "MMSingleStreamBlock.forward → apply_rotary_emb(img_q, img_k)",
        ],
        "circular_mode_b": "symmetric_temporal_freqs_roll",
        "global_block_indexing": "double[0..D) then single[D..D+S)",
    }


def roll_hunyuan_freqs_cis(
    freqs_cis: Tuple[Tensor, Tensor] | Tensor,
    *,
    tt: int,
    th: int,
    tw: int,
    time_shift: int,
) -> Tuple[Tensor, Tensor] | Tensor:
    """Roll Hunyuan RoPE freqs along temporal tokens.

    Args:
        freqs_cis: ``(cos, sin)`` each ``[L, D]`` (use_real=True), or complex
            ``[L, D/2]`` (not used by default HunyuanVideo-1.5 path).
        tt, th, tw: latent patch grid sizes (``L`` must equal ``tt*th*tw``).
        time_shift: temporal roll; ``0`` is identity.

    Returns:
        Same type as input, with temporal axis rolled.
    """
    if time_shift == 0:
        return freqs_cis

    expected = int(tt) * int(th) * int(tw)
    if isinstance(freqs_cis, tuple):
        cos, sin = freqs_cis
        if cos.shape[0] != expected:
            raise ValueError(
                f"freqs L={cos.shape[0]} != tt*th*tw={expected}; "
                "sequence-parallel sharded freqs are not supported in H1 v1"
            )
        cos3 = cos.view(tt, th, tw, cos.shape[-1])
        sin3 = sin.view(tt, th, tw, sin.shape[-1])
        cos3 = roll_temporal_freqs_3d(cos3, time_shift)
        sin3 = roll_temporal_freqs_3d(sin3, time_shift)
        return cos3.reshape(expected, -1), sin3.reshape(expected, -1)

    # complex cis
    if freqs_cis.shape[0] != expected:
        raise ValueError(
            f"freqs L={freqs_cis.shape[0]} != tt*th*tw={expected}; "
            "sequence-parallel sharded freqs are not supported in H1 v1"
        )
    x3 = freqs_cis.view(tt, th, tw, freqs_cis.shape[-1])
    x3 = roll_temporal_freqs_3d(x3, time_shift)
    return x3.reshape(expected, -1)
