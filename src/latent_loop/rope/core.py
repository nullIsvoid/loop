"""Loopy-style temporal RoPE（旋转位置编码）roll — model-independent core.

Terminology boundary:
  This is *Loopy-style RoPE roll* (per-block temporal phase reordering via
  ``torch.roll`` on expanded ``freqs_3d``). It is **not** claimed to be a
  mathematically periodic / circular RoPE. Closed-loop behaviour must be
  validated on real Wan runs; stricter periodic encodings remain future work.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def roll_temporal_freqs_3d(freqs_3d: Tensor, time_shift: int) -> Tensor:
    """Roll only the temporal axis (dim 0) of expanded freqs ``[F, H, W, ...]``.

    ``time_shift == 0`` returns the input unchanged (anchor behaviour).
    Never rolls H or W.
    """
    if time_shift == 0:
        return freqs_3d
    if freqs_3d.ndim < 1:
        raise ValueError("freqs_3d must have a temporal dimension")
    return torch.roll(freqs_3d, shifts=int(time_shift), dims=0)


def expand_wan_freqs_3d(
    freqs: Tensor,
    f: int,
    h: int,
    w: int,
    *,
    head_half: int,
) -> Tensor:
    """Build Wan-style 3D freqs grid from the concatenated cis table.

    ``freqs`` layout matches official Wan: temporal / height / width slices
    along dim=1, complex dtype, shape ``[M, head_half]``.
    Returns ``[F, H, W, 1, head_half]``.
    """
    c = int(head_half)
    freqs_f, freqs_h, freqs_w = freqs.split(
        [c - 2 * (c // 3), c // 3, c // 3], dim=1
    )
    freqs_i = torch.cat(
        [
            freqs_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    )
    return freqs_i.view(f, h, w, 1, -1)


@torch.amp.autocast("cuda", enabled=False)
def rope_apply_loopy_roll(
    x: Tensor,
    grid_sizes: Tensor | Sequence[Sequence[int]],
    freqs: Tensor,
    time_shift: int = 0,
) -> Tensor:
    """Apply Wan 3D RoPE to Q/K with optional Loopy-style temporal freqs roll.

    Matches WeChatCV/Loopy ``rope_apply_loop`` behaviour for numerical parity
    when ``L == F * H * W`` (typical single-clip unpadded path).

    Args:
        x: ``[B, L, N, C]`` real Q or K (C = head dim).
        grid_sizes: ``[B, 3]`` of ``(F, H, W)`` patch grid sizes.
        freqs: ``[M, C/2]`` complex Wan freqs table.
        time_shift: temporal roll amount; 0 leaves freqs unrolled.

    Returns:
        Tensor with the same shape / device as ``x``, float32 stacked like Wan.
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    if isinstance(grid_sizes, Tensor):
        fwh_list = grid_sizes.tolist()
    else:
        fwh_list = [list(g) for g in grid_sizes]

    output: list[Tensor] = []
    for i, (f, h, w) in enumerate(fwh_list):
        f, h, w = int(f), int(h), int(w)
        seq_len = f * h * w

        x_i = torch.view_as_complex(
            x[i, :s].to(torch.float64).reshape(s, n, -1, 2)
        )
        freqs_3d = expand_wan_freqs_3d(freqs, f, h, w, head_half=c)
        freqs_3d = roll_temporal_freqs_3d(freqs_3d, time_shift)
        freqs_i = freqs_3d.reshape(seq_len, 1, -1)

        rotated = torch.view_as_real(x_i * freqs_i).flatten(2)
        rotated = torch.cat([rotated, x[i, s:]])
        output.append(rotated)

    return torch.stack(output).float()


def rope_apply_baseline(
    x: Tensor,
    grid_sizes: Tensor | Sequence[Sequence[int]],
    freqs: Tensor,
) -> Tensor:
    """Official Wan ``rope_apply`` equivalent (no temporal roll)."""
    return rope_apply_loopy_roll(x, grid_sizes, freqs, time_shift=0)
