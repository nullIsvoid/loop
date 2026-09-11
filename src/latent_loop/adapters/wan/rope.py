"""Official Wan ``rope_apply`` / Loopy-style roll adapter.

Comfy ``rope_encode`` / ``apply_rope1`` is intentionally out of scope.
"""

from __future__ import annotations

from typing import Sequence

from torch import Tensor

from latent_loop.rope.core import rope_apply_baseline, rope_apply_loopy_roll


def wan_rope_apply(
    x: Tensor,
    grid_sizes: Tensor | Sequence[Sequence[int]],
    freqs: Tensor,
    *,
    time_shift: int = 0,
    enabled: bool = True,
) -> Tensor:
    """Wan Q/K RoPE entry used by Mode B.

    When ``enabled`` is False or ``time_shift == 0``, behaviour matches official
    Wan ``rope_apply`` (baseline / Mode A path through the same function).
    """
    if not enabled:
        return rope_apply_baseline(x, grid_sizes, freqs)
    return rope_apply_loopy_roll(x, grid_sizes, freqs, time_shift=time_shift)
