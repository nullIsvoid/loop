"""Injectable I2V first-frame (temporal index 0) anchor strength schedules.

Official Wan I2V hard-clamps latent[:, 0] to the reference image latent after
every denoise step. These schedules parameterize a soft blend:

    latent0 = a * reference0 + (1 - a) * generated0

``a = 1`` is hard clamp; ``a = 0`` is fully released.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class Frame0AnchorSchedule(Protocol):
    def strength(self, step_index: int, total_steps: int) -> float: ...


@dataclass(frozen=True)
class HardAnchorSchedule:
    """Official Wan I2V: frame 0 hard-clamped every step (a=1)."""

    def strength(self, step_index: int, total_steps: int) -> float:
        _ = step_index, total_steps
        return 1.0


@dataclass(frozen=True)
class LateAnchorReleaseSchedule:
    """D1: keep hard anchor early/mid, release during late denoise.

    Default table is for ``total_steps == 20`` as agreed:

        0..12 → 1.00
        13 → 0.90, 14 → 0.75, 15 → 0.60, 16 → 0.45,
        17 → 0.30, 18 → 0.15, 19 → 0.00

    For other step counts, indices are linearly mapped onto that 20-step curve.
    """

    # Explicit 20-step strengths (index = denoise step index).
    strengths_20: tuple[float, ...] = (
        *(1.0 for _ in range(13)),
        0.90,
        0.75,
        0.60,
        0.45,
        0.30,
        0.15,
        0.00,
    )

    def strength(self, step_index: int, total_steps: int) -> float:
        if step_index < 0:
            raise ValueError("step_index must be >= 0")
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if step_index >= total_steps:
            raise ValueError("step_index must be < total_steps")

        table = self.strengths_20
        if total_steps == len(table):
            return float(table[step_index])

        # Map step onto [0, 19] curve.
        pos = step_index * (len(table) - 1) / max(total_steps - 1, 1)
        i0 = int(pos)
        i1 = min(i0 + 1, len(table) - 1)
        t = pos - i0
        return float(table[i0] * (1.0 - t) + table[i1] * t)


def apply_frame0_anchor(
    latent: Tensor,
    reference: Tensor,
    strength: float,
) -> Tensor:
    """Blend temporal index 0 toward ``reference`` with strength ``a`` in [0, 1].

    ``latent`` / ``reference``: ``[C, F, H, W]``. Other frames unchanged.
    """
    a = float(strength)
    if a < 0.0 or a > 1.0:
        raise ValueError(f"strength must be in [0, 1], got {a}")
    if latent.shape != reference.shape:
        raise ValueError(
            f"shape mismatch latent={tuple(latent.shape)} ref={tuple(reference.shape)}"
        )
    if a == 1.0:
        out = latent.clone()
        out[:, 0] = reference[:, 0]
        return out
    if a == 0.0:
        return latent
    out = latent.clone()
    out[:, 0] = a * reference[:, 0] + (1.0 - a) * latent[:, 0]
    return out


def list_anchor_strengths(
    total_steps: int,
    schedule: Frame0AnchorSchedule | None = None,
) -> list[float]:
    sched: Frame0AnchorSchedule = schedule or HardAnchorSchedule()
    return [sched.strength(i, total_steps) for i in range(total_steps)]
