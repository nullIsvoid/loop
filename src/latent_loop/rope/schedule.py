"""Injectable per-block temporal shift schedules for Loopy-style RoPE roll.

Default policy mirrors Loopy; adapters must accept any ``TemporalShiftSchedule``
so later experiments (fixed offsets, symmetric ± shifts, true periodic RoPE,
Mobius latent shift) do not require rewriting attention hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class TemporalShiftSchedule(Protocol):
    """Maps ``(block_idx, num_latent_frames)`` → temporal freqs roll amount."""

    def time_shift(self, block_idx: int, num_latent_frames: int) -> int: ...


@dataclass(frozen=True)
class IdentityShiftSchedule:
    """Always 0 — Mode A / Mode B disabled path."""

    def time_shift(self, block_idx: int, num_latent_frames: int) -> int:
        _ = block_idx, num_latent_frames
        return 0


@dataclass(frozen=True)
class LoopyShiftSchedule:
    """Loopy default: block 0 anchor; others ``(block_idx - 1) % (F - 1) + 1``."""

    def time_shift(self, block_idx: int, num_latent_frames: int) -> int:
        if block_idx < 0:
            raise ValueError("block_idx must be >= 0")
        if num_latent_frames < 2:
            raise ValueError("num_latent_frames must be >= 2 for Loopy schedule")
        if block_idx == 0:
            return 0
        return (block_idx - 1) % (num_latent_frames - 1) + 1


@dataclass(frozen=True)
class FixedShiftSchedule:
    """Constant shift for non-anchor blocks; block 0 stays unshifted."""

    shift: int = 1

    def time_shift(self, block_idx: int, num_latent_frames: int) -> int:
        if num_latent_frames < 2:
            raise ValueError("num_latent_frames must be >= 2")
        if block_idx == 0:
            return 0
        return int(self.shift) % int(num_latent_frames)


@dataclass(frozen=True)
class SymmetricShiftSchedule:
    """Bidirectional circular propagation with block-0 anchor.

    Sequence of shifts (before modulo ``F``)::

        0, +1, -1, +2, -2, +3, -3, ...

    Applied as ``s % F`` so torch.roll stays on the temporal ring.
    """

    def time_shift(self, block_idx: int, num_latent_frames: int) -> int:
        if block_idx < 0:
            raise ValueError("block_idx must be >= 0")
        if num_latent_frames < 2:
            raise ValueError("num_latent_frames must be >= 2")
        if block_idx == 0:
            return 0
        k = (block_idx + 1) // 2
        signed = k if (block_idx % 2 == 1) else -k
        return int(signed % num_latent_frames)


def list_layer_time_shifts(
    num_layers: int,
    num_latent_frames: int,
    schedule: TemporalShiftSchedule | None = None,
) -> list[int]:
    """Materialise per-block shifts for logging / tests."""
    sched: TemporalShiftSchedule = schedule or LoopyShiftSchedule()
    return [sched.time_shift(i, num_latent_frames) for i in range(num_layers)]
