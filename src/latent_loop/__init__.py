"""latent-loop — Circular Temporal RoPE for looping video (Wan Mode B).

Mainline exports are under ``latent_loop.rope`` and ``latent_loop.adapters.wan``.
``RingLatentProcessor`` residual mix is experimental / Mode C only.
"""

from .adapters.wan import disable_mode_b_on_wan_model, enable_mode_b_on_wan_model
from .ring import (
    RingLatentProcessor,
    RingMixConfig,
    circular_indices,
    circular_pad,
    circular_temporal_mix,
    cosine_denoise_strength,
    ring_windows,
)
from .rope.schedule import (
    FixedShiftSchedule,
    IdentityShiftSchedule,
    LoopyShiftSchedule,
    SymmetricShiftSchedule,
)

__all__ = [
    "SymmetricShiftSchedule",
    "LoopyShiftSchedule",
    "FixedShiftSchedule",
    "IdentityShiftSchedule",
    "enable_mode_b_on_wan_model",
    "disable_mode_b_on_wan_model",
    "RingLatentProcessor",
    "RingMixConfig",
    "circular_indices",
    "circular_pad",
    "circular_temporal_mix",
    "cosine_denoise_strength",
    "ring_windows",
]
