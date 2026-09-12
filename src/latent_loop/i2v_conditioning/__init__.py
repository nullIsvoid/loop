"""I2V first-frame conditioning experiments (Circular I2V Conditioning track)."""

from .anchor_schedule import (
    Frame0AnchorSchedule,
    HardAnchorSchedule,
    LateAnchorReleaseSchedule,
    apply_frame0_anchor,
    apply_frame0_timestep_mask,
    frame0_timestep_factor,
    list_anchor_strengths,
)
from .circular_soft import (
    Radius2CosineProfile,
    apply_circular_latent_conditioning,
    apply_circular_timestep_mask,
    ring_distance,
)

__all__ = [
    "Frame0AnchorSchedule",
    "HardAnchorSchedule",
    "LateAnchorReleaseSchedule",
    "Radius2CosineProfile",
    "apply_circular_latent_conditioning",
    "apply_circular_timestep_mask",
    "apply_frame0_anchor",
    "apply_frame0_timestep_mask",
    "frame0_timestep_factor",
    "list_anchor_strengths",
    "ring_distance",
]
