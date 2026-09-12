"""D2: Circular Soft I2V Conditioning — ring weights around temporal index 0.

Spreads the reference image latent (and matching timestep factors) onto
neighbors by ring distance ``d(i) = min(i, F - i)``. Latent and timestep
stay coupled: ``t_i = (1 - w_i) * current_t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def ring_distance(index: int, length: int) -> int:
    """Circular distance from temporal index 0 on a ring of ``length``."""
    if length < 1:
        raise ValueError("length must be >= 1")
    i = int(index) % length
    return min(i, length - i)


@dataclass(frozen=True)
class Radius2CosineProfile:
    """Fixed first D2 profile (no denoise schedule).

    d=0 → 1.00, d=1 → 0.75, d=2 → 0.25, d≥3 → 0.
    """

    weights_by_distance: tuple[float, ...] = (1.0, 0.75, 0.25)

    def weight_at_distance(self, d: int) -> float:
        if d < 0:
            raise ValueError("distance must be >= 0")
        table = self.weights_by_distance
        if d < len(table):
            return float(table[d])
        return 0.0

    def weights(self, length: int) -> list[float]:
        if length < 1:
            raise ValueError("length must be >= 1")
        return [
            self.weight_at_distance(ring_distance(i, length)) for i in range(length)
        ]


def apply_circular_latent_conditioning(
    latent: Tensor,
    reference: Tensor,
    weights: list[float] | Tensor,
) -> Tensor:
    """Blend each temporal index toward ``reference[:, 0]`` by weight ``w_i``.

    ``latent``: ``[C, F, H, W]``.
    ``reference``: ``[C, 1, H, W]`` or ``[C, F, H, W]`` (uses index 0).
    ``weights``: length ``F``, values in ``[0, 1]``.
    """
    if latent.ndim != 4 or reference.ndim != 4:
        raise ValueError("latent and reference must be [C, F, H, W]")
    if latent.shape[0] != reference.shape[0] or latent.shape[2:] != reference.shape[2:]:
        raise ValueError(
            f"C/H/W mismatch latent={tuple(latent.shape)} ref={tuple(reference.shape)}"
        )
    if reference.shape[1] not in (1, latent.shape[1]):
        raise ValueError(
            f"reference F must be 1 or {latent.shape[1]}, got {reference.shape[1]}"
        )

    f = latent.shape[1]
    if isinstance(weights, Tensor):
        w_list = [float(x) for x in weights.detach().cpu().flatten().tolist()]
    else:
        w_list = [float(x) for x in weights]
    if len(w_list) != f:
        raise ValueError(f"weights length {len(w_list)} != latent F {f}")
    for wi in w_list:
        if wi < 0.0 or wi > 1.0:
            raise ValueError(f"weight must be in [0, 1], got {wi}")

    ref0 = reference[:, 0]
    w = torch.tensor(w_list, device=latent.device, dtype=latent.dtype).view(1, f, 1, 1)
    ref_b = ref0.unsqueeze(1).expand(-1, f, -1, -1)
    return (1.0 - w) * latent + w * ref_b


def apply_circular_timestep_mask(
    official_mask: Tensor,
    weights: list[float] | Tensor,
) -> Tensor:
    """Per-token timestep scale: ``scale_i = 1 - w_i`` so ``t_i = scale_i * t``.

    ``official_mask``: Wan ``mask2[0]`` shape ``[C, F, H, W]`` (values ignored;
    only shape / device used).
    """
    if official_mask.ndim != 4:
        raise ValueError("official_mask must be [C, F, H, W]")
    f = official_mask.shape[1]
    if isinstance(weights, Tensor):
        w_list = [float(x) for x in weights.detach().cpu().flatten().tolist()]
    else:
        w_list = [float(x) for x in weights]
    if len(w_list) != f:
        raise ValueError(f"weights length {len(w_list)} != mask F {f}")

    out = official_mask.to(dtype=torch.float32).clone()
    for i, wi in enumerate(w_list):
        if wi < 0.0 or wi > 1.0:
            raise ValueError(f"weight must be in [0, 1], got {wi}")
        out[:, i] = 1.0 - wi
    return out
