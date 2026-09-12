"""D3: Circular Reference Residual Conditioning.

Propagate a reference *correction* around the ring instead of copying ref0:

    delta = ref0 - generated_0
    x_i = generated_i + w_i * delta

with smooth full-ring cosine weights (no finite radius cutoff).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .circular_soft import ring_distance


@dataclass(frozen=True)
class FullRingCosineProfile:
    """Periodic cosine over the whole temporal ring.

    ``w(d) = 0.5 * (1 + cos(pi * d / Dmax))`` with ``d = min(i, F-i)``
    and ``Dmax = F // 2`` so ``w(0)=1`` and ``w(Dmax)=0``.
    """

    def weights(self, length: int) -> list[float]:
        if length < 2:
            raise ValueError("length must be >= 2")
        dmax = length // 2
        out: list[float] = []
        for i in range(length):
            d = ring_distance(i, length)
            # Guard float edges: d==0 → 1, d==dmax → 0
            w = 0.5 * (1.0 + math.cos(math.pi * d / dmax))
            out.append(float(w))
        return out


def apply_circular_reference_residual(
    latent: Tensor,
    reference: Tensor,
    weights: list[float] | Tensor,
) -> Tensor:
    """Apply ``x_i = g_i + w_i * (ref0 - g_0)``.

    With ``w_0 = 1``, this forces ``x_0 = ref0``. Other frames keep their
    own generated content plus a shared residual correction.
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
    delta = ref0 - latent[:, 0]
    w = torch.tensor(w_list, device=latent.device, dtype=latent.dtype).view(1, f, 1, 1)
    return latent + w * delta.unsqueeze(1)
