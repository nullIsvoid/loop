"""Circular temporal operators for latent-space video loops.

Time is treated as a ring, not a line. Latent 0 is not copied to the end;
instead temporal indexing wraps modulo T so the last and first latents are
true neighbours throughout the operation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import Tensor


def _normalize_dim(ndim: int, dim: int) -> int:
    dim = dim if dim >= 0 else ndim + dim
    if dim < 0 or dim >= ndim:
        raise IndexError(f"temporal_dim={dim} out of range for ndim={ndim}")
    return dim


def circular_indices(
    length: int,
    centers: Tensor | Iterable[int],
    offsets: Tensor | Iterable[int],
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return modulo-wrapped temporal indices.

    Example: T=6, center=0, offsets=(-1, 0, 1) -> [5, 0, 1].
    """
    if length < 2:
        raise ValueError("a temporal ring requires at least two positions")

    centers_t = torch.as_tensor(centers, dtype=torch.long, device=device).reshape(-1, 1)
    offsets_t = torch.as_tensor(offsets, dtype=torch.long, device=device).reshape(1, -1)
    return torch.remainder(centers_t + offsets_t, length)


def circular_pad(x: Tensor, pad: int, *, temporal_dim: int = 2) -> Tensor:
    """Circularly pad a tensor on the temporal axis.

    Common video-latent layout is [B, C, T, H, W], but any dimensionality is
    supported through ``temporal_dim``.
    """
    if pad < 0:
        raise ValueError("pad must be >= 0")
    if pad == 0:
        return x

    dim = _normalize_dim(x.ndim, temporal_dim)
    t = x.shape[dim]
    if t < 2:
        raise ValueError("temporal dimension must contain at least two latents")
    if pad > t:
        raise ValueError("pad cannot be larger than the temporal length")

    left = x.narrow(dim, t - pad, pad)
    right = x.narrow(dim, 0, pad)
    return torch.cat((left, x, right), dim=dim)


def ring_windows(x: Tensor, radius: int, *, temporal_dim: int = 2) -> Tensor:
    """Return a wrapped temporal neighbourhood for every latent position.

    For [B,C,T,H,W] with radius=1, result shape is [B,C,T,3,H,W]. At t=0,
    the neighbourhood is [T-1, 0, 1].
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")

    dim = _normalize_dim(x.ndim, temporal_dim)
    t = x.shape[dim]
    if t < 2:
        raise ValueError("temporal dimension must contain at least two latents")

    offsets = torch.arange(-radius, radius + 1, device=x.device)
    centers = torch.arange(t, device=x.device)
    idx = circular_indices(t, centers, offsets, device=x.device)

    moved = x.movedim(dim, 0)   # [T, dims-before-T, dims-after-T]
    windows = moved[idx]        # [T, K, dims-before-T, dims-after-T]

    # Rebuild the original dimension order and insert K directly after T.
    # Example for [B,C,T,H,W]: [T,K,B,C,H,W] -> [B,C,T,K,H,W].
    before = list(range(2, 2 + dim))
    after = list(range(2 + dim, windows.ndim))
    windows = windows.permute(before + [0, 1] + after)
    return windows


def circular_temporal_mix(
    x: Tensor,
    *,
    radius: int = 1,
    strength: float = 0.15,
    temporal_dim: int = 2,
    sigma: float | None = None,
) -> Tensor:
    """Mix each latent with ring neighbours while preserving sequence length.

    This is not endpoint interpolation and never duplicates a frame. It is a
    small residual coupling primitive intended for use inside denoising.
    """
    if radius < 1:
        raise ValueError("radius must be >= 1")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if strength == 0.0:
        return x

    original_temporal_dim = _normalize_dim(x.ndim, temporal_dim)
    windows = ring_windows(x, radius, temporal_dim=temporal_dim)
    k = 2 * radius + 1

    if sigma is None:
        sigma = max(radius / 2.0, 0.5)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    offsets = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
    weights = torch.exp(-0.5 * (offsets / sigma) ** 2)
    weights[radius] = 0.0  # exclude self from neighbour residual
    weights /= weights.sum()
    weights = weights.to(dtype=x.dtype)

    window_dim = original_temporal_dim + 1
    shape = [1] * windows.ndim
    shape[window_dim] = k
    neighbour_mean = (windows * weights.reshape(shape)).sum(dim=window_dim)
    return torch.lerp(x, neighbour_mean, strength)


def cosine_denoise_strength(
    step: int,
    total_steps: int,
    *,
    maximum: float = 0.15,
    active_fraction: float = 0.70,
) -> float:
    """Fade circular coupling out before final-detail denoising steps."""
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")
    if step < 0 or step >= total_steps:
        raise ValueError("step must satisfy 0 <= step < total_steps")
    if not 0.0 < active_fraction <= 1.0:
        raise ValueError("active_fraction must be in (0, 1]")
    if not 0.0 <= maximum <= 1.0:
        raise ValueError("maximum must be in [0, 1]")

    active_steps = max(1, math.ceil(total_steps * active_fraction))
    if step >= active_steps:
        return 0.0
    progress = step / max(active_steps - 1, 1)
    return maximum * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class RingMixConfig:
    radius: int = 1
    maximum_strength: float = 0.15
    active_fraction: float = 0.70
    sigma: float | None = None
    temporal_dim: int = 2


class RingLatentProcessor:
    """Sampler-facing adapter for step-wise circular latent coupling."""

    def __init__(self, config: RingMixConfig | None = None) -> None:
        self.config = config or RingMixConfig()

    def __call__(self, latents: Tensor, *, step: int, total_steps: int) -> Tensor:
        strength = cosine_denoise_strength(
            step,
            total_steps,
            maximum=self.config.maximum_strength,
            active_fraction=self.config.active_fraction,
        )
        if strength == 0.0:
            return latents
        return circular_temporal_mix(
            latents,
            radius=self.config.radius,
            strength=strength,
            temporal_dim=self.config.temporal_dim,
            sigma=self.config.sigma,
        )
