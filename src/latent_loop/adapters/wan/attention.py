"""Hook official Wan ``WanSelfAttention`` for Mode B Loopy-style RoPE roll.

Does not import Wan at module load time. The patch expects the official
signature::

    forward(x, seq_lens, grid_sizes, freqs) -> Tensor

and modules with ``q/k/v/o``, ``norm_q/norm_k``, ``num_heads``, ``head_dim``,
optional ``window_size``.

Schedule is injected (default ``SymmetricShiftSchedule``), never hard-coded into
the attention body.
"""

from __future__ import annotations

import types
from typing import Any, Callable

import torch
from torch import Tensor, nn

from latent_loop.adapters.wan.rope import wan_rope_apply
from latent_loop.rope.schedule import (
    IdentityShiftSchedule,
    SymmetricShiftSchedule,
    TemporalShiftSchedule,
)


def _find_self_attn_modules(model: nn.Module) -> list[tuple[int, nn.Module]]:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError("model has no .blocks; not an official WanModel layout")

    found: list[tuple[int, nn.Module]] = []
    for idx, block in enumerate(blocks):
        attn = getattr(block, "self_attn", None) or getattr(block, "attn", None)
        if attn is None:
            for name, child in block.named_children():
                if "self" in name.lower() and hasattr(child, "q") and hasattr(child, "k"):
                    attn = child
                    break
        if attn is None:
            raise AttributeError(f"blocks[{idx}] has no self-attention module")
        found.append((idx, attn))
    return found


def enable_mode_b_on_wan_model(
    model: nn.Module,
    *,
    schedule: TemporalShiftSchedule | None = None,
    flash_attention_fn: Callable[..., Tensor] | None = None,
    enabled: bool = True,
) -> list[int]:
    """Patch each ``WanSelfAttention.forward`` for Mode B.

    When ``enabled`` is False, installs Identity schedule behaviour (baseline
    RoPE only). Returns the shift table for the current ``num_latent_frames``
    preview using ``F`` from the first forward's ``grid_sizes`` is runtime;
    here we return shifts only if ``model`` already stores
    ``_latent_loop_num_latent_frames``, else an empty list after patching.

    Returns:
        Per-block shifts if ``num_latent_frames`` is known on the model,
        otherwise ``[]`` (shifts are still computed per forward from grid_sizes).
    """
    if flash_attention_fn is None:
        try:
            from wan.modules.attention import flash_attention as flash_attention_fn  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised when wan missing
            raise ImportError(
                "flash_attention_fn is required when wan.modules.attention "
                "is not importable"
            ) from exc

    sched: TemporalShiftSchedule = schedule or SymmetricShiftSchedule()
    if not enabled:
        sched = IdentityShiftSchedule()

    for block_idx, attn in _find_self_attn_modules(model):
        _patch_one_attn(attn, block_idx, sched, flash_attention_fn, enabled=enabled)

    model._latent_loop_mode_b = bool(enabled)  # type: ignore[attr-defined]
    model._latent_loop_shift_schedule = sched  # type: ignore[attr-defined]

    f = getattr(model, "_latent_loop_num_latent_frames", None)
    if f is None:
        return []
    return [sched.time_shift(i, int(f)) for i in range(len(model.blocks))]


def disable_mode_b_on_wan_model(model: nn.Module) -> int:
    """Restore original attention forwards. Returns number of restored layers."""
    restored = 0
    for _, attn in _find_self_attn_modules(model):
        original = getattr(attn, "_latent_loop_original_forward", None)
        if original is None:
            continue
        attn.forward = original  # type: ignore[method-assign]
        for attr in (
            "_latent_loop_mode_b_patched",
            "_latent_loop_original_forward",
            "_latent_loop_block_idx",
            "_latent_loop_schedule",
            "_latent_loop_enabled",
            "_latent_loop_flash_attention",
        ):
            if hasattr(attn, attr):
                delattr(attn, attr)
        restored += 1
    for attr in ("_latent_loop_mode_b", "_latent_loop_shift_schedule"):
        if hasattr(model, attr):
            delattr(model, attr)
    return restored


def _patch_one_attn(
    attn: nn.Module,
    block_idx: int,
    schedule: TemporalShiftSchedule,
    flash_attention_fn: Callable[..., Tensor],
    *,
    enabled: bool,
) -> None:
    attn._latent_loop_schedule = schedule  # type: ignore[attr-defined]
    attn._latent_loop_enabled = enabled  # type: ignore[attr-defined]
    attn._latent_loop_flash_attention = flash_attention_fn  # type: ignore[attr-defined]
    attn._latent_loop_block_idx = block_idx  # type: ignore[attr-defined]

    if getattr(attn, "_latent_loop_mode_b_patched", False):
        return

    original_forward = attn.forward

    def forward_mode_b(
        self: Any,
        x: Tensor,
        seq_lens: Tensor,
        grid_sizes: Tensor,
        freqs: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        f = int(grid_sizes[0][0].item()) if torch.is_tensor(grid_sizes) else int(grid_sizes[0][0])
        sched: TemporalShiftSchedule = self._latent_loop_schedule
        use_b = bool(self._latent_loop_enabled)
        shift = sched.time_shift(int(self._latent_loop_block_idx), f) if use_b else 0
        flash = self._latent_loop_flash_attention

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        window_size = getattr(self, "window_size", (-1, -1))
        out = flash(
            q=wan_rope_apply(q, grid_sizes, freqs, time_shift=shift, enabled=use_b),
            k=wan_rope_apply(k, grid_sizes, freqs, time_shift=shift, enabled=use_b),
            v=v,
            k_lens=seq_lens,
            window_size=window_size,
        )
        out = out.flatten(2)
        return self.o(out)

    attn.forward = types.MethodType(forward_mode_b, attn)  # type: ignore[method-assign]
    attn._latent_loop_mode_b_patched = True  # type: ignore[attr-defined]
    attn._latent_loop_original_forward = original_forward  # type: ignore[attr-defined]
