"""Enable/disable Symmetric Circular Temporal RoPE on HunyuanVideo-1.5.

Patches ``MMDoubleStreamBlock`` / ``MMSingleStreamBlock`` so each block rolls
shared ``freqs_cis`` on the temporal axis before ``apply_rotary_emb`` (img Q/K).

Global block index: double blocks first, then single blocks — matches
``SymmetricShiftSchedule`` ``0,+1,-1,+2,-2,...``.
"""

from __future__ import annotations

import types
from typing import Any

from torch import nn

from latent_loop.adapters.hunyuan.rope import roll_hunyuan_freqs_cis
from latent_loop.rope.schedule import (
    IdentityShiftSchedule,
    SymmetricShiftSchedule,
    TemporalShiftSchedule,
)


def _rope_sizes(transformer: nn.Module) -> tuple[int, int, int]:
    sizes = getattr(transformer, "_latent_loop_rope_sizes", None)
    if sizes is None:
        raise RuntimeError(
            "Hunyuan Mode B: rope sizes unknown — get_rotary_pos_embed "
            "wrapper did not run yet (enable Mode B before transformer forward)"
        )
    tt, th, tw = (int(sizes[0]), int(sizes[1]), int(sizes[2]))
    if tt < 2:
        raise ValueError(f"need tt>=2 for circular temporal RoPE, got {tt}")
    return tt, th, tw


def _wrap_get_rotary_pos_embed(transformer: nn.Module) -> None:
    if getattr(transformer, "_latent_loop_original_get_rotary", None) is not None:
        return
    original = transformer.get_rotary_pos_embed
    transformer._latent_loop_original_get_rotary = original  # type: ignore[attr-defined]

    def wrapped(_self: nn.Module, rope_sizes):
        out = original(rope_sizes)
        transformer._latent_loop_rope_sizes = tuple(int(x) for x in rope_sizes)  # type: ignore[attr-defined]
        transformer._latent_loop_num_latent_frames = int(rope_sizes[0])  # type: ignore[attr-defined]
        return out

    transformer.get_rotary_pos_embed = types.MethodType(wrapped, transformer)  # type: ignore[method-assign]


def _maybe_roll_freqs(
    freqs_cis: Any,
    *,
    transformer: nn.Module,
    global_idx: int,
    schedule: TemporalShiftSchedule,
    enabled: bool,
) -> Any:
    if freqs_cis is None or not enabled:
        return freqs_cis
    tt, th, tw = _rope_sizes(transformer)
    shift = schedule.time_shift(global_idx, tt)
    return roll_hunyuan_freqs_cis(
        freqs_cis, tt=tt, th=th, tw=tw, time_shift=shift
    )


def _patch_block_forward(
    block: nn.Module,
    *,
    global_idx: int,
    schedule: TemporalShiftSchedule,
    enabled: bool,
    transformer: nn.Module,
) -> None:
    if getattr(block, "_latent_loop_mode_b_patched", False):
        # Update live knobs if re-enabled.
        block._latent_loop_global_idx = global_idx  # type: ignore[attr-defined]
        block._latent_loop_schedule = schedule  # type: ignore[attr-defined]
        block._latent_loop_enabled = enabled  # type: ignore[attr-defined]
        block._latent_loop_transformer = transformer  # type: ignore[attr-defined]
        return

    original = block.forward
    block._latent_loop_original_forward = original  # type: ignore[attr-defined]
    block._latent_loop_mode_b_patched = True  # type: ignore[attr-defined]
    block._latent_loop_global_idx = global_idx  # type: ignore[attr-defined]
    block._latent_loop_schedule = schedule  # type: ignore[attr-defined]
    block._latent_loop_enabled = enabled  # type: ignore[attr-defined]
    block._latent_loop_transformer = transformer  # type: ignore[attr-defined]

    def forward(self: nn.Module, *args, **kwargs):
        if "freqs_cis" in kwargs and kwargs["freqs_cis"] is not None:
            kwargs = {
                **kwargs,
                "freqs_cis": _maybe_roll_freqs(
                    kwargs["freqs_cis"],
                    transformer=self._latent_loop_transformer,
                    global_idx=self._latent_loop_global_idx,
                    schedule=self._latent_loop_schedule,
                    enabled=self._latent_loop_enabled,
                ),
            }
        return original(*args, **kwargs)

    block.forward = types.MethodType(forward, block)  # type: ignore[method-assign]


def enable_mode_b_on_hunyuan_transformer(
    transformer: nn.Module,
    *,
    schedule: TemporalShiftSchedule | None = None,
    enabled: bool = True,
) -> list[int]:
    """Patch Hunyuan double/single stream blocks for Mode B.

    Returns preview shifts using ``transformer._latent_loop_num_latent_frames``
    if set, else ``[]`` (shifts still computed at runtime from rope sizes).
    """
    double = getattr(transformer, "double_blocks", None)
    single = getattr(transformer, "single_blocks", None)
    if double is None or single is None:
        raise AttributeError(
            "transformer missing double_blocks/single_blocks — not HunyuanVideo_1_5"
        )

    sched: TemporalShiftSchedule = schedule or SymmetricShiftSchedule()
    if not enabled:
        sched = IdentityShiftSchedule()

    _wrap_get_rotary_pos_embed(transformer)

    n_double = len(double)
    for i, block in enumerate(double):
        _patch_block_forward(
            block,
            global_idx=i,
            schedule=sched,
            enabled=enabled,
            transformer=transformer,
        )
    for j, block in enumerate(single):
        _patch_block_forward(
            block,
            global_idx=n_double + j,
            schedule=sched,
            enabled=enabled,
            transformer=transformer,
        )

    transformer._latent_loop_mode_b = bool(enabled)  # type: ignore[attr-defined]
    transformer._latent_loop_shift_schedule = sched  # type: ignore[attr-defined]
    transformer._latent_loop_n_double = n_double  # type: ignore[attr-defined]

    f = getattr(transformer, "_latent_loop_num_latent_frames", None)
    n_layers = n_double + len(single)
    if f is None:
        return []
    return [sched.time_shift(i, int(f)) for i in range(n_layers)]


def disable_mode_b_on_hunyuan_transformer(transformer: nn.Module) -> int:
    """Restore original block forwards / get_rotary_pos_embed. Returns restore count."""
    restored = 0
    for group_name in ("double_blocks", "single_blocks"):
        blocks = getattr(transformer, group_name, None)
        if blocks is None:
            continue
        for block in blocks:
            original = getattr(block, "_latent_loop_original_forward", None)
            if original is None:
                continue
            block.forward = original  # type: ignore[method-assign]
            for attr in (
                "_latent_loop_mode_b_patched",
                "_latent_loop_original_forward",
                "_latent_loop_global_idx",
                "_latent_loop_schedule",
                "_latent_loop_enabled",
                "_latent_loop_transformer",
            ):
                if hasattr(block, attr):
                    delattr(block, attr)
            restored += 1

    orig_rope = getattr(transformer, "_latent_loop_original_get_rotary", None)
    if orig_rope is not None:
        transformer.get_rotary_pos_embed = orig_rope  # type: ignore[method-assign]
        delattr(transformer, "_latent_loop_original_get_rotary")

    for attr in (
        "_latent_loop_mode_b",
        "_latent_loop_shift_schedule",
        "_latent_loop_n_double",
        "_latent_loop_rope_sizes",
        "_latent_loop_num_latent_frames",
    ):
        if hasattr(transformer, attr):
            delattr(transformer, attr)
    return restored
