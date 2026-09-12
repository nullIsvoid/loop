"""Enable/disable Symmetric Circular Temporal RoPE on HunyuanVideo-1.5.

Uses ``register_forward_pre_hook(..., with_kwargs=True)`` so Diffusers group
offloading hooks on ``forward`` stay intact (replacing ``forward`` breaks them).

Global block index: double blocks first, then single blocks — matches
``SymmetricShiftSchedule`` ``0,+1,-1,+2,-2,...``.

Actual **480p_i2v** runtime (H0/H1 measured): ``double_blocks=54``,
``single_blocks=0`` — do not assume the older 20+40 layout.
"""

from __future__ import annotations

import types
from typing import Any, Iterable

from torch import nn
from torch.utils.hooks import RemovableHandle

from latent_loop.adapters.hunyuan.rope import roll_hunyuan_freqs_cis
from latent_loop.rope.schedule import (
    IdentityShiftSchedule,
    SymmetricShiftSchedule,
    TemporalShiftSchedule,
)


def parse_active_block_spec(spec: str | None, n_layers: int) -> set[int] | None:
    """Parse ``all`` / empty → None (all layers); else ``0-17`` / ``0,5,9``.

    Indices are **global** block indices (0 .. n_layers-1). Inclusive ranges.
    """
    if spec is None:
        return None
    text = str(spec).strip().lower()
    if not text or text in {"all", "*", "full"}:
        return None
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_s, b_s = part.split("-", 1)
            a, b = int(a_s), int(b_s)
            if a > b:
                a, b = b, a
            for i in range(a, b + 1):
                out.add(i)
        else:
            out.add(int(part))
    if not out:
        raise ValueError(f"empty active block set from spec={spec!r}")
    bad = [i for i in sorted(out) if i < 0 or i >= n_layers]
    if bad:
        raise ValueError(
            f"active blocks out of range [0, {n_layers}): {bad[:8]}..."
            if len(bad) > 8
            else f"active blocks out of range [0, {n_layers}): {bad}"
        )
    return out


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


def _make_pre_hook(
    *,
    transformer: nn.Module,
    global_idx: int,
    schedule: TemporalShiftSchedule,
    enabled: bool,
):
    def pre_hook(_module: nn.Module, args: tuple, kwargs: dict):
        freqs = kwargs.get("freqs_cis", None)
        if freqs is None:
            return args, kwargs
        kwargs = dict(kwargs)
        kwargs["freqs_cis"] = _maybe_roll_freqs(
            freqs,
            transformer=transformer,
            global_idx=global_idx,
            schedule=schedule,
            enabled=enabled,
        )
        return args, kwargs

    return pre_hook


def _attach_block_hook(
    block: nn.Module,
    *,
    global_idx: int,
    schedule: TemporalShiftSchedule,
    enabled: bool,
    transformer: nn.Module,
) -> None:
    # Remove previous Mode-B hook if re-enabling.
    old: RemovableHandle | None = getattr(block, "_latent_loop_pre_hook", None)
    if old is not None:
        old.remove()
        delattr(block, "_latent_loop_pre_hook")

    handle = block.register_forward_pre_hook(
        _make_pre_hook(
            transformer=transformer,
            global_idx=global_idx,
            schedule=schedule,
            enabled=enabled,
        ),
        with_kwargs=True,
    )
    block._latent_loop_pre_hook = handle  # type: ignore[attr-defined]
    block._latent_loop_global_idx = global_idx  # type: ignore[attr-defined]
    block._latent_loop_schedule = schedule  # type: ignore[attr-defined]
    block._latent_loop_enabled = enabled  # type: ignore[attr-defined]


def enable_mode_b_on_hunyuan_transformer(
    transformer: nn.Module,
    *,
    schedule: TemporalShiftSchedule | None = None,
    enabled: bool = True,
    active_block_indices: Iterable[int] | None = None,
) -> list[int]:
    """Install temporal freqs-roll pre-hooks on double/single stream blocks.

    ``active_block_indices``: if set, only those **global** indices roll.
    Other blocks keep native freqs (depth localization for H1-A/B/C).
    Shift amount still uses the full Symmetric schedule at that global index
    (not reindexed within the active subset).
    """
    double = getattr(transformer, "double_blocks", None)
    single = getattr(transformer, "single_blocks", None)
    if double is None:
        raise AttributeError(
            "transformer missing double_blocks — not HunyuanVideo_1_5"
        )
    if single is None:
        single = []

    sched: TemporalShiftSchedule = schedule or SymmetricShiftSchedule()
    if not enabled:
        sched = IdentityShiftSchedule()

    _wrap_get_rotary_pos_embed(transformer)

    n_double = len(double)
    n_single = len(single)
    n_layers = n_double + n_single
    active: set[int] | None
    if active_block_indices is None:
        active = None
    else:
        active = {int(i) for i in active_block_indices}
        bad = [i for i in sorted(active) if i < 0 or i >= n_layers]
        if bad:
            raise ValueError(
                f"active_block_indices out of range [0, {n_layers}): {bad}"
            )

    for i, block in enumerate(double):
        on = bool(enabled) and (active is None or i in active)
        _attach_block_hook(
            block,
            global_idx=i,
            schedule=sched,
            enabled=on,
            transformer=transformer,
        )
    for j, block in enumerate(single):
        g = n_double + j
        on = bool(enabled) and (active is None or g in active)
        _attach_block_hook(
            block,
            global_idx=g,
            schedule=sched,
            enabled=on,
            transformer=transformer,
        )

    transformer._latent_loop_mode_b = bool(enabled)  # type: ignore[attr-defined]
    transformer._latent_loop_shift_schedule = sched  # type: ignore[attr-defined]
    transformer._latent_loop_n_double = n_double  # type: ignore[attr-defined]
    transformer._latent_loop_n_single = n_single  # type: ignore[attr-defined]
    transformer._latent_loop_active_blocks = (  # type: ignore[attr-defined]
        None if active is None else sorted(active)
    )

    f = getattr(transformer, "_latent_loop_num_latent_frames", None)
    if f is None:
        return []
    # Effective shifts: inactive / disabled → 0
    out: list[int] = []
    for i in range(n_layers):
        on = bool(enabled) and (active is None or i in active)
        out.append(sched.time_shift(i, int(f)) if on else 0)
    return out


def disable_mode_b_on_hunyuan_transformer(transformer: nn.Module) -> int:
    """Remove Mode-B pre-hooks and restore get_rotary_pos_embed. Returns hook count."""
    restored = 0
    for group_name in ("double_blocks", "single_blocks"):
        blocks = getattr(transformer, group_name, None)
        if not blocks:
            continue
        for block in blocks:
            handle: RemovableHandle | None = getattr(block, "_latent_loop_pre_hook", None)
            if handle is not None:
                handle.remove()
                delattr(block, "_latent_loop_pre_hook")
                restored += 1
            for attr in (
                "_latent_loop_global_idx",
                "_latent_loop_schedule",
                "_latent_loop_enabled",
            ):
                if hasattr(block, attr):
                    delattr(block, attr)

    orig_rope = getattr(transformer, "_latent_loop_original_get_rotary", None)
    if orig_rope is not None:
        transformer.get_rotary_pos_embed = orig_rope  # type: ignore[method-assign]
        delattr(transformer, "_latent_loop_original_get_rotary")

    for attr in (
        "_latent_loop_mode_b",
        "_latent_loop_shift_schedule",
        "_latent_loop_n_double",
        "_latent_loop_n_single",
        "_latent_loop_active_blocks",
        "_latent_loop_rope_sizes",
        "_latent_loop_num_latent_frames",
    ):
        if hasattr(transformer, attr):
            delattr(transformer, attr)
    return restored
