"""Hunyuan attention Mode-B enable/disable — skeleton only.

Official attention path:

- ``parallel_attention`` / ``sequence_parallel_attention``
  in ``hyvideo.models.transformers.modules.attention``
- RoPE is applied **before** attention on image Q/K only

No monkey-patch until Phase D.
"""

from __future__ import annotations


def enable_mode_b_on_hunyuan_transformer(*_args, **_kwargs) -> None:
    raise NotImplementedError(
        "Hunyuan Symmetric Circular Temporal RoPE is Phase D — "
        "complete A14B control + Hunyuan native H0 probe first."
    )


def disable_mode_b_on_hunyuan_transformer(*_args, **_kwargs) -> None:
    raise NotImplementedError(
        "Hunyuan Mode-B disable is Phase D — not implemented yet."
    )
