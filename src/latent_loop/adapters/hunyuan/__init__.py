"""HunyuanVideo-1.5 adapters (skeleton).

Circular Temporal RoPE is **not** implemented here yet.
See ``notes/hunyuan_i2v_call_chain.md`` and ``call_chain.md``.
"""

from . import attention, conditioning, rope

__all__ = ["attention", "conditioning", "rope"]
