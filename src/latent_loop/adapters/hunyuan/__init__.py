"""HunyuanVideo-1.5 adapters.

Circular Temporal RoPE: ``attention.enable_mode_b_on_hunyuan_transformer``
(see ``notes/hunyuan_h1_rope_call_chain.md``).
"""

from . import attention, conditioning, rope

__all__ = ["attention", "conditioning", "rope"]
