"""Hunyuan 3D RoPE helpers — skeleton only.

Do not implement Symmetric Circular Temporal RoPE here until Phase D
(after native H0 latent probe). Official hooks:

- ``hyvideo.models.transformers.hunyuanvideo_1_5_transformer
   .HunyuanVideo_1_5_Transformer.get_rotary_pos_embed``
- ``hyvideo.models.transformers.modules.posemb_layers
   .get_nd_rotary_pos_embed`` / ``apply_rotary_emb``

``rope_dim_list`` default ``[16, 56, 56]`` = (temporal, height, width).
"""

from __future__ import annotations


def describe_official_rope_layout() -> dict:
    """Return static metadata for docs/tests; no model import."""
    return {
        "rope_dim_list_default": [16, 56, 56],
        "axes": ["temporal", "height", "width"],
        "apply_sites": [
            "MMDoubleStreamBlock.forward → apply_rotary_emb(img_q, img_k)",
            "MMSingleStreamBlock.forward → apply_rotary_emb(img_q, img_k)",
        ],
        "circular_mode_b": "not_implemented",
    }
