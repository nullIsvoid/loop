"""Hunyuan I2V conditioning notes as code — do not patch.

Official non-destructive path (Phase C must keep this intact):

1. ``get_image_condition_latents`` — VAE encode reference frame
2. ``_prepare_cond_latents`` — ref only at temporal 0 + mask channel
3. Denoise: ``cat([latents, cond_latents], dim=1)`` each step
4. ``scheduler.step`` updates **noise latents only**

Contrast Wan TI2V-5B: no post-step ``latent[:,:,0] = ref`` and no ``t0=0``.
"""

from __future__ import annotations


def describe_i2v_conditioning() -> dict:
    return {
        "style": "channel_concat_condition_plus_vision_tokens",
        "destructive_frame0_overwrite": False,
        "frame0_timestep_forced_zero": False,
        "pipeline_fns": [
            "HunyuanVideoPipeline.get_image_condition_latents",
            "HunyuanVideoPipeline._prepare_cond_latents",
            "HunyuanVideoPipeline._prepare_vision_states",
        ],
        "edit_policy": "forbidden_until_phase_D_greenlight",
    }
