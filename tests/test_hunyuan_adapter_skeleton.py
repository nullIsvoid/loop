"""Smoke tests for Hunyuan adapter skeleton (no model load)."""

from latent_loop.adapters.hunyuan.conditioning import describe_i2v_conditioning
from latent_loop.adapters.hunyuan.rope import describe_official_rope_layout


def test_hunyuan_rope_layout_metadata():
    meta = describe_official_rope_layout()
    assert meta["rope_dim_list_default"] == [16, 56, 56]
    assert meta["circular_mode_b"] == "not_implemented"


def test_hunyuan_conditioning_is_non_destructive():
    meta = describe_i2v_conditioning()
    assert meta["destructive_frame0_overwrite"] is False
    assert meta["frame0_timestep_forced_zero"] is False
