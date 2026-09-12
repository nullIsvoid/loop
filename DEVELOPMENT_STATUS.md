# Development status

> **D1 in flight:** Late Anchor Release vs hard-anchor Symmetric I2V (person + environment).  
> RoPE unchanged. No Circular Temporal Context. No D2 yet.

## Causal chain (agreed)

T2V + Symmetric RoPE → latent seam flat.  
I2V + per-step hard clamp of frame 0 → late F-1→0 **and** 0→1 spike.

## D1 = Late Anchor Release

`latent0 = a(step)*ref0 + (1-a)*gen0`  
a: 0–12 → 1.0; then 0.90…0.00 by step 19.  
Module: `latent_loop.i2v_conditioning`. Script: `scripts/cloud_mode_b_i2v_d1_compare.py`.

## Still held

- No F-1 dual-nail to reference  
- No Circular Temporal Context as first D  
- No S4/S5  

Next after D1 results: if late double-spike dies → design D2 circular soft conditioning; else revisit.
