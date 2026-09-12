# Development status

> **D1 done:** Late Anchor Release lowers `seam_vs_adj` a bit but **does not** kill the late F-1→0 / 0→1 double spike.  
> Next design candidate (not started): **D2 circular soft conditioning** around index 0.

## D1 late numbers

| case | B seam_vs_adj | D1 seam_vs_adj | note |
|------|--------------:|---------------:|------|
| person | 1.77 | 1.37 | F-1→0↓ but 0→1↑ |
| environment | 1.56 | 1.42 | same pattern |

## Active stack

Symmetric Circular Temporal RoPE + official/Wan-style I2V frame-0 clamp (B) or D1 late release.  
Module: `latent_loop.i2v_conditioning`.

## Held

- No Circular Temporal Context as first fix  
- No S4/S5  
- No D2 until green-light  

Details: `artifacts/mode_b_i2v_d1/RESULT.md`
