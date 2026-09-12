# Development status

> **D2 in flight:** Circular Soft I2V Conditioning (radius-2, fixed every step).  
> Symmetric RoPE unchanged. No late release. No Circular Temporal Context.

## Settled so far

| knife | change | late outcome |
|-------|--------|--------------|
| B | hard latent + hard t0=0 | double spike at 0 |
| D1 | soft latent + hard t0 | F-1→0 ↓, 0→1 ↑ |
| D1b | soft latent + soft t0 | still double spike; env worse |

Single-point conditioning path closed (`1a8f63a`).

## D2 = Circular Soft Conditioning

Ring distance `d=min(i,F-i)`; weights 1.0 / 0.75 / 0.25 / 0 for d=0..≥3.  
Coupled: `x_i=w·ref+(1-w)·gen`, `t_i=(1-w)·t`. Same weights all 20 steps.  
Probe F-4..4 to detect pushed seams. Compare B vs D2 only.

Script: `scripts/cloud_mode_b_i2v_d2_compare.py` → `artifacts/mode_b_i2v_d2/`.
