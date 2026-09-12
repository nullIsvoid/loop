# Development status

> **Next: residual-seam localization (B only).** No S4/S5. No sample expansion.  
> Decoded frame+flow diagnostics landed; latent early/mid/late probes next when cloud is up.

## Current latent handling

**Active:** Circular Temporal RoPE on attention Q/K (`SymmetricShiftSchedule`), V untouched.  
**Not active:** residual ring mix, true periodic RoPE, circular temporal context.  
Details: `notes/current_latent_path.md`.

## Mode B I2V — human 2026-09-12

| case | A | B | winner |
|--|--|--|--|
| person | more obvious jitter | slight jitter | **B** |
| environment | more obvious jitter | slight jitter | **B** |

## Seam diagnostics (B only) — in progress

Decoded 75–80 / 0–5 + flow strips: `artifacts/mode_b_i2v_seam_diag/`.  
Person: seam flow ≫ adj + **dx sign flip**. Environment: milder mag bump.  
Latent probes (steps 2/10/19): script ready, cloud run pending.

## Still forbidden

- No new schedules for curiosity  
- No seam_metrics → Gate/Repair  
- No implementing D (periodic RoPE / circular context) until latent stage is classified  
