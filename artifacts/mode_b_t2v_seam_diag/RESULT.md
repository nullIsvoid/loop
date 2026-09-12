# Mode B Symmetric T2V latent probe — control result

## Question

Does the late seam spike exist **without** Wan I2V first-frame hard anchoring?

## Setup

- Mode B `SymmetricShiftSchedule`
- **T2V** (`img=None`), official `masks_like(..., zero=False)` — no per-step clamp of latent 0
- seed=42, 1280×704, 81 frames (F=21), 20 steps, probes 2 / 10 / 19
- candle prompt (same family as early T2V runs)

## Latent L2 chain (T2V)

| stage | F-3→F-2 | F-2→F-1 | F-1→0 | 0→1 | 1→2 | seam_vs_adj |
|-------|--------:|--------:|------:|----:|----:|------------:|
| early | 560 | 560 | 563 | 563 | 560 | **1.004** |
| middle | 470 | 470 | 473 | 473 | 471 | **1.004** |
| late | 328 | 316 | 321 | 311 | 323 | **1.006** |

## vs I2V Mode B (prior)

I2V late: F-1→0 and 0→1 both spike (`seam_vs_adj` ~1.6–1.8).  
T2V late: all five gaps stay flat (~1.0).

## Decision

**T2V late stays normal** → residual I2V seam is primarily the **first-frame hard conditioning boundary**, not missing Circular Temporal Context.

**Next research (when scoped):** Circular I2V Conditioning — not Circular Temporal Context as the first D.  
Still no S4/S5. Still no D implementation in this commit beyond recording the control.
