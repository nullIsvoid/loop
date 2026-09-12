# Hunyuan anchor probe — first gate (blocks 0–4 only)

Benchmark: `person_loop_v1`. Method: Loopy-style half-cycle single-block temporal RoPE + frozen native transformer inputs.  
**`--blocks all` not run** (batch parent that would chain into all was killed).

## α_l (mean stepwise MSE)

| block | α |
|------:|--:|
| 0 | 0.000322 |
| 1 | 0.000530 |
| **2** | **0.004106** |
| 3 | 0.000506 |
| 4 | 0.000341 |

`alpha_summary.json`: `anchor_layer=2`, `anchor_alpha≈0.00411`.

## Gate vs Loopy prior

Paper prior for HunyuanVideo-1.5: **Block 2** strongest.  
Our 480p_i2v on this fixed prompt: **α2 ≫ α0,1,3,4** (~8× next-best α1).

→ Measurement looks **aligned** enough to consider full 0–53 — **only after explicit approval**.

## Caveat

Single prompt (`person_loop_v1`). Paper averages many prompts; treat block 2 as **candidate**, not universal yet.
