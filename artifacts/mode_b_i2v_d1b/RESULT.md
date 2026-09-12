# D1b Coupled Anchor Release — result

## Question

Does releasing **frame-0 timestep** together with **latent** (same `a` schedule) flatten late **F-1→0 / 0→1** vs D1 latent-only and B hard?

## Setup

- Symmetric Circular Temporal RoPE unchanged
- B = hard latent + hard `t0=0`
- D1 = soft latent + hard `t0=0`
- D1b = soft latent + soft `t0=(1-a)·t`
- person / environment, seed=42, 81 frames, 20 steps
- No F-1 / ring conditioning (no D2)

## Late latent L2 (step 19)

| case | variant | F-2→F-1 | F-1→0 | 0→1 | 1→2 | seam_vs_adj |
|------|---------|--------:|------:|----:|----:|------------:|
| person | B | 104 | 213 | 169 | 105 | **1.77** |
| person | D1 | 108 | 175 ↓ | 192 ↑ | 106 | **1.37** |
| person | D1b | 122 ↑ | 192 | 198 ↑ | 122 ↑ | **1.37** |
| environment | B | 157 | 258 | 218 | 142 | **1.56** |
| environment | D1 | 161 | 247 ↓ | 247 ↑ | 144 | **1.42** |
| environment | D1b | 168 | 271 ↑ | 260 ↑ | 151 | **1.49** |

Early/middle (a=1) identical across B/D1/D1b — release only hits late.

## Verdict

**D1b does not clear the double spike.** Coupling `t0` to `(1-a)·t` does not fix D1’s failure mode:

- person: D1b ≈ D1 on `seam_vs_adj`, but both seam edges stay high (~192 / ~198); neighbors F-2→F-1 / 1→2 rise.
- environment: D1b **worse** than D1 and worse than B on absolute F-1→0 / 0→1.

So: single-point frame-0 conditioning — whether hard, latent-soft, or latent+timestep soft — still manufactures a temporal boundary. Softening the official `(latent0=ref, t0=0)` pair together is **not** enough.

## Implication for D2

This is the cleanest justification so far for **D2 = Circular Soft Conditioning** (spread soft reference conditioning onto neighboring ring positions, not only index 0). Do not invent more D1.x single-point knobs unless a new hypothesis appears.

Artifacts under this directory; script: `scripts/cloud_mode_b_i2v_d1b_compare.py`.
