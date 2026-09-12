# D3 Circular Reference Residual — result

## Question

Does full-ring smooth **reference residual** (`x_i = g_i + w_i·(ref0−g0)`) keep the circular adjacent-gap profile smooth vs B hard single-point?

## Setup

- Symmetric Circular Temporal RoPE unchanged
- B = official hard frame-0 + `t0=0`
- D3 = `w(d)=0.5·(1+cos(π·d/Dmax))`, `Dmax=F//2=10`, `x_i=g_i+w_i·delta`, **`t0=0` only**
- Full-ring 21 edges; `full_latent_step19.pt` on cloud (not in git)
- person / environment, seed=42, 81 frames, 20 steps

## Late full-ring summary (step 19)

| case | variant | max edge | max L2 | median | max/med | F-1→0 | 0→1 |
|------|---------|----------|-------:|-------:|--------:|------:|----:|
| person | B | 20→0 | 213 | 103 | **2.07** | 213 | 169 |
| person | D3 | 20→0 | **562** | 103 | **5.43** | 562 | 521 |
| environment | B | 20→0 | 258 | 142 | **1.82** | 258 | 218 |
| environment | D3 | 20→0 | **629** | 146 | **4.30** | 629 | 603 |

D3 top gaps are still the old seam (`20→0`, `0→1`); median elsewhere stays ~B. So this is **not** a moved wall — it is a **worse seam at the same place**.

## Verdict

**D3 fails.** Full-ring residual correction does not smooth the ring; it amplifies the discontinuity at index 0 (~2.5–3× B on the seam edges) while leaving the rest of the ring near the B median.

Likely reason (diagnostic, not proven): each step the model fights `x_0=ref0`, producing a large `delta`; nearly full `w` on ±1 then injects that fight into neighbors every step, compounding the 0-boundary instead of removing it.

## Implication

Absolute ref copy (D2) and relative residual (D3) both fail under **post-step latent surgery** that keeps hard `t0=0` / `x_0=ref0`. Next hypothesis should probably leave this post-hoc latent rewrite path — e.g. Circular Temporal Context / attention-side coupling — rather than another weight profile on `delta`.

Do not iterate D3.x cosine shapes without a new mechanism.
