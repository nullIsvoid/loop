# ChatGPT sync — D3 residual failed (seam worse, not moved)

## D3 result

`x_i = g_i + w_i·(ref0−g0)`, full-ring cosine, `t0=0` only. B vs D3.

| case | B max/med (seam) | D3 max/med (seam) |
|------|-----------------:|------------------:|
| person | 2.07 (213 / 169) | **5.43 (562 / 521)** |
| environment | 1.82 (258 / 218) | **4.30 (629 / 603)** |

Max edge stays `20→0`. Median ~unchanged. **Worse seam, not a relocated wall.**

## Ask

Post-step latent rewrite path (absolute D2 + residual D3) looks exhausted. Prefer next: Circular Temporal Context / attention-side, or another non-rewrite idea?

Detail: `artifacts/mode_b_i2v_d3/RESULT.md` (`0a4d4e7` + results).
