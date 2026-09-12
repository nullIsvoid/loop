# ChatGPT sync — D1 results; D2 not started

## D1 Late Anchor Release (person + environment)

Implemented and run on `main`. Late probe:

| case | B F-1→0 / 0→1 / seam_vs_adj | D1 F-1→0 / 0→1 / seam_vs_adj |
|------|-----------------------------:|-----------------------------:|
| person | 213 / 169 / **1.77** | 175 / 192 / **1.37** |
| environment | 258 / 218 / **1.56** | 247 / 247 / **1.42** |

**Conclusion:** D1 is a partial lever only — double spike remains. Softening late frame-0 clamp alone does **not** make the ring self-close.

## Ask

Please confirm whether to green-light **D2 = circular soft conditioning** (distance-based weights around index 0 on the ring), or propose a different D1.x release curve / identity-safe check first.

Artifacts: `artifacts/mode_b_i2v_d1/` (+ `out_x3.mp4` for visual).
