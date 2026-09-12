# ChatGPT sync — D2 moved the seam; do not widen radius

## D2 result (B vs radius-2 circular soft, fixed every step)

Person late F-4..4 (abridged):

| | F-1→0 | 0→1 | 2→3 | max edge |
|--|------:|----:|----:|----------|
| B | 213 | 169 | 88 | F-1→0 |
| D2 | 29 | 19 | **163** | **2→3** |

Environment: F-1→0 258→23, 0→1 218→21, but **3→4 = 229** becomes the new max.

**Verdict:** D2 flattens the old seam by pinning neighbors to `ref0`, then recreates a discontinuity at the soft-window edge. Still piecewise conditioning. **Do not** tune radius 2→3→4→5 as the next move.

Detail: `artifacts/mode_b_i2v_d2/RESULT.md` (code `20032ee` + results commit).

## Ask

What hypothesis next? (e.g. Circular Temporal Context, generated-relative blend, something else — not radius sweep.)
