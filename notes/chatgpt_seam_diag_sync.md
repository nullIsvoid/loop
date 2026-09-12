# ChatGPT sync — D2 Circular Soft running

## Prior

D1 / D1b closed single-point soft conditioning (`1a8f63a`).

## Now: D2 (green-lit)

Radius-2 ring soft weights (fixed every step, no late release):

| d | w | t factor |
|---|---|----------|
| 0 | 1.00 | 0 |
| 1 | 0.75 | 0.25 |
| 2 | 0.25 | 0.75 |
| ≥3 | 0 | 1 |

Compare **B vs D2** × person + environment. Probe **F-4..4**. Watch for spike push to ±2/±3 and visual pause at seam.

Artifacts: `artifacts/mode_b_i2v_d2/` (in flight).
