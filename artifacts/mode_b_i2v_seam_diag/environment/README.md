# Seam frame diagnostics — `B_x3.mp4`

Auxiliary only. Compare seam strip 80→0 vs neighbors: position jump (big RGB change), velocity jump (mag spike), direction reverse (hue flip in flow), local desync (regional).

cycle_frames=81, seam_vs_adj_flow_ratio=1.366

| pair | seam? | mean_mag | median | p95 | mean_dx | mean_dy |
|---|---|---:|---:|---:|---:|---:|
| 78->79 | False | 1.545 | 0.249 | 11.067 | -0.979 | -0.150 |
| 79->80 | False | 1.808 | 0.613 | 11.066 | -1.127 | -0.221 |
| 80->0 | True | 2.424 | 0.751 | 13.398 | -0.944 | -0.103 |
| 0->1 | False | 2.089 | 0.356 | 13.421 | -1.543 | -0.222 |
| 1->2 | False | 1.655 | 0.403 | 9.836 | -1.281 | 0.123 |

See `frames/` and `flow/strip_*.png`.
