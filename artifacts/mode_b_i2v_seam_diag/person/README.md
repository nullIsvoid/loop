# Seam frame diagnostics — `B_x3.mp4`

Auxiliary only. Compare seam strip 80→0 vs neighbors: position jump (big RGB change), velocity jump (mag spike), direction reverse (hue flip in flow), local desync (regional).

cycle_frames=81, seam_vs_adj_flow_ratio=3.395

| pair | seam? | mean_mag | median | p95 | mean_dx | mean_dy |
|---|---|---:|---:|---:|---:|---:|
| 78->79 | False | 0.151 | 0.003 | 0.897 | 0.117 | 0.006 |
| 79->80 | False | 0.154 | 0.003 | 0.880 | 0.100 | -0.054 |
| 80->0 | True | 0.586 | 0.011 | 2.656 | -0.306 | -0.072 |
| 0->1 | False | 0.220 | 0.008 | 0.864 | 0.129 | -0.087 |
| 1->2 | False | 0.165 | 0.008 | 0.703 | 0.055 | -0.065 |

See `frames/` and `flow/strip_*.png`.
