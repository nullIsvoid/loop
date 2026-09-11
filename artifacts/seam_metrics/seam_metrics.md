# Seam metrics (multi-detector)

Auxiliary multi-detector seam metrics. Not a substitute for human / ChatGPT visual review. Lower D1/D3/D5/D8 better.

| scene | schedule | D1 mae | D3 flow_seam | D5 flow_ratio | D8 pixel_ratio |
|---|---|---:|---:|---:|---:|
| human_sway | S0_identity | 5.49 | 1.558 | 5.138 | 4.231 |
| human_sway | S1_loopy | 3.18 | 0.482 | 2.636 | 2.668 |
| human_sway | S2_fixed1 | 3.76 | 0.236 | 1.630 | 3.198 |
| human_sway | S3_symmetric | 2.27 | 0.270 | 1.214 | 1.808 |
| pendulum | S0_identity | 9.90 | 5.698 | 6.410 | 2.252 |
| pendulum | S1_loopy | 7.93 | 2.075 | 2.354 | 1.692 |
| pendulum | S2_fixed1 | 7.49 | 1.518 | 1.597 | 1.565 |
| pendulum | S3_symmetric | 4.81 | 1.075 | 1.035 | 1.176 |
| rotating_fan | S0_identity | 9.57 | 1.309 | 1.235 | 2.040 |
| rotating_fan | S1_loopy | 7.46 | 1.655 | 2.367 | 1.413 |
| rotating_fan | S2_fixed1 | 7.57 | 0.921 | 1.734 | 1.692 |
| rotating_fan | S3_symmetric | 5.98 | 0.549 | 0.650 | 1.159 |
| candle | S0_identity | 20.34 | 5.784 | 1.999 | 1.743 |
| candle | S1_loopy | 14.88 | 2.688 | 1.220 | 0.942 |
| candle | S2_fixed1 | 15.67 | 3.275 | 1.158 | 1.033 |
| candle | S3_symmetric | 14.89 | 3.452 | 1.333 | 0.885 |

## Majority ranking (D1+D3+D5+D8 ranks)

- **candle**: winner `S1_loopy` — rank_sum={'S0_identity': 12.0, 'S1_loopy': 2.0, 'S2_fixed1': 5.0, 'S3_symmetric': 5.0}
- **human_sway**: winner `S3_symmetric` — rank_sum={'S0_identity': 12.0, 'S1_loopy': 6.0, 'S2_fixed1': 5.0, 'S3_symmetric': 1.0}
- **pendulum**: winner `S3_symmetric` — rank_sum={'S0_identity': 12.0, 'S1_loopy': 8.0, 'S2_fixed1': 4.0, 'S3_symmetric': 0.0}
- **rotating_fan**: winner `S3_symmetric` — rank_sum={'S0_identity': 9.0, 'S1_loopy': 8.0, 'S2_fixed1': 7.0, 'S3_symmetric': 0.0}
