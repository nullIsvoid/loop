# ChatGPT independent review request

Please score the loop seam yourself. Do **not** copy the human verdict blindly.

## Why

User asked for multi-check accuracy: human eye alone is not enough. Repo now has:

1. Human notes (partial — S3 focus on phase scenes)
2. Automated multi-detector metrics (`artifacts/seam_metrics/`)
3. **Your** independent visual scores (this file — fill in)

## Watch (repo `*_x3.mp4`)

### Phase scenes — full S0–S3 matrix

`artifacts/mode_b_phase_scenes/`

- pendulum / rotating_fan / human_sway × S0_identity / S1_loopy / S2_fixed1 / S3_symmetric

Ignore image beauty. Score **seam / phase continuity only** (0=broken, 1=usable, 2=smooth).

### Candle schedules (optional cross-check)

`artifacts/mode_b_schedules/` — same four schedules.

## Automated metrics (already computed)

See `artifacts/seam_metrics/seam_metrics.md`.

Detectors: pixel MAE/RMSE at last↔first, Farneback flow at x3 joins, flow_ratio vs adjacent motion, pixel_ratio. Majority rank (D1+D3+D5+D8):

| scene | metric winner |
|-------|----------------|
| candle | S1_loopy (S2/S3 tied 2nd) |
| pendulum | **S3_symmetric** |
| rotating_fan | **S3_symmetric** |
| human_sway | **S3_symmetric** |

S0 is worst almost everywhere. Metrics are auxiliary — they can disagree with eyes on soft motion.

## Please fill

```text
ChatGPT visual scores (0–2), date:

pendulum:     S0=? S1=? S2=? S3=?
rotating_fan: S0=? S1=? S2=? S3=?
human_sway:   S0=? S1=? S2=? S3=?
candle:       S0=? S1=? S2=? S3=?   (optional)

Preferred default schedule after your pass:
Agree / disagree with elevating S3 on phase-hard scenes:
Notes:
```

Reply by editing this file or writing `notes/chatgpt_review_response.md`, then update `DEVELOPMENT_STATUS.md`.
