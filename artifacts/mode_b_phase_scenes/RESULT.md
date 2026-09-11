# Phase-clear scenes × S0–S3 — multi-check verdict

## Research question

Is Loopy’s layer-wise schedule necessary, or does any non-zero temporal RoPE roll close the loop — and which geometry wins on **phase-clear** motion?

## Shared

seed=42, 1280×704, 81 frames (F=21), 20 steps, CFG=5, UniPC.

## Check 1 — Human (2026-09-12) — focus on S3 Symmetric

Image quality set aside; seam / jitter only:

| Scene | S3 Symmetric |
|-------|----------------|
| pendulum | no visible jitter |
| rotating_fan | no visible jitter |
| human_sway | **smallest jitter; almost none** |

Candle round previously: S1/S2/S3 all OK, S0 bad.

## Check 2 — Automated multi-detector (`../seam_metrics/`)

Eight detectors on each `*_x3.mp4` (pixel MAE/RMSE, Farneback seam flow ×2 joins, ratios vs adjacent motion). Majority rank of D1+D3+D5+D8 (lower better):

| scene | winner | rank_sum (S0 / S1 / S2 / S3) |
|-------|--------|------------------------------|
| pendulum | **S3_symmetric** | 12 / 8 / 4 / **0** |
| rotating_fan | **S3_symmetric** | 9 / 8 / 7 / **0** |
| human_sway | **S3_symmetric** | 12 / 6 / 5 / **1** |
| candle (schedules folder) | S1_loopy | 12 / **2** / 5 / 5 |

S0 Identity is worst on every scene. Metrics elevate **S3** on all three phase-hard scenes; candle soft motion still prefers Loopy slightly with S2/S3 close.

## Check 3 — ChatGPT independent visual scores

**Pending.** Fill `notes/chatgpt_review_request.md` after watching the 12 phase `*_x3.mp4` (and optionally candle). Do not rubber-stamp Check 1.

## Videos in repo

12 triple-play previews under this folder + metrics under `artifacts/seam_metrics/`.
