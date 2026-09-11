# Phase-clear scenes × S0–S3 — human verdict

## Research question

Is Loopy’s layer-wise schedule necessary, or does any non-zero temporal RoPE roll close the loop — and which geometry wins on **phase-clear** motion?

## Shared

seed=42, 1280×704, 81 frames (F=21), 20 steps, CFG=5, UniPC.

## Human verdict (2026-09-12) — focus on S3 Symmetric

Image quality set aside; seam / jitter only:

| Scene | S3 Symmetric |
|-------|----------------|
| pendulum | no visible jitter |
| rotating_fan | no visible jitter |
| human_sway | **smallest jitter; almost none** |

**Takeaway from this pass:** `SymmetricShiftSchedule` (`0,+1,-1,+2,-2,…` mod F) holds the seam on all three phase-hard scenes; human sway is where it looks strongest.

Candle round previously: S1/S2/S3 all OK, S0 bad. This round elevates **S3** as the preferred geometry candidate for next default — pending any explicit S0/S1/S2 contrast notes from the same watcher.

## Videos in repo (`*_x3.mp4`)

12 triple-play previews under this folder for ChatGPT review.
