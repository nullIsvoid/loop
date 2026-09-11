# Cross-seed S1 vs S3 — ready for human / ChatGPT review

## Scope (frozen)

S1 Loopy vs S3 Symmetric only. Seeds **123** / **888**.  
Scenes: pendulum, rotating_fan, human_sway.  
Shared: 1280×704, 81 frames / F=21, 20 steps, CFG=5, UniPC.  
Cloud run: `CROSS_SEED_OK` (12/12). Metrics stay auxiliary — do not auto-pick winner.

## Decision rule

6 pairs (seed × scene). Compare `*_S1_loopy_x3.mp4` vs `*_S3_symmetric_x3.mp4` on **seam only**.

- S3 wins or ties most → promote `SymmetricShiftSchedule` as Wan default  
- S1 more stable across seeds → keep Loopy  
- Clear scene split → selectable policy  

## Score sheet (fill)

| pair | S1 | S3 | winner |
|------|----|----|--------|
| seed123 × pendulum | | | |
| seed123 × rotating_fan | | | |
| seed123 × human_sway | | | |
| seed888 × pendulum | | | |
| seed888 × rotating_fan | | | |
| seed888 × human_sway | | | |

Videos: twelve `*_x3.mp4` in this folder (full single-loop MP4s remain on cloud).
