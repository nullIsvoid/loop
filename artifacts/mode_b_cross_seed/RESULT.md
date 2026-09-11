# Cross-seed S1 vs S3 — pending results

## Scope (frozen)

S1 Loopy vs S3 Symmetric only. Seeds 123 / 888. Scenes: pendulum, rotating_fan, human_sway.  
No new schedules. Metrics stay auxiliary. Same 1280×704 / 81 / 20 / CFG5 / UniPC.

## Decision rule

6 pairs (seed×scene). If S3 wins or ties most → promote Symmetric as Wan default.

## Artifacts

Fill after cloud run: `experiment.json`, twelve `*_x3.mp4`, then human (+ optional ChatGPT) scores here.
