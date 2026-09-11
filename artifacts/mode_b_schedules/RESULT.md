# Schedule geometry compare (S0–S3) — awaiting human seam verdict

## Correction

`LoopyShiftSchedule` has **no parameters** to sweep. This round compares **shift geometries**:

| ID | Schedule | Intent |
|----|----------|--------|
| S0 | Identity | baseline (all 0) |
| S1 | Loopy | prior winner (monotonic) |
| S2 | FixedShift(1) | constant +1 after anchor |
| S3 | Symmetric | `0,+1,-1,+2,-2,…` then `% F` |

## Shared (identical to first A/B)

seed=42, 1280×704, 81 frames (F=21), 20 steps, CFG=5, UniPC, same candle prompt.

## Cloud videos (not in git)

`/root/latent-loop/artifacts/mode_b_schedules/`

- `S0_identity_x3.mp4`
- `S1_loopy_x3.mp4`
- `S2_fixed1_x3.mp4`
- `S3_symmetric_x3.mp4`

## Repo keeps

- `experiment.json` / `schedule_tables.json`
- first/last PNG previews
- this `RESULT.md`

## Human check

Watch the four `*_x3.mp4` on cloud. Focus **last→first** only.

Pick winner among S0–S3 (or tie notes). Especially: **does S3 beat S1?**

## Verdict

_(pending user)_
