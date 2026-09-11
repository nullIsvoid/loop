# Cross-seed S1 vs S3 — human verdict

## Scope (frozen)

S1 Loopy vs S3 Symmetric only. Seeds **123** / **888**.  
Scenes: pendulum, rotating_fan, human_sway.  
Shared: 1280×704, 81 frames / F=21, 20 steps, CFG=5, UniPC.  
Cloud run: `CROSS_SEED_OK` (12/12). Metrics stay auxiliary.

## Human verdict (2026-09-12)

Seam only; image quality ignored.

| pair | observation | winner |
|------|-------------|--------|
| seed123 × pendulum | cannot tell S1 vs S3 | tie |
| seed123 × rotating_fan | cannot tell | tie |
| seed123 × human_sway | cannot tell | tie |
| seed888 × pendulum | cannot tell | tie |
| seed888 × rotating_fan | cannot tell | tie |
| seed888 × human_sway | both have **very slight** jitter; no clear S1 vs S3 gap | tie |

**Summary:** 6/6 ties. No pair where Loopy beats Symmetric across seeds.  
Per prior decision rule (S3 wins **or ties** most) → **promote `SymmetricShiftSchedule` as Wan default**.

Loopy remains available as an explicit schedule option; Fixed(1) stays parked.

## Decision

`SymmetricShiftSchedule` is the frozen Wan Mode B default after this cross-seed gate.
