# Development status

> ACTIVE — Mode B / Circular Temporal RoPE. **Do not freeze S3 as default yet.**  
> Next gate: cross-seed S1 vs S3 only.

## Freeze (ChatGPT + human, 2026-09-12)

- Do **not** add S4/S5/S6 or new schedule geometries
- Do **not** expand `seam_metrics` into auto-score / auto-winner / Gate / Repair
- Human seam watch = primary; automated metrics = auxiliary only
- Do **not** raise steps/resolution for this research question
- Do **not** implement Comfy / residual / “true periodic RoPE” yet
- S2 Fixed(1) parked (no phase-hard win) — not in next round

## Evidence so far (seed 42)

| | Candle | Pendulum | Fan | Human sway |
|--|--|--|--|--|
| S0 Identity | ❌ | ❌ | ❌ | ❌ |
| S1 Loopy | ✅ | 次 | 次 | 次 |
| S2 Fixed1 | ✅ | 次 | 次 | 次 |
| S3 Symmetric | ✅ | ★ | ★ | ★ |

Hypothesis **rejected**: “any non-zero roll is the same” — shift geometry matters on phase-clear motion. Symmetric is the strongest candidate, not yet the frozen default.

## Next experiment — **READY TO SCORE**

**S1 Loopy vs S3 Symmetric only** — cloud `CROSS_SEED_OK`, 12/12 x3 in `artifacts/mode_b_cross_seed/`.

- Scenes: pendulum, rotating_fan, human_sway  
- Seeds: 123, 888  
- Shared: 1280×704, 81 frames / F=21, 20 steps, CFG=5, UniPC  

Decision rule:

- S3 wins or ties most of 6 pairs → promote `SymmetricShiftSchedule` as Wan default  
- S1 more stable across seeds → keep Loopy default  
- Clear scene split → selectable policy, not a single forced default  

Fill `artifacts/mode_b_cross_seed/RESULT.md` (human + ChatGPT).

## Naming (forming)

If S3 holds: this is no longer a Loopy copy — it is **Circular Temporal RoPE + Bidirectional Layer Phase Distribution**.

## Git policy

Full single-loop MP4s on cloud. Repo: metadata + `*_x3.mp4` + RESULT (+ optional auxiliary metrics for this research only).
