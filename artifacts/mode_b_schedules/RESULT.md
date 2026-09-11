# Schedule geometry compare (S0–S3) — human verdict

## Correction (context)

`LoopyShiftSchedule` has no tunable parameters. This round compared **shift geometries**.

| ID | Schedule |
|----|----------|
| S0 | Identity (baseline) |
| S1 | Loopy monotonic |
| S2 | FixedShift(1) |
| S3 | Symmetric `0,+1,-1,+2,-2,…` (mod F) |

## Shared

seed=42, 1280×704, 81 frames (F=21), 20 steps, CFG=5, UniPC, same candle prompt.

## Human verdict (2026-09-11)

**S0 (`S0_identity_x3.mp4`): clear jitter / seam problem.**

**S1 / S2 / S3: no obvious seam problem to the eye** — all three look acceptable; cannot rank a clear winner among them from this pass.

Implication:
- Any non-identity RoPE-roll schedule (Loopy / Fixed(1) / Symmetric) beats baseline Identity on this clip.
- Monotonic Loopy is **not uniquely required** for a usable seam here; Fixed(1) and Symmetric also work visually.
- Next should refine **how to discriminate S1/S2/S3** (harder prompt, more steps, second seed, or motion-heavy subject) — not jump to Comfy / residual / true periodic RoPE yet.

## Videos in repo (for ChatGPT review)

- `S0_identity_x3.mp4`
- `S1_loopy_x3.mp4`
- `S2_fixed1_x3.mp4`
- `S3_symmetric_x3.mp4`

Full single-loop mp4s remain on cloud only.
