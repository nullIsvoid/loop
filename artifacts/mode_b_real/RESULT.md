# Mode B real A/B generation — result summary

Run: `4083434` scaffold + this generate script. Cloud RTX 4090, official Wan2.2 TI2V-5B.

## What ran

| | A | B |
|---|---|---|
| Pipeline | WanTI2V full denoise + VAE | same + `enable_mode_b_on_wan_model(LoopyShiftSchedule)` |
| Seed | 42 | 42 |
| Size | 1280×704 | same |
| Frames | 81 (latent F=21) | same |
| Steps | 20 | same |
| CFG / shift / solver | 5.0 / 5.0 / unipc | same |
| Time | ~194s | ~193s |

Attention: `flash_attn` not on cloud env → Wan official `attention()` SDPA fallback (not the smoke stub). RoPE path still Mode A vs Loopy-style roll Mode B.

## Artifacts

See `artifacts/mode_b_real/`:

- `A_baseline.mp4` / `B_rope_roll.mp4`
- `A_x3.mp4` / `B_x3.mp4` (triple play for seam)
- first/last PNGs + `run.json`

## Visual check (human) — VERDICT 2026-09-11

**Decision: (1) A jumps, B clearly smoother.**

User watched `A_x3.mp4` / `B_x3.mp4` and confirmed Mode B (Loopy-style RoPE roll) is **clearly better** on the last→first seam than baseline A.

Implication (agreed branch):
1. Keep Mode B as primary track
2. Next: **LoopyShiftSchedule parameter experiments** (not residual mix, not Comfy)
3. Then: normal steps → normal resolution → TI2V / T2V / I2V coverage

Do **not** pivot to residual mix. Do **not** jump to “true periodic RoPE” until schedule/params are explored on this winning geometry.
