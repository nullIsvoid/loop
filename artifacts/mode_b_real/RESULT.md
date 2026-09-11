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

## Visual check (human)

Play **`A_x3.mp4`** then **`B_x3.mp4`**. Focus only on **last → first** cut.

Do **not** treat this as scored; decide among:

1. A jumps, B clearly smoother → continue Loopy schedule experiments  
2. Both jump differently → try stricter periodic RoPE variants  
3. B worse → Loopy schedule/geometry likely wrong for TI2V-5B; do not pivot to residual mix
