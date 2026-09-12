# D1 Late Anchor Release — result

## Question

Does releasing frame-0 hard clamp in late denoise remove the late **F-1→0 / 0→1** double spike?

## Setup

- Symmetric Circular Temporal RoPE unchanged
- B = `HardAnchorSchedule` (a=1 every step)
- D1 = `LateAnchorReleaseSchedule` (0–12: 1.0 → … → 19: 0.0)
- Same person / environment I2V, seed=42, 81 frames, 20 steps

## Late latent L2 (step 19)

| case | variant | F-1→0 | 0→1 | 1→2 | seam_vs_adj |
|------|---------|------:|----:|----:|------------:|
| person | B hard | 213 | 169 | 105 | **1.77** |
| person | D1 release | 175 ↓ | 192 ↑ | — | **1.37** ↓ |
| environment | B hard | 258 | 218 | 142 | **1.56** |
| environment | D1 release | 247 ↓ | 247 ↑ | — | **1.42** ↓ |

## Verdict

**D1 does not clear the double spike.**  
`F-1→0` improves modestly; `0→1` stays high or rises; `seam_vs_adj` still ≫ 1.

So: merely softening late frame-0 clamp is **not** enough for the ring to self-close. Causal story (I2V anchor boundary) still stands, but the first knife D1 is only a **partial** lever.

## Next (not implemented here)

Per prior plan: design **D2 = circular soft conditioning** around index 0 (symmetric distance weights on the ring) — only after ChatGPT/human green-light. Still no F-1=ref hard dual-nail; no Circular Temporal Context as default.

Human: also watch `*/out_x3.mp4` for seam vs identity drift when a→0 at the end.
