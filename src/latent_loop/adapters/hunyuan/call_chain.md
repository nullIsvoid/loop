# Hunyuan adapter call-chain pointer

Full analysis: `notes/hunyuan_i2v_call_chain.md` (official `hyvideo` tree).

## Intended adapter surfaces (Phase D — not yet)

| module | future hook | status |
|--------|-------------|--------|
| `rope.py` | temporal coords in `get_nd_rotary_pos_embed` / `apply_rotary_emb` | stub |
| `attention.py` | enable/disable Mode-B on double/single stream blocks | stub |
| `conditioning.py` | document-only; **do not** mutate I2V cond in Phase B/C | stub |

## Official path (summary)

```text
image → VAE cond_latents (+ mask) + VisionEncoder states
noise latents → cat(latents, cond) → PatchEmbed → blocks
  → img Q/K + 3D RoPE → attention → scheduler.step(latents only)
```
