# Model switch plan (post-D3)

Locked at commit lineage through `5c7f59c` (D3 failed).  
This document is the phase map; it does **not** authorize Circular Temporal Context or further TI2V-5B latent surgery.

## Keep

- Symmetric Circular Temporal RoPE (block shifts `0, +1, -1, +2, -2, …`)
- Existing Wan TI2V-5B artifacts and adapters as baseline / failure case
- Full-ring latent gap diagnostics (not scoring gates)

## Stop

- D1 / D1b / D2 / D3 and all D*.x weight / radius / timestep-release variants
- Post-step rewriting of generated latents on TI2V-5B
- S4/S5 RoPE schedules, RepairPlan, auto winner metrics, Comfy adapters (until phases A–C clear)

## Role map

| model | role |
|-------|------|
| Wan2.2 TI2V-5B | baseline + Circular RoPE PoC + hard-conditioning failure case |
| Wan2.2 I2V-A14B | **architecture control** (independent `y` cond; no hard `latent[0]=ref`) |
| HunyuanVideo-1.5 | **primary development** target |
| LTX-2.x | later guiding-latent reference only |

## Phase A — A14B native control

**Question:** Does native A14B late latent still show TI2V-5B-style **index-0 double spike** (`F-1→0` and `0→1` both anomalous)?

- Official `WanI2V` only; **no** Mode-B RoPE; **no** conditioning edits
- Full temporal latent dumps + full-ring adjacent L2
- Script: `scripts/cloud_wan_a14b_i2v_probe.py`
- Cloud ckpt (already present): `/model/HuggingFace/Wan-AI/Wan2.2-I2V-A14B`
- Allow resolution/steps tweaks for VRAM; keep person image / prompt / seed=42 / 81 frames when possible

### How to read A14B

| late pattern | interpretation |
|--------------|----------------|
| Both `F-1→0` and `0→1` abnormally high | problem broader than TI2V hard clamp |
| Only `F-1→0` elevated; `0→1` ≈ other adjacents | TI2V hard conditioning was an extra artificial boundary |

## Phase B — Hunyuan source + skeleton

- Analyze official `Tencent-Hunyuan/HunyuanVideo-1.5` I2V path (done → `notes/hunyuan_i2v_call_chain.md`)
- Skeleton only: `src/latent_loop/adapters/hunyuan/`
- **No** Circular RoPE implementation in this phase
- **No** conditioning edits

## Phase C — Hunyuan native probe (H0)

- Native I2V only; full-ring gaps early/middle/late
- Ask: does reference conditioning create a sharp discontinuity at a forced temporal index?

## Phase D — Hunyuan + Symmetric Circular RoPE (H1)

Only if H0 looks clean enough. Re-implement Mode B against Hunyuan’s own Q/K + 3D RoPE (`rope_dim_list` t/h/w), **do not** copy Wan adapter blindly.

Ideal minimal stack:

```text
HunyuanVideo-1.5
  + native non-destructive I2V conditioning
  + Symmetric Circular Temporal RoPE
  → seamless I2V candidate
```

Then — and only then — consider Circular Temporal Context.
