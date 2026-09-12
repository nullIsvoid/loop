# Development status

> **Phase switch:** Wan2.2 TI2V-5B conditioning surgery **closed** (`5c7f59c`).  
> Next: architecture control (A14B) + **HunyuanVideo-1.5** as primary I2V backbone.  
> Keep: Symmetric Circular Temporal RoPE (proven). Do not invent D3.x / D2 radius / S4/S5 / CTC yet.

## Goal (unchanged)

Image-to-Video seamless loop: last→first temporal motion continuity.  
No identity scoring, RepairPlan, product Gates, or RoPE schedule sweeps.

## Settled on Wan TI2V-5B

| piece | status |
|-------|--------|
| Symmetric Circular Temporal RoPE | **keep** (T2V flat; I2V improves vs A) |
| Hard / soft / residual post-step conditioning | **closed** (B→D3) |

TI2V-5B remains baseline + RoPE PoC + conditioning failure case. Do not delete artifacts.

## Next phases

| phase | model | action |
|-------|-------|--------|
| **A** | Wan2.2 I2V-A14B | Native I2V latent probe only (no RoPE patch). Architecture control. |
| **B** | HunyuanVideo-1.5 | Source call-chain + adapter skeleton (no Circular RoPE yet). |
| **C** | Hunyuan | Native I2V full-ring latent probe (H0). |
| **D** | Hunyuan | Only if H0 clean → Symmetric Circular Temporal RoPE (H1). |

Plan detail: `notes/model_switch_plan.md`.  
Hunyuan call chain: `notes/hunyuan_i2v_call_chain.md`.

## Hypotheses under test

- **H1:** TI2V-5B residual seam largely from per-step hard frame0 conditioning.  
- **H2:** Independent conditioning channels reduce that artificial boundary.  
- **H3:** If conditioning is clean, Symmetric Circular Temporal RoPE may be enough as the main loop mechanism.
