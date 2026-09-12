# Development status

> **Phase A done (architecture only):** A14B native I2V — **no double spike** (`0→1`≈median; only `F-1→0` elevated ~1.93×).  
> **Not visually seamless** — `out_x3.mp4` still jumps at the loop. “Pass” ≠ closed loop.  
> Next: HunyuanVideo-1.5 native H0 probe (Phase C). Keep Symmetric Circular Temporal RoPE for later H1.

## Goal

I2V seamless loop (last→first motion continuity). No scoring Gates / RepairPlan / S4–S5.

## Settled

| item | status |
|------|--------|
| Symmetric Circular Temporal RoPE | keep |
| TI2V-5B post-step conditioning (B→D3) | **closed** |
| A14B architecture control | **done** — supports H1/H2; **not** visual seamless |

## A14B late (native, no RoPE)

| | F-1→0 | 0→1 | median | max/med |
|--|------:|----:|-------:|--------:|
| A14B | 236 | **123≈med** | 122 | 1.93 |

vs TI2V-5B hard I2V: both F-1→0 and 0→1 high.  
**Visual:** loop still jumps; residual seam = `F-1→0` only.

## Next

1. Hunyuan native I2V latent probe (H0) — no Circular RoPE yet  
2. If H0 clean → H1 = Hunyuan + Symmetric Circular Temporal RoPE  

Docs: `notes/model_switch_plan.md`, `artifacts/wan_a14b_i2v_probe/RESULT.md`.
