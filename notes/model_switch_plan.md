# Model switch plan (post-D3) → fixed benchmark phase

Locked at commit lineage through `5c7f59c` (D3 failed) and A14B architecture probe `cc8b12f` / clarification `50c96a5`.

**Current execution:** `notes/loop_benchmark.md` — `person_loop_v1` natives W5-0 / WA-0 / H0 before any Circular RoPE port.

## Keep

- Symmetric Circular Temporal RoPE (block shifts `0, +1, -1, +2, -2, …`) for later W5-1 / WA-1 / H1
- Existing Wan TI2V-5B / A14B artifacts as **OLD RESEARCH EVIDENCE**
- Full-ring latent gap diagnostics (not scoring gates)

## Stop

- D1 / D1b / D2 / D3 and all D*.x weight / radius / timestep-release variants
- Post-step rewriting of generated latents on TI2V-5B
- S4/S5 RoPE schedules, RepairPlan, auto winner metrics, Comfy adapters
- Treating substitute-image A14B video as a fair visual baseline

## Role map

| model | role |
|-------|------|
| Wan2.2 TI2V-5B | W5 baseline + hard-conditioning failure case |
| Wan2.2 I2V-A14B | primary I2V backbone: independent `y` conditioning + future Loopy anchor/grouped shifts |
| HunyuanVideo-1.5 | second backbone + cross-model validator for non-destructive conditioning and Loopy layer sensitivity |
| LTX-2.x | official design reference for Guiding Latents versus Replacing Latents |
| Loopy | primary circular-time method: alpha measurement, anchor selection, grouped per-layer temporal RoPE shifts |
| Mobius | fallback circular-time method based on denoising-time latent cycle/rotation |

## Phase map (new IDs)

1. **Benchmark lock** — `assets/loop_benchmark/` ✅  
2. **Native baselines** — W5-0 / WA-0 / H0 (this commit: scripts)  
3. **Circular RoPE** — W5-1 / WA-1 / H1 only after natives  
4. CTC / surgery — not authorized

Target main stack:

```text
Wan2.2-I2V-A14B
  + native non-destructive image conditioning
  + Loopy anchor-based grouped temporal RoPE shifts
  → seamless I2V candidate
```

`SymmetricShiftSchedule` (`0,+1,-1,+2,-2,...`) is retained only as early experimental evidence. It is not the formal Loopy algorithm and is not the target production schedule.

See `RESEARCH_DIRECTION.md` for the authoritative source/method/model mapping.
