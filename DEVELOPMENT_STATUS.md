# Development status

> **Control done:** Mode B Symmetric T2V late `seam_vs_adj≈1.0` (flat).  
> I2V late spike is tied to **first-frame hard anchor**. Next research: **Circular I2V Conditioning**. HOLD Circular Temporal Context as first D.

## Current latent handling

**Active:** Circular Temporal RoPE on Q/K (`SymmetricShiftSchedule`), V untouched.  
I2V still uses official Wan per-step clamp of temporal index 0 to image latent.

## Diagnostics summary

| path | late seam_vs_adj | F-1→0 & 0→1 |
|------|-----------------:|-------------|
| I2V Mode B | ~1.6–1.8 | both high |
| T2V Mode B control | **~1.006** | flat / normal |

## Next (not started)

Design/experiment **Circular I2V Conditioning** only after ChatGPT/human green-light.  
No S4/S5. No Circular Temporal Context as default next step.

Sync: `notes/chatgpt_seam_diag_sync.md` · T2V result: `artifacts/mode_b_t2v_seam_diag/RESULT.md`
