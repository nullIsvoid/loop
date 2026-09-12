# Development status

> Residual seam on Mode B I2V localizes to **late denoise**, not early RoPE topology.  
> Next candidate when scoped: Circular Temporal Context (content across F-1→0). No S4/S5.

## Current latent handling

**Active:** Circular Temporal RoPE on Q/K (`SymmetricShiftSchedule`), V untouched.  
**Not active:** residual ring mix, true periodic RoPE, circular temporal context.  
Details: `notes/current_latent_path.md`.

## Seam diagnostics (B only) — filled

Decoded flow: person strong seam/adj + dx flip; environment milder.  
Latent `seam_vs_adj`: early/mid **&lt;1**, late **~1.6–1.8** (person + environment).

→ Prefer investigating **circular temporal context / latent shift** over more schedule geometry or periodic RoPE-as-first-fix.

## Mode B I2V — human

A clearer jitter, B slight residual — B wins. Artifacts under `artifacts/mode_b_i2v/`.

## Still forbidden until explicitly scoped

- No S4/S5  
- No Gate/Repair metrics  
- No implementing D until you green-light the candidate  

## ChatGPT

Sync note: `notes/chatgpt_seam_diag_sync.md` — please confirm late-seam readout and next D = Circular Temporal Context.
