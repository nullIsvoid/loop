# Mode B I2V seam diagnostics (B only)

No new schedule. No auto Gate. Goal: locate residual seam.

## Current latent path (short)

Active: **Circular Temporal RoPE on Q/K** + Symmetric layer phase.  
Not active: residual `RingLatentProcessor`, periodic RoPE, circular latent context.  
See `notes/current_latent_path.md`.

## Step 1–2 — decoded frames + optical flow

From existing `B_x3.mp4` (cycle=81):

| case | seam 80→0 mean_mag | adj mean_mag | seam/adj | note |
|------|-------------------:|-------------:|---------:|------|
| person | 0.586 | ~0.17 | **3.40** | dx sign flip at seam |
| environment | 2.42 | ~1.77 | **1.37** | milder mag bump |

See `frames/` and `flow/strip_*.png`.

## Step 3–4 — latent probes (done)

Mode B only, steps **2 / 10 / 19**, slices `F-3..F-1 | 0..2`.  
`seam_vs_adj` = L2(F-1→0) / mean L2 of other consecutive pairs.

| case | early (2) | middle (10) | late (19) |
|------|----------:|------------:|----------:|
| person | 0.84 | 0.83 | **1.77** |
| environment | 0.88 | 0.87 | **1.56** |

**Readout:** early/middle seam is **not** worse than adjacent temporal gaps; **late** denoise is where seam_vs_adj spikes — and **0→1 rises with F-1→0**, so index 0 is special under I2V.

**T2V control** (`artifacts/mode_b_t2v_seam_diag/`): late `seam_vs_adj≈1.006`, gaps flat → spike is **I2V first-frame hard anchor**, not missing Circular Temporal Context. Next research: **Circular I2V Conditioning** (when scoped).

Artifacts: `person/latent/`, `environment/latent/` (montages + diffs JSON; `.pt` stay on cloud).
