# Mode B I2V seam diagnostics (B only)

No new schedule. No auto Gate. Goal: locate residual seam.

## Current latent path (short)

Active: **Circular Temporal RoPE on Q/K** + Symmetric layer phase.  
Not active: residual `RingLatentProcessor`, periodic RoPE, circular latent context.  
See `notes/current_latent_path.md`.

## Step 1–2 — decoded frames + optical flow (done locally)

From existing `B_x3.mp4` (cycle=81):

| case | seam 80→0 mean_mag | adj mean_mag | seam/adj | mean_dx seam vs adj |
|------|-------------------:|-------------:|---------:|---------------------|
| person | 0.586 | ~0.17 | **3.40** | seam **−0.31** vs adj **+0.10** (sign flip) |
| environment | 2.42 | ~1.77 | **1.37** | seam −0.94 vs adj ~−1.2 (same sign) |

Artifacts per case under this folder:

- `frames/f075..f080`, `f000..f005`
- `flow/strip_078_to_079.png` … `strip_080_to_000.png` … `strip_001_to_002.png`
- `diagnostics.json`

**Human-facing read (aux):**  
person looks like a **velocity jump + horizontal direction flip** at 80→0 (not only local hair).  
environment looks milder — closer to **magnitude bump on already-moving water/fur**, less clear direction reversal.

## Step 3–4 — latent probes early/middle/late

Script: `scripts/cloud_mode_b_i2v_latent_probe.py`  
Probes denoise steps **2 / 10 / 19**, saves slices `F-3..F-1 | 0..2` + montage + L2 diffs.

Status: **pending cloud run** (upload + execute when GPU host reachable).  
Interpretation rule once filled:

- high `seam_vs_adj` already at **early** → topology / RoPE phase  
- only **late** blows up → late denoise detail convergence → prefer circular context / content coupling over more schedule tweaks
