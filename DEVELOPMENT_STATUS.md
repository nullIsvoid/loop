# Development status

> **H1 full-depth Symmetric FAILED** (near dual spike).  
> **Depth loc DONE**: no single third reproduces dual spike; H1-A slightly improves `F-1→0`.  
> **Anchor probe 0–4**: α2 ≫ neighbors (Loopy-like). Do **not** run `--blocks all` until approved.
> **H2-B2 generation DONE**: Block 2 half-period perturbation is a diagnostic generation, not the final Loopy schedule. Late `F-1→0` improved from 130.14 to 123.57; `0→1` from 87.00 to 84.22. The seam remains.

Authoritative research direction: `RESEARCH_DIRECTION.md`. The primary target is Wan2.2-I2V-A14B non-destructive conditioning plus a Loopy anchor/grouped-shift circular-time policy. Hunyuan is the second backbone and cross-model validator.

## H0 / H1 / depth thirds (late)

| | H0 | H1-A (0–17) | H1-B (18–35) | H1-C (36–53) | H1 full |
|--|---:|------------:|-------------:|-------------:|--------:|
| median | 54.7 | 62.2 | 58.7 | 55.1 | 96.5 |
| `0→1` | 87 | 90 | 90 | 103 | **303** |
| `F-1→0` | 130 | **115** | 131 | 142 | **309** |

480p_i2v: **54 double / 0 single**.

## Anchor probe α0…α4

| 0 | 1 | **2** | 3 | 4 |
|--:|--:|------:|--:|--:|
| 0.000322 | 0.000530 | **0.004106** | 0.000506 | 0.000341 |

## Next

1. Finish or deliberately widen Hunyuan `alpha_l` measurement before designing its grouped shift table; Block 2 is only the strongest candidate among 0–4.
2. Port the same strict same-input `alpha_l` measurement to Wan2.2-I2V-A14B's 40-block dual-expert runtime.
3. Implement an explicit A14B grouped Loopy schedule only after the measured layer map and high-/low-noise expert indexing are verified.
4. Keep Mobius latent rotation as route B; do not mix it into the Loopy measurements.

No S4/S5 until decided.

## Artifacts

- Depth summary: `artifacts/loop_benchmark_v1/H1_DEPTH_LOC_RESULT.md`
- H1-A/B/C: `artifacts/loop_benchmark_v1/hunyuan15_symmetric_depth_*`
- Anchor 0–4: `artifacts/loop_benchmark_v1/hunyuan15_anchor_probe/`
- Sync: `notes/chatgpt_seam_diag_sync.md`
