# Development status

> **H1 full-depth Symmetric FAILED** (near dual spike).  
> **Depth loc DONE**: no single third reproduces dual spike; H1-A slightly improves `F-1→0`.  
> **Anchor probe 0–4**: α2 ≫ neighbors (Loopy-like). Do **not** run `--blocks all` until approved.

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

## Next (pick one before coding)

1. **Early 二分/扩窗** — stress-test H1-A (e.g. 0–8 / 0–26)
2. **Early+mid 交互** — find when dual spike appears (still no new schedule)
3. **Pause Hunyuan RoPE** — run W5-0 / WA-0 natives on `person_loop_v1`
4. Or: approve full anchor scan `0–53` (separate from RoPE variants)

No S4/S5 until decided.

## Artifacts

- Depth summary: `artifacts/loop_benchmark_v1/H1_DEPTH_LOC_RESULT.md`
- H1-A/B/C: `artifacts/loop_benchmark_v1/hunyuan15_symmetric_depth_*`
- Anchor 0–4: `artifacts/loop_benchmark_v1/hunyuan15_anchor_probe/`
- Sync: `notes/chatgpt_seam_diag_sync.md`
