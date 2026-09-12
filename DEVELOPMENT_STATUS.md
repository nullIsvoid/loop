# Development status

> **H1 v1 FAILED.** Symmetric Circular RoPE on Hunyuan worsened both `F-1→0` and `0→1` (near dual spike).  
> H0 remains the best Hunyuan baseline on `person_loop_v1`.

## Late table

| | H0 | H1 |
|--|---:|---:|
| median | 54.7 | 96.5 |
| `0→1` | 87 (1.59×) | **303 (3.14×)** |
| `F-1→0` | 130 (2.38×) | **309 (3.20×)** |

## Next

Diagnose H1 failure layer before more schedule knobs. Optional: W5-0 / WA-0 natives for three-way table.

## Artifacts

- H0: `artifacts/loop_benchmark_v1/hunyuan15_native/`
- H1: `artifacts/loop_benchmark_v1/hunyuan15_symmetric/`
- Sync: `notes/chatgpt_seam_diag_sync.md`
