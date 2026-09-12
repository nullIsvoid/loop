# Development status

> **H0 done on person_loop_v1.** Late: `F-1→0` max (~2.38× med); `0→1` mild 2nd (~1.59×).  
> Next eligible: **H1** (Hunyuan + Symmetric Circular Temporal RoPE) — not conditioning surgery.  
> Still pending fair natives: **W5-0 / WA-0** on the same benchmark.

## Confirmed

- TI2V-5B hard frame0 conditioning creates an extra `0→1` boundary.
- Native A14B (architecture probe) removes that extra wall; still has real `F-1→0`.
- Previous cross-model visuals were uncontrolled → fixed `person_loop_v1`.
- **H0 (Hunyuan native, locked benchmark):** dominant `F-1→0`; `0→1` only mildly high — not TI2V double spike.

## Current task

1. Optionally finish **W5-0 / WA-0** natives on the same benchmark for three-way table.  
2. Then **H1** Circular RoPE on Hunyuan (only after deciding W5/WA order).

## Experiment IDs

| id | status |
|----|--------|
| W5-0 | script ready — not yet run on v1 |
| WA-0 | script ready — not yet run on v1 |
| H0 | **done** — `artifacts/loop_benchmark_v1/hunyuan15_native/` |
| W5-1 / WA-1 / H1 | not started |

## Artifacts

| path | role |
|------|------|
| `assets/loop_benchmark/` | locked inputs |
| `artifacts/loop_benchmark_v1/hunyuan15_native/` | H0 strict run |
| `artifacts/wan_a14b_i2v_probe/` | OLD RESEARCH EVIDENCE |

Docs: `notes/loop_benchmark.md`, `notes/hunyuan_h0_runbook.md`, `notes/chatgpt_seam_diag_sync.md`.
