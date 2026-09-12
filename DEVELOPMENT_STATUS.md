# Development status

> **H1 authorized.** H0 on `person_loop_v1` done; next is Hunyuan + Symmetric Circular Temporal RoPE.  
> Refined rule: non-destructive cond avoids TI2V dual walls, but may leave a mild `0→1` bump.

## Confirmed

- TI2V-5B hard frame0 → severe extra `0→1` wall; surgery closed.
- Hunyuan H0 (strict benchmark): `F-1→0` 2.38× med (dominant); `0→1` 1.59× (mild).
- Fixed `person_loop_v1` locked.

## Experiment IDs

| id | status |
|----|--------|
| H0 | **done** — `artifacts/loop_benchmark_v1/hunyuan15_native/` |
| H1 | **in progress** — Symmetric Circular Temporal RoPE |
| W5-0 / WA-0 | scripts ready; not required before H1 |
| W5-1 / WA-1 | later |

## Docs

- `notes/hunyuan_h1_rope_call_chain.md`
- `notes/chatgpt_seam_diag_sync.md`
- `scripts/benchmark/run_hunyuan15_symmetric.py`
