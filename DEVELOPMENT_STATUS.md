# Development status

> **Phase switch complete.** Fixed benchmark `person_loop_v1` is locked.  
> Next: run native baselines **W5-0 / WA-0 / H0** (no Circular RoPE in this phase).

## Confirmed

- TI2V-5B hard frame0 conditioning creates an extra `0→1` boundary.
- Native A14B removes that extra `0→1` wall (architecture evidence).
- A14B still has a real `F-1→0` loop seam; prior substitute-image video was **not** visually seamless.
- Previous cross-model visual comparisons were **not** fully controlled (source/steps/CFG/resolution differed).

## Current task

Build fixed benchmark and rerun native baselines:

```text
W5-0 / WA-0 / H0
```

Only after native baseline:

```text
W5-1 / WA-1 / H1  + Symmetric Circular Temporal RoPE
```

## Goal

I2V seamless loop (last→first motion continuity). No scoring Gates / RepairPlan / S4–S5 / CTC / conditioning surgery.

## Settled

| item | status |
|------|--------|
| Symmetric Circular Temporal RoPE | keep for later W5-1/WA-1/H1 |
| TI2V-5B post-step conditioning (B→D3) | **closed** |
| Fixed benchmark `person_loop_v1` | **locked** |
| A14B Phase A (substitute image) | **OLD RESEARCH EVIDENCE** |

## Experiment IDs

| id | meaning |
|----|---------|
| W5-0 | Wan TI2V-5B native |
| W5-1 | + Symmetric Circular Temporal RoPE |
| WA-0 | Wan A14B native |
| WA-1 | + Symmetric Circular Temporal RoPE |
| H0 | HunyuanVideo-1.5 native |
| H1 | + Symmetric Circular Temporal RoPE |

## Artifacts

| path | role |
|------|------|
| `assets/loop_benchmark/` | locked inputs |
| `scripts/benchmark/` | common + W5-0 / WA-0 / H0 |
| `artifacts/loop_benchmark_v1/` | strict benchmark outputs |
| `artifacts/mode_b_i2v*` / `wan_a14b_i2v_probe` | OLD RESEARCH EVIDENCE |

Docs: `notes/loop_benchmark.md`, `notes/hunyuan_h0_runbook.md`, `notes/chatgpt_seam_diag_sync.md`.
