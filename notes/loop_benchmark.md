# Fixed loop benchmark (person_loop_v1)

## Why

Previous A14B Phase A used a **substitute** still; TI2V runs used the real person image, different steps/CFG/resolution. Architecture evidence stands, but **cross-model visual/seam comparison was not controlled**.

## Locked inputs

`assets/loop_benchmark/` — see README there. SHA256 pinned in `benchmark.json`.

## Experiment IDs

| id | script | status in this commit |
|----|--------|------------------------|
| W5-0 | `scripts/benchmark/run_wan_ti2v5b_native.py` | runnable |
| WA-0 | `scripts/benchmark/run_wan_a14b_native.py` | runnable |
| H0 | `scripts/benchmark/run_hunyuan15_native.py` | runnable path; needs text/vision encoders |
| W5-1 / WA-1 / H1 | — | **not** in this commit |

## Output root

```text
artifacts/loop_benchmark_v1/{wan_ti2v5b_native,wan_a14b_native,hunyuan15_native}/
```

## Old evidence

Keep, but label **OLD RESEARCH EVIDENCE / NOT STRICT CROSS-MODEL BENCHMARK**:

- `artifacts/mode_b_i2v*`
- `artifacts/wan_a14b_i2v_probe/`

## Phase-1 question only

> Under identical source/prompt/seed/frame_num, what late temporal seam structure do W5-0 / WA-0 / H0 show?
