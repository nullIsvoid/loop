# Loop benchmark — person_loop_v1

Fixed I2V inputs for cross-model seam studies.

## Locked inputs

| field | value |
|-------|-------|
| source | `person.png` (sha256 in `benchmark.json`) |
| prompt | `person_prompt.txt` |
| seed | 42 |
| frame_num | 81 (when the model supports it) |
| motion | gentle hair / clothing hem sway |
| camera | fixed |
| goal | last→first temporal continuity |

## Rules

1. All native / Circular-RoPE runs read this directory via `scripts/benchmark/common.py`.
2. **Do not** hardcode a different prompt or person image inside model scripts.
3. Model-specific `steps` / CFG / resolution / scheduler may differ; record them in each `run.json`.
4. Old artifacts under `artifacts/mode_b_i2v*` and `artifacts/wan_a14b_i2v_probe/` are **OLD RESEARCH EVIDENCE**, not this benchmark.

## Experiment IDs

| id | meaning |
|----|---------|
| W5-0 | Wan2.2 TI2V-5B native I2V |
| W5-1 | W5 + Symmetric Circular Temporal RoPE |
| WA-0 | Wan2.2 I2V-A14B native |
| WA-1 | WA + Symmetric Circular Temporal RoPE |
| H0 | HunyuanVideo-1.5 native I2V |
| H1 | H + Symmetric Circular Temporal RoPE |

Phase-1 commit ships W5-0 / WA-0 / H0 scripts only.
