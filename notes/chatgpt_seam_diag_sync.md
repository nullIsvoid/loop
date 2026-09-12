# ChatGPT sync — fixed benchmark phase (person_loop_v1)

## Correction

A14B Phase A (`artifacts/wan_a14b_i2v_probe/`) remains **architecture evidence only**:

- native A14B: late `0→1` ≈ median; `F-1→0` elevated (~1.93×)
- **not** a fair visual/seam comparison vs TI2V-5B (different source image, steps, CFG, resolution)

Visual loop was still broken on that substitute-image run.

## New phase

Locked benchmark: `assets/loop_benchmark/` (`person_loop_v1`).

- source = original mode_b person.png (sha256 pinned)
- prompt = fixed `person_prompt.txt`
- seed = 42, frame_num = 81

Scripts (native only this commit):

| id | script |
|----|--------|
| W5-0 | `scripts/benchmark/run_wan_ti2v5b_native.py` |
| WA-0 | `scripts/benchmark/run_wan_a14b_native.py` |
| H0 | `scripts/benchmark/run_hunyuan15_native.py` |

Shared loader/probes: `scripts/benchmark/common.py`  
Outputs: `artifacts/loop_benchmark_v1/<run>/`

**Not in this commit:** W5-1 / WA-1 / H1 (Circular RoPE), CTC, conditioning surgery.

## Research question

Under identical I2V inputs, what late temporal seam structure do different conditioning architectures show — and can Symmetric Circular Temporal RoPE later flatten the real `F-1→0` ring seam without surgery?
