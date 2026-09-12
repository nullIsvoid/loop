# ChatGPT sync — loop project (commit `2a4d7ce`)

Repo: https://github.com/nullIsvoid/loop  
Tip: `main` @ `2a4d7ce`

---

## Research goal (only)

Image-to-Video **seamless loop**: last→first temporal motion continuity.

Not now: identity/face scores, Gates, RepairPlan, S4/S5, CTC, conditioning surgery.

---

## Locked benchmark: `person_loop_v1`

```text
assets/loop_benchmark/
  person.png          # original Wan I2V person (sha256 pinned)
  person_prompt.txt
  benchmark.json      # seed=42, frame_num=81
```

All models must load via `scripts/benchmark/common.py`.  
Do **not** compare runs that swapped source/prompt/steps silently.

---

## Experiment IDs

| id | meaning | status |
|----|---------|--------|
| W5-0 | Wan TI2V-5B native | script ready, **not run on v1 yet** |
| W5-1 | W5 + Symmetric Circular Temporal RoPE | not started |
| WA-0 | Wan A14B native | script ready, **not run on v1 yet** |
| WA-1 | WA + Symmetric Circular Temporal RoPE | not started |
| **H0** | HunyuanVideo-1.5 native | **done on person_loop_v1** |
| H1 | Hunyuan + Symmetric Circular Temporal RoPE | **next candidate** |

---

## Prior architecture evidence (NOT strict cross-model benchmark)

| run | note |
|-----|------|
| TI2V-5B B→D3 | hard frame0 clamp creates **extra** `0→1` wall; conditioning surgery closed |
| A14B Phase A (`artifacts/wan_a14b_i2v_probe/`) | substitute image; late `0→1≈median`, `F-1→0`↑ — architecture only; **video still not seamless** |

---

## H0 result (strict `person_loop_v1`)

Path: `artifacts/loop_benchmark_v1/hunyuan15_native/`  
Video: `out_x3.mp4`  
Detail: `RESULT.md`

Late latent full-ring (`F_latent=21`):

| metric | value |
|--------|------:|
| median | 54.7 |
| `0→1` | **87.0** (~1.59× median) — rank **2** / 21 |
| `F-1→0` | **130.1** (~2.38× median) — **max** |

### Read carefully

1. **Not** TI2V-5B equal double spike (`F-1→0` and `0→1` both huge).
2. **Not** as clean as A14B architecture probe (`0→1≈median`) — Hunyuan has a **mild** `0→1` bump.
3. Dominant failure mode = real ring seam **`F-1→0`**.
4. Visual loop is still expected to jump; H0 did **not** claim seamless.

### Implication

Next authorized step: **H1 = Hunyuan native conditioning + Symmetric Circular Temporal RoPE**  
(block shifts `0,+1,-1,+2,-2,…` on Hunyuan’s own ND RoPE — **do not** copy Wan adapter blindly).

Do **not** reopen TI2V conditioning surgery.  
Optional before H1: run W5-0 / WA-0 on the same benchmark for a three-way native table.

---

## Ask ChatGPT

Given H0 above, confirm:

1. Is H1 the right next knife (vs first finishing W5-0/WA-0 natives)?
2. For H1, what must be re-verified in Hunyuan `rope_dim_list=[16,56,56]` / `apply_rotary_emb` path before coding?
3. How should we judge H1 success using the same late full-ring metrics + `out_x3` visual checklist only (jump / stall / reverse / speed pop)?
