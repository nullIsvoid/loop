# H1 depth localization — RESULT

Same Symmetric schedule as failed full H1; only active block thirds differ.
Benchmark: `person_loop_v1` (strict). Topology: **54 double / 0 single**.

## Late table

| run | active | median | `0→1` | `F-1→0` | shape vs H0 |
|-----|--------|------:|------:|--------:|-------------|
| H0 | — | 54.7 | 87 (1.59×) | 130 (2.38×) | mild 0→1 bump; main ring seam |
| H1-A | 0–17 | 62.2 | 90 (1.45×) | **115 (1.85×)** | **no dual spike**; seam slightly better |
| H1-B | 18–35 | 58.7 | 90 (1.53×) | 131 (2.23×) | ≈ H0 |
| H1-C | 36–53 | 55.1 | 103 (1.87×) | 142 (2.57×) | mild worsen both; still single main seam |
| H1 full | 0–53 | 96.5 | **303 (3.14×)** | **309 (3.20×)** | near dual spike |

## Verdict

1. **No single third reproduces H1’s dual spike.** Early / mid / late alone stay near H0 shape.
2. Full-depth harm is therefore **cumulative / interaction across depth**, not “one toxic zone.”
3. **H1-A** is the only slice that clearly **lowers** `F-1→0` vs H0 without creating a second wall (`0→1` absolute ≈ H0).
4. Do **not** declare Circular RoPE dead on Hunyuan; declare **full-depth Wan-style all-blocks roll** failed, with evidence that **shallow early roll is not the failure mode**.

## Not next (until decided)

- S4/S5 schedule sweep
- Blind “all blocks” retries
- W5/WA (still optional controls)

## Plausible next (ask first)

- Bisect / expand around **early** (e.g. 0–8, 0–26) to see if H1-A’s seam win is robust
- Or early+mid combo without late — only if we want interaction tests, not schedule changes

## Videos

- `hunyuan15_symmetric_depth_0_17/out_x3.mp4`
- `hunyuan15_symmetric_depth_18_35/out_x3.mp4`
- `hunyuan15_symmetric_depth_36_53/out_x3.mp4`
