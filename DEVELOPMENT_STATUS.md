# Development status

> **FROZEN Wan default:** `SymmetricShiftSchedule` (Circular Temporal RoPE + Bidirectional Layer Phase Distribution).  
> Mode B is no longer a Loopy copy. Loopy / Fixed remain selectable; S2 parked.

## Freeze hygiene (2026-09-12 audit)

| Item | Status |
|--|--|
| RoPE padded / heterogeneous batch (`seq_len` per sample) | **fixed** |
| `cloud_mode_b_generate.py` default schedule | **Symmetric** (`--schedule`) |
| `schedule=None` → Symmetric regression test | **added** |
| README mainline | **Circular Temporal RoPE** (ring mix = experimental) |
| seam_metrics → Gate/Repair | still forbidden |
| Comfy / residual / true periodic RoPE | not started |

## Cross-seed gate (seeds 123 / 888) — human 2026-09-12

6/6 pairs: **tie** (cannot discriminate S1 vs S3 on seam).  
Only `seed888 × human_sway` shows very slight jitter on **both** S1 and S3.

→ Decision rule satisfied: S3 wins or ties most → **Symmetric is Wan default**.

## Evidence map

| | Candle (42) | Phase (42) | Cross-seed (123/888) |
|--|--|--|--|
| S0 Identity | ❌ | ❌ | (not rerun) |
| S1 Loopy | ✅ | 次 | tie vs S3 |
| S2 Fixed1 | ✅ | 次 | parked |
| S3 Symmetric | ✅ | ★ | **default** (ties) |

Hypothesis **rejected**: “any non-zero roll is the same” (phase-42 ranked S3 above S1/S2).  
Cross-seed: S3 **generalizes at least as well as** Loopy under human seam watch.

## Git policy

Full single-loop MP4s on cloud. Repo: metadata + `*_x3.mp4` + RESULT.

## Next (optional, not started)

Product/Comfy adapter, residual Mode C, or true periodic RoPE — only when explicitly scoped. No more schedule invention for the current default. Optional: GitHub Actions pytest (none yet).
