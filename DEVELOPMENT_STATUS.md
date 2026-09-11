# Development status

> **FROZEN Wan default:** `SymmetricShiftSchedule` (Circular Temporal RoPE + Bidirectional Layer Phase Distribution).  
> Mode B is no longer a Loopy copy. Loopy / Fixed remain selectable; S2 parked.

## Cross-seed gate (seeds 123 / 888) — human 2026-09-12

6/6 pairs: **tie** (cannot discriminate S1 vs S3 on seam).  
Only `seed888 × human_sway` shows very slight jitter on **both** S1 and S3.

→ Decision rule satisfied: S3 wins or ties most → **Symmetric is Wan default**.

## Freeze still in force

- Do **not** add S4/S5/S6
- Do **not** expand `seam_metrics` into auto-score / Gate / Repair (auxiliary only)
- Do **not** implement Comfy / residual / “true periodic RoPE” until a new research question is opened

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

Product/Comfy adapter, residual Mode C, or true periodic RoPE — only when explicitly scoped. No more schedule invention for the current default.
