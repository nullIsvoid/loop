# ChatGPT review — closed

## Outcome (2026-09-12)

Phase + metrics + cross-seed human watch completed. **Wan default = `SymmetricShiftSchedule`.**

Follow-up audit (this thread) found engineering gaps — fixed in the same freeze pass:

- P1: `rope_apply_loopy_roll` now uses per-sample `seq_len = F*H*W` (Wan padding semantics)
- P1: `cloud_mode_b_generate.py` defaults to Symmetric (`--schedule`)
- P1: default-schedule regression test when `schedule=None`
- P2: README / package exports rewritten around Circular Temporal RoPE mainline

Visual score sheet for historical phase x3s is no longer blocking; see `DEVELOPMENT_STATUS.md` and `artifacts/mode_b_cross_seed/RESULT.md`.
