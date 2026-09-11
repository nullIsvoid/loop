# Development status

> ACTIVE — Mode B (Loopy-style RoPE roll) **validated by human seam check**.
> Next: **shift schedule geometry** comparison (not “Loopy parameter sweep”).

## Terminology

- **Loopy-style RoPE roll**: per-block `torch.roll` on expanded `freqs_3d` temporal axis only.
- **Not claimed:** mathematical periodic / circular RoPE.
- **LoopyShiftSchedule has no tunable parameters** — it is a fixed formula. “Parameter sweep” is wrong wording; compare **schedule geometries** instead.

## Owned now

ChatGPT-owned: `ring.py` / residual helpers / `tests/test_ring.py`  
Cursor-owned: `rope/`, `adapters/wan/`, generate/compare scripts, experiment notes

## Experiment priority

| Mode | Meaning | Role |
|------|---------|------|
| A / S0 | Wan + Identity shifts | baseline |
| B / S1 | Wan + Loopy schedule | **validated winner (first A/B)** |
| S2 | FixedShift(1) | geometry对照 |
| S3 | Symmetric ± shifts | geometry对照 |
| C | residual ring mix | CONTROL (later) |
| E | Mobius latent shift | later |

## Current work

1. ~~Loopy parity + Wan adapter + full A/B generate~~
2. ~~Human verdict (199f111): B clearly smoother than A~~
3. **Now:** S0 Identity / S1 Loopy / S2 Fixed(1) / S3 Symmetric — same seed/prompt/F/steps
4. Git policy: **do not commit every schedule MP4**; keep metadata + RESULT + selected previews only. First A/B videos in `artifacts/mode_b_real/` may stay.

## Schedule formulas

- Identity: all `0`
- Loopy: `0`, then `(i-1)%(F-1)+1`
- Fixed(1): `0`, then `1` (mod F)
- Symmetric: `0, +1, -1, +2, -2, ...` applied as `s % F`
