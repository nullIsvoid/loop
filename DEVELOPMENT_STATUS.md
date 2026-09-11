# Development status

> ACTIVE — Mode B works. Emerging candidate: **SymmetricShiftSchedule (S3)** on phase-clear motion.

## Human results

### Candle (seed 42)

| S0 Identity | S1 Loopy | S2 Fixed(1) | S3 Symmetric |
|--|--|--|--|
| ❌ jitter | ✅ | ✅ | ✅ |

### Phase scenes (seed 42) — 2026-09-12

User report (seam only; ignore image quality):

- **pendulum / S3**: no visible jitter  
- **rotating_fan / S3**: no visible jitter  
- **human_sway / S3**: **least jitter, almost none**

→ S3 is the current preferred geometry for harder periodic motion.

## Open product implication

If S3 stays best after any remaining S1/S2 contrast notes:

- Default candidate may become **Symmetric** (or still Fixed(1) if later shown equal — not yet claimed equal on phase scenes)
- Loopy monotonic schedule is **not uniquely required** for usable seams

## Git policy

Full single-loop MP4s on cloud. Repo: metadata + `*_x3.mp4` + RESULT.

## Next (after ChatGPT sync)

Confirm full matrix notes for S0/S1/S2 on phase scenes if needed; then decide default schedule + optional second-seed only on survivors. Still not Comfy / residual / “true periodic RoPE” yet.
