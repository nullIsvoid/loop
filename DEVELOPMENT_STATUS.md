# Development status

> ACTIVE — Mode B works. Multi-check in progress: human + automated metrics both lean **S3 Symmetric** on phase-clear scenes; **ChatGPT visual pass pending**.

## Multi-check results

### Candle (seed 42)

| Check | S0 | S1 Loopy | S2 Fixed(1) | S3 Symmetric |
|--|--|--|--|--|
| Human | ❌ jitter | ✅ | ✅ | ✅ |
| Metrics majority | worst | **best** | tied 2nd | tied 2nd |

### Phase scenes (seed 42)

| Check | Result |
|--|--|
| Human (S3 focus) | pendulum/fan no jitter; human_sway almost none |
| Metrics majority | **S3 wins** all three scenes; S0 worst |
| ChatGPT visual | **pending** — see `notes/chatgpt_review_request.md` |

## Open product implication

Pending ChatGPT scores. If S3 stays best across checks:

- Default candidate → **SymmetricShiftSchedule**
- Loopy monotonic is **not uniquely required** for usable seams

## Git policy

Full single-loop MP4s on cloud. Repo: metadata + `*_x3.mp4` + RESULT + seam_metrics.

## Next

1. ChatGPT fills independent 0–2 seam scores for full S0–S3 matrix  
2. Merge three checks → freeze default schedule  
3. Optional second-seed only on survivors  
Still not Comfy / residual / “true periodic RoPE”.
