# Development status

> ACTIVE — Mode B scaffold approved (2026-09-11 ChatGPT).
> Primary: official Wan `rope_apply` / `WanSelfAttention` Loopy-style RoPE roll.
> Comfy adapter deferred. Residual mix = Mode C control only.

## Terminology

- **Loopy-style RoPE roll**: per-block `torch.roll` on expanded `freqs_3d` temporal axis only.
- **Not claimed:** mathematical periodic / circular RoPE. Closed-loop effect needs real Wan experiments.
- Stricter periodic encodings remain a later comparable strategy under the same schedule interface.

## Owned now

ChatGPT-owned (do not rewrite without sync):

- `src/latent_loop/ring.py`
- `tests/test_ring.py`
- residual helpers currently in `ring.py` (`RingLatentProcessor`, etc.)

Cursor-owned (Mode B):

- `src/latent_loop/rope/`
- `src/latent_loop/adapters/wan/`
- `tests/test_rope_roll.py`, `tests/test_wan_rope_adapter.py`
- `notes/wan_rope_call_chain.md`

## Layout

```text
src/latent_loop/
├── ring.py                 # topology primitives (ChatGPT)
├── rope/
│   ├── core.py             # F-axis freqs roll + Wan-shaped apply
│   └── schedule.py         # injectable TemporalShiftSchedule
└── adapters/wan/
    ├── rope.py             # wan_rope_apply
    └── attention.py        # Mode B hook for WanSelfAttention
```

## Experiment priority

| Mode | Meaning | Role |
|------|---------|------|
| A | Wan baseline | baseline |
| B | Wan + Loopy-style RoPE roll | **PRIMARY** |
| C | Wan + residual ring mix | CONTROL |
| D | RoPE + residual | later |
| E | Mobius-style latent shift | later对照 |

## Rules (Mode B)

1. Match Loopy behaviour first — no new circular RoPE formula yet.
2. Roll temporal F only; never H/W.
3. Apply to Q/K multipliers only; never V.
4. Block 0 unshifted (anchor).
5. Disabled Mode B == baseline `rope_apply`.
6. Do not modify `ring.py` / `RingLatentProcessor`.
7. No Comfy adapter yet.
8. Do not touch diffusion `t`, scheduler, token order, or global `rope_params`.
9. Shift schedule injectable (`LoopyShiftSchedule` default; Fixed/Identity ready).
10. Tests: F-only roll + shape/dtype/device + Loopy numerical parity.

## Current work

1. ~~Analysis + ChatGPT inject-point ack~~
2. ~~Mode B scaffold + Loopy parity unit tests~~
3. **Done (cloud smoke):** official `WanModel` TI2V-5B on RTX 4090 — Mode A vs Mode B one-step forward OK (`SMOKE_OK`). Script: `scripts/cloud_mode_b_smoke.py`
4. **Next:** short real generate (few steps / low res) + triple-play seam look; still not Comfy


## Collaboration protocol

Disagreement → Exchange inbox → wait → then code. See prior resolved Q1–Q3.

### Exchange inbox

_(empty — Mode B scaffold approved)_
