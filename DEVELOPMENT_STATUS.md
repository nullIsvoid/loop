# Development status

> ACTIVE DEVELOPMENT — ring-latent core from ChatGPT `loop-v0.1.0` zip.
> Cursor landed the package into this GitHub repo because ChatGPT cannot push here.

## Owned now

- `src/latent_loop/ring.py`
- `tests/test_ring.py`
- sampler-facing ring latent API

## Implemented

- modulo-wrapped temporal indexing
- circular temporal padding
- per-frame ring neighbourhood windows
- residual circular latent mixing
- denoising-step strength schedule
- sampler-facing `RingLatentProcessor`
- unit tests for the 0 <-> T-1 boundary

## Next integration work

1. Add a model adapter rather than treating circular mixing as a final algorithm.
2. Hook circular indexing into temporal attention / temporal convolution context.
3. Integrate one real open-source video model first (Wan-family candidate).
4. Compare three modes separately:
   - baseline
   - latent residual ring mix
   - native circular temporal context (RoPE roll / Loopy-style — separate track)
5. Keep first and last output frames distinct. Never solve looping by copying frame 0 to frame T-1.

## Collaboration

- ChatGPT: continue algorithm / API design; ship zips or patches when GitHub push is unavailable.
- Cursor: land zips into this repo, run tests, commit/push; only implement after status sync.
- Do not independently reimplement files under "Owned now" without syncing intent first.

### Disagreement protocol (user rule)

1. Before coding, read this file + README + owned modules for ChatGPT intent.
2. If Cursor disagrees with ownership, next step, API shape, or geometry (ring mix vs RoPE roll vs Mobius latent shift), **do not code first**.
3. Write the disagreement into **Exchange inbox** below (or a dated note), commit/push so ChatGPT can see it.
4. Wait for ChatGPT reply / user decision, then implement.

### Exchange inbox

#### 2026-09-11 — Cursor notes (await ChatGPT)

- **Agree:** ring topology ≠ copy frame0→last; residual mix is only a first primitive; target is model temporal context, not RGB crossfade; keep modes separable for ablation.
- **Agree ownership:** `ring.py` / `test_ring.py` / sampler-facing ring API stay ChatGPT-owned unless ChatGPT hands them off.
- **Open question (do not code yet):** product research earlier ranked **Loopy-style RoPE-only roll** as primary geometry on Wan, with Mobius-style latent roll as对照. This repo’s next step list puts **model adapter + residual ring mix** first, RoPE as “separate track”. Need ChatGPT confirm:
  1. Is `latent_loop` the home for both tracks (e.g. `ring/` + `rope_loop/`), or only the residual-ring track?
  2. Should Wan adapter first wire `RingLatentProcessor` into denoise, or first wire RoPE roll into Wan temporal attention?
  3. Any API freeze on `RingMixConfig` / `RingLatentProcessor` before adapter work?
- **Not claiming:** Cursor will not rewrite `ring.py` or demote residual mix without ChatGPT reply.
