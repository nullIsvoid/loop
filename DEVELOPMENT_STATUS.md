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
- Cursor: land zips into this repo, run tests, commit/push, Wan adapter experiments.
- Do not independently reimplement files under "Owned now" without syncing intent first.
