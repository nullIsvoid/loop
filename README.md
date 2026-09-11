# latent-loop

**Status: active development.** The current code implements the first model-agnostic circular latent primitives. It is not yet a complete Wan/VACE/ComfyUI integration.

## Core idea

A looping video should not be represented as a linear timeline whose final frame is forced to equal the first frame. Treat temporal latent positions as a ring:

```text
      0
   /     \
  1       5
  |       |
  2 ----- 4
     \
      3
```

For six temporal positions the neighbourhoods are therefore:

```text
0 -> [5, 0, 1]
1 -> [0, 1, 2]
2 -> [1, 2, 3]
3 -> [2, 3, 4]
4 -> [3, 4, 5]
5 -> [4, 5, 0]
```

There is no duplicated endpoint.

## What exists now

`src/latent_loop/ring.py` provides:

- `circular_indices()` — modulo temporal indexing
- `circular_pad()` — circular temporal context padding
- `ring_windows()` — wrapped neighbourhoods for every latent position
- `circular_temporal_mix()` — small residual ring coupling
- `cosine_denoise_strength()` — coupling fades out before final detail steps
- `RingLatentProcessor` — sampler-facing entry point

The residual mixer is intentionally only a first integration primitive. The target design is to inject ring topology directly into a video model's temporal context/attention rather than post-process decoded frames.

## Install

```bash
pip install -e .[dev]
pytest
```

## Minimal integration

```python
from latent_loop import RingLatentProcessor, RingMixConfig

ring = RingLatentProcessor(
    RingMixConfig(radius=1, maximum_strength=0.12, active_fraction=0.70)
)

for step in range(total_steps):
    latents = scheduler_step(...)
    latents = ring(latents, step=step, total_steps=total_steps)
```

## Non-goals

- do not copy the first frame to the last frame
- do not crossfade decoded RGB frames as the primary solution
- do not hide a discontinuity with a single seam interpolation

See `DEVELOPMENT_STATUS.md` before working on the same files with Cursor.
