# latent-loop

**Research direction:** **Wan2.2-I2V-A14B non-destructive image conditioning + Loopy anchor-based circular temporal positions**.

Read [`RESEARCH_DIRECTION.md`](RESEARCH_DIRECTION.md) first. It is the source of truth for the research goal, borrowed methods, model roles, and acceptance criteria.

**Current status:** HunyuanVideo-1.5 is the second backbone and cross-model validator. Its Loopy-style anchor measurement found Block 2 as the strongest candidate among Blocks 0–4; the full 54-block scan is not complete.

## Early experimental implementation (not the final Loopy policy)

```text
Circular Temporal RoPE
        ↓
WanSelfAttention Q/K only (V untouched)
        ↓
expanded freqs_3d temporal roll
        ↓
Early symmetric layer phase: 0, +1, -1, +2, -2, …  (% F)
        ↓
Wan2.2 TI2V-5B real T2V generation (batch=1 path validated)
```

Hook:

```python
from latent_loop.adapters.wan import enable_mode_b_on_wan_model

# default schedule = SymmetricShiftSchedule
enable_mode_b_on_wan_model(model, flash_attention_fn=flash_attention)
```

Selectable schedules currently include `SymmetricShiftSchedule`, `LoopyShiftSchedule`, `FixedShiftSchedule`, and `IdentityShiftSchedule`. `SymmetricShiftSchedule` is our early experiment. The current `LoopyShiftSchedule` mirrors the simple non-distributed Loopy path, not the grouped 40-block A14B schedule.

RoPE follows official Wan per-sample `seq_len = F*H*W` (padding left untouched). Equal-length unpadded batches still match historical Loopy `rope_apply_loop`.

## Experimental / control only

`RingLatentProcessor` residual ring mix in `ring.py` is **not** the product mainline. It remains Mode C / control scaffolding.

## Install

```bash
pip install -e .[dev]
pytest
```

## Cloud generate

```bash
# Text-to-video A/B
python scripts/cloud_mode_b_generate.py --schedule symmetric

# Image-to-video A/B (same TI2V-5B; pass --image)
python scripts/cloud_mode_b_i2v.py --image /path/to/wallpaper.png --case-id person \
  --prompt "Subtle hair and clothing sway, seamless loop, preserve identity"
```

## Non-goals

- do not copy the first frame to the last frame
- do not crossfade decoded RGB as the primary loop fix
- do not turn seam metrics into auto Gate / Repair
- Comfy adapter / true periodic RoPE — not started

See `DEVELOPMENT_STATUS.md` before working on the same files with Cursor / ChatGPT.
