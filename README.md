# latent-loop

**Status:** Wan Mode B default frozen as **Circular Temporal RoPE + Bidirectional Layer Phase Distribution** (`SymmetricShiftSchedule`).

## Mainline (validated)

```text
Circular Temporal RoPE
        ↓
WanSelfAttention Q/K only (V untouched)
        ↓
expanded freqs_3d temporal roll
        ↓
Symmetric layer phase: 0, +1, -1, +2, -2, …  (% F)
        ↓
Wan2.2 TI2V-5B real T2V generation (batch=1 path validated)
```

Hook:

```python
from latent_loop.adapters.wan import enable_mode_b_on_wan_model

# default schedule = SymmetricShiftSchedule
enable_mode_b_on_wan_model(model, flash_attention_fn=flash_attention)
```

Selectable schedules: `SymmetricShiftSchedule` (default), `LoopyShiftSchedule`, `FixedShiftSchedule`, `IdentityShiftSchedule`.

RoPE follows official Wan per-sample `seq_len = F*H*W` (padding left untouched). Equal-length unpadded batches still match historical Loopy `rope_apply_loop`.

## Experimental / control only

`RingLatentProcessor` residual ring mix in `ring.py` is **not** the product mainline. It remains Mode C / control scaffolding.

## Install

```bash
pip install -e .[dev]
pytest
```

## Cloud generate (A vs B)

```bash
python scripts/cloud_mode_b_generate.py --schedule symmetric   # default
python scripts/cloud_mode_b_generate.py --schedule loopy
```

## Non-goals

- do not copy the first frame to the last frame
- do not crossfade decoded RGB as the primary loop fix
- do not turn seam metrics into auto Gate / Repair
- Comfy adapter / true periodic RoPE — not started

See `DEVELOPMENT_STATUS.md` before working on the same files with Cursor / ChatGPT.
