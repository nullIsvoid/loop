# Wan2.2 I2V-A14B native probe — how to run

Cloud already has weights:

```text
/model/HuggingFace/Wan-AI/Wan2.2-I2V-A14B
```

Code (this repo → sync to `/root/latent-loop/`):

```text
scripts/cloud_wan_a14b_i2v_probe.py
```

## Constraints

- Official `WanI2V` only
- **No** Mode-B / Symmetric Circular RoPE
- **No** conditioning surgery
- Full-ring gaps + `stepXX_full_latent.pt` at early/middle/late

## Example (24GB-friendly)

```bash
/usr/local/miniconda3/envs/mobius/bin/python -u \
  /root/latent-loop/scripts/cloud_wan_a14b_i2v_probe.py \
  --image /root/latent-loop/artifacts/mode_b_i2v/_inputs/person.png \
  --seed 42 \
  --frame-num 81 \
  --steps 40 \
  --max-area $((480*832)) \
  --out-root /root/latent-loop/artifacts/wan_a14b_i2v_probe
```

Use `--max-area $((720*1280))` only if VRAM allows.

## Readout

From late `full_ring_gaps.json` / `experiment.json`:

| pattern | meaning |
|---------|---------|
| `seam_F-1->0` and `seam_0->1` both ≫ median | double spike survives without hard clamp |
| only `F-1→0` high; `0→1` ≈ median | TI2V-5B hard conditioning was the extra boundary |

This is diagnostics, not a Gate.
