# Hunyuan H0 runbook (native benchmark)

## Goal

Run **H0** = HunyuanVideo-1.5 native I2V on `person_loop_v1` with early/middle/late full-ring latent probes.

No Circular RoPE. No conditioning edits. Prompt rewrite **OFF**.

## Script

```bash
cd /root/latent-loop
/usr/local/miniconda3/envs/mobius/bin/python \
  scripts/benchmark/run_hunyuan15_native.py
```

Check layout only:

```bash
.../python scripts/benchmark/run_hunyuan15_native.py --check-only
```

## Model layout required under `HY_MODEL_PATH`

Default: `/model/ModelScope/Tencent-Hunyuan/HunyuanVideo-1.5`

Must exist:

```text
transformer/480p_i2v/   # (or 720p_i2v if --resolution 720p)
vae/
text_encoder/           # Qwen2.5-VL + byT5/Glyph layout per official docs
vision_encoder/         # SigLIP from FLUX.1-Redux-dev
```

Official download notes: `/root/HunyuanVideo-1.5/checkpoints-download.md`

As of 2026-09-12 cloud snapshot: DiT + VAE present; **text_encoder / vision_encoder dirs missing** — H0 will exit with `prerequisites.json` until those are installed (prefer non-C: cloud paths).

## Fixed benchmark inputs

Loaded from `assets/loop_benchmark/` via `scripts/benchmark/common.py`:

| field | value |
|-------|-------|
| source | person.png |
| prompt | person_prompt.txt |
| seed | 42 |
| frames | 81 (`--video-length` override allowed; official prefers 121) |

## Model-specific defaults (recorded in run.json)

| knobs | default |
|-------|---------|
| resolution | 480p |
| aspect_ratio | inferred from person.png (≈9:16) |
| steps | 50 |
| sr | off |
| rewrite | off |

## Output

```text
artifacts/loop_benchmark_v1/hunyuan15_native/
  source.png out.mp4 out_x3.mp4 first.png last.png
  run.json RESULT.md prerequisites.json
  latent/{early,middle,late}_full_latent.pt
  latent/{early,middle,late}_full_ring_gaps.json
```
