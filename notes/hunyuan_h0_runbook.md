# Hunyuan H0 runbook (native benchmark)

## Goal

Run **H0** = HunyuanVideo-1.5 native I2V on `person_loop_v1` with early/middle/late full-ring latent probes.

No Circular RoPE. No conditioning edits. Prompt rewrite **OFF**.

## Cloud paths (important)

`/model` is **read-only** UPFS. Do **not** write encoders there.

Writable overlay used for H0:

```text
/workspace/hunyuan-ckpts/HunyuanVideo-1.5/
  transformer -> /model/.../transformer
  vae/          # local clean config.json + symlink weight
                # (ModelScope vae/config.json was duplicated JSON)
  text_encoder/llm              # Qwen2.5-VL-7B-Instruct (ModelScope)
  text_encoder/byt5-small
  text_encoder/Glyph-SDXL-v2
  vision_encoder/siglip -> FLUX.1-Redux-dev
```

Env:

```bash
export HY_MODEL_PATH=/workspace/hunyuan-ckpts/HunyuanVideo-1.5
export HY_ROOT=/root/HunyuanVideo-1.5
```

## Script

```bash
cd /root/latent-loop
/usr/local/miniconda3/envs/mobius/bin/python \
  scripts/benchmark/run_hunyuan15_native.py \
  --model-path /workspace/hunyuan-ckpts/HunyuanVideo-1.5
```

`--check-only` validates nested encoder paths + large LLM shards.

## Probe note

Hunyuan `__call__` **recreates** `scheduler` via `_create_scheduler(flow_shift)`.  
The runner wraps both the current scheduler and `_create_scheduler` so early/middle/late probes stick.

## Status

H0 on `person_loop_v1` **completed** — see `artifacts/loop_benchmark_v1/hunyuan15_native/RESULT.md`.
