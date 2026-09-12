#!/usr/bin/env python3
"""Force-complete Qwen2.5-VL-7B into Hunyuan text_encoder/llm via ModelScope."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

LLM = Path("/workspace/hunyuan-ckpts/HunyuanVideo-1.5/text_encoder/llm")
os.environ["MODELSCOPE_CACHE"] = "/workspace/.cache/modelscope"
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Incomplete HF leftovers (config only) — wipe and refetch.
if LLM.exists():
    # Keep only if a large weight shard exists.
    weights = list(LLM.glob("*.safetensors")) + list(LLM.glob("*.bin"))
    big = [p for p in weights if p.stat().st_size > 100_000_000]
    if big:
        print("LLM already has weights:", [p.name for p in big])
        sys.exit(0)
    print("removing incomplete LLM dir", LLM)
    shutil.rmtree(LLM)

LLM.mkdir(parents=True, exist_ok=True)
print("disk free GB", shutil.disk_usage("/workspace").free / 1e9)

from modelscope.hub.snapshot_download import snapshot_download

p = snapshot_download(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir=str(LLM),
)
print("LLM_DONE", p)
weights = list(LLM.glob("*.safetensors")) + list(LLM.glob("model-*.safetensors"))
print("weights", sorted(w.name for w in LLM.glob("*"))[:30])
print("total_bytes", sum(f.stat().st_size for f in LLM.rglob("*") if f.is_file()))
