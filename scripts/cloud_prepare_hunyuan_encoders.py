#!/usr/bin/env python3
"""Prepare writable Hunyuan ckpt tree + download missing encoders onto /workspace.

/model is read-only UPFS; DiT/VAE stay symlinked from ModelScope.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

CKPT = Path("/workspace/hunyuan-ckpts/HunyuanVideo-1.5")
SRC = Path("/model/ModelScope/Tencent-Hunyuan/HunyuanVideo-1.5")
FLUX = Path("/model/HuggingFace/black-forest-labs/FLUX.1-Redux-dev")
LOG_DIR = Path("/workspace/hunyuan-ckpts")

os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"
os.environ["MODELSCOPE_CACHE"] = "/workspace/.cache/modelscope"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
# Prefer HF mirror if direct HF is slow (common on CN clouds).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MODELSCOPE_CACHE"]).mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
(CKPT / "text_encoder").mkdir(parents=True, exist_ok=True)
(CKPT / "vision_encoder").mkdir(parents=True, exist_ok=True)


def link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        print(f"exists {dst} -> {dst.resolve() if dst.exists() else 'broken'}")
        return
    dst.symlink_to(src, target_is_directory=src.is_dir())
    print(f"linked {dst} -> {src}")


def main() -> None:
    print("disk /workspace free GB", shutil.disk_usage("/workspace").free / 1e9)
    for name in (
        "transformer",
        "vae",
        "scheduler",
        "assets",
        "model_index.json",
        "configuration.json",
    ):
        src = SRC / name
        if src.exists():
            link(src, CKPT / name)
        else:
            print("missing src", src)

    link(FLUX, CKPT / "vision_encoder" / "siglip")

    # Spawn downloads as separate processes so we can poll logs.
    py = "/usr/local/miniconda3/envs/mobius/bin/python"
    jobs = [
        (
            "download_llm.log",
            [
                py,
                "-c",
                "import os; from huggingface_hub import snapshot_download; "
                "os.environ['HF_HOME']='/workspace/.cache/huggingface'; "
                "os.environ['HUGGINGFACE_HUB_CACHE']='/workspace/.cache/huggingface/hub'; "
                "os.environ['HF_ENDPOINT']=os.environ.get('HF_ENDPOINT','https://hf-mirror.com'); "
                "p=snapshot_download(repo_id='Qwen/Qwen2.5-VL-7B-Instruct',"
                "local_dir='/workspace/hunyuan-ckpts/HunyuanVideo-1.5/text_encoder/llm',"
                "local_dir_use_symlinks=False); print('LLM_DONE', p)",
            ],
        ),
        (
            "download_byt5.log",
            [
                py,
                "-c",
                "import os; from huggingface_hub import snapshot_download; "
                "os.environ['HF_HOME']='/workspace/.cache/huggingface'; "
                "os.environ['HUGGINGFACE_HUB_CACHE']='/workspace/.cache/huggingface/hub'; "
                "os.environ['HF_ENDPOINT']=os.environ.get('HF_ENDPOINT','https://hf-mirror.com'); "
                "p=snapshot_download(repo_id='google/byt5-small',"
                "local_dir='/workspace/hunyuan-ckpts/HunyuanVideo-1.5/text_encoder/byt5-small',"
                "local_dir_use_symlinks=False); print('BYT5_DONE', p)",
            ],
        ),
        (
            "download_glyph.log",
            [
                py,
                "-c",
                "import os; "
                "os.environ['MODELSCOPE_CACHE']='/workspace/.cache/modelscope'; "
                "from modelscope.hub.snapshot_download import snapshot_download; "
                "p=snapshot_download('AI-ModelScope/Glyph-SDXL-v2',"
                "local_dir='/workspace/hunyuan-ckpts/HunyuanVideo-1.5/text_encoder/Glyph-SDXL-v2'); "
                "print('GLYPH_DONE', p)",
            ],
        ),
    ]

    for log_name, cmd in jobs:
        log_path = LOG_DIR / log_name
        # skip if already complete
        target_hints = {
            "download_llm.log": CKPT / "text_encoder" / "llm" / "config.json",
            "download_byt5.log": CKPT / "text_encoder" / "byt5-small" / "config.json",
            "download_glyph.log": CKPT
            / "text_encoder"
            / "Glyph-SDXL-v2"
            / "checkpoints"
            / "byt5_model.pt",
        }
        done_marker = target_hints[log_name]
        if done_marker.is_file():
            print("already done", done_marker)
            continue
        print("starting", log_name, cmd[-1][:80], "...")
        with log_path.open("w", encoding="utf-8") as logf:
            subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                start_new_session=True,
            )

    print("SETUP_STARTED")
    print("CKPT", CKPT)
    for child in sorted(CKPT.iterdir()):
        print(" ", child.name, "->", os.path.realpath(child))


if __name__ == "__main__":
    main()
