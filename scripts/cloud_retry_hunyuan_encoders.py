#!/usr/bin/env python3
"""Retry Hunyuan encoder downloads onto /workspace (writable)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

CKPT = Path("/workspace/hunyuan-ckpts/HunyuanVideo-1.5")
LOG = Path("/workspace/hunyuan-ckpts")
PY = "/usr/local/miniconda3/envs/mobius/bin/python"

os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"
os.environ["MODELSCOPE_CACHE"] = "/workspace/.cache/modelscope"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# Try official HF first after disabling xet; mirror can break CAS.
os.environ.pop("HF_ENDPOINT", None)

Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MODELSCOPE_CACHE"]).mkdir(parents=True, exist_ok=True)


def run(cmd: list[str], log_name: str) -> int:
    log_path = LOG / log_name
    print("RUN", " ".join(cmd[:4]), "... ->", log_path)
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy())
    print("EXIT", log_name, p.returncode)
    print(log_path.read_text(encoding="utf-8", errors="replace")[-2000:])
    return p.returncode


def ensure_modelscope() -> None:
    try:
        import modelscope  # noqa: F401

        print("modelscope ok")
    except ImportError:
        print("pip install modelscope ...")
        subprocess.check_call(
            [PY, "-m", "pip", "install", "-q", "modelscope"],
            env=os.environ.copy(),
        )


def main() -> None:
    llm_cfg = CKPT / "text_encoder" / "llm" / "config.json"
    byt5_cfg = CKPT / "text_encoder" / "byt5-small" / "config.json"
    glyph_pt = (
        CKPT / "text_encoder" / "Glyph-SDXL-v2" / "checkpoints" / "byt5_model.pt"
    )

    if not byt5_cfg.is_file():
        rc = run(
            [
                PY,
                "-c",
                "from huggingface_hub import snapshot_download; "
                "p=snapshot_download(repo_id='google/byt5-small',"
                "local_dir='/workspace/hunyuan-ckpts/HunyuanVideo-1.5/text_encoder/byt5-small'); "
                "print('BYT5_DONE', p)",
            ],
            "download_byt5.log",
        )
        if rc != 0:
            sys.exit(rc)
    else:
        print("byt5 already ok")

    if not llm_cfg.is_file():
        # Prefer ModelScope mirror for Qwen (CN-friendly); fall back to HF.
        ensure_modelscope()
        rc = run(
            [
                PY,
                "-c",
                "from modelscope.hub.snapshot_download import snapshot_download; "
                "p=snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct',"
                "local_dir='/workspace/hunyuan-ckpts/HunyuanVideo-1.5/text_encoder/llm'); "
                "print('LLM_DONE', p)",
            ],
            "download_llm.log",
        )
        if rc != 0:
            print("modelscope llm failed; trying HF with HF_HUB_DISABLE_XET=1")
            rc = run(
                [
                    PY,
                    "-c",
                    "from huggingface_hub import snapshot_download; "
                    "p=snapshot_download(repo_id='Qwen/Qwen2.5-VL-7B-Instruct',"
                    "local_dir='/workspace/hunyuan-ckpts/HunyuanVideo-1.5/text_encoder/llm'); "
                    "print('LLM_DONE', p)",
                ],
                "download_llm_hf.log",
            )
            if rc != 0:
                sys.exit(rc)
    else:
        print("llm already ok")

    if not glyph_pt.is_file():
        ensure_modelscope()
        rc = run(
            [
                PY,
                "-c",
                "from modelscope.hub.snapshot_download import snapshot_download; "
                "p=snapshot_download('AI-ModelScope/Glyph-SDXL-v2',"
                "local_dir='/workspace/hunyuan-ckpts/HunyuanVideo-1.5/text_encoder/Glyph-SDXL-v2'); "
                "print('GLYPH_DONE', p)",
            ],
            "download_glyph.log",
        )
        if rc != 0:
            sys.exit(rc)
    else:
        print("glyph already ok")

    print("ALL_ENCODERS_READY")
    for p in [llm_cfg, byt5_cfg, glyph_pt, CKPT / "vision_encoder" / "siglip"]:
        print(p, "OK" if p.exists() else "MISSING")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("elapsed_sec", round(time.time() - t0, 1))
