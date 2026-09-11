#!/usr/bin/env python3
"""Cloud Mode A vs Mode B smoke on official Wan TI2V-5B backbone.

Loads only WanModel (no T5/VAE decode). One tiny forward each for A/B.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

WAN_ROOT = Path(os.environ.get("WAN_ROOT", "/root/Wan2.2"))
CKPT = Path(
    os.environ.get(
        "WAN_TI2V_CKPT",
        "/model/HuggingFace/Wan-AI/Wan2.2-TI2V-5B",
    )
)
LOOP_SRC = Path(os.environ.get("LOOP_SRC", "/root/latent-loop/src"))

sys.path.insert(0, str(WAN_ROOT))
sys.path.insert(0, str(LOOP_SRC))


def sdpa_attention(q=None, k=None, v=None, q_lens=None, k_lens=None, window_size=(-1, -1), **kwargs):
    """Drop-in for wan flash_attention when flash_attn is unavailable.

    Supports both ``flash_attention(q, k, v, ...)`` and keyword form.
    """
    _ = q_lens, window_size, kwargs
    if q is None or k is None or v is None:
        raise TypeError("sdpa_attention requires q, k, v")
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)
    return out.transpose(1, 2).contiguous()


def _import_wan_model():
    """Import WanModel without triggering wan/__init__ heavy deps (T5/ftfy)."""
    import importlib.util
    import types

    pkg_root = WAN_ROOT / "wan"
    modules_root = pkg_root / "modules"
    for name, path in (
        ("wan", pkg_root),
        ("wan.modules", modules_root),
    ):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [str(path)]  # type: ignore[attr-defined]
            sys.modules[name] = m

    attn_path = modules_root / "attention.py"
    attn_spec = importlib.util.spec_from_file_location(
        "wan.modules.attention", attn_path
    )
    assert attn_spec and attn_spec.loader
    attn_mod = importlib.util.module_from_spec(attn_spec)
    sys.modules["wan.modules.attention"] = attn_mod
    attn_spec.loader.exec_module(attn_mod)

    model_path = modules_root / "model.py"
    spec = importlib.util.spec_from_file_location("wan.modules.model", model_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wan.modules.model"] = mod
    spec.loader.exec_module(mod)
    return mod.WanModel


def main() -> None:
    WanModel = _import_wan_model()
    from latent_loop.adapters.wan.attention import (
        disable_mode_b_on_wan_model,
        enable_mode_b_on_wan_model,
    )
    from latent_loop.rope.schedule import LoopyShiftSchedule, list_layer_time_shifts

    device = torch.device("cuda")
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    print("ckpt", CKPT)
    print("wan_root", WAN_ROOT)

    # Cross-attn still calls wan.modules.attention.flash_attention; patch globally.
    import wan.modules.attention as wan_attn
    import wan.modules.model as wan_model_mod

    wan_attn.flash_attention = sdpa_attention
    wan_model_mod.flash_attention = sdpa_attention

    t0 = time.time()
    print("loading WanModel.from_pretrained ...")
    model = WanModel.from_pretrained(str(CKPT))
    model.eval()
    model.to(device)
    print(f"loaded in {time.time() - t0:.1f}s  layers={len(model.blocks)} type={model.model_type}")

    # Tiny latent: keep tokens small for VRAM. TI2V in_dim from config.
    in_dim = int(model.config.in_dim)
    # F,H,W after patch embedding grid — input is [C,F,H,W] before patch
    # Use small spatial; F=8 latent frames
    f, h, w = 8, 16, 16
    x = [torch.randn(in_dim, f, h, w, device=device, dtype=torch.float32)]
    # seq_len must cover F'*H'*W' after patch; patch_size usually (1,2,2)
    pf, ph, pw = model.patch_size
    f_p, h_p, w_p = f // pf, h // ph, w // pw
    seq_len = f_p * h_p * w_p
    context = [torch.randn(16, model.text_dim, device=device, dtype=torch.float32)]
    t = torch.tensor([500], device=device)

    print("grid_patches", f_p, h_p, w_p, "seq_len", seq_len)
    print(
        "loopy shifts preview",
        list_layer_time_shifts(len(model.blocks), f_p, LoopyShiftSchedule())[:8],
        "...",
    )

    with torch.no_grad():
        # Mode A: baseline (mode B disabled)
        enable_mode_b_on_wan_model(
            model,
            schedule=LoopyShiftSchedule(),
            flash_attention_fn=sdpa_attention,
            enabled=False,
        )
        out_a = model(x, t=t, context=context, seq_len=seq_len)[0]
        disable_mode_b_on_wan_model(model)

        enable_mode_b_on_wan_model(
            model,
            schedule=LoopyShiftSchedule(),
            flash_attention_fn=sdpa_attention,
            enabled=True,
        )
        out_b = model(x, t=t, context=context, seq_len=seq_len)[0]
        disable_mode_b_on_wan_model(model)

    diff = (out_a.float() - out_b.float()).abs()
    report = {
        "ok": True,
        "out_a_shape": list(out_a.shape),
        "out_b_shape": list(out_b.shape),
        "shapes_match": list(out_a.shape) == list(out_b.shape),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "mode_b_differs": bool(diff.max().item() > 0),
        "num_layers": len(model.blocks),
        "latent_frames_patched": f_p,
    }
    print(json.dumps(report, indent=2))
    if not report["shapes_match"]:
        raise SystemExit("shape mismatch")
    if not report["mode_b_differs"]:
        raise SystemExit("Mode B output identical to Mode A — unexpected for Loopy schedule")
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
