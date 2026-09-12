#!/usr/bin/env python3
"""W5-0: Wan2.2 TI2V-5B native I2V on fixed loop benchmark.

No Symmetric Circular Temporal RoPE. No conditioning surgery.
Official hard frame0 I2V path (latent[0] re-pinned each step).
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image

_SCRIPTS = Path(__file__).resolve().parents[1]
_BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_SCRIPTS))

from common import (  # noqa: E402
    DEFAULT_ARTIFACTS_ROOT,
    DEFAULT_BENCHMARK_DIR,
    default_probe_steps,
    finalize_media,
    load_benchmark,
    prepare_run_dir,
    probe_tag_map,
    save_latent_probe,
    write_json,
    write_result_stub,
)
from cloud_mode_b_generate import (  # noqa: E402
    CKPT,
    LOOP_SRC,
    WAN_ROOT,
    _install_official_attention_backend,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)


def _generate_native_with_probes(
    pipe,
    *,
    prompt: str,
    img: Image.Image,
    n_prompt: str,
    seed: int,
    max_area: int,
    frame_num: int,
    shift: float,
    sample_solver: str,
    sampling_steps: int,
    guide_scale: float,
    probe_dir: Path,
    probe_steps: tuple[int, ...],
) -> tuple[torch.Tensor, list[dict]]:
    import torchvision.transforms.functional as TF
    from tqdm import tqdm
    from wan.utils.fm_solvers import (
        FlowDPMSolverMultistepScheduler,
        get_sampling_sigmas,
        retrieve_timesteps,
    )
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from wan.utils.utils import best_output_size, masks_like

    ih, iw = img.height, img.width
    dh = pipe.patch_size[1] * pipe.vae_stride[1]
    dw = pipe.patch_size[2] * pipe.vae_stride[2]
    ow, oh = best_output_size(iw, ih, dw, dh, max_area)
    scale = max(ow / iw, oh / ih)
    img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
    x1 = (img.width - ow) // 2
    y1 = (img.height - oh) // 2
    img = img.crop((x1, y1, x1 + ow, y1 + oh))
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(pipe.device).unsqueeze(1)

    f_pix = frame_num
    seq_len = (
        ((f_pix - 1) // pipe.vae_stride[0] + 1)
        * (oh // pipe.vae_stride[1])
        * (ow // pipe.vae_stride[2])
        // (pipe.patch_size[1] * pipe.patch_size[2])
    )
    seq_len = int(math.ceil(seq_len / pipe.sp_size)) * pipe.sp_size

    seed_g = torch.Generator(device=pipe.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        pipe.vae.model.z_dim,
        (f_pix - 1) // pipe.vae_stride[0] + 1,
        oh // pipe.vae_stride[1],
        ow // pipe.vae_stride[2],
        dtype=torch.float32,
        generator=seed_g,
        device=pipe.device,
    )

    if not pipe.t5_cpu:
        pipe.text_encoder.model.to(pipe.device)
        context = pipe.text_encoder([prompt], pipe.device)
        context_null = pipe.text_encoder([n_prompt], pipe.device)
        pipe.text_encoder.model.cpu()
    else:
        context = pipe.text_encoder([prompt], torch.device("cpu"))
        context_null = pipe.text_encoder([n_prompt], torch.device("cpu"))
        context = [t.to(pipe.device) for t in context]
        context_null = [t.to(pipe.device) for t in context_null]

    z = pipe.vae.encode([img])
    tags = probe_tag_map(probe_steps)
    probe_metas: list[dict] = []

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(pipe.model, "no_sync", noop_no_sync)

    with (
        torch.amp.autocast("cuda", dtype=pipe.param_dtype),
        torch.no_grad(),
        no_sync(),
    ):
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(
                sampling_steps, device=pipe.device, shift=shift
            )
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=pipe.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler, device=pipe.device, sigmas=sampling_sigmas
            )
        else:
            raise NotImplementedError(sample_solver)

        latent = noise
        _mask1, mask2 = masks_like([noise], zero=True)
        # Official TI2V-5B hard frame0 clamp
        latent = (1.0 - mask2[0]) * z[0] + mask2[0] * latent

        arg_c = {"context": [context[0]], "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        pipe.model.to(pipe.device)
        torch.cuda.empty_cache()

        for step_i, t in enumerate(tqdm(timesteps)):
            latent_model_input = [latent.to(pipe.device)]
            timestep = torch.stack([t]).to(pipe.device)
            temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
            temp_ts = torch.cat(
                [
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep,
                ]
            )
            timestep = temp_ts.unsqueeze(0)

            noise_pred_cond = pipe.model(latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = pipe.model(
                latent_model_input, t=timestep, **arg_null
            )[0]
            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latent.unsqueeze(0),
                return_dict=False,
                generator=seed_g,
            )[0]
            latent = temp_x0.squeeze(0)
            latent = (1.0 - mask2[0]) * z[0] + mask2[0] * latent

            if step_i in tags:
                tag = tags[step_i]
                logging.info("probe step=%s tag=%s F=%s", step_i, tag, latent.shape[1])
                probe_metas.append(
                    save_latent_probe(
                        latent, probe_dir / "latent", tag=tag, step_i=step_i
                    )
                )

        pipe.model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        videos = pipe.vae.decode([latent])

    return videos[0], probe_metas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W5-0 Wan TI2V-5B native benchmark")
    p.add_argument("--benchmark-dir", type=str, default=str(DEFAULT_BENCHMARK_DIR))
    p.add_argument(
        "--out-root",
        type=str,
        default=str(
            Path(os.environ.get("LOOP_ARTIFACTS_ROOT", DEFAULT_ARTIFACTS_ROOT))
        ),
    )
    p.add_argument("--run-name", type=str, default="wan_ti2v5b_native")
    p.add_argument("--ckpt", type=str, default=str(CKPT))
    p.add_argument("--max-area", type=int, default=704 * 1280)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guide-scale", type=float, default=5.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--sample-solver", type=str, default="unipc")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bench = load_benchmark(args.benchmark_dir)
    assert bench.frame_num % 4 == 1, "Wan frame_num must be 4n+1"

    out_dir = prepare_run_dir(args.out_root, args.run_name, bench)
    probe_steps = default_probe_steps(args.steps)

    sys.path.insert(0, str(WAN_ROOT))
    sys.path.insert(0, str(LOOP_SRC))
    attn_info = _install_official_attention_backend()
    attn_info.pop("flash_fn", None)

    from wan.configs import WAN_CONFIGS
    from wan.textimage2video import WanTI2V

    cfg = WAN_CONFIGS["ti2v-5B"]
    n_prompt = cfg.sample_neg_prompt
    fps = int(cfg.sample_fps)

    logging.info("W5-0 building WanTI2V from %s", args.ckpt)
    pipe = WanTI2V(
        config=cfg,
        checkpoint_dir=str(args.ckpt),
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        init_on_cpu=True,
        convert_model_dtype=True,
    )

    img = Image.open(bench.source_path).convert("RGB")
    t0 = time.time()
    video, probes = _generate_native_with_probes(
        pipe,
        prompt=bench.prompt,
        img=img,
        n_prompt=n_prompt,
        seed=bench.seed,
        max_area=args.max_area,
        frame_num=bench.frame_num,
        shift=args.shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.steps,
        guide_scale=args.guide_scale,
        probe_dir=out_dir,
        probe_steps=probe_steps,
    )
    elapsed = time.time() - t0
    media = finalize_media(video, out_dir, fps=fps)
    late = next((p for p in probes if p["tag"] == "late"), None)

    run = {
        "experiment_id": "W5-0",
        "name": "wan_ti2v5b_native",
        "mode_b_rope": False,
        "conditioning_edits": False,
        "benchmark": bench.as_metadata(),
        "model": {
            "family": "Wan2.2",
            "variant": "TI2V-5B",
            "ckpt": args.ckpt,
        },
        "model_specific": {
            "steps": args.steps,
            "guide_scale": args.guide_scale,
            "shift": args.shift,
            "sample_solver": args.sample_solver,
            "max_area": args.max_area,
            "probe_steps": list(probe_steps),
            "attention_backend": attn_info,
        },
        "elapsed_sec": round(elapsed, 2),
        "media": media,
        "probes": probes,
        "late": None
        if late is None
        else {
            "max_gap_edge": late["max_gap_edge"],
            "max_gap_l2": late["max_gap_l2"],
            "median_gap_l2": late["median_gap_l2"],
            "max_vs_median": late["max_vs_median"],
            "seam_F-1->0": late["seam_F-1->0"],
            "seam_0->1": late["seam_0->1"],
        },
        "question": (
            "On person_loop_v1, does native TI2V-5B late latent show double spike "
            "(F-1→0 and 0→1 both elevated)?"
        ),
    }
    write_json(out_dir / "run.json", run)
    write_result_stub(
        out_dir,
        experiment_id="W5-0",
        question=run["question"],
        late=run["late"],
        notes=[
            "Native hard frame0 conditioning only.",
            "Diagnosis metrics — not a scoring Gate.",
            "OLD RESEARCH EVIDENCE under artifacts/mode_b_i2v* is not this benchmark.",
        ],
    )
    print(json_dumps_ok(run))
    print("W5_0_OK")


def json_dumps_ok(run: dict) -> str:
    import json

    return json.dumps({"ok": True, "late": run.get("late")}, indent=2)


if __name__ == "__main__":
    main()
