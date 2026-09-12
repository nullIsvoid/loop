# HunyuanVideo-1.5 I2V call chain

Source of truth: official clone `/root/HunyuanVideo-1.5`  
(`https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5`, analyzed 2026-09-12).

Weights on cloud (incomplete HF/ModelScope layout may still need full download for run):  
`/model/ModelScope/Tencent-Hunyuan/HunyuanVideo-1.5/` (transformer/480p_i2v, vae, …).

This note is **analysis only**. No Circular RoPE. No conditioning patches.

---

## Entry

```text
generate.py --task i2v
  → hyvideo.pipelines.hunyuan_video_pipeline.HunyuanVideoPipeline.__call__
```

Key modules:

| piece | path |
|-------|------|
| Pipeline | `hyvideo/pipelines/hunyuan_video_pipeline.py` |
| Transformer | `hyvideo/models/transformers/hunyuanvideo_1_5_transformer.py` |
| Patch embed | `hyvideo/models/transformers/modules/embed_layers.py` (`PatchEmbed`) |
| RoPE | `hyvideo/models/transformers/modules/posemb_layers.py` |
| Attention | `hyvideo/models/transformers/modules/attention.py` |
| Vision encoder | `hyvideo/models/vision_encoder/` |
| VAE | `hyvideo/models/autoencoders/hunyuanvideo_15_vae.py` |

---

## I2V conditioning (non-destructive)

Unlike Wan TI2V-5B, Hunyuan does **not** overwrite `latents[:, :, 0]` after each scheduler step.

### 1. Reference → condition latent

`get_image_condition_latents(task_type="i2v", …)`:

```text
PIL reference
  → resize / center-crop to target H×W
  → VAE encode → cond_latents  [B, C, 1, h, w]
  → × vae.scaling_factor
```

### 2. Expand + mask (still condition only)

`_prepare_cond_latents`:

```text
latents_concat = cond_latents.repeat(..., T, ...)
latents_concat[:, :, 1:, :, :] = 0          # only temporal index 0 holds ref
mask: 1 at index 0, 0 elsewhere (i2v task mask)
cond_latents = cat([latents_concat, mask], dim=1)   # channel concat
```

### 3. Denoise loop (generated latents stay free)

```text
latents = random noise   # prepare_latents — full T, not pinned to ref
for t in timesteps:
    latents_concat = cat([latents, cond_latents], dim=1)   # channel axis
    noise_pred = transformer(latents_concat, t, text…, vision_states=…)
    latents = scheduler.step(noise_pred, t, latents)      # updates noise stream only
```

`cond_latents` is **fixed** side-channel input every step.  
No `frame0 timestep = 0` token mask equivalent to Wan TI2V-5B.

### 4. Vision semantic tokens (second channel)

`_prepare_vision_states` → `VisionEncoder.encode_images` →  
`VisionProjection` → concatenated into text/encoder stream with `cond_type_embedding` id=2.

So I2V uses **both**:

1. Channel-concat VAE cond + mask into `PatchEmbed`
2. Vision encoder tokens into joint attention

---

## Transformer / RoPE / attention

### Patch embed

`PatchEmbed` with `concat_condition=True`:

```text
in_chans_effective = in_chans * 2 + 1   # (or reshape_temporal variant)
# Conv3d projects noisy latent channels + cond channels + mask
```

### 3D RoPE

`HunyuanVideo_1_5_Transformer.get_rotary_pos_embed(rope_sizes=(tt, th, tw))`:

```text
rope_dim_list default = [16, 56, 56]   # temporal, height, width
rope_theta = 256
freqs_cos, freqs_sin = get_nd_rotary_pos_embed(...)
```

Applied in double/single stream blocks:

```text
img_q, img_k = Linear → reshape [B, L, H, D]
img_q, img_k = apply_rotary_emb(img_q, img_k, (freqs_cos, freqs_sin), head_first=False)
# V has no RoPE
→ parallel_attention((img_q, txt_q), (img_k, txt_k), (img_v, txt_v), ...)
```

`apply_rotary_emb` lives in `posemb_layers.py` (real cos/sin or complex cis).  
Frequencies are **linear meshgrid** positions — **not** circular yet.  
Temporal axis is **first** size in `rope_sizes` / `rope_dim_list[0]`.

### Q/K shapes (per block, image stream)

```text
img_q, img_k: [B, L_img, heads_num, head_dim]
head_dim = sum(rope_dim_list)   # e.g. 16+56+56
L_img = tt * th * tw   (after patch; may be SP-chunked)
```

---

## Contrast vs Wan TI2V-5B / A14B

| | TI2V-5B | A14B `WanI2V` | HunyuanVideo-1.5 |
|--|---------|---------------|------------------|
| Generated latent | noise, then **hard-clamp frame0** each step | noise end-to-end | noise end-to-end |
| Ref path | overwrite `latent[:,0]` + `t0=0` tokens | `y=cat(mask, vae(img+zeros))` channel cond | `cat(latents, cond+mask)` + vision tokens |
| Timestep special | frame0 forced 0 | uniform `t` | uniform `t` |
| RoPE | Wan 3-split freqs on Q/K | same family | ND RoPE `rope_dim_list` on img Q/K |

---

## Implications for Mode B port (later Phase D)

Hook points (do not implement yet):

1. `get_rotary_pos_embed` / `get_nd_rotary_pos_embed` — temporal coords before cos/sin  
2. or `apply_rotary_emb` call sites inside `MMDoubleStreamBlock` / `MMSingleStreamBlock`  
3. Per-block shift schedule must follow Hunyuan block index (double then single streams)

Do **not** copy `adapters/wan/rope.py` blindly — grid / head layout differ.
