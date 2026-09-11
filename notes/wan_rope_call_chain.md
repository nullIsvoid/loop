# Wan RoPE / temporal attention call chain

Analysis only (2026-09-11). No algorithm code yet.
Sources: Loopy-vendored Wan2.2 under lovelygirls-video `.tmp_research/Loopy/Wan2.2`, and ComfyUI `comfy/ldm/wan/model.py`.

## A. Official Wan (TI2V / T2V / I2V share `WanModel`)

Pipeline entry (TI2V example): `wan2/textimage2video.py` → `WanTI2V` loads `WanModel.from_pretrained` → denoise loop:

```text
latents (list of [C, F, H, W])
  → sample_scheduler timesteps
  → for each t:
        noise_pred = WanModel(x=latents, t=timestep, context=..., seq_len=...)
        latents = scheduler.step(noise_pred, t, latents)
  → VAE decode
```

### Inside `WanModel.forward` (`wan2/modules/model.py`)

```text
x: List[[C_in, F, H, W]]
  │  optional I2V: cat(y) on channel
  ▼
patch_embedding = Conv3d(patch_size)     # tokens
  ▼
grid_sizes = stack(u.shape[2:])         # [B, 3] = (F_patches, H_patches, W_patches)
  ▼
flatten → pad to seq_len                # video tokens [B, L, dim]
  ▼
timestep t → sinusoidal_embedding_1d → time_embedding → time_projection  # e0
  ▼
text context → text_embedding
  ▼
kwargs = {e0, seq_lens, grid_sizes, freqs=self.freqs, context, ...}
  ▼
for block in blocks:                    # WanAttentionBlock × num_layers
      block(x, **kwargs)
  ▼
Head → unpatchify(grid_sizes)
```

### Where `freqs` come from (construction, once)

In `WanModel.__init__`:

```text
d = dim // num_heads
self.freqs = cat(
  rope_params(1024, d - 4*(d//6)),   # temporal slice of head
  rope_params(1024, 2*(d//6)),       # height
  rope_params(1024, 2*(d//6)),       # width
)                                    # complex cis table, max_seq=1024
```

`rope_params` = `torch.polar(1, outer(arange(max_seq), theta^{-2i/d}))`.
This buffer is **linear** positions `0..1023`. It is **not** circular yet.

### Self-attention + RoPE apply (per block)

```text
WanAttentionBlock.forward
  → modulate x with e0
  → WanSelfAttention.forward(x, seq_lens, grid_sizes, freqs)
        q,k,v = Linear + (RMSNorm on q/k)
        q' = rope_apply(q, grid_sizes, freqs)
        k' = rope_apply(k, grid_sizes, freqs)
        out = flash_attention(q', k', v)   # V has no RoPE
        return o(out)
  → cross-attn (text) — no video RoPE
  → FFN
```

### Exact `rope_apply` geometry (baseline)

```text
freqs → split into [f_axis, h_axis, w_axis] by head dim thirds
for each sample (F,H,W) in grid_sizes:
    freqs_i = cat(
      freqs_f[:F] expanded over H,W,
      freqs_h[:H] expanded over F,W,
      freqs_w[:W] expanded over F,H,
    ) → shape [F*H*W, ...]
    q/k_complex *= freqs_i
```

Temporal neighbours in RoPE space are **adjacent indices on the F axis of this expanded table**. Token order is F-major flatten (`F*H*W`). There is **no separate temporal-only attention module** — time mixes inside the same full 3D self-attention via these positions.

### Loopy circularization (reference implementation)

File: `wan2/modules/model_roll.py`

```text
rope_apply_loop(...):
    build freqs_i as above
    freqs_3d = freqs_i.view(F, H, W, 1, -1)
    if time_shift != 0:
        freqs_3d = torch.roll(freqs_3d, shifts=time_shift, dims=0)  # TEMPORAL ONLY
    apply to q/k

WanSelfAttention:
    block_idx == 0 → time_shift = 0          # anchor layer
    else → time_shift = (block_idx-1) % (F-1) + 1

WanModel_roll.forward:
    for i, block in enumerate(blocks):
        block(..., block_idx=i)
```

So Loopy does **not** change:

- `rope_params` / `self.freqs` construction
- position id tensors (official path has none; only `grid_sizes`)
- V, cross-attn, scheduler, or latent layout

It changes **only** the temporal axis of the **already-expanded** 3D RoPE multiplier, per layer, at Q/K apply time.

## B. Comfy Wan path (product / Kijai stack)

Different wiring — same idea (3D RoPE on tokens), different objects:

```text
WanModel._forward(x [B,C,T,H,W], timestep, context, ...)
  → pad_to_patch_size
  → freqs = rope_encode(T, H, W, ...)
        img_ids[t,h,w] = (t_coord, h_coord, w_coord)   # linspace
        freqs = rope_embedder(img_ids)                  # EmbedND
  → forward_orig(... freqs=freqs ...)
  → WanAttentionBlock → WanSelfAttention
        q = apply_rope1(q, freqs)
        k = apply_rope1(k, freqs)
```

Here circularization candidates are:

1. Remap / roll **temporal coords in `img_ids[..., 0]`** before `rope_embedder`
2. Or reshape encoded `freqs` back to `(F,H,W,*)`, `torch.roll` on F, flatten again
3. Not a drop-in of Loopy `rope_apply_loop` without a `grid_sizes`-style apply path

Comfy already exposes `transformer_options["rope_options"]` with `shift_t` / `scale_t` — linear shift/scale, **not** circular roll.

## C. Inject-layer decision matrix

| Layer | Circularize here? | Notes |
|-------|-------------------|-------|
| Diffusion timestep `t` | No | Noise schedule, not video time topology |
| `rope_params` / global freqs table | Possible but coarse | Would circularize all uses; hard to do per-layer anchor schedule |
| `grid_sizes` / token order | No (for Loopy parity) | Layout stays linear FHW; topology via RoPE |
| Expanded `freqs_3d` roll (Loopy) | **Yes — recommended** | Matches proven Wan2.2 A14B Loopy code; per-layer shifts |
| `apply_rope1` wrapper (Comfy) | Later adapter | Need F-aware reshape or id remapping |
| Attention bias / mask | Optional later | Orthogonal; Loopy doesn't need it for closed loop |
| Post-step `RingLatentProcessor` | Control only (mode C) | Not primary |

## D. What Cursor should implement next (after ChatGPT ack)

1. Scaffold `src/latent_loop/rope/` with a pure function mirroring `rope_apply_loop` (no Wan import required for unit tests).
2. `adapters/wan/rope.py` + `attention.py` targeting **official** `WanSelfAttention` / `rope_apply` signature (`grid_sizes`, `freqs`, `block_idx`).
3. Leave Comfy adapter stub documented, not required for first smoke.
4. Keep `RingLatentProcessor` untouched as mode C.
