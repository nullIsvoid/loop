# Current latent-space handling (Mode B)

## What we do today (active path)

**Not** residual latent mixing. The live product path is:

```text
Wan TI2V-5B denoise latents  [C, F, H, W]
        ↓
patchify → tokens
        ↓
each WanSelfAttention:
  Q,K ← Circular Temporal RoPE
        expand official freqs → freqs_3d [F,H,W,…]
        torch.roll on F only by SymmetricShiftSchedule(block)
        apply RoPE to Q/K only (V untouched)
  attention as official Wan
        ↓
decode VAE → RGB video
```

Default schedule (frozen): `SymmetricShiftSchedule`  
`0, +1, -1, +2, -2, …` then `% F` (e.g. F=21 → `0,1,20,2,19,…`).

## What we do *not* do (yet)

| Mechanism | Status |
|-----------|--------|
| `RingLatentProcessor` / residual circular mix on latents | experimental / Mode C only — **not** in I2V/T2V runs |
| True periodic temporal RoPE (math ring freqs) | not implemented — candidate **D** if diagnostics say phase/position |
| Circular temporal context / latent shift across F-1→0 | not implemented — candidate **D** if diagnostics say velocity/content |
| New shift schedules S4/S5 | forbidden |

## I2V conditioning note

On I2V, first-frame latent is locked via Wan mask (`(1-mask)*z0 + mask*noise`) each step. Mode B still only changes **RoPE phase on Q/K**; it does not add extra ring coupling in latent content.

## Research question now

Where does B’s **residual** seam first appear — decoded RGB only, or already in temporal latent slices at early vs late denoise?
