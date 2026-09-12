# Hunyuan H1 — Symmetric Circular Temporal RoPE call chain

Status: **H1 full-depth FAILED** (`2c7ecb1`). Circular RoPE not declared dead;  
**full-depth Wan-style port on Hunyuan is failed.** Next: depth localization H1-A/B/C.

Do **not** copy `adapters/wan/*` blindly. Do **not** start S4/S5 schedule sweeps.

## Goal

Same inputs as H0 (`person_loop_v1`, 50 steps, 480p, …).  
Only change: per-block Symmetric temporal freqs roll on **image** Q/K RoPE.

Success criteria (vs H0 late) — for any variant that claims success:

| check | want |
|-------|------|
| `F-1→0` | clearly closer to ordinary adjacent |
| `0→1` | **not worse** than H0 (~1.59×) |
| other edges | no new wall |
| `out_x3` | jump / stall / reverse / speed-pop only |

## Actual 480p_i2v runtime (measured H0/H1)

```text
double_blocks = 54
single_blocks = 0
latent F (tt) = 21   # 81 pixel frames, VAE t-factor 4
```

**Do not** analyze with the older assumption `20 double + 40 single = 60`.  
That layout may exist in other Hunyuan variants/docs; it is **not** what
`get_transformer_version(480p, i2v, …)` loads on our cloud path.

Official RoPE fields (still correct):

```text
rope_dim_list = [16, 56, 56]   # temporal, height, width
rope_theta    = 256
use_real      = True          # freqs_cos, freqs_sin

get_rotary_pos_embed((tt, th, tw))
  → get_nd_rotary_pos_embed → meshgrid ij → [L, D] cos/sin
  L = tt * th * tw   (token order: t-major)

MMDoubleStreamBlock (×54 on 480p_i2v): apply_rotary_emb(img_q, img_k, freqs_cis)
MMSingleStreamBlock (×0 on 480p_i2v):  n/a for this runtime
```

Transformer forward builds **one** shared `freqs_cis` then passes it to every block.  
Therefore Mode B must **roll per block** at apply time (pre-hook), not rebuild freqs once.

## Global block index (Symmetric schedule)

```text
double_blocks[i]  → block_idx = i                 # 0 .. 53 on 480p_i2v
single_blocks[j]  → block_idx = n_double + j      # unused when single=0
```

Shifts: `SymmetricShiftSchedule` → `0, +1, -1, +2, -2, …` then `% tt`.

## Roll semantics

`freqs_cos/sin` shaped `[L, D]` → view `[tt, th, tw, D]` → `torch.roll(..., dims=0)` → flatten.

- Roll **temporal axis only** (never H/W).
- `time_shift == 0` → identity (block 0 anchor).
- Sequence-parallel (`sp_size != 1`) not supported in v1 (H0/H1 runs are single-GPU).

## H1 full-depth result (closed as failed port)

| | H0 | H1 full |
|--|---:|---:|
| median | 54.7 | 96.5 |
| `0→1` | 87 / 1.59× | **303 / 3.14×** |
| `F-1→0` | 130 / 2.38× | **309 / 3.20×** |

Near dual spike. **Verdict:** full-depth Symmetric on all 54 blocks is harmful.

## Depth localization (DONE)

Same Symmetric schedule; limit which global indices roll:

| id | active blocks | late vs H0 |
|----|---------------|------------|
| H1-A | `0-17` | `F-1→0` **115** (better); no dual spike |
| H1-B | `18-35` | ≈ H0 |
| H1-C | `36-53` | mild worsen; still single seam |
| H1 full | `0-53` | dual spike (failed) |

**Finding:** no single third creates the dual spike → full-depth harm is cumulative/interaction.
See `artifacts/loop_benchmark_v1/H1_DEPTH_LOC_RESULT.md`.

CLI: `scripts/benchmark/run_hunyuan15_symmetric.py --experiment-id H1-A|H1-B|H1-C`

## Adapter surfaces

| module | role |
|--------|------|
| `adapters/hunyuan/rope.py` | `roll_hunyuan_freqs_cis` |
| `adapters/hunyuan/attention.py` | enable/disable + `active_block_indices` |
| `rope/schedule.py` | reuse `SymmetricShiftSchedule` (shared) |

## Out of scope for this pass

- New schedules (S4/S5 / Fixed / Identity as “next algorithm”)
- Conditioning edits / prompt rewrite
- W5-0 / WA-0 / W5-1 / WA-1 (paused while locating depth)
- Circular Temporal Context
- flash-attn install (torch attn fallback OK, as in H0)
