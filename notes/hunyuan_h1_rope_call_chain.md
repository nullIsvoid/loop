# Hunyuan H1 — Symmetric Circular Temporal RoPE call chain

Status: **authorized after H0** (`2a4d7ce` / sync `140d9f7`).  
Do **not** copy `adapters/wan/*` blindly.

## Goal

Same inputs as H0 (`person_loop_v1`, 50 steps, 480p, …).  
Only change: per-block Symmetric temporal freqs roll on **image** Q/K RoPE.

Success criteria (vs H0 late):

| check | want |
|-------|------|
| `F-1→0` | clearly closer to ordinary adjacent |
| `0→1` | **not worse** than H0 (~1.59×) |
| other edges | no new wall |
| `out_x3` | jump / stall / reverse / speed-pop only |

## Official layout (verified on `/root/HunyuanVideo-1.5`)

```text
rope_dim_list = [16, 56, 56]   # temporal, height, width
rope_theta    = 256
use_real      = True          # freqs_cos, freqs_sin

get_rotary_pos_embed((tt, th, tw))
  → get_nd_rotary_pos_embed → meshgrid ij → [L, D] cos/sin
  L = tt * th * tw   (token order: t-major)

MMDoubleStreamBlock (×20): apply_rotary_emb(img_q, img_k, freqs_cis)
MMSingleStreamBlock (×40): apply_rotary_emb(img_q, img_k, freqs_cis)  # img tokens only
```

Transformer forward builds **one** shared `freqs_cis` then passes it to every block.  
Therefore Mode B must **roll per block** at apply time (same idea as Wan), not rebuild freqs once.

## Global block index (Symmetric schedule)

```text
double_blocks[i]  → block_idx = i                 # 0 .. 19
single_blocks[j]  → block_idx = 20 + j            # 20 .. 59
```

Shifts: `SymmetricShiftSchedule` → `0, +1, -1, +2, -2, …` then `% tt`.

## Roll semantics

`freqs_cos/sin` shaped `[L, D]` → view `[tt, th, tw, D]` → `torch.roll(..., dims=0)` → flatten.

- Roll **temporal axis only** (never H/W).
- `time_shift == 0` → identity (block 0 anchor).
- Sequence-parallel (`sp_size != 1`) not supported in v1 (H0/H1 runs are single-GPU).

## Adapter surfaces

| module | role |
|--------|------|
| `adapters/hunyuan/rope.py` | `roll_hunyuan_freqs_cis` |
| `adapters/hunyuan/attention.py` | enable/disable Mode B on transformer blocks |
| `rope/schedule.py` | reuse `SymmetricShiftSchedule` (shared) |

## Out of scope for H1

- Conditioning edits / prompt rewrite
- W5-1 / WA-1
- Circular Temporal Context
- flash-attn install (torch attn fallback OK, as in H0)
