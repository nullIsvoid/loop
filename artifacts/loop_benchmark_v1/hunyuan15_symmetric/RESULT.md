# H1 — RESULT (FAILED vs H0 success criteria)

## Question

On `person_loop_v1`, does Symmetric Circular Temporal RoPE reduce Hunyuan `F-1→0` **without** worsening `0→1`?

## Late comparison

| metric | H0 | H1 | delta |
|--------|---:|---:|------|
| median | 54.7 | 96.5 | ↑ |
| `0→1` | 87.0 (1.59×) | **302.6** (3.14×) | **much worse** |
| `F-1→0` | 130.1 (2.38×) | **309.1** (3.20×) | **worse** |
| max edge | `20→0` | `20→0` | — |
| shape | mild 0→1 bump | **near dual spike** (`20→0`≈`0→1`) | regress |

Top edges H1: `20→0` ≈ `0→1` ≫ everything else.

## Verdict

**H1 v1 does not pass.** Symmetric temporal freqs roll on this 480p_i2v stack (`double_blocks=54`, `single_blocks=0`) **inflated** the ring seam and **created** a near-TI2V-style second wall at `0→1`.

Do **not** treat this as “Circular RoPE fails forever” — first diagnose:

1. Schedule depth vs 54 blocks / `tt=21` (too many large wraps?)
2. Roll semantics vs Hunyuan t-major meshgrid (direction / which tokens?)
3. Whether Mode B should apply only to a subset of blocks
4. Interaction with I2V cond at index 0

## Visual

`out_x3.mp4` — expect worse loop jump than H0; confirm with eyes.

## Next

**Depth localization** (same Symmetric schedule): H1-A `0-17`, H1-B `18-35`, H1-C `36-53`.  
No S4/S5. W5/WA paused until depth answer exists.
