# Wan2.2 I2V-A14B native probe — RESULT (Phase A)

## Question

Does **native A14B I2V** (independent `y` conditioning, **no** Mode-B RoPE, **no** conditioning edits) still show the TI2V-5B-style **index-0 double spike** (both `F-1→0` and `0→1` anomalous)?

## Setup

- Official `WanI2V` / ckpt `Wan2.2-I2V-A14B`
- seed=42, 81 frames, 40 steps, max_area=480×832, UniPC, guide=3.5
- Full-ring adjacent L2; probes early=2 / middle=19 / late=39
- Note: source image was a substitute loop still (original person.png was cleared with artifacts). Architecture readout only.

## Late (step 39)

| metric | value |
|--------|------:|
| median gap | 122 |
| `0→1` | **123** ≈ median |
| `F-1→0` (`20→0`) | **236** |
| max edge | `20→0` |
| max/median | **1.93** |

Early/middle rings are nearly flat (max/median ≈ 1.00).

## Verdict

**Architecture pass only — not a visual seamless loop.**

Eye check of `out_x3.mp4` still shows a clear last→first jump. That matches the numbers: `F-1→0` ≈ **1.93×** median. We did **not** claim pixel/temporal seamless.

What *did* pass the probe question:

**Not a TI2V-5B-style double spike.**

- `0→1` tracks ordinary adjacent motion.
- Only `F-1→0` is the elevated loop seam.

So independent channel conditioning (`y = mask + VAE(ref+zeros)`) **removes the artificial second wall at 0→1** that TI2V-5B hard frame0 clamp + `t0=0` created. The **ring seam itself remains**.

This supports **H1/H2** (conditioning-implementation-specific extra wall on TI2V-5B), not “A14B already loops clean.”

## Next

Proceed with HunyuanVideo-1.5 as primary (Phase C native H0 probe). Do not invest further in TI2V-5B latent surgery. A14B stays architecture control / reference — **not** proof of seamless success.

## Classification

**OLD RESEARCH EVIDENCE / NOT STRICT CROSS-MODEL BENCHMARK** (substitute source image). Strict reruns: rtifacts/loop_benchmark_v1/.
