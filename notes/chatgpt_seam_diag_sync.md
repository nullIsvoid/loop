# ChatGPT sync — A14B Phase A: architecture pass ≠ visual seamless

## Do not misread

**Eye check of `out_x3.mp4` still shows a clear loop jump.**  
Phase A did **not** claim pixel/temporal seamless loop. The residual ring seam is real.

Late latent full-ring (F=21):

| edge | L2 gap | note |
|------|-------:|------|
| ordinary adjacent (median) | **122** | normal motion |
| `0→1` | **123** | ≈ median — **not** a second wall |
| `20→0` (`F-1→0`) | **236** | **~1.93× median** — still the loop seam |

So: **video is not seamless**; only the *shape* of the latent gap profile changed vs TI2V-5B.

## What “passed”

Question asked: does native A14B I2V (independent `y` cond, **no** Mode-B RoPE, **no** cond surgery) still show TI2V-5B-style **index-0 double spike** (both `F-1→0` **and** `0→1` anomalous)?

**Answer: No.** Only `F-1→0` is elevated; `0→1` tracks median.

That supports **H1/H2** (hard frame-0 / fused conditioning created an *extra* boundary on TI2V-5B). It does **not** mean A14B closed the loop.

## Implication for next work

- Stop TI2V-5B latent conditioning surgery (already closed).
- A14B = architecture control / reference, not “seamless achieved”.
- Primary next: **HunyuanVideo-1.5 native H0** (no Circular RoPE yet). Visual/latent seam must still be fixed there (+ later Symmetric Circular Temporal RoPE if H0 is clean enough).

## Artifacts

- Numbers: `artifacts/wan_a14b_i2v_probe/RESULT.md`, `latent/step39_late_full_ring_gaps.json`
- Video (local, not necessarily in git): `artifacts/wan_a14b_i2v_probe/out_x3.mp4`
- Cloud: `/root/latent-loop/artifacts/wan_a14b_i2v_probe/`
