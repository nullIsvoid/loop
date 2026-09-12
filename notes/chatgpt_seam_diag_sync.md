# ChatGPT sync — H1 first attempt FAILED (8058b61+)

## H0 baseline (pass architecture read)

| | H0 |
|--|---:|
| median | 54.7 |
| `0→1` | 87 (1.59×) |
| `F-1→0` | 130 (2.38×) |

## H1 = same inputs + Symmetric Circular Temporal RoPE

480p_i2v: **54 double blocks, 0 single**; schedule `0,+1,-1,...` via temporal freqs roll pre-hooks.

| | H1 | vs H0 |
|--|---:|------|
| median | 96.5 | ↑ |
| `0→1` | **302.6** (3.14×) | **much worse** |
| `F-1→0` | **309.1** (3.20×) | **worse** |

Shape: **near dual spike** (`20→0` ≈ `0→1`).

## Verdict

H1 v1 **fails** the pre-agreed success bar (`F-1→0`↓ and `0→1` not worse).

## Ask

1. Is the failure more likely **schedule over-shift** (54 layers × F=21), **wrong roll axis/order** for Hunyuan ND RoPE, or **Mode B fighting I2V index-0 cond**?
2. Smallest next experiment: Identity schedule sanity, FixedShift(1) only, or block-subset (e.g. last N blocks)?
3. Should W5-0/WA-0 natives on person_loop_v1 run before more Hunyuan RoPE variants?

Video: `artifacts/loop_benchmark_v1/hunyuan15_symmetric/out_x3.mp4`  
Detail: `RESULT.md`
