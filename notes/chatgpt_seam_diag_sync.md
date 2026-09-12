# ChatGPT sync — H0 native Hunyuan on person_loop_v1

## Benchmark

Locked `person_loop_v1` (original Wan person.png, fixed prompt, seed=42, 81 frames).

## H0 late latent full-ring (F_latent=21)

| | value |
|--|------:|
| median | 54.7 |
| `0→1` | **87.0** (~1.59×) — rank 2 |
| `F-1→0` | **130.1** (~2.38×) — **max** |

## Read

- **Not** TI2V-5B equal double spike.
- **Not** as clean as A14B’s `0→1≈median` either — Hunyuan shows a **mild** `0→1` bump.
- Dominant problem remains the real ring seam `F-1→0`.

Supports next step **H1 = Symmetric Circular Temporal RoPE** (no conditioning surgery).  
W5-0 / WA-0 on the same benchmark still pending for a three-way table.

## Video

`artifacts/loop_benchmark_v1/hunyuan15_native/out_x3.mp4`

Detail: `artifacts/loop_benchmark_v1/hunyuan15_native/RESULT.md`
