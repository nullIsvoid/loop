# ChatGPT sync — H1 approved after H0 (commit pending)

Repo tip follows this note on `main`.

## Refined conditioning conclusion (locked)

```text
TI2V-5B:  destructive frame0 write     → severe artificial 0→1 wall
A14B:     independent y cond           → 0→1 ≈ median (arch probe; not v1 yet)
Hunyuan:  non-destructive cond+vision  → mild 0→1 bump (1.59×); dominant F-1→0 (2.38×)
```

**More accurate rule:** non-destructive conditioning avoids TI2V-style severe dual walls, but index-0 reference asymmetry can still leave a **weak** local `0→1` effect. Do not claim “non-destructive ⇒ 0→1 always flat.”

## H0 baseline (person_loop_v1, strict)

| edge | L2 | vs med |
|------|---:|-------:|
| median | 54.7 | 1.00× |
| `0→1` | 87.0 | 1.59× (rank 2) |
| `F-1→0` | 130.1 | 2.38× (**max**) |

## H1 authorization

**Approved.** Do **not** wait for W5-0/WA-0.

```text
H1 = H0 inputs + Symmetric Circular Temporal RoPE only
     schedule 0,+1,-1,+2,-2,... on Hunyuan ND RoPE (img Q/K)
```

Success vs H0:

1. `F-1→0` clearly closer to ordinary adjacent  
2. `0→1` **not worse**  
3. no new ring walls  
4. `out_x3`: jump / stall / reverse / speed-pop only  

Call chain: `notes/hunyuan_h1_rope_call_chain.md`  
Runner: `scripts/benchmark/run_hunyuan15_symmetric.py`

## Still pending (not blocking H1)

- W5-0 / WA-0 on person_loop_v1 (three-way native table later)
