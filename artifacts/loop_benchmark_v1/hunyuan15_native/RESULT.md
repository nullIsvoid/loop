# H0 — RESULT (HunyuanVideo-1.5 native on person_loop_v1)

## Question

On locked `person_loop_v1`, what late temporal seam structure does **native** Hunyuan I2V show
(`F-1→0` vs `0→1` vs median)? No Circular RoPE. No conditioning edits. Prompt rewrite OFF.

## Setup

| knob | value |
|------|-------|
| source | `assets/loop_benchmark/person.png` (sha256 pinned) |
| prompt | `person_prompt.txt` |
| seed | 42 |
| frames | 81 |
| resolution | 480p, aspect 3:4 → 528×768 |
| steps | 50 |
| sr / rewrite | off |
| ckpt tree | `/workspace/hunyuan-ckpts/HunyuanVideo-1.5` (writable; DiT/VAE linked from read-only `/model`) |

## Late (step 49)

| metric | value |
|--------|------:|
| median | **54.7** |
| mean | 59.9 |
| `0→1` | **87.0** (~1.59× median) — rank **2** / 21 |
| `F-1→0` (`20→0`) | **130.1** (~2.38× median) — **max** |
| max/median | **2.38** |

Top edges: `20→0` ≫ `0→1` ≫ ordinary (~55–64).

## Verdict

**Not a TI2V-5B-style equal double spike**, and **not** as flat on `0→1` as native A14B.

- Dominant discontinuity = real loop seam `F-1→0`.
- `0→1` is mildly elevated (2nd highest), consistent with Hunyuan’s non-destructive cond channel still marking temporal index 0 — but much weaker than the ring seam, and not a second wall of similar height.

So H0 still supports moving to **H1 = Symmetric Circular Temporal RoPE** to attack the real `F-1→0` seam, without conditioning surgery.

## Visual

Check `out_x3.mp4` for last→first jump / stall / direction flip only (not identity/quality).

## Classification

Strict **person_loop_v1** benchmark run (not OLD RESEARCH EVIDENCE).
