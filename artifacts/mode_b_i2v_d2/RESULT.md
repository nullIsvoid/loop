# D2 Circular Soft Conditioning — result

## Question

Does radius-2 ring soft reference conditioning (coupled latent + timestep, **fixed every step**) flatten adjacent gaps on **F-4..4**, or only move the discontinuity to the edge of the conditioning window?

## Setup

- Symmetric Circular Temporal RoPE unchanged
- B = hard single-point (index 0 only, t0=0)
- D2 = `d∈{0,1,2}` → w=`1.00/0.75/0.25`, `x_i=w·ref+(1-w)·gen`, `t_i=(1-w)·t`
- Same weights all 20 steps (no late release)
- person / environment, seed=42, 81 frames → latent F=21
- Probe half_window=4

## Late adjacent L2 (step 19)

### person

| edge | B | D2 |
|------|--:|---:|
| F-4→F-3 | 100 | **159** ↑ |
| F-3→F-2 | 103 | **139** ↑ |
| F-2→F-1 | 104 | 111 |
| F-1→0 | **213** | 29 ↓ |
| 0→1 | **169** | 19 ↓ |
| 1→2 | 105 | 91 |
| 2→3 | 88 | **163** ↑ |
| 3→4 | 83 | **144** ↑ |
| max edge | F-1→0 | **2→3** |
| max_vs_median | 2.05 | 1.30 |
| seam_vs_adj | 2.19 | 0.22 (misleading — old seam only) |

### environment

| edge | B | D2 |
|------|--:|---:|
| F-4→F-3 | 149 | **226** ↑ |
| F-3→F-2 | 142 | **173** ↑ |
| F-2→F-1 | 157 | 138 |
| F-1→0 | **258** | 23 ↓ |
| 0→1 | **218** | 21 ↓ |
| 1→2 | 142 | 112 |
| 2→3 | 129 | **156** ↑ |
| 3→4 | 133 | **229** ↑ |
| max edge | F-1→0 | **3→4** |
| max_vs_median | 1.77 | 1.56 |
| seam_vs_adj | 1.81 | 0.13 (misleading) |

## Verdict

**D2 removes the old F-1|0|1 spike by pinning neighbors to the reference — and recreates a new boundary at the soft-window edge (±2 / ±3).**

Shape is exactly the “seam moved” failure:

```
B:   ... ~100  ~100  213  169  ~100 ...
D2:  ... ~150  ~120   25   20  ~160 ...
              ↑ flatten     ↑ new wall
```

So: laying the **same reference latent** onto a finite ring neighborhood is still **piecewise conditioning**. Widening radius (2→3→4→5) is the wrong next knob — it would likely relocate the wall again.

## Visual note

Latent already predicts a near-static patch around index 0 (very small F-1→0 / 0→1). Prefer watching `out_x3.mp4` for pause/stall; do not treat `seam_vs_adj≈0.2` as success.

## Next (hypothesis change, not radius tune)

Need a different idea than “copy ref0 into neighbors,” e.g. softer / generated-relative blending, Circular Temporal Context, or another mechanism that does not create a hard free/conditioned interface.
