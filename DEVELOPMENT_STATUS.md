# Development status

> **D3 in flight:** Circular Reference Residual Conditioning (full-ring cosine).  
> Not D2 radius widen. Absolute ref copy excluded.

## Settled exclusions

| knife | failure |
|-------|---------|
| B / D1 / D1b | boundary at index 0 |
| D2 absolute ref soft window | boundary at window edge |

## D3 = Circular Reference Residual

```
delta = ref0 - g0
x_i = g_i + w_i * delta
w(d) = 0.5 * (1 + cos(pi * d / Dmax))   # full ring, Dmax=F//2
```

`w_0=1` → `x_0=ref0`. Timestep: **t0=0 only**. Probe: full F adjacent gaps + step19 full latent.

Script: `scripts/cloud_mode_b_i2v_d3_compare.py` → `artifacts/mode_b_i2v_d3/`.
