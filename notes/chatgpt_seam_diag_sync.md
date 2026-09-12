# ChatGPT sync — D3 Circular Reference Residual running

## Prior

D2 (`343b64f`) moved the seam to the soft-window edge. No radius widen.

## Now: D3

```
delta = ref0 - generated_0
x_i = generated_i + w_i * delta
```

Full-ring cosine `w(d)=0.5*(1+cos(pi*d/Dmax))`. `t0=0` only (no soft timestep). Compare **B vs D3**. Full-ring 21-edge gaps + `full_latent_step19.pt`.

Artifacts: `artifacts/mode_b_i2v_d3/` (in flight).
