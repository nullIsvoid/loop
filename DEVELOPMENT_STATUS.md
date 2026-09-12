# Development status

> **D3 done:** Circular Reference Residual **worsened** the seam (~2.5–3×).  
> Absolute copy (D2) and residual (D3) both fail as post-step latent rewrites.

## Exclusion list

| knife | outcome |
|-------|---------|
| B / D1 / D1b single-point | boundary at 0 |
| D2 absolute soft window | boundary moves to window edge |
| **D3 full-ring residual** | same seam location, **larger** gap |

## Still held

- Symmetric Circular Temporal RoPE mainline  
- No S4/S5 / metric gate  

Next: change mechanism class (likely attention/context), not another `w(d)` on latent post-step.
