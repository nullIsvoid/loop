# Development status

> **D1b done:** Coupled Anchor Release failed to flatten late double spike.  
> Next candidate (await green-light): **D2 Circular Soft Conditioning**. Symmetric RoPE unchanged.

## Causal chain (agreed)

T2V + Symmetric RoPE → latent seam flat.  
I2V hard clamp of frame 0 → late F-1→0 **and** 0→1 spike.

| knife | change | late outcome |
|-------|--------|--------------|
| B | hard latent + hard t0=0 | double spike |
| D1 | soft latent + hard t0 | F-1→0 ↓, 0→1 ↑ |
| D1b | soft latent + soft t0=(1-a)t | still double spike; env worse |

## Still held

- No Circular Temporal Context as first knife  
- No S4/S5  

D2 is now justified if product wants to continue: soft reference conditioning on the ring around index 0, not further single-point D1.x.
