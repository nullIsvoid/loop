# Development status

> **I2V phase started.** Wan Mode B default remains `SymmetricShiftSchedule`.  
> T2V schedule research is closed; next gate is **wallpaper I2V** A/B seam + identity.

## Freeze hygiene (done on `cf3f42f`)

Padded RoPE `seq_len`, Symmetric default, generate `--schedule`, README mainline — closed.

## Current gate — Mode B I2V

Script: `scripts/cloud_mode_b_i2v.py`  
Same TI2V-5B with `img=PIL.Image` (official I2V path).

- A = baseline Wan I2V  
- B = Wan I2V + Symmetric Circular Temporal RoPE  
- Cases: `person` (hair/cloth sway), `environment` (water/ambient)  
- Artifacts: `artifacts/mode_b_i2v/<case>/`

## Prior T2V evidence (closed)

Symmetric default after phase + cross-seed ties. See history under `artifacts/mode_b_*`.

## Still forbidden

- No new schedules for curiosity  
- No seam_metrics → Gate/Repair  
- No Comfy / residual / true periodic RoPE until scoped  
