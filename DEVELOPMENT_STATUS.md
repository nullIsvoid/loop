# Development status

> **I2V gate: B beats A on seam** (person + environment).  
> Wan Mode B default remains `SymmetricShiftSchedule`. Residual slight B jitter noted.

## Mode B I2V — human 2026-09-12

Same TI2V-5B with `img=PIL.Image`. A = baseline, B = Symmetric.

| case | A | B | winner |
|--|--|--|--|
| person | more obvious jitter | slight jitter | **B** |
| environment | more obvious jitter | slight jitter | **B** |

→ Circular Temporal RoPE helps wallpaper I2V loop seam, not only T2V.  
Open: further reduce B’s slight residual jitter (not by inventing S4 yet).  
**ChatGPT independent I2V scores:** pending — `notes/chatgpt_i2v_review_request.md`.

## Freeze hygiene (done on `cf3f42f`)

Padded RoPE `seq_len`, Symmetric default, generate `--schedule`, README mainline — closed.

## Prior T2V evidence (closed)

Symmetric default after phase + cross-seed ties. See `artifacts/mode_b_*`.

## Still forbidden

- No new schedules for curiosity  
- No seam_metrics → Gate/Repair  
- No Comfy / residual / true periodic RoPE until scoped  
