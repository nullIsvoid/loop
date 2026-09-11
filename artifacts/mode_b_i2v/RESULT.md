# Mode B I2V — ready for human / ChatGPT review

## Question

Can Circular Temporal RoPE (Symmetric) keep an existing wallpaper’s identity /
composition while adding gentle motion and a loopable last→first seam?

## Setup

- Model: Wan2.2 TI2V-5B **I2V** (`img=PIL.Image`)
- A = baseline · B = Symmetric
- seed=42, 81 frames, 20 steps, CFG=5, UniPC, max_area=704×1280
- Cloud: `ALL_I2V_OK`

## Cases

| case | source | output size | watch |
|------|--------|-------------|-------|
| `person/` | product live-wallpaper still | 800×1088 | hair/cloth sway + seam + identity |
| `environment/` | Wan official `i2v_input` (cat/coast water) | 800×1088 | water/ambient + seam + identity |

## Score sheet

| case | A seam | B seam | identity vs source | winner |
|------|--------|--------|--------------------|--------|
| person | | | | |
| environment | | | | |

Play `A_x3.mp4` / `B_x3.mp4` next to `source.png`. Full single-loop mp4s stay on cloud.
