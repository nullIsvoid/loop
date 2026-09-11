# Mode B I2V — first product gate

## Question

Can Circular Temporal RoPE (Symmetric) keep an existing wallpaper’s identity /
composition while adding gentle motion and a loopable last→first seam?

## Setup

- Model: Wan2.2 TI2V-5B **I2V** path (`img=PIL.Image`, not `img=None`)
- A = baseline Wan I2V
- B = Wan I2V + `SymmetricShiftSchedule`
- Shared: seed=42, 81 frames, 20 steps, CFG=5, UniPC, max_area=704×1280

## Cases

| case | source | motion ask |
|------|--------|------------|
| `person` | product live-wallpaper still | hair / clothing hem slight sway |
| `environment` | official Wan `i2v_input.JPG` (water/coast) | water / light / ambient motion |

## Watch

For each case: `source.png` vs `A_x3.mp4` vs `B_x3.mp4` — seam **and** identity.
