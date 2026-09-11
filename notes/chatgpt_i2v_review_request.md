# ChatGPT I2V independent review

Please watch the I2V artifacts yourself. Do **not** copy the human verdict blindly.

Human already reported (2026-09-12): both cases A has clearer jitter, B only slight → B wins. Confirm or disagree.

## Artifacts (`artifacts/mode_b_i2v/`)

### person/

- `source.png` — input wallpaper
- `A_x3.mp4` — baseline Wan I2V (triple play)
- `B_x3.mp4` — Symmetric Circular Temporal RoPE

Focus: last→first seam **and** identity/composition vs source (hair/cloth motion only).

### environment/

- `source.png` — Wan official I2V still (cat + coast water)
- `A_x3.mp4` / `B_x3.mp4`

Focus: seam + identity; water/ambient motion only.

## Shared settings

TI2V-5B I2V path (`img` set), seed=42, 81 frames, 20 steps, CFG=5, UniPC, max_area=704×1280.  
B schedule = `SymmetricShiftSchedule`. Metrics stay auxiliary — visual judgment is primary.

## Please fill

```text
ChatGPT I2V scores (date):

person:
  A seam (0=broken,1=usable,2=smooth): ?
  B seam: ?
  identity vs source (A / B / tie / unclear): ?
  winner: A | B | tie

environment:
  A seam: ?
  B seam: ?
  identity vs source: ?
  winner: A | B | tie

Agree with human (B better seam, slight residual B jitter)? yes / no / partial
Notes:
```

Reply by editing this file or writing `notes/chatgpt_i2v_review_response.md`, then update `DEVELOPMENT_STATUS.md` / `artifacts/mode_b_i2v/RESULT.md`.
