# Mode B I2V — human verdict

## Question

Can Circular Temporal RoPE (Symmetric) keep an existing wallpaper’s identity /
composition while adding gentle motion and a loopable last→first seam?

## Setup

- Model: Wan2.2 TI2V-5B **I2V** (`img=PIL.Image`)
- A = baseline · B = Symmetric
- seed=42, 81 frames, 20 steps, CFG=5, UniPC, max_area=704×1280
- Cloud: `ALL_I2V_OK`

## Human verdict (2026-09-12)

Seam / jitter focus:

| case | A (baseline) | B (Symmetric) | winner |
|------|--------------|---------------|--------|
| person | **more obvious jitter** | slight jitter | **B** |
| environment | **more obvious jitter** (same pattern) | slight jitter | **B** |

**Takeaway:** On wallpaper I2V, Circular Temporal RoPE + Symmetric improves the last→first seam vs baseline for both person and environment. Residual slight jitter on B remains; identity/composition notes not separately scored in this pass.

## Check 2 — ChatGPT independent visual scores

**Pending.** Fill `notes/chatgpt_i2v_review_request.md` after watching person/environment `A_x3` vs `B_x3` (and `source.png`). Do not rubber-stamp Check 1.

## Score sheet (human filled)

| case | A seam | B seam | identity vs source | winner |
|------|--------|--------|--------------------|--------|
| person | clear jitter | slight | (not ranked) | **B** |
| environment | clear jitter | slight | (not ranked) | **B** |
