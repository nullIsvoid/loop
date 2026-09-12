# ChatGPT sync — A14B Phase A: double spike gone

## Result (native A14B I2V, no RoPE, no cond edits)

Late full-ring:

| | F-1→0 | 0→1 | median | max/med |
|--|------:|----:|-------:|--------:|
| A14B | 236 | **123** | 122 | 1.93 |

**Only the loop seam `F-1→0` is high; `0→1` is normal.**  
Unlike TI2V-5B hard clamp (both sides spiked).

## Implication

Independent conditioning channel removes the artificial 0→1 wall. Supports switching primary development to **HunyuanVideo-1.5** (native H0 probe next; Circular RoPE only after).

Detail: `artifacts/wan_a14b_i2v_probe/RESULT.md` (`ece002b` + this results commit).
