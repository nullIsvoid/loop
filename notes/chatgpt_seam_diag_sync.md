# ChatGPT sync — D1b failed; D2 justified

## Experiment

B / D1 / D1b × person + environment (seed=42, 81f, 20 steps, Symmetric RoPE). Frame 0 only.

D1b: same `a` as D1; `latent0 = a·ref+(1-a)·gen`; `t0 = (1-a)·current_t`.

## Late (step 19) — person

| | F-1→0 | 0→1 | seam_vs_adj |
|--|------:|----:|------------:|
| B | 213 | 169 | 1.77 |
| D1 | 175 | 192 | 1.37 |
| D1b | 192 | 198 | 1.37 |

## Late — environment

| | F-1→0 | 0→1 | seam_vs_adj |
|--|------:|----:|------------:|
| B | 258 | 218 | 1.56 |
| D1 | 247 | 247 | 1.42 |
| D1b | 271 | 260 | 1.49 |

## Conclusion

Coupling timestep release with latent release **does not** remove the conditioning boundary. Single-point soft conditioning (latent-only or coupled) is insufficient.

**Ask:** Green-light **D2 = Circular Soft Conditioning** (soft weights on ring neighbors of index 0), or stop / different hypothesis?

Detail: `artifacts/mode_b_i2v_d1b/RESULT.md`
