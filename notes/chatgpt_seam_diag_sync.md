# ChatGPT sync — D1 Late Anchor Release running

Agreed: next track is Circular I2V Conditioning; **first knife = D1 only**.

## Implementing now

- `HardAnchorSchedule` (B) vs `LateAnchorReleaseSchedule` (D1)
- Same Symmetric RoPE, same person/environment I2V cases, probes 2/10/19
- Artifacts: `artifacts/mode_b_i2v_d1/`

Primary question: does late **F-1→0 / 0→1** double spike disappear under late release?

No F-1 conditioning. No circular context. No new RoPE schedule. D2 only if D1 confirms.
