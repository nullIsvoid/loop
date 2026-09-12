# 无缝 I2V 单变量实验协议

> 目标：每次实验只回答一个问题，且只改变一个独立变量。任何同时改变模型权重、条件通道和环形策略的比较，都不得作因果结论。

## 1. 每个实验对必须记录

```json
{
  "pair_id": "HY-I2V-COND-01",
  "question": "VAE condition latent 是否在双循环中制造第 0 位双尖峰？",
  "independent_variable": {
    "name": "vae_condition_enabled",
    "control": false,
    "treatment": true
  },
  "frozen": {},
  "outputs": {}
}
```

必填字段：

- `pair_id`：一对对照共用的唯一标识。
- `question`：这一对只允许回答的问题。
- `independent_variable`：唯一允许变化的变量及两个值。
- `frozen`：必须一致的所有配置。
- `outputs`：两个 arm 的指标和视频地址。

## 2. 强制冻结项

每对实验必须相同：

- 模型 family、variant 和 checkpoint 文件集。
- source SHA256（需要参考图时）、prompt 文本及 SHA256。
- seed 与初始 noise tensor SHA256；不能只相信相同 seed。
- 分辨率、宽高比、帧数、latent 形状和 dtype。
- sampler / scheduler、timesteps、steps、CFG、flow shift。
- attention backend、offload / cache / compile 开关。
- 循环 schedule 类型、完整 Block shift 表和 active Block 列表。
- 代码 Git commit 与云端脚本 SHA256。

若上述任意一项在两个 arm 之间不同，除非它就是声明的唯一独立变量，否则整对实验无效。

## 3. Hunyuan 参考条件定位顺序

全部使用同一份 `480p_i2v` 权重。不再用 `480p_t2v` 与 `480p_i2v` 作为参考条件因果对照。

### Pair A：双循环本身

| | Control | Treatment |
|---|---|---|
| 唯一变量 | Circular RoPE 关 | Circular RoPE 开 |
| VAE condition | 全 0，形状不变 | 全 0，形状不变 |
| vision states | 关 | 关 |

回答：在同一 I2V 权重上，当没有参考信息时，双循环自身是否制造首尾问题？

### Pair B：VAE condition latent

| | Control | Treatment |
|---|---|---|
| 唯一变量 | VAE condition 全 0 | 官方位置 0 VAE condition + mask |
| Circular RoPE | 开 | 开 |
| vision states | 关 | 关 |

回答：位置 0 的 VAE 条件是否制造或放大双尖峰？

### Pair C：vision states

| | Control | Treatment |
|---|---|---|
| 唯一变量 | vision states 关 | 官方 vision states 开 |
| Circular RoPE | 开 | 开 |
| VAE condition | 全 0 | 全 0 |

回答：无时间位置的视觉语义条件是否影响环形闭合？

### Pair D：两条参考路径的交互

这不是一次同时改两个变量：

- D1：vision-only → VAE + vision，唯一打开 VAE。
- D2：VAE-only → VAE + vision，唯一打开 vision。

两个方向都显示只有联合时恶化，才可以报告交互风险。

## 4. 输出与结论约束

每个 arm 必须产生：

- `run.json`：完整 contract、冻结项和实际运行值。
- `first.png` / `last.png`。
- `out.mp4` / `out_x3.mp4`；报告时必须给出本地可点击地址。
- latent early / middle / late 的 `F-1→0`、`0→1`和相邻中位数。
- decoded-frame 全 81 帧相邻差异、`F-1→0`、`0→1`和 seam/median。

结论只能写：

```text
在冻结配置 X 下，将唯一变量 V 从 A 改为 B，观测到指标/视觉现象如何变化。
```

不得由单个 latent 指标宣称视频已无缝，不得由不同 checkpoint 的比较宣称某条件通道是唯一原因。

## 5. 现有结果的证据等级

- H0 I2V → H1 I2V：同一 I2V 权重，可用于评估全层双循环与完整参考条件共存时的变化；但无法单独归因于双循环或参考条件。
- H1 I2V → H1-T2V：同时改变 checkpoint 和参考条件，只作跨任务辅助证据，不作因果结论。
- H2-B2：只证明对 Block 2 做半周期扰动时的局部响应，不证明 Block 2 是 54 层全局锚点，也不证明正式分组 schedule 有效。

## 6. 执行顺序

1. 先实现同一 `480p_i2v` checkpoint 下对 VAE condition 和 vision states 的独立开关，保持形状与推理路径不变。
2. 先跑 Pair A，确定双循环在无参考信息的 I2V 权重上是否独立失败。
3. 再跑 Pair B / C，分别定位 VAE 和 vision 条件。
4. 最后做 D1 / D2 交互对照。
5. 条件因果未分离前，不改 Block schedule，不加 step 门控，不换模型。
