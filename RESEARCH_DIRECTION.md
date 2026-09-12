# Seamless I2V 研究总纲

> 本文是项目研究目标、借鉴来源和模型角色的权威入口。旧实验文档与本文冲突时，以本文为准。

每次实验的变量、冻结项和可允许结论，必须遵循 [`notes/controlled_experiment_protocol.md`](notes/controlled_experiment_protocol.md)。

## 研究目标

把非破坏式图像条件与环形时间建模结合，得到原生无缝循环的图生视频（I2V）：

```text
参考图 ──→ 独立条件通道 ──→ Transformer
                                  ↑
生成视频 latent ──→ 自由去噪 + 环形时间 ──┘
```

参考图可以在 latent 空间提供条件，但不得在每个去噪步覆盖 generated latent（生成潜变量）。项目同时诊断参考条件在时间索引 0 产生的隐性边界，重点观测 `F-1→0` 与 `0→1` 双尖峰。

## 借鉴来源与项目角色

| 要解决的问题 | 主要来源 | 原方案 | 本项目借鉴 | 不照搬 |
|---|---|---|---|---|
| 参考图进入 I2V，不直接破坏生成 latent | Wan2.2-I2V-A14B | VAE 图像条件与 mask 作为额外 `y` 通道，主 latent 正常去噪 | **首选 I2V 底模**；保持参考条件和生成 latent 分离 | Wan TI2V-5B 每步硬写 `latent[0]` |
| 验证另一种非破坏式条件和跨模型环化 | HunyuanVideo-1.5 I2V | 时间位置 0 的 condition latent + mask，再加 vision states；主 latent 独立 | **第二底模 + 跨模型验证器**；验证 Loopy 层级时间控制能否泛化 | 不把 Wan 的 shift 表直接套到 54 个 Hunyuan Block |
| 参考图只引导、不替换生成 latent 的明确设计 | LTX-2 Guiding Latents | 区分 Replacing Latents 和 Guiding Latents | 作为非破坏式参考条件的理论与工程依据 | 不优先走普通首帧 Replacing Latents |
| 线性时间改造为环形时间 | Loopy | 测量逐层时间控制强度，确定锚点层，其余层按分组位移时间 RoPE | **主循环机制**：`alpha_l` → anchor → 分组 shift → 环形时间认知 | 不再把早期 `0,+1,-1,+2,-2...` 当成 Loopy 正式算法 |
| 定位锚点层 | Loopy 论文的 `alpha_l` 测量方法 | 单独扰动第 `l` 层约半周期时间位置，对同一 `x_t` 比较去噪预测差异 | 自行工程化重建该测量，用于 Hunyuan，后续用于 A14B | Loopy 未公开自动定位脚本，不声称复制了该代码 |
| 不依赖逐层 RoPE 的环形备用方法 | Mobius | 去噪过程中持续旋转 latent 时间顺序 | **B 路线/备用方案**，用于评估更模型无关的 latent cycle | Mobius 主要是 T2V，不能单独解决 I2V 参考条件 |

## 固定主路线

> **Wan2.2-I2V-A14B 的非破坏式图像条件 + Loopy 的锚点式环形时间位置机制。**

HunyuanVideo-1.5 是第二底模和跨模型验证器；LTX-2 提供 Guiding Latents 的参考条件设计依据；Mobius 是不依赖逐层 RoPE 的备用环形路线。

## Loopy 复现边界

Loopy 的两个阶段必须分开：

1. **分析阶段**：对每层计算 `alpha_l = mean_n MSE(v'_n, v_n)`，`argmax(alpha_l)` 是当前模型的锚点层。
2. **推理阶段**：锚点层 `shift=0`，其余层使用与当前模型层级敏感性匹配的分组 shift 表。

本地 Loopy A14B 参考实现已核对：分布式 40 层路径将 Block 0 作为锚点，Block 2–32 每两层共享一组 raw shift，Block 33–39 每三层共享一组。项目当前的 `LoopyShiftSchedule` 是普通非分布式 `model_roll.py` 的简化逐层实现，**不是 A14B 的完整分组策略**。

Hunyuan 的 `run_hunyuan15_anchor_probe.py` 已复现分析阶段，并且用缓存的同一份原生 Transformer 输入比较扰动前后预测，避免采样轨迹分叉污染 `alpha_l`。

## 验收观测

- 参考图不在每个去噪步覆盖主 latent。
- 对同一 source / prompt / seed / frame count 比较原生和环形版本。
- 同时报告 `F-1→0`、`0→1`、中位相邻帧差异和视觉连续性；不用单一自动分数宣称成功。
- 双尖峰是诊断现象，不预设它只由参考条件或只由 RoPE 引起。
