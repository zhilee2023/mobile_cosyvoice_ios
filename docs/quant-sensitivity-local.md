# CosyVoice3 本地分组件量化敏感性（2026-10-16）

状态：只完成 Windows / MNN 3.6.1 CPU 正确性筛查；**不是 iPhone Metal 的内存、速度或听感验收**。

## 实验固定条件

- 模型：公开的 Fun-CosyVoice3-0.5B-2512 蒸馏两步 Flow MNN 包，固定同一个提示音色。
- 三组真实 LLM 生成的 speech token：中文短句 74 个（2.96 秒），中文长句 123 个（4.92 秒），英文短句 74 个（2.96 秒）。
- 每次只替换一个组件，其余组件使用公开基线：LLM 原包混合 W4/W8、Conditioner FP32、Flow FP16、HiFT-F0 FP32、HiFT-core FP32。
- 对内部 FP32 MNN 图分别使用 `MNNConvert -f MNN --weightQuantBits 8 --weightQuantBlock 128` 和 W4 `--hqq`；另测 HiFT-core W4 `block=32`。
- 指标：中间张量相对 L2、最终 PCM 逐样本相关，以及 FFT 1024 的对数幅度谱相关。后者对激励相位变化较稳健，但仍不是主观音质分数。

| 组件 | FP32/FP16 原包 | W8 | W4 HQQ | 本地敏感性与暂定配方 |
| --- | ---: | ---: | ---: | --- |
| LLM Transformer / head / embedding | 已是 Transformer W4、head W8、embedding BF16 | 未重导出 | 未重导出 | 保持原包；没有同权重的可比 FP32/W8 源图，不能宣称另一配方更好。 |
| Conditioner | 4.40 MB | 2.70 MB | 2.41 MB | W8 的 mu 相对 L2 约 1.1%，三句最终对数谱相关 0.983–0.987；W4 mu 相对 L2 约 16–17%，相关 0.967–0.978。只省约 2 MB，暂保留 FP32。 |
| 蒸馏两步 Flow | 664.59 MB（图+外置 FP16 权重） | **无有效模型** | **无有效模型** | 当前 `MNN→MNN` 转换生成的 `.weight` 为空，不能用其 NaN 输出评价 W8/W4。保留 FP16；需要蒸馏学生的原始 FP32/ONNX 权重再导出。 |
| HiFT F0 预测 | 13.27 MB | 3.41 MB | 1.76 MB | W8 F0 相关 0.991–0.999；W4 仅 0.822–0.895。F0 小偏差会累积成相位差；暂保留 FP32，W8 仅作候选。 |
| HiFT core | 70.22 MB | 18.11 MB | 9.37 MB | W8 三句最终对数谱相关 0.998 左右；W4 只有 0.873–0.910。W4 block32 提升到 0.941–0.952，仍不达稳健替换要求。优先保留 FP32，W8 可在盲听及手机内存测量后考虑。 |
| 激励、STFT/iSTFT | 非神经网络 FP32 运算 | 不适用 | 不适用 | 保持 FP32，尤其相位积分与逆变换。 |

HiFT-core 的云端 CPU 复验使用同样的 W8/W4 配方，2.96 秒样本的原始 PCM 相关分别约 0.989 与 0.162；与本地趋势一致。W8 并未带来清晰的 CPU 核心执行时间收益。MNN 权重量化缩小的是磁盘表示，**不能据此推断手机 Metal 会话峰值内存也按相同比例下降**。

本地完整数值见 `builds/local-sensitivity/report.json` 与 `builds/local-sensitivity/audio-metrics.json`；24 个 WAV 均在同目录。量化模型只放在忽略的 `builds/` 中，不进入 iOS App 或发布权重。

## 下一步门槛

1. 先人工听三组原版 / W8 / W4 对照，尤其听齿音、音高和尾音。
2. 如 W8 HiFT-core 听感可接受，再在 iPhone 单独测该组件的 CPU/Metal 峰值内存与端到端 RTF；不要同时改多个组件。
3. Flow 先做 256 帧手机内存探针；要量化 Flow，必须取得与现有蒸馏学生相同权重的 FP32/ONNX 导出，建立非空权重、finite、Mel/WAV 一致性门槛后再上设备。
