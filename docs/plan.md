# CosyVoice3-MNN 验证顺序

1. **Windows PC 先行（当前）**：下载公开的 `cosyvoice3-mnn-mobile-fp16-complete.zip`，核对 ZIP SHA-256；只提取清单文件并逐项核对大小/SHA-256。测试 MNN 3.6.1 对 Conditioner、LLM、Flow、HiFT 的加载和前向；生成 WAV 并试听后，才宣布 PC 全链路通过。
2. **PC 上验证真分块**：LLM 按 25 speech-token 左右出块；Flow/HiFT 按官方缓存、lookahead 和交叠逻辑逐块运行。验收首包延迟、块间停顿、波形连续性和各阶段峰值内存。简单把整句 WAV 切开播放，不算流式。
3. **iPhone 移植**：PC 的分块结果与全句基线一致后，再把相同 MNN 图、缓存与调度搬到 iOS。iPhone 15 上分别验证冷启动、首包、RTF、内存峰值、后台/锁屏和多轮生成。此阶段不能用 Android 性能代替真机数据。

项目独立于 `C:\ace-step-mnn-ios`。ACE 项目不移动、不改名、不删除；本项目只复用它已有的 Windows MNN Python 环境作为测试工具。

## 已做的 PC 单组件测试

`flow-conditioner.fp32.mnn` 与 `prompt-speech-tokens.csv` 已从公开 ZIP 的 HTTP Range 单独取得并通过清单 SHA-256。Windows MNN 3.6.1 CPU High 下，87 个提示 speech token 分别加上 12、25、50 个**合成测试 token**，动态输入长度为 99、112、137；输出长度分别为 15,840、17,920、21,920 个 float，均为有限值。两次前向只有在每次重新传入 token 输入后才逐值一致；这是后续分块循环必须保留的操作。

这些合成 token **不是** LLM 根据文字生成的 speech token，因此此测试不能证明语音内容、音质、完整流式合成或 iPhone 可运行。记录见 `reports/pc-conditioner-*.json`。

同一会话中按 12→25→50→25 个 target token 改变输入长度也通过；返回 25 时输出与此前逐值一致。记录见 `reports/pc-conditioner-dynamic.json`。这只说明 Conditioner 图支持当前测试范围内的动态长度，不代表 Flow/HiFT 已可逐块解码。

Flow 权重 663,253,312 B 已从断点 ZIP 提取且与清单 SHA-256 一致。在**暂用已记录哈希差异的 Flow 图**的前提下，Windows CPU 用 25 个合成目标 token 做了 Conditioner→Flow 两步前向，输出 80×50 Mel，均为有限值；Flow 前向约 19.09 秒。报告 `reports/pc-flow-synthetic.json`。这是图功能烟雾测试，并非 LLM 文字转语音，也不表示流式或 iPhone 性能。

HiFT 两个图经清单 SHA-256 验证后，使用 87 个合成目标 token 跑通 Conditioner→Flow→HiFT，写出 3.48 秒 PCM16 WAV：`reports/synthetic-3.48s.wav`。该次 Windows CPU 测得 Flow 前向 6.39 秒、HiFT core 0.70 秒、总耗时 8.79 秒，PCM 峰值 0.746、RMS 0.092，均为有限值。这不是文字朗读或听感验收，亦不代表 iPhone 的 RTF。报告 `reports/pc-synthetic-wav.json`。

完整 ZIP（1,399,083,563 B）已通过发布清单 SHA-256。ZIP 内 `flow.cfg-student-2step.batch1.fp16.mnn` 的实际 SHA-256 为 `B74ECE0A1A2E165EEF53DBC8D628310A1B464323D4729959069744BAECF418D7`，而 ZIP 内 `manifest.json` 和 Android 仓库清单均写 `C70084684EC71908C38FAD043B07534F3BCF1999810353361405A0564F14C1EC`。17 个必需文件中仅此一项不符；文件长度、ZIP CRC 与直接从源站 Range 重取的字节一致。`reports/zip-audit.json` 保存逐项结果；提取脚本只在完整 ZIP 哈希通过且显式指定例外时接受该实际图哈希。上述 Flow 合成 token 前向已通过，但仍需真实文字和听感验证。

首次 LLM 测试曾因我方清单漏列 `llm_config.json` 出现 `/blocks.0/Reshape_output_0` 形状错误。该 456 B 文件已从完整 ZIP 提取并通过 SHA-256；补齐后形状错误消失，但最初的 Python `generate()` 测试仍返回 0 个 token（`gen_seq_len=0`，decode 未执行）。单独 `forward()` 的 logits 形状为 1×1×158720，语音 token 区有有限且非零的高分值，促使我们继续检查生成接口；原因和修复见下。

进一步核对 [MNN 3.6.1 Python 绑定](https://github.com/alibaba/MNN/blob/3.6.1/pymnn/src/llm.h) 与 [C++ 生成逻辑](https://github.com/alibaba/MNN/blob/3.6.1/transformers/llm/engine/src/llm.cpp) 后，确认 Python 包装层 `generate(ids)` 把底层 `max_tokens` 留在默认值 0；只有负值才会回退到配置文件的 `max_new_tokens`。测试脚本现显式调用底层绑定并传入上限 256。相同中文句“你好，欢迎来到离线语音测试。”生成 74 个有效 speech token、无非法 token，第 75 个为 EOS；LLM 总生成约 1.32 秒，其中 prefill 约 0.33 秒、decode 约 0.90 秒，记录于 `reports/pc-llm.json`。

将这 74 个真实 LLM speech token 接入 Conditioner→两步 Flow→HiFT，Windows CPU 写出 `reports/pc-real-text.wav`：2.96 秒、峰值 0.722、RMS 0.091，Flow 约 12.38 秒、整体约 14.91 秒，记录于 `reports/pc-real-text.json`。这证明文件与数值链路跑通；**声音是否听得清、内容是否正确和自然仍需人工试听**。完整输出仍在 LLM 全句结束后才解码，尚非流式；此速度也不能外推 iPhone。

## 公开基线与待验证项

- [官方 CosyVoice](https://github.com/QwenAudio/CosyVoice)：按 25 speech-token 左右的块加 lookahead/cache 进行流式语音合成。
- [公开 CosyVoice3-MNN Android 工程](https://github.com/Nian27/CosyVoice3-MNN)：提供 MNN 模型包、LLM、Flow、HiFT 的全句合成实现，可参考导出方式和资产清单；现有流程先产生整句 token，不可直接当成真流式 iPhone 实现。
- 当前公开 Flow MNN 图的输入是 `x / mask / mu / spks / cond / t`，未见官方流式 `flow_cache` 等状态输入。因此不能只把整句图按 25 token 多跑几次就宣称等价于官方流式；需在 PC 上验证缓存/重叠质量，必要时重新导出带状态的图。此项属于根据两份源码作出的工程判断。
- Windows 未完成项：真实文字 WAV 的听感验收、真实分块与全句数值/听感对照。整包审计及一条完整文字→WAV 链路已完成。
- iPhone 未完成项：完整 MNN 推理接线、编译与真机生成。
