# v0.5.0：句级生成即播与 RTF 优化候选

## 真机基线已经通过

v0.4.1 iPhone 15 用户日志：当前文字 → 手机 LLM → Conditioner CPU High → Flow CPU High → HiFT CPU High，149 个有效 speech token，正常 EOS，5.96 秒 WAV，耗时 7.834 秒，RTF 1.314。日志采样峰值 1,558 MB、最低可用 1,663 MB；所有输出有限。LLM load 0.391 秒、prefill 0.247 秒、decode 1.754 秒，Flow 两步分别约 1.33/1.36 秒。单次成功不证明持续运行的热稳定性或音质。

## 对照 Android 后的选择

2026-10-16 阅读公开实现：

- [Android Runtime](https://github.com/Nian27/CosyVoice3-MNN/blob/main/app/src/main/java/com/fun/cosyvoice/CosyVoiceRuntime.kt) 先等待整句 LLM 输出，再做 Conditioner、Flow、HiFT；不是 token 级流式。两步学生 Flow 已在本 iOS 工程使用，不再重复计算 10→2 步收益。
- [Flow persistent runtime](https://github.com/Nian27/CosyVoice3-MNN/blob/main/mnn-jni/CosyVoiceFlowPersistentBenchmark.cpp) 复用会话与相同形状的 resize，使用 Power_High、可配置线程和两步求解。iOS 本版保留 Memory_Low、CPU High，优化模式请求 Power_High；它不是 iOS 必然使用性能核的保证。
- [HiFT persistent runtime](https://github.com/Nian27/CosyVoice3-MNN/blob/main/mnn-jni/CosyVoiceHiFTPersistentBenchmark.cpp) 复用 F0/core 并按形状调整；[硬件策略](https://github.com/Nian27/CosyVoice3-MNN/blob/main/app/src/main/java/com/fun/cosyvoice/CosyVoiceHardwarePolicy.kt) 的线程设置依设备变化，不能把 Android 的线程数或 OpenCL/Hexagon 速度照搬到苹果。

跨阶段常驻会改变已验证的串行内存预算，且会与现有 2800 MB 初始门槛冲突，本次不默认引入。LLM、Conditioner、Flow、HiFT 每句仍顺序创建/释放，避免未经测量就把 Android 常驻策略搬过来。后续需要独立的共驻探针。

## 已实现

1. **按句切分**：中文/英文句末、换行（含 CRLF），长句优先在逗号/空格附近拆分；默认每段最多24个 UTF-16 单元，不切开组合字符。首轮总输入≤2000单元、≤128段。这是软件测试边界，不是已验证最长音频。
2. **生成和播放流水**：首句 WAV 完成后，200 ms 一块读入 AVAudioEngine，目标待播≤2秒、播放器硬上限4秒。播放前句时，后台串行生成下一句；最多一条生成完成但尚未交给播放器的句段，两个交替临时文件。不会并行运行两套模型。
3. **有限内存/磁盘**：不累计所有 PCM、token 或 Mel；完整 WAV 边读边写盘，只有两个句段文件和有界缓冲，最终完整音频文件随文本时长增长。模型权重不变。句间需要2800 MB可用内存连续两次达标，原有各阶段门槛不降低。
4. **长度失败处理**：只有生成了≥160个合法 speech token 并达到输出上限，才允许拆短重做；最多3级/128段，不将未遇到 EOS 的前缀截成“完成语音”。失败计算也计入 RTF；非法 token/内存失败不自动重试。
5. **优化候选**：Flow 从256/384/512桶改为最小32帧对齐（最大仍512）；DSP 使用预计算16点 DFT 的窗和三角函数、每个频谱点只计算一次幅相转换。模型权重、Flow 两步/时间表、各模块精度、音频格式不变。可关闭回到旧公式/形状。
6. **线程对照**：可选4线程（原基线）和2线程（实验）；不声称2线程一定更快。关闭优化＋4线程是本版句级调度下的旧推理基线。
7. **停止/失败**：停止立即取消继续入队和下一句，当前有限时长 native 推理不强杀，返回后销毁。每次 stop 的播放器 epoch 不同，旧完成回调不会误减新队列。遇到播放中断20秒无推进会停止，不无限等待；需保持 App 前台。

这不是保持句内 KV/声码器状态的真流式。每段重新条件化，韵律、句尾/接缝需试听；不通过任意裁剪或交叉淡化掩盖漏字。

## 日志定义

- `last-sentence-stream.log`：首块入队时间、每句音频长度/原生调用耗时、句间内存回收、采样 footprint/available、队列断供。
- **计算 RTF**：所有 native 调用墙钟之和（含失败尝试）÷成功音频长度。不包含播放排队；不等于纯模型算子时间。
- **生成墙钟 RTF**：从开始到最后一句生成并交给调度的总墙钟÷音频长度，含回收和背压等待。二者都不把播完所需时间算作推理。
- 首块是送入播放器的时间，不是麦克风测出的首次扬声器发声时间。断供统计来自完成回调/新入队的时间差，含调度误差，不等于精密声学间隙测量。
- `last-sentence-diagnostics.log` 合并每次调用的生成日志/LLM原始日志，避免被下一句覆盖。采样峰值不是连续采样的真实瞬时峰值。
- `last-sentence-stream.wav` 只在整篇模型生成成功并关闭文件后发布；失败/取消保留 `.partial.wav`。播放器失败不改变已经成功生成的完整 WAV。旧的“最近完整长文 WAV”不代表本轮成功。

当生成 RTF>1，句间可能断供；句级流水主要降低等待整篇的延迟，不会自动把计算 RTF降到1以下。

## 测试与验收

Windows `scripts/pc_flow_aligned_check.py`：同一组输入比较旧桶/32对齐，25/74/149/160 token 的 Mel 相对 L2 分别约0.000049、0.000035、0、0，全部有限。149/160的输入重复真实 token 序列仅做形状敏感性测试，不是自然语音质量证据。25 token 本次更小尺寸并未更快；PC计时波动明显，不能据此承诺 iPhone 提速比例。

云构建在打包前编译运行**生产 Swift SentencePlanner**（中英文、长句、CRLF、emoji、十进制、空输入、2000字符/128段边界、重拆不丢字）和**生产 C++ DSP header**（1/2.98/5.96秒长度、有限值、与旧式 STFT/ISTFT 最大误差≤1e-5、RMS误差≤1e-7）。云端测试不替代真机 AVAudioEngine、热降频和模型内存验收。

真机顺序：

1. 用同一签名/Bundle ID 覆盖安装v0.5.0，确认17个模型，无需新权重。
2. 点“填入三句测试文字”；关闭优化、4线程，按句生成。应首句完成即播放，下一句推理重叠；分享汇总日志、各句诊断、完整WAV。
3. 同文开优化、4线程再测；随后单独测2线程。采样有随机性，应重复3次、按实际音频秒数归一化，不用不同句子的总耗时冒充提速。
4. 再测试约10句或30–60秒音频，核对最后几句 footprint 是否持续上升、最低可用内存、句间停顿/漏字/音色。音频长度由模型输出决定，不承诺指定秒数。
5. 测中途停止/立即重开（须等当前native返回）、连续两轮、空输入/长句自动拆分、超过上限、播放被打断。旧文件不得冒充当轮成功。

当前没有v0.5.0真机结果，不声称RTF已降低或长文内存已证明稳定。
