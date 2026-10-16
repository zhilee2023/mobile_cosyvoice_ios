# 端侧流式可行性：官方、Android 移植与当前 iPhone

2026-10-16。当前 iPhone 已验证固定 25 speech token → 1 秒 WAV；v0.4.0 已编译短句文字链，但还没有真机文字生成日志。本文是流式调查，不宣布流式已完成。

## Android 的两步不是流式

[公开 Android Runtime](https://github.com/Nian27/CosyVoice3-MNN/blob/main/app/src/main/java/com/fun/cosyvoice/CosyVoiceRuntime.kt) 先等待 LLM 任务结束和 token 质量检查，再读取整句 `speech-tokens-0.csv`，选择 Flow 桶，顺序调用 Conditioner、两步 Flow、HiFT。可借鉴模型和调度，但该流程不是逐 token 出音频。

两步蒸馏描述的是一次 Flow 求解的迭代数。当前 iPhone 已使用两步学生模型；不能再把 10→2 步的收益重复计入流式优化。

## 官方 CosyVoice3 如何出流

[官方 model.py](https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/cli/model.py) 中 LLM 任务向 token 队列追加数据。输出循环初始 hop=25 token、最大 hop=100、倍率=2；25 Hz 的 token 对应约 1 秒音频，但首块还受提示 token 对齐和 3-token 前瞻影响。

本工程提示 token 为 87 个。官方循环的首块对齐补量是 `ceil(87/25)*25-87=13`，因此首块等待 `25+13+3=41` 个目标 token，稳定输出区域对应 38 token、约 1.52 秒 Mel 音频，最终波形还受声码器前瞻影响。这不是首包耗时保证。

[官方 flow.py](https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/flow/flow.py) 在非最后一块时将末尾 3 个 token 当作 pre-lookahead 的 context；[官方 DiT](https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/flow/DiT/dit.py) 的 `streaming=True` 使用分块注意力掩码，`False` 使用整段掩码。`CosyVoice3Model.token2wav` 还累积 Mel，并通过声码器输出偏移只交付新增波形。

因此，官方参考实现本身仍会传入越来越长的 token 前缀、保留累计 Mel；它不是“固定一个 1 秒窗口、永久固定内存”的证明。CV3 的这条实现也不能简单等同于 CV1 的显式 `flow_cache` 接口。是否缺少名为 flow_cache 的输入，不足以独立判断能否流式。

## 当前 MNN 导出差异

[Android Conditioner 导出](https://github.com/Nian27/CosyVoice3-MNN/blob/main/research/mnn-cosyvoice3/scripts/export_flow_conditioner.py) 对整个输入调用 pre-lookahead 层；[Estimator 导出](https://github.com/Nian27/CosyVoice3-MNN/blob/main/research/mnn-cosyvoice3/scripts/export_flow_estimator_variant.py) 调用 estimator 的六参数形式，没有传入 `streaming=True`。当前发布图输入仅为 x/mask/mu/t/spks/cond，没有暴露官方流式开关。不能把这个离线学生图分几次运行，就认定与官方 causal streaming 等价。

本地 `scripts/pc_stream_prefix_check.py` 使用刚刚由 LLM 产生的同一条 72-token 短句，取前 41 个目标 token，Conditioner 后裁去最后 3 token 的编码，得到 38-token/1.52 秒首块；与看到完整 72 token 后的同一区域 Mel 相比，相对 L2 为 0.032365。完整音频长 2.88 秒。输出与报告位于 `builds/stream-prefix-feasibility/`。

该实验说明当前离线图存在前缀敏感性；它没有与官方 streaming=True 的结果比较，不能据此宣布首块质量合格或官方流式失败。需要接缝、音色、基频、最终 PCM 及人工试听验证。

## 可执行的推进顺序

1. 先在 v0.4.0 验证真实手机文字 LLM、prefill/decode、内存回收与 CPU High 短句生成；该步已经有可安装的构建。
2. 当前图可先尝试**句级播放流水**：长文按完整短句切开，每句复用同一音色，生成即写盘/播放，播放队列有限长；每句完成后释放模型和音频缓冲。它能让总朗读长度超过单句上限，但不是一句内部的 token 级流式，句间韵律与停顿必须验收。
3. token 级流式需要 LLM 保留 KV、增量 decode，token 队列与 Flow/HiFT 的因果分块正确接通。先测 LLM 与 CPU High Flow 同时驻留的加载/前向峰值；当前磁盘文件大小不能替代该测量。CPU 上并发也可能争抢核心，可先使用保留 KV 的协作式串行调度。
4. 要让单句很长时仍有严格内存上限，还需有限左上下文或带状态缓存的兼容图，并重新验证蒸馏学生的分块行为。当前公开包没有提供已验证的这种 iPhone MNN 流式图；原始同权重学生 checkpoint/ONNX 会降低重新导出的难度。

现有 1 秒样例的 Flow 两步本身约 1.114 秒，尚不能保证持续实时播放；较大块是否改善 RTF、首包延迟和内存，要用实际真机日志判断。
