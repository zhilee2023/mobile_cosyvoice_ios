# v0.4.0：短句文字到语音

## 已有真机基线

v0.3.3 已在 iPhone 上完成固定 25 speech token → Conditioner CPU High → Flow CPU High 256 帧、两步 → HiFT CPU High → 24 kHz 单声道 1 秒 WAV。总耗时 1.970 秒，采样峰值 1,514 MB，最低可用 1,707 MB。WAV 写入和有限值检查通过；音质与逐采样误差尚未验收。

## 新接线

本版本将 MNN LLM 编入相同的静态框架，关闭 LLM demo 与 HTTP 资源支持。复用原有 17 个模型文件，不下载另一套模型。

文字按 PC 已跑通的固定参考音色模板处理，经 `apply_chat_template`、分词，再附加 87 个参考 speech token 的词表偏移 ID。LLM 使用发布包原有的 W4/W8 混合模型及 BF16 嵌入文件，CPU Low、Memory Low、四线程与 mixed sampler；Flow 与 HiFT 保持 CPU High。逐个输出 token 的 LLM decode 使用内部 KV cache；该 cache 不与 Flow 的扩散状态混为一谈。

LLM 正常结束后只保留合法的 speech token，保存 `last-generated-speech-tokens.csv`，销毁 LLM 并观察可用内存连续两次达到 2,500 MB（最多 5 秒），然后执行 Conditioner → Flow 两步 → HiFT → WAV。生成输入全部来自当前文字的手机 LLM 输出，不回退到 PC 的固定 token。

## 范围与限制

- 首版文字长度 1–40 个 UTF-16 单元；输出最多 160 个 speech token，相当于 320 个 Mel 帧、6.4 秒音频。
- LLM 最多采样 161 token，预留 EOS；没有 EOS、出现非法 ID 或超过 160 speech token 时不合成半句。
- Flow 有效长度为 `174 + 2 × speech_token_count`，选择不小于该长度的 256/384/512 帧；最大有效长度 494。
- 这些是软件限制，不是 iPhone 的已测最大时长。384/512 帧 CPU High、文字 LLM 本身及其阶段切换均待真机验证。
- 这是先完成整句 token、释放 LLM、再生成 WAV 的串行实现；不宣称首包流式、增量文字输入或长句衔接已完成。
- Qwen 与 Flow 尚未做同时驻留测试，不能把模型文件大小相加当成运行内存结论。

## 本地验证与真机验收

`scripts/pc_text_chain_check.py` 实际调用 PC MNN LLM，从默认中文短句产生 72 个合法 speech token、正常 EOS；Conditioner 318 有效帧、384 帧 Flow → HiFT，生成 2.88 秒 WAV。384 相对 512 帧 Mel L2 为 4.47e-5，PCM RMS 0.10661、峰值 0.81960。采样有随机性，token 数不要求每次完全一致；这些不是 iPhone 性能或数值结果。

安装后确认版本 0.4.0，先用默认短句点击“用当前文字生成 · CPU”。分享 `last-generation.log`、成功 WAV 与可选的 speech token CSV。日志需包括 LLM 加载、prefill/decode、token 合法性/EOS、释放门槛、Flow 桶、各阶段输出有限值、WAV 时长及整个流程 RTF。如果失败，保留最后检查点；旧 WAV 只代表最近一次成功生成。
