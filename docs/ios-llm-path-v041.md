# v0.4.1：LLM 路径修复与真机验收

## 已定位的故障

用户 v0.4.0 日志在开始加载后约 2 ms 返回失败，footprint 一直为 32 MB。没有进入 prefill/decode，也没有 Flow 前向；这份日志不是 OOM 的证据。

旧桥把 llm_model、llm_weight、tokenizer_file 改成绝对路径。MNN 3.6.1 的 [LlmConfig](https://github.com/alibaba/MNN/blob/3.6.1/transformers/llm/engine/src/llmconfig.hpp) 使用 `base_dir_ + filename`，不特殊处理绝对路径；[Llm::load](https://github.com/alibaba/MNN/blob/3.6.1/transformers/llm/engine/src/llm.cpp) 在 tokenizer 检查时立即失败。

PC 回归实际捕获：

```
[Error]: tokenizer file not found: C:/cosyvoice-ios/models/CosyVoice3/C:/cosyvoice-ios/models/CosyVoice3/tokenizer.mtok
```

## 修改

- 模型目录作为明确的 base_dir（末尾 `/`），五项文件均保留相对文件名。
- 加载前记录目录、每个文件名、字节数、可读性；拒绝绝对文件名和向上跳转。
- runtime 配置仍写在实际模型目录，不覆盖发布包配置和模型权重。
- 区分 createLLM 失败与 load() 失败；捕获 stdout/stderr 到 Documents/last-llm-runtime.log，失败时把末尾内容附入生成日志。UI 增加原始日志分享入口。
- LLM、Flow、HiFT 精度与内存门槛不变；Flow 保持 CPU High，未启用有 NaN 问题的 CPU Low。

原始日志捕获是进程级重定向，仅在 UI 排他运行 LLM 时有效，退出 LLM 阶段自动恢复；它能保存 MNN 打印信息，但无法保证捕获系统在强制终止前尚未刷出的缓冲区。

## PC 回归

```
C:/ace-step-mnn-ios/.venv/Scripts/python.exe scripts/pc_llm_path_check.py
```

2026-10-16 Windows CPU：旧配置状态 NOT_LOADED (-1)，加载 0.000289 秒；新配置加载 0.929829 秒，输入 125 token（含音色提示），输出 70 个有效 speech token、0 非法 token、EOS、状态 NORMAL_FINISHED (1)。采样有随机性，token 数不是固定断言。

脚本把两组配置、原始输出、报告存于 builds/llm-path-v041，分别在独立进程运行；旧配置失败后绝不调用 tokenizer。回归只证明 Windows 配置修复与语音 token 有效，不证明 iPhone 精度、音质、内存安全或流式。

## iPhone 下一次测试

1. 用同一 Bundle ID 覆盖安装 v0.4.1（build 9），确认页面版本；不先卸载，以免删掉 App 文件区模型。
2. 刷新确认 17 个模型已识别，无需重新下载权重。
3. 冷启动后先输入默认短句“你好，欢迎来到离线语音测试。”，点击“用当前文字生成 · CPU”。
4. 分享“生成日志”和“LLM 原始日志”；成功后播放并分享新 WAV，确认不是最近成功的旧固定样例。
5. 检查五文件可读、LLM 加载完成、prefill/decode、有效 speech token/EOS、释放门槛、Flow 两步有限值、HiFT 有限值、WAV。若失败保留具体阶段，不取消门槛。

仍为整句串行，不是流式；6.4 秒只是软件试验上限，不是已验证手机最大长度。
