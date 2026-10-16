# CosyVoice3 → MNN → iPhone

**Windows PC 已完成一条真实文字→语音的 MNN 烟雾测试。** 听感与真流式尚未验收；iPhone 工程只是后续移植骨架，不应当作已完成的端侧流式 TTS。ACE-Step 工程 `C:\ace-step-mnn-ios` 保留且不受本项目改动。

独立于 `C:\ace-step-mnn-ios` 的端侧流式语音合成工程。首选模型为公开的 **Fun-CosyVoice3-0.5B-2512**，运行时只使用 **MNN**；不调用云端 TTS，也不把 MLX 当作 iPhone 推理后端。

## 当前状态

- 已搭建 iOS 17+ SwiftUI 应用、MNN C++ 桥、模型文件检查、24 kHz 单声道分块播放与首包/RTF 指标框架。
- `试听播放链路` 只播放本地生成的测试音，不是 CosyVoice 推理。
- `探测 Conditioner` 只验证一个 MNN 小图能在手机上加载、创建 CPU 会话。**完整的 LLM → Flow → HiFT 流式推理尚未接入**；界面会明确标记“未接通”，不会把预生成音频冒充端侧推理。
- Windows 不能编译 iOS App；仓库内的 GitHub Actions 流程用于 macOS 云端 unsigned IPA 构建。尚无真机运行结论。
- Windows CPU 已跑通 LLM→Conditioner→Flow→HiFT：固定中文测试句生成 74 个有效 speech token、正常 EOS，并写出 2.96 秒 WAV；**是否清晰、自然仍需试听确认，且不是流式**。详见 [PC 验证记录](docs/plan.md)。
- 1.40 GB 发布 ZIP 的整包 SHA-256 已通过；17 个必需文件中 16 个与上游清单匹配，Flow 图存在已记录的清单哈希差异。MNN 3.6.1 Python 包装层生成接口默认只做 prefill；PC 测试脚本显式传入最大 token 数后才获得有效输出。

## 技术选择

参考官方 CosyVoice3 的 25-token 分块和缓存机制，以及公开的 [CosyVoice3-MNN Android 移植](https://github.com/Nian27/CosyVoice3-MNN)。后者提供 MNN 模型包和导出经验，但现有合成管线是“先生成整句 token，再跑 Flow/HiFT”，**不是 iPhone 可直接复用的真流式实现**。本项目不复制 Android JNI 或 Qualcomm 专用代码。细节和验收门槛见 [实施方案](docs/plan.md)。

## 模型准备

从 [公开 MNN 完整包](https://github.com/Nian27/CosyVoice3-MNN/blob/main/release-manifest.json) 下载约 1.40 GB ZIP 到本项目的 `downloads/`；断网后重复执行同一命令可续传。模型和音色资产**不纳入 Git**。下载后先核对整包与 17 个必需文件，再测推理：

```powershell
python scripts/download_model.py
python scripts/audit_zip.py downloads/cosyvoice3-mnn-mobile-fp16-complete.zip
python scripts/extract_model.py downloads/cosyvoice3-mnn-mobile-fp16-complete.zip --accept-known-flow-graph-mismatch
python scripts/verify_model.py models/CosyVoice3 --sha256 --accept-known-flow-graph-mismatch
C:\ace-step-mnn-ios\.venv\Scripts\python.exe scripts/pc_probe.py models/CosyVoice3 --target-tokens 25
C:\ace-step-mnn-ios\.venv\Scripts\python.exe scripts/pc_llm_probe.py models/CosyVoice3 --max-new-tokens 256 --output reports/pc-llm.json --tokens-output reports/pc-speech-tokens.csv
C:\ace-step-mnn-ios\.venv\Scripts\python.exe scripts/pc_flow_hift_probe.py models/CosyVoice3 reports/pc-speech-tokens.csv reports/pc-real-text.wav --report reports/pc-real-text.json
```

`audit_zip.py` 会因那一个 Flow 图哈希差异返回非零码；只有整包发布 SHA-256 已通过、审计报告确认仅该差异时，才使用显式例外标志。将来部署到 iPhone 时，可把提取的 `models/CosyVoice3/` 文件通过 Apple Devices 文件共享导入 App。这个 Android 包以 MNN 3.6.1 制作，只是 iOS 兼容性候选，尚未证明其中每个图都能在 iPhone/Metal 上运行，更未证明它们能分块流式运行。不要直接按 Android 的 0.79–0.96 RTF 估算 iPhone 速度。

## 构建

项目由 `project.yml` 生成 Xcode 工程。macOS 上需要 Xcode、XcodeGen、CMake；`Vendor/MNN.framework` 由云端工作流构建，不提交到仓库。将项目推送到**新的** GitHub 仓库后运行 `Build CosyVoice iOS`；产物是 unsigned IPA，可沿用当前 ACE-Step 项目的 Windows + Sideloadly 安装方式。构建成功仅证明 App 和 MNN 桥可编译，不代表 TTS 已能生成。

## 首期验收

1. Windows 上模型包 SHA-256 与 17 个必需文件核对；Flow 图清单差异须单独记录和验证。
2. Windows 上 LLM → Conditioner → Flow → HiFT 能生成可听 WAV，并与公开实现的输出/音质对照。
3. Windows 上增量 token、Flow 分块、HiFT 分块缓存/重叠淡化跑通；只有第一块**在最后 token 生成前**可播放，才算真流式。
4. 云端 unsigned IPA 构建通过，iPhone 15 文件共享可见模型；再验证同一套推理图、正确性、耗时和内存。
