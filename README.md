# CosyVoice3 → MNN → iPhone

**iPhone 已完成当前文字→手机 LLM→Flow CPU High→HiFT：5.96 秒音频，耗时 7.834 秒、RTF 1.314、采样峰值 1,558 MB。** v0.5.0 新增有限队列的按句生成即播，以及 Flow 尺寸/DSP/2或4线程对照；新版本仍须真机验收，不声称RTF已降低。ACE-Step 工程 `C:\ace-step-mnn-ios` 保留且不受本项目改动。

独立于 `C:\ace-step-mnn-ios` 的端侧流式语音合成工程。首选模型为公开的 **Fun-CosyVoice3-0.5B-2512**，运行时只使用 **MNN**；不调用云端 TTS，也不把 MLX 当作 iPhone 推理后端。

## 当前状态

- v0.7.6（build20）根据v0.7.5两次Metal Low/NoTune 256帧内存探针（最低余量1129/1159MB）接入真实两步Flow测试；256/384分别冷启动，记录首次/预热与input/run/read-sync，GPU释放后对照同输入CPU High目标Mel。保留800MB采样底线，前向前要求1100MB规划余量；已有会话内存探针从UI归档。尚未取得GPU前向/整链RTF结果，见 [GPU测试说明](docs/metal-forward-v076.md)。
- v0.7.4（build18）根据v0.7.3真机报告新增“三句分阶段／唯一Flow”整链对照：默认三句先获取完整EOS并释放LLM，再共用一套Flow384逐句解码；两条已测路线从实验按钮归档。保留800MB采样底线、有限拆句、播放队列与统一报告。旧两句路线计算RTF为1.744/1.668、峰值约1.61GB；三句新路线的首句延迟、RTF和内存必须真机实测。见 [三句对照说明](docs/three-sentence-staged-v074.md)。
- v0.7.5（build19）新增独立的 Flow Metal Low/NoTune 256帧安全探针：20ms采样峰值与最低可用内存，800MB底线不过即停止；只测会话与resize，不做GPU前向，不将结果冒充整链RTF。保持现有CPU生成路线不变，结果进入统一报告。
- v0.7.3（build17）修复token超限动态错误信息跨autoreleasepool返回的生命周期隐患，短句WAV限制为单句；新增两个真实整链候选：LLM逐句清KV复用、两句token预生成后释放LLM并复用唯一Flow384。默认High两步/HiFT Low/4线程，800MB采样底线不变；计入准备/等待/失败/关闭耗时、记录实际复用及规划释放，仍只分享一个报告。新RTF、共驻和长文稳定性待真机验收。见 [接线、预算与验收](docs/pipeline-reuse-v073.md)。
- v0.7.2（build16）归档已结论的线程/尺寸/BF16/独立LLM复用探针，撤下未证明收益的自适应Flow常驻入口；默认4线程、Flow High两步、HiFT Low，保留音质开关。以后只显示已实现且待验证的实验，计划不冒充可运行按钮；历史记录及一个统一报告不变。本版未新增加速引擎。下一步为完整链路LLM复用＋逐句清KV，先验证共驻余量，再研究唯一Flow复用。见 [实测结论与下一轮验收](docs/optimization-review-v072.md)。
- v0.7.1（build15）统一导出一个 `cosyvoice-test-report.json`，保存分次模块测试及生成记录，带版本/编号/状态和匹配日志；分模块按钮保留，Flow2/3/4改为独立入口。修正LLM累计计时、BF16未完成两步却显示两步耗时的问题。默认推理配置不变；近期RTF1.5–1.7只是按现有数据的条件估算，不是新实测。见 [报告、估算和后续实验方向](docs/next-experiments-v071.md)。
- v0.7.0（build14）清理已有结论的探针UI，增加六条独立路线（Flow尺寸/线程/BF16、LLM复用/线程、HiFT线程），固定输入记录首次/预热、数值偏差和内存；不自动替换默认配置。常驻句级新增1800 MB加载规划与释放Flow转串行回退，800 MB底线不变。所有新配置的真机收益仍待测，W8缺原始导出链暂不可运行。见 [路线、验收与边界](docs/experiments-v070.md)。
- v0.6.1（build13）按用户试听选择默认HiFT core Low，F0/Flow仍High，保留High开关；普通和唯一Flow常驻的句级流水均接通，所有生成增加20ms采样与HiFT resize后800 MB余量保护。PC Low三句回归通过。另完成EOS前出PCM的KV暂停/续推实验，但现有离线图的声学一致性门槛未通过，**不把句级版冒充正式token级流式**；手机内存/RTF仍待真机。见 [Low版、内存策略与token级阻碍](docs/low-stream-v061.md)。
- v0.6.0 新增独立“按句生成 · 常驻Flow384”入口：整批唯一CPU High Flow，4线程、两步不变，每句重新prefill并销毁LLM后才运行HiFT High；超105 speech token（约4.2秒）拆句重做，结束/取消/失败后关闭Flow，保留原逐句释放基线。只有短KV共驻探针已由iPhone确认（峰值2308 MB、最低可用913 MB）；完整整句/HiFT/长文共驻与RTF尚待新包真机验收，不承诺更快或绝不OOM。17个模型不变。见 [常驻句级版与测试步骤](docs/resident-sentences-v060.md)。
- v0.5.1 新增两项隔离真机探针：HiFT core High/Low相同Mel/F0输入、各3次前向和两段2.96秒WAV；唯一384帧Flow会话在LLM加载前/存活时/释放后复用，LLM只decode2个token，约20ms采样内存。普通生成仍保持High和原内存门槛，不自动开启常驻或延长输出。PC调用顺序回归通过，iPhone精度/共驻尚待实测。见 [测试入口与验收边界](docs/optimization-probes-v051.md)。
- v0.4.1 已由真机日志确认完整手机文字链路成功，149个有效speech token、正常EOS、5.96秒WAV，最低可用1,663 MB。v0.5.0 按句流水支持输入≤2000 UTF-16单元/128段，首句完成后开始播放、随后逐句释放模型；待播队列目标≤2秒/硬上限4秒。优化候选保持CPU High和两步Flow，只缩小补零尺寸、优化DSP，另提供2/4线程对照。详见 [句级流水、安卓参考与验收](docs/sentence-stream-v050.md)。它不是句内token级流式；时长、听感、RTF和内存稳定性仍须真机验证。
- 已搭建 iOS 17+ SwiftUI 应用、MNN C++ 桥、模型文件检查、24 kHz 单声道分块播放与首包/RTF 指标框架。
- `试听播放链路` 只播放本地生成的测试音，不是 CosyVoice 推理。
- `探测 Conditioner` 只验证一个 MNN 小图能在手机上加载、创建 CPU 会话。Flow Metal High（256 帧）与 HiFT core CPU Low（160 帧）的单独会话内存试验只做创建/resize。此前 512 帧 Flow 真机 resize 占 2,957 MB、仅余 264 MB，因此不再提供 512 帧按钮。日志保存在文件共享区 `last-memory-probe.log`。
- v0.3.0 的“生成 1 秒固定样例”使用 PC 文字模型已生成的 25 个真实 speech token，在 iPhone 上执行 Conditioner CPU → 两步 Flow Metal High（224 有效帧、256 帧图）→ HiFT F0/core CPU → 24 kHz WAV。逐阶段记录耗时和可用内存到 `last-generation.log`，成功后生成 `last-generated.wav`。首份真机日志在 Flow 256 resize 后 footprint=2,757 MB、available=464 MB，低于 650 MB 安全门槛，**尚未执行 Flow 前向或生成音频**；不能为求通过而直接取消门槛。v0.3.1 增加隔离的 Flow Metal High NoTune 与 CPU Low 256 帧会话/resize 探针，逐项运行并分享 `last-memory-probe.log`，再决定生成后端。它们不执行前向。**手机文字 LLM 与真流式均未接通**。
- v0.3.1 真机：Metal High NoTune resize footprint=2,745 MB、available=476 MB，未改善内存；CPU Low resize footprint=792 MB、available=2,429 MB。v0.3.2 将固定 1 秒生成切换为 **Flow CPU Low**（Conditioner/HiFT 保持 CPU High），resize 后需至少 1,200 MB、第一步后需至少 800 MB 可用内存，逐步检查有限值并记录 `last-generation.log`。同时增加 Metal Low NoTune 256 帧隔离 resize 探针，以区分后端差异与精度差异；生成按钮仍使用 CPU Low。Windows 相同 25 token、256 帧 normal/low 回归均写出有效 1 秒 WAV；Windows CPU 不支持原生 FP16，不能替代 iPhone ARM 低精度正确性验证。
- v0.3.2 真机 CPU Low 第一次 Flow 前向后未通过输出检查，footprint=805 MB、available=2,416 MB；旧日志不能区分形状/读取错误与 NaN/Inf，尚不能断定 FP16 溢出。v0.3.3 默认生成切换 **Flow CPU High**，保留 CPU Low 复现按钮，详细记录输出 shape、逻辑类型、字节数、复制状态和首个非有限值，取消对未知后端缓冲的原始 memcpy 兜底；内存门槛不降低。先分享 CPU High 的 `last-generation.log` 与成功 WAV，再决定优化。ACE/Flow 的模型格式差异见 [内存调查](docs/metal-flow-memory-audit.md)。
- v0.3.3 真机固定 25 token：CPU High Flow 两步输出、HiFT F0/core 输出全部有限；完成 1.00 秒 WAV 总耗时 1.970 秒，采样峰值 1,514 MB、最低可用 1,707 MB，RMS=0.09183、PCM 峰值=0.66577。舍入后的两个统计量与 PC 参考一致，不证明逐采样相同或听感已验收。
- v0.4.0 新增“用当前文字生成 · CPU”、真实生成 WAV 播放、LLM prefill/decode/内存日志。首版接收 1–40 个 UTF-16 单元，最多 160 个 speech token（约 6.4 秒）；长度越界或未遇到 EOS 会停止，**不截断并冒充完整语音**。Flow 按真实有效长度选择 256/384/512 帧；LLM（发布包混合量化，CPU Low）和 Flow/HiFT（CPU High）串行释放。6.4 秒是软件测试上限，不是已验证的手机时长或模型理论上限。详见 [短句接线与验收](docs/ios-text-v040.md)。
- v0.4.0 真机 LLM 在约 2 毫秒内加载失败，尚未进入 prefill/decode。原因是桥接层把文件名改成绝对路径，而 MNN 3.6.1 又直接拼接模型目录。v0.4.1 使用明确的 `base_dir` 与相对文件名，逐项预检图、权重、tokenizer、metadata、embedding，并可分享 `last-llm-runtime.log`。PC 回归复现旧路径失败，修复配置成功生成 70 个有效 speech token、正常 EOS；不代表 iPhone 已通过。详见 [修复与验收](docs/ios-llm-path-v041.md)。
- Windows 不能编译 iOS App；仓库内的 GitHub Actions 流程用于 macOS 云端 unsigned IPA 构建。新生成实验需要真机日志验收，云端编译成功不等于手机生成成功。
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

真机内存试验需导入全部 **17 个**必需文件，包括 `llm_config.json`。先单独运行 HiFT，再单独运行 Flow；两项都只测会话创建及固定形状 resize，可能仍因内存压力退出。结束后分享 `last-memory-probe.log`；如 App 退出，重开后分享该文件。不能用这两个数值推断 LLM 与两模块同时常驻或实际前向峰值安全，更不能据此宣称流式已接通。

v0.2.1 另有手动/推送触发的 Linux 云测 `Probe CosyVoice Flow W8 W4`：在公开 FP16 MNN 图上分别尝试 W8 block-128 与 W4 HQQ block-128，核查**外部权重实际是否变小**，再用同一批 74 个真实 speech token 测 256/512 帧 Mel、HiFT PCM 差异并导出对照 WAV。它不把云端 CPU 结果当作 iPhone Metal 内存结论；如果 FP16 源图不支持直接再量化，会记录转换无效，后续需从原始 FP32/ONNX 重新导出。量化模型不打包入 App，也不自动替换已验证的 FP16 模型。

## 构建

项目由 `project.yml` 生成 Xcode 工程。macOS 上需要 Xcode、XcodeGen、CMake；`Vendor/MNN.framework` 由云端工作流构建，不提交到仓库。将项目推送到**新的** GitHub 仓库后运行 `Build CosyVoice iOS`；产物是 unsigned IPA，可沿用当前 ACE-Step 项目的 Windows + Sideloadly 安装方式。构建成功仅证明 App 和 MNN 桥可编译，不代表 TTS 已能生成。

## 首期验收

1. Windows 上模型包 SHA-256 与 17 个必需文件核对；Flow 图清单差异须单独记录和验证。
2. Windows 上 LLM → Conditioner → Flow → HiFT 能生成可听 WAV，并与公开实现的输出/音质对照。
3. Windows 上增量 token、Flow 分块、HiFT 分块缓存/重叠淡化跑通；只有第一块**在最后 token 生成前**可播放，才算真流式。
4. 云端 unsigned IPA 构建通过，iPhone 15 文件共享可见模型；再验证同一套推理图、正确性、耗时和内存。
