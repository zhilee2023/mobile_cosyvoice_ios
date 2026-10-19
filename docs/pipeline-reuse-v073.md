# v0.7.3 / build17 — 超限错误生命周期修复与两个整链候选

## 范围与未验证项

修复已发现的动态错误对象生命周期隐患；原始`.ips`尚未提供，因此不声称
所有闪退根因已获崩溃栈确认。新入口真实调用完整文字LLM→Conditioner→Flow→
HiFT Low→WAV，并沿用有限队列的句级播放；不是token级流式。未通过iPhone
新一轮测试前，不声称RTF更低、共驻安全或分钟级稳定。17个模型、ACE工程不变。

## 超限错误返回修复

旧`NSString **failure`在ARC下隐式为autoreleasing指针。函数内有自己的
`@autoreleasepool`，动态格式化的超限消息可能在返回caller writeback之前
被释放。用户日志正停在161个speech token、无EOS后的失败路径。

改为共享的`CosyVoiceFailureOutput = NSString *__strong *`，直接保存到调用方
强引用；动态消息工厂亦由测试与生产共用。所有调用点均为强引用局部变量。
macOS回归在内层pool销毁之后读取消息1000次，并启用NSZombieEnabled。
这验证该类对象生命周期修复，不代替真机触发超限后的稳定性验收。

UI短句WAV仅接受一条≤40 UTF-16单元句子；多句使用按句入口。native仍检查
160-token/EOS，超限失败不运行Flow/HiFT、不截断，并让句级调度有限拆句重做。

## ① 整链LLM复用

- 每批唯一LLM，后续句逐句reset KV并重新prefill。没有跨句KV、没有并行模型计算。
- 每句声学创建一套Flow、运行两步后销毁，再做F0 High/core Low与WAV。
- 加载新LLM前规划1700 MB；LLM保留时Flow resize前规划2300 MB（约1.5GB
  增长预算＋800MB底线）。不足则先释放LLM，保留已正常EOS的token，不重跑LLM，
  等待2500 MB再做声学。
- Flow resize后，若LLM仍存活但余量不足1000 MB，则释放LLM；同一Flow会话保留，
  等待1200 MB运行工作区。非共驻仍要求1200 MB；与LLM共驻1000 MB仅是本候选
  独立规划值（预留200MB工作区），并非降低最终800 MB采样底线。
- 任何20ms采样低于800 MB均不能算候选通过，即使之后回升。未知/零余量失败关闭。
- 报告区分实际复用请求、LLM规划释放次数、Flow期间是否持有LLM。全部回退生成
  成功不等于LLM共驻提速成功。关闭/失败/取消后销毁保留的LLM。

## ② 两句分阶段、唯一Flow384

- 每小批最多两句。唯一LLM逐请求清KV，先生成各自正常EOS的完整speech token，
  每句≤105 token/4.2s；然后销毁LLM，等待至少2500 MB余量。此阶段没有Flow/HiFT。
- 创建唯一Flow384，仅resize一次；先声学解码句1并播放，再复用它解码句2。
  重新传入所有六个输入，High/两步不变；HiFT/F0逐句运行。Flow与HiFT可能内存
  共驻，依旧必须实际采样验收，而不是从独立峰值推算安全。
- 小批结束释放Flow及token队列，下一小批重新调度。末尾单句正常flush。
  无进程全局模型池，不允许覆盖仍有未消费句段的队列。
- 首句前先做两句LLM，因此可能增加TTFA；是否减少总RTF和断供必须实测，不能
  隐藏这个首包代价。播放PCM队列仍目标2s/硬上限4s，至多一句待播文件。
- 准备阶段某句超限则记录具体failedTextIndex，销毁LLM并丢弃未播小批token，
  有限拆该句后重做。失败/重新准备耗时计入计算RTF，不丢字或强制EOS。

## 共用记录与清理

新入口沿用当前输入、HiFT选择与4线程，默认三句fixture拆分仍为3句。
保留正常串行基线；已结论实验不重新露出按钮。
统一报告新增`pipelineMode`、每次准备/生成的`attempts`、生成墙钟时间/RTF与
`lifecycle`计数。生命周期源文件自动汇入同一个报告，不新增分散分享入口。
Trace仍保留关键落盘记录；此版未混入日志缓冲等额外变量。

初始化、准备、生成、保护等待、失败重试、候选close都计入候选计算耗时；
生成墙钟含复制/合并/队列等待，播放完成墙钟另由报告elapsed字段记录。
停止不强杀正在执行的native调用；返回后取消队列、保留partial并关闭唯一owner。
错误/取消路径也await close，不以取消Task为理由跳过模型销毁。

## 验证

- Cloud：拆句/报告/归档/数值指标/内存策略单测、ARC动态错误消息回归、iOS实际编译。
- PC：`scripts/pc_pipeline_reuse_check.py` 用同一penalty+greedy采样和384帧控制输入，
  比较fresh / LLM保留穿过声学 / 两句分阶段三种生命周期；检查speech token完全
  一致、正常EOS、Mel/PCM相对L2<1e-3，另强制两个token无EOS后销毁并恢复。
  Windows无法验证iPhone原生FP16、ARC、可用内存或播放性能。
  本轮已实际通过：三句分别20/42/27 speech token；三种生命周期均EOS且逐token
  相同，九段受控Mel/PCM相对参考L2均为0，强制无EOS销毁两次后可恢复生成。
- Phone：分别冷启动后，先默认三句LLM复用，再默认三句两句分阶段；只分享一个
  `cosyvoice-test-report.json`。看真实复用次数、完整EOS、声音、RTF、首入队、
  断供、峰值/最低余量与清理。随机采样时长可能不同，不能拿一次RTF差当严格A/B。
- 三句通过后再测30/60秒、连续两批与取消/恢复；任一保护失败不直接扩大长文。
- 短句WAV另用一条句子测试正常返回；超限原生生命周期测试已在mac执行，但需要
  真机复测曾触发问题的长度。多句输入不再直接交给短句按钮。

参考机制：[Android持久LLM](https://github.com/Nian27/CosyVoice3-MNN/blob/main/mnn-jni/CosyVoiceLlmPersistentBenchmark.cpp)、
[Android持久Flow](https://github.com/Nian27/CosyVoice3-MNN/blob/main/mnn-jni/CosyVoiceFlowPersistentBenchmark.cpp)、
[Clang ARC对象生命周期](https://clang.llvm.org/docs/AutomaticReferenceCounting.html#storage-duration-of-autoreleasing-objects)。
本包未照搬Android专用API/硬件设置，也未更换匹配的2步学生模型。
