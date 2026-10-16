# v0.5.1：HiFT 配对与唯一 Flow 共驻探针

本版是独立诊断，不自动开启长文模型常驻、不改变已验证生成精度、不放宽160 speech token上限。ACE项目不变，17个模型文件不变。

## 真机使用

覆盖安装0.5.1（build11），保留模型；每项从冷启动单独运行，保持App前台。两项不能和播放、句级生成或其他探针同时运行。

1. 点“HiFT High / Low 同输入对照 · 2.96秒”。完成后分享 `last-hift-precision.log`、High参考WAV与Low候选WAV，试听是否有音高、齿音、响度、尾音变化。
2. 冷启动后点“Flow常驻＋LLM前向 · 内存与复用探针”。分享 `last-resident-flow.log`，如LLM失败再分享原始LLM日志。该项不生成音频，LLM只decode2个token，正常可能没有EOS；不能当完整语音生成。

如果App被系统终止，重新打开后仍可分享对应阶段日志。没有检查能够保证绝不被系统终止；可用内存会动态改变，阶段门槛与20ms采样不是操作系统峰值或硬预算证明。

## HiFT只改一个变量

使用 `testdata/pc-real-74-speech-tokens.csv` 同一组真实PC语音token（嵌入原生代码，不需要新增模型），一次性运行 Conditioner High → 两步Flow High → F0 High与确定性声源/STFT，得到148帧Mel、2.96秒音频输入。

High与Low只改变HiFT core的计算精度，4线程、Memory Low、优化DSP和输入完全相同。分别创建/释放独立core会话，每组前向3次；记录第一次与后两次均值、resize时间和backend。不会同时留下High和Low两套core模型。

记录频谱相对L2、PCM相对L2、去均值相关、RMS增益差、拟合最佳标量后的残差与接近削波比例。`candidate ~= scale*reference` 的残差单独报告；试听WAV不作增益对齐，避免掩盖音量变化。当前ISTFT与已验证生成一致，输出裁到±0.99，因此需要同时看频谱差异与削波比例，不能仅凭PCM有限值认证音质。

`success=true`只表示两组前向、数值检查和WAV保存完成，**不批准切换默认Low**。候选非有限/静音/写入失败时不会将这轮候选开放为配对试听；旧文件不证明当轮成功。UI两段试听仅在当轮配对成功后显示。

加载前至少1800 MB、resize后1000 MB、前向前800 MB可用；20ms采样曾低于800 MB也会停止。准备Flow仍沿用原有串行门槛。

## Flow/LLM共驻与复用

只创建**一套**384帧 CPU High Flow，输入为同一74-token固定样例（有效322帧）。顺序：

1. 调整一次形状并前向两步，保留基线latent。
2. 不释放Flow，加载手机LLM，当前固定短句prefill＋2步decode，验证合法speech token。
3. LLM权重与短KV仍存活时，同一个Flow再跑两步；始终串行算，不启动模型并行。
4. 释放LLM，同一个Flow第三轮两步前向；两轮均与基线相对L2≤1e-3。
5. 释放唯一Flow，观察0.25秒后的内存。函数结束不保留全局会话池。

初始可用≥2900 MB；Flow准备后及LLM加载前≥1500 MB；LLM前向、共驻Flow前向前≥800 MB。约20ms采样曾低于800 MB时不会继续下一轮Flow。本探针不套用普通串行的2500 MB回收门槛，因为Flow本就刻意存活；**普通生成的门槛没有降低**。

探针通过只证明此固定尺寸、短KV、三轮调用条件下可以共驻，不能直接推出161-token decode、HiFT同时存活、512帧Flow、并发计算或分钟级长文安全。下一步需按实际日志再决定常驻方案。

## PC与云端验证

`scripts/pc_v051_probe_check.py` 在Windows用同一固定74-token样例、唯一384帧Flow会话，验证三阶段调用、同输入复用、HiFT High/Low各3次前向及两段WAV；输出在 `builds/v051-probes/`。

2026-10-16 PC：LLM合法输出2个speech token，Flow只resize一次，复用相对L2=0；两组PCM相对L2=0，2.96秒。Windows报告不支持原生FP16，所以此结果不代表ARM Low正确性或内存。无iPhone共驻实测时不得填写“已通过手机内存”。

云构建另运行生产比较指标header的单元测试：完全相同、纯1.3倍增益（相对L2约0.3而对齐残差近0）、NaN、空/异形、静音、削波计数；原来的分句与DSP回归继续保留。

## 时长与prefill口径

- 单段最多160 speech token，即6.4秒软件上限；单段真机已经验证5.96秒。
- 当前句级累计最长日志8.60秒（三段），总输入2000 UTF-16单元/128段只是测试入口限制，不证明分钟级上限。
- 当前每个句段重新prefill，句内decode复用本句KV；未来模型常驻也应与跨句KV累计分开。安卓LLM持久实例同样每次请求reset。此版没有开启跨句KV累计。
