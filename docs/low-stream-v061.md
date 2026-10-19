# v0.6.1（build 13）：HiFT Low句级流水与token级准入实验

## 已实现的手机入口

用户对2.96秒配对WAV试听反馈“没啥区别”，本版将“HiFT core Low”开关默认选中；只影响当前文字、普通句级流水与常驻Flow384生成。关闭开关回到High。Flow仍CPU High两步、F0仍High、权重文件不变；旧固定样例与精度探针不受开关影响。不会把Flow切成已出现NaN的Low。

首句完整生成后播放、下一句推理与播放重叠，仍是句级流式。普通入口每句释放模型，常驻入口整批唯一Flow384；每句LLM独立prefill/EOS后销毁，不保留跨句KV。队列目标≤2秒、硬上限4秒，两条交替短文件，磁盘流式合并，不累积整篇PCM。

现有17个模型不变，覆盖安装不卸载。默认Low没有手机端完整长文验收，不能承诺RTF或绝不OOM；High对照始终可选。

## 内存与计时

- 所有生成入口（不止常驻）启用约20ms内存采样，记峰值footprint与最低available。
- 常驻初始化仍需2900 MB稳定余量，Flow初始化后和每次加载/释放LLM需1500 MB；HiFT加载前1000 MB，resize后及完成后检查当前和本次采样最低值均≥800 MB。
- 保持旧串行阶段门槛；没有为通过测试降低门槛。800 MB是本项目保护策略，不是iOS硬限制；单次MNN调用无法强杀，检查可能在调用返回后才生效，仍需真机验证。
- 常驻最大105个speech token/4.2秒，超长拆分重试，不截断、不伪造EOS；输入2000 UTF-16/128段上限保留。
- Flow每步新增input/run/read与进程CPU累计耗时、thermal状态。共驻探针也增加同样字段，帮助排查此前第三轮第一步约10.33秒空档。CPU是全进程多线程累计值，可以大于wall，不是单核执行时长；read包含输出诊断写盘，不能当作纯传输耗时。
- `CosyVoiceBudget.hpp`为生产使用的最低余量与容量策略，云端测试覆盖边界、采样曾跌破后恢复仍拒绝、0余量、105/106token边界。

## 本地验证结果与边界

2026-10-19：`pc_resident_sentence_check.py --hift-low`通过，三句分别25/44/26个有效speech token，均EOS，共3.80秒WAV。唯一Flow只resize一次；更换句长后重放首输入、新建Flow对照相对L2均为0。F0 High/core Low路径PCM有限且非静音。Windows不支持本机ARM FP16实现，不能替代iPhone Low数值或内存验证。

`pc_kv_chunk_gate.py`另做真正的暂停/续推研究：用声明的确定性greedy+重复惩罚前向循环，不冒充生产mixed采样器。一句只prefill一次，在第41个目标token时暂停，运行前缀Conditioner/两步Flow/HiFT Low出首块，然后继续LLM至EOS；同一LLM的KV没有因声学模型调用变化，token序列与不中断重放完全一致。

实测“今天天气很好。”46token/1.84秒，首块在EOS前得到1.32秒PCM（研究用0.2秒尾部扣留，并非已导出的官方finalize）。尾部精确交付剩余12480采样，无丢失/重复长度。**但声学准入失败**：prefix Mel相对最终上下文L2约0.02519，拼接PCM相对同token整句PCM L2约0.84922，未通过1e-3一致性门槛。因此`approved_for_phone_token_stream=false`，未开放为手机正式token流式入口。

这个严格数值门槛失败不等于主观音质差84.9%，但证明不能声称与完整上下文一致。PC试听文件在`builds/kv-chunk-gate-v061/`，报告明确标记离线前缀实验，不当作官方因果流式、iPhone内存合格或实时证明。

“欢迎来到离线语音测试。”的首轮greedy实验到105token仍未EOS，程序按设计拒绝截断；随后使用更短测试句完成暂停/续推实验。不能据短句通过推断任意文本或跨句KV可无限增长。

## token级下一步依赖

官方[调度/HiFT处理](https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/cli/model.py)与[Flow](https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/flow/flow.py)显式处理streaming/finalize/前瞻及累计Mel偏移。当前[公开MNN导出](https://github.com/Nian27/CosyVoice3-MNN/blob/main/research/mnn-cosyvoice3/scripts/export_flow_estimator_variant.py)不显式启用streaming=True。

须获得与现有两步学生匹配的原始checkpoint/补丁或可重新导出的ONNX，导出/验证流式注意力和HiFT状态，再按同token做前缀/尾块、边界音质与内存对照。当前公开HF完整发布目录只有17文件ZIP与音色扩展，没有独立学生checkpoint/流式ONNX；本轮不自动用官方十步教师替换两步学生。

先复用权重、限队列与工作区，再考虑异步预取。当前CPU链路图读取不是整个外部权重读取耗时，resize仍含权重/内存准备，不能凭小图读取毫秒数判断Flash毫无成本。也不启用三个模型并行或未经验证的全常驻。

## 手机验收

1. 冷启动0.6.1，确认Low开关选中，填入三句文字，先跑原逐句释放入口；分享句级汇总/各句诊断/WAV。
2. 单独冷启动再跑常驻Flow384入口；分享上述文件及生命周期，核对HiFT core Low、Flow High、EOS、800 MB最低余量、队列断供与总/稳态RTF。
3. 通过后扩展10句，测停止/重开/长句拆分。内存持续上升、保护停止或闪退时保留日志，不降低门槛强跑。
4. 必须等这些日志，才能说手机Low流水达到内存/速度要求。不要用隔离HiFT的1.81倍core加速当作整链路加速。
