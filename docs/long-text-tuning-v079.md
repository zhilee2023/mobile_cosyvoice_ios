# v0.7.9 / build23：长文本摊销与独立 GPU 对照

## 已有结论

build22 混合 GPU 的 256/384 帧都连续三轮两步有限、重复 L2=0。预热两步分别约1.90秒 / 2.63–2.75秒，采样峰值约2.1GB。GPU释放后内存残留使2500MB CPU参考门槛无法通过；这不等于GPU前向OOM。CPU三句分阶段最新8.48秒音频，完整计算RTF1.397，首块7.11秒，断供0。不要把声学单句RTF当完整链路RTF。

## MNN 是否支持调优

支持。固定使用3.6.1，不混用更新版本行为：

- [Metal线程组搜索](https://github.com/alibaba/MNN/blob/3.6.1/source/backend/metal/MNNMetalContext.mm)有Never/Fast/Normal/Wide/Heavy等级；搜索运行候选配置并计时，重调会增加初始化成本，不修复数值溢出。
- [Interpreter缓存接口](https://github.com/alibaba/MNN/blob/3.6.1/include/MNN/Interpreter.hpp)支持setCacheFile/updateCacheFile，可持久化运行时调优信息。未来应按引擎/模型/形状/精度/设备隔离缓存并验证命中，不声称本版已经实现持久调优缓存。
- [Metal Attention实现](https://github.com/alibaba/MNN/blob/3.6.1/source/backend/metal/MetalAttention.mm)有融合注意力路径，但其中Flash Attention prefill shader带因果边界假设，不能盲套非因果Flow；现有导出是分离QK/Add/Softmax，开启编译选项不等于该图已融合。
- [Android经验](https://github.com/Nian27/CosyVoice3-MNN/blob/main/docs/PITFALLS_AND_FIXES.md)的OpenCL Buffer和固定桶具有参考价值，不等于iPhone Metal可用同一内存模式/速度。

优先次序：正确性参考拆进独立冷启动；测较长负载的摊销；若混合GPU仍慢，再研究保留非因果mask/原缩放的GPU FP32累加融合，避免22层反复拷贝。不能只把Inf清零或改变Softmax温度。

## 本版实现

### 独立模块

按①CPU384→冷启动②GPU384→冷启动③CPU512→冷启动④GPU512。每个测试只有一套Flow，没有CPU/GPU模型重叠。

CPU参考将完整目标Mel保存在原模块JSON。GPU必须匹配版本/build、fixture、实际五个非时间输入的SHA256、全部7个模型/条件文件的大小与mtime，再读取参考。mtime/大小只是本地身份线索，不是权重密码学证明；变更模型请重新运行CPU参考。GPU结束后不加载CPU。

- 384：原真实70-token前缀，目标Mel140帧/2.8秒。
- 512：原真实74-token序列重复两次，共148 token，含提示区有效470帧，目标Mel296帧/5.92秒。明确标记重复容量负载，不是自然长句、不是可听长文。
- 均三轮两步，列首次/预热、setup、Flow-only冷/热RTF、回调读取/FP32计算/回传、CPU相对L2及温度。
- 不降低前向规划1100MB、20ms采样800MB底线；512的注意力临时矩阵更大，不能保证能前向。有限值与重复一致不等于CPU正确性通过；严格L2仍为1e-3。
- 256已证明吞吐不利，归档按钮，保留实现和结果；384正确性尚待验收。

### 可听长文整链

增加12句/24句预置文本（粗估约30/60秒，真实时长由EOS测得），同一输入分别运行原三句批量与新六句批量。每句仍105 speech-token上限、Flow384 High两步，HiFT可选Low，超限有限拆句，不丢字或强行截断。

六句批量只增大已完成EOS的token队列（≤6×105个int）；LLM逐句reset KV生成后销毁，再加载唯一Flow，按句声学和播放；本批结束释放Flow再下一批，不保留分钟级KV或WAV队列。播放目标2秒/硬限4秒和至多一条待播句段不变。六句可能降低LLM/Flow重复初始化，却增加首块等待，也可能受热降频拖慢；未自动替换正式入口。

统一报告增加：每3句检查点（累计音频/计算RTF/温度/峰值与最低余量/断供）；LLM批量准备与逐句prefill/decode指标；声学调用总时长；Conditioner、Flow、F0、source/STFT、HiFT、ISTFT分项；关闭成本独立列出且算入最终RTF。检查点注明cleanupPending，不能与含关闭成本的最终RTF混淆。墙钟另含播放背压/文件/报告等待；阶段分项不是总耗时之和。

只需分享同一个cosyvoice-test-report.json。原20次/8MB保留策略不变。本版Windows检查/云端编译不代替真机长文、内存或听感验收；没有宣称RTF0.85已达成。

Windows CPU High实际形状回归已通过：384帧/70token输出80×140，峰值10.95635；512帧/148token输出80×296，峰值10.99629，均有限。前向分别1.946/3.940秒，仅表示这两个容量输入能在PC CPU运行，不外推iPhone性能。
