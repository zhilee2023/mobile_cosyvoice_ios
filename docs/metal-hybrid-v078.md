# v0.7.8 局部 FP32 注意力候选

## 来自真机的定位证据

v0.7.7在256帧第一步检查289个算子、668个浮点tensor，未跳过tensor：

- 早期UnaryOp70先产生Inf，但后续激活恢复，未在此产生NaN。
- 第三个Transformer块（blocks.2）的QK输入峰值37.53125/278.25，均有限；QK矩阵乘法输出414192个Inf，有限部分最大65504。
- 加mask后Inf增至484755；随后Softmax首次产生408576个NaN，输入尚无NaN。
- 整个诊断最低可用1108MB。直接失效边界为注意力分数溢出进入Softmax；仍需验证上游Low误差与最终音质。

只对已经溢出的score改Softmax不能恢复原值，因此此版从仍有限的Q/K重算注意力概率。

## 候选实现

保留原FP16模型文件和Metal Low/NoTune主干，以MNN公开算子回调取得每层已缩放的Q/K和原mask。CPU Accelerate SGEMM计算FP32 QK，加入mask，再做减去行最大值的稳定Softmax。概率位于0–1范围，回传到原Softmax输出后继续GPU计算。不是纯GPU路线。

不对logit剪裁、不把NaN清零、不修改全图权重或正式CPU生成入口。原GPU QK/Softmax仍会被执行；同步完成后丢弃其输出，以下游覆盖的FP32重算概率继续。其额外计算和复制开销全部计入候选实测。

每步严格要求22次QK捕获、22次mask和22次概率替换，验证层名/形状对应；任一输入已有非有限值、复制失败、层缺失或内存不足即失败，不把部分覆盖计成功。只保留当前层的host缓冲，不创建第二套模型会话。Accelerate内部线程由系统管理。

## 验证

- 云端单测覆盖：QK分数65536/65538（均超FP16范围）仍保持正确概率比；mask广播/独立head mask、行归一化、零输入和NaN/尺寸拒绝。
- 真机先冷启动测256帧，再测384帧。两步各三轮，首次/预热分开，记录注意力读取/CPU计算/同步回传及整个run+最终read。
- 前向规划1100MB与20ms采样800MB底线保持。结束GPU会话后，余量恢复到2500MB才启动CPU High参考；保存目标Mel到同一个报告，严格L2门槛仍为1e-3，超限标需质量复核。
- 此次只运行固定token的Flow；尚无新的整链RTF或试听验收，不将目标0.85当作实测。
- MNN接口依据：[Interpreter callbacks](https://github.com/alibaba/MNN/blob/3.6.1/include/MNN/Interpreter.hpp)。

操作：安装build22，单独测“① GPU主干＋FP32注意力 · 256帧”；若仍有NaN或内存停止先发报告，否则冷启动测384帧；最后只分享统一JSON。
