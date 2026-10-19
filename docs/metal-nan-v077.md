# v0.7.7 Metal 数值定位

## v0.7.6 真机结果

| Flow形状 | 有效帧 | 第一步非有限值 | 峰值MB | 最低余量MB |
|---|---:|---:|---:|---:|
| 256 | 224 | 17920/20480 | 2106 | 1115 |
| 384 | 314 | 25120/30720 | 2115 | 1106 |

两组Conditioner均有限，Flow第一步即失败，未完成两步或CPU对照。无有效GPU RTF；失败调用的耗时不能用作候选加速比。NaN数量分别等于80×有效帧数，但这一模式不能单独定位到注意力或mask算子。

本地只读检查9501个算子：45个LayerNorm epsilon均为1e-6；FP16注意力mask常量为有限值-65504。尚不能据此确定是哪一个中间计算首次失效，不盲改常量或重复相同前向配置。

## 新诊断

冷启动后单独点“定位 GPU 首个 NaN 算子（256帧）”，再分享统一JSON报告。

- 同样25个固定speech token、256帧、Metal Low/NoTune与六个真实输入，时间t=0；只定位第一步。
- 使用MNN 3.6.1默认Session_Debug和runSessionWithCallBackInfo，同步读取算子前输入、后输出；记录首次Inf并继续观察，遇到首个观测NaN后返回。有些Inf可经后续激活饱和恢复，因此不能自动将首次Inf认定为最终NaN根因。依据[官方Interpreter接口](https://github.com/alibaba/MNN/blob/3.6.1/include/MNN/Interpreter.hpp)。
- host采用CAFFE逻辑布局，避免把NC4HW4未初始化padding误判成NaN；读取float32/float16，记录NaN/Inf数量、首个位置、有限值峰值、形状和类型。
- 每个tensor上限8MB，保留最近8个算子统计，不保留中间tensor。大tensor会跳过，并显式记录跳过数，因此只声称首个“观测到”的异常边界。
- 最多60秒，1100MB前向规划、20ms采样800MB底线。诊断会频繁同步，耗时不能与普通runSession基准比较。若同步后异常消失，也需保留该结果以进一步查执行/同步差异。
- 报告状态diagnostic-completed仅表示完成定位，不表示GPU数值通过。没有更改模型或正式CPU生成设置。
- 旧256/384按钮从UI归档，已有日志与实现保留。
