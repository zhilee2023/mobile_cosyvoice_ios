# ACE-Step 与 CosyVoice Flow 的 Metal 内存调查

2026-10-16。当前优先推进 CPU High 生成；下面保留 Metal 调查结果，不把推测写成已修复。

## 本地项目与模型对比

ACE 与 CosyVoice 工程均启用了 `MNN_LOW_MEMORY=ON`、`MNN_SUPPORT_TRANSFORMER_FUSE=ON`，会话均设置 `Memory_Low`、CPU backup，并在阶段结束释放会话/解释器。没有发现 CosyVoice 漏开 ACE 曾使用的低内存开关。两工程的 MNN 锁定版本不同，尚未完成同版本 A/B，不能据此认定版本回归。

只读工具 `scripts/audit_mnn_weight_layout.py` 根据 MNN 3.6.1 FlatBuffers schema 检查实际图文件，未加载外部权重：

| 项目 | 卷积权重块 | 编码 | 卷积权重元素 | 外部卷积权重字节 |
| --- | ---: | --- | ---: | ---: |
| ACE `dit_w4.mnn` | 273 | quant_type=1，bits=4 | 1,574,961,152 | 787,487,274 |
| Cosy Flow FP16 | 191 | quant_type=3，bits=0 | 330,842,112 | 661,684,224 |

这只统计卷积/线性转换后的 Convolution2D 权重；不是模型全参数量，不含 scale、bias、常量或激活。ACE 整个外部权重文件为 1,003,698,366 B；Cosy Flow 为 663,253,312 B。

## 源码已确认的机制

- [MetalConvolution1x1 低内存选择](https://github.com/alibaba/MNN/blob/3.6.1/source/backend/metal/MetalConvolution1x1.mm#L29)：`Memory_Low` 下整数权重可走压缩加载，但明确排除 type=3（FP16）和 type=8。因此 FP16 文件不享有 ACE W4 的同一路径。
- [Metal High 的存储选择](https://github.com/alibaba/MNN/blob/3.6.1/source/backend/metal/MetalBackend.mm#L1515)：只有 `Precision_High` 令 `useFp16AsFp32=false`；浮点张量缓冲大小按此选择 2 或 4 字节。
- [Metal 权重加载/转换](https://github.com/alibaba/MNN/blob/3.6.1/source/backend/metal/MetalConvolutionCommon.mm#L282)：部分浮点权重先加载到 FP32 源缓冲，再写入重排后的目标缓冲；转换通过异步 Metal 命令提交。

Flow 的上述权重全部展开成 FP32 约 1,323,368,448 B。若初始化时源/目标两份都存在，约为 2,646,736,896 B，与真机 resize 后 2,745 MB 的量级高度吻合。**这是机制支持的解释，尚未逐分配测量**：不能断言恰有两份完整权重永久常驻，也不能把所有差额都归因于 FP32；激活、对齐、算子临时内存及缓冲生命周期仍未拆开测量。

## ACE 的可复用经验

- ACE W4 DiT 在 Metal High、Memory Low 下实际峰值约 1.95–2.03 GB；High 对非量化计算精度的要求不意味着把所有 W4 权重永久展开成 FP32。
- ACE 每个扩散步重传全部输入，CPUHigh/MetalHigh 两步相对 L2 降到约 1.8e-5、6.1e-5。本工程同样每步重传六个 Flow 输入。
- ACE VAE 的 NoTune 曾显著减少初始化时间；Cosy Flow NoTune 却没有减少内存（2,757→2,745 MB）。它不是通用的省内存开关。
- ACE 采用分阶段串行释放；本工程保持该做法。串行释放解决组件叠加，不能消除单个 Flow 的加载峰值。

上游 [CosyVoice3-MNN README](https://github.com/Nian27/CosyVoice3-MNN) 的 Flow CPU 路径使用 normal 精度、OpenCL 使用 high；其 CPU Low/FP16 行为不能直接外推 iPhone。MNN [#3883](https://github.com/alibaba/MNN/issues/3883) 也有人报告 Metal 内存高，但模型、平台与配置不同，没有可直接套用到本图的已证实修复。

## 当前执行方向

真机 CPU Low 256 resize 约 792–805 MB，但第一步输出检查失败；详细原因待 v0.3.3 日志确认。优先用 CPU High 跑固定 25 token→1 秒 WAV，保持数值/内存检查。Metal 后续若恢复研究，应先区分 High/Low、初始化暂存和稳态缓冲，再讨论选择性量化或带状态流式图；不重复无效的外部 FP16 MNN 直接 W8 转换。
