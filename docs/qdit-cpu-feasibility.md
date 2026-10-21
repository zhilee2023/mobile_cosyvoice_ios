# Q-DiT and CPU: keep algorithms, backend and measured evidence separate

Official authors' [Q-DiT repository](https://github.com/Juanerx/Q-DiT) includes
`qdit/gptq.py`, `qdit/quant.py`, activation-wrapped `qdit/qBlock.py` and
image DiT calibration/quantization scripts. It is not a ready CosyVoice or MNN
export recipe. `qdit/qLinearLayer.py` calls floating `F.linear`: simulated
precision quality does not establish native integer-kernel throughput.
The official GPTQ helper uses full Hessian/cross-block error propagation and
unconditionally calls `torch.cuda.synchronize`; a CPU port needs a device guard
and FP32-compatible calibration/evaluation, rather than a device string alone.

MNN3.6.1 official [compression guide](https://github.com/alibaba/MNN/blob/3.6.1/docs/tools/compress.md)
provides `MNNConvert --weightQuantBits 4 --hqq --weightQuantBlock 64`,
`auto_quant.py`, and low-memory dynamic-quant runtime support. No ready named
Q-DiT integration found in the reviewed official entry points. Support for
W4/W8 and generic dynamic input quantization is not proof of Q-DiT's exact
sample-wise/group-wise activation quantization semantics or full reproduction.

Local study status: RTN, NumPy proximal HQQ, per-group MSE clipping and
BLOCK-DIAGONAL GPTQ approximation are independent native-IDST experiments.
The latter uses110 layer block Hessians from20 calibration utterances/two real
steps,32 valid sampled positions per snapshot,1280 rows/layer, no held-out fit.
It omits cross-block correlations/feedback and is NOT original Q-DiT or full
GPTQ. Failure of these approximations must not be used to reject the original.
All5 held-out texts stay separate; one prompt voice is limited coverage.

The block32 MSE sweep is only a granularity pilot, not evolutionary per-layer
allocation. CPU can run offline calibration and optimized MNN inference;
quality tests/packing compatibility and target-device performance are separate.
Per the user's clarification, these PTQ runs use the local PC CPU. No GPU rental
or new cloud PTQ workflow was dispatched. iOS build30 used the existing macOS
build workflow separately; that does not benchmark CPU quantization quality.
