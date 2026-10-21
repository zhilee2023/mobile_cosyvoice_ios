# Local official MNN W4 conversion — 2026-10-21

The installed `_tools.cp312-win_amd64.pyd` imports successfully under the
bundled Python 3.12.14 x64, and reports MNN 3.6.1. The historical Windows
application-control failure does not describe the current working environment.
No security policy, interpreter installation or original model was changed.

## Input and converter protocol

- Inputs live under `C:/cosyvoice-ios/builds/flow-official-w4-20261021/`.
- `fp32-source` materializes the original external FP16 convolution values
  exactly as FP32, without changing the graph computation. All five held-out
  cases replay with zero teacher-velocity and closed-loop Mel error under
  CPU High / Memory Low. This is not a 25-case acceptance result.
- The official MNN-to-MNN `addBizCode` path only unpacks the graph. `postTreat`
  reads external parameters from `.__convert_external_data.bin`; it does not
  resolve our source `.weight` automatically. Each conversion therefore uses
  a unique working directory and a hash-verified scratch copy. The converter
  removes only that generated scratch file on normal completion.
- The pinned Windows bool parser uses `stream >> bool` without `boolalpha`.
  Use `--weightQuantAsymmetric=1`, not a bare flag or `=true`; check the
  `Use HQQ to quant weight` log line, not only command-line intent.
- HQQ with `optimizeLevel=0` hit native access violation `0xC0000005`.
  Upstream HQQ reads `Global<modelConfig>`, initialized by `optimizeNet`.
  The subsequent successful invocation uses level 1 and disables C4 fusion;
  this source-level explanation is not a symbolized native-crash diagnosis.
- Official HQQ g32 with level 1 completed, with 191 W4 convolution payloads,
  9,501 operators and 248,353,462 external-weight bytes. This is **all-conv
  W4**, not the intended selective FFN-W4/projection-W8 deployment recipe.
  Successful conversion is not accuracy or phone acceptance.

Early `official-hqq-g32` and `official-hqq-g32-staged` candidates are invalid:
HQQ was disabled, and the first also lacked external input. Keep their logs
only for diagnosis; never import or label them as a valid HQQ deployment.
`official-hqq-g32-verified` failed natively and has no valid output.

## Current screen and next boundary

The all-convolution official minmax g32 and HQQ g32 runs show gross held-out
output errors. They must not replace the phone's accepted selective W8 model.
Do not interpret this as proof that selective official W4 or Q-DiT cannot work.
Before a new phone version, isolate official W4 to selected FFN/projection
layers, retain sensitive conditioning/output layers, audit preserved bytes,
then replay the frozen 20 calibration + 5 validation corpus. Do not loosen the
existing gates or claim a W4 speedup from this x64 PC run.

## Phone baseline (report 20, v0.7.16/build 30)

Same-process five-sentence A/B/C: identical 370 speech tokens and five valid
EOS completions. A compute RTF 1.332634; B projection W8 1.123375; C projection
W8 plus bounded HiFT reuse 1.080535. C has 1,065.421 MB sampled peak footprint
and 2,155.804 MB minimum available memory. B/C Mel and PCM hashes are exact.
This is one sequential same-process set, not an independent cold-start
ranking, listening acceptance, RTF 0.85 achievement or token-level streaming.

Official sources inspected:

- https://github.com/alibaba/MNN/blob/3.6.1/tools/converter/include/cxxopts.hpp
- https://github.com/alibaba/MNN/blob/3.6.1/tools/converter/source/common/cli.cpp
- https://github.com/alibaba/MNN/blob/3.6.1/tools/converter/source/MNN/addBizCode.cpp
- https://github.com/alibaba/MNN/blob/3.6.1/tools/converter/source/common/writeFb.cpp
- https://github.com/alibaba/MNN/blob/3.6.1/tools/converter/source/common/HQQQuantizer.cpp
- https://github.com/alibaba/MNN/blob/3.6.1/tools/converter/source/optimizer/PostConverter.cpp
