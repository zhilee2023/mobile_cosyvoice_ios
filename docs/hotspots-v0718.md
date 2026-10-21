# v0.7.18 / build32 — current W8 vs AWQ CPU hotspots

No new weights; no production precision, memory gate, two-step schedule,
streaming scheduler or residency change. ACE and original models are untouched.

## Actual production backend

All current generation components request CPU: LLM Low, Conditioner High,
Flow High/Memory Low, F0 High, HiFT core Low when selected. DSP source/STFT/
ISTFT execute on CPU as well. Metal experiments remain archived. A compiled
Metal framework is not evidence of GPU execution. The hotspot route rejects
a non-CPU main backend; this does not identify individual GEMM microkernels.

## Phone procedure

1. Cover-install build32. Existing projection W8 pair, AWQ pair and
   `flow-w8-fixtures.f32.bin` are reused, not replaced or renamed.
2. Cool/cold launch; run **投影 W8 · 五条 CPU 热点** alone, foreground.
3. Close/cold launch; run **AWQ W4 · 五条 CPU 热点** alone, foreground.
4. Export the single unified `cosyvoice-test-report.json` once after both.
   Do not select both routes as a back-to-back performance ranking.

Both routes use the projection manifest's identical SHA256-checked fixture,
five cases, exact offsets/schedules, four threads, CPU High/Memory Low, two
steps and all six inputs reuploaded per step. Each selected model is separately
SHA256-verified. AWQ preserves the explicit review and failed PC numeric flag;
missing/wrong assets stop, never fall back. No LLM/HiFT runs, no song/speech WAV.

Per case: one first ordinary run, one warm ordinary run, one callback run.
Warm vs first and callback vs warm target Mel relative L2 must each be <=1e-5.
Cross-W8/AWQ Mel equality is NOT required: quantization intentionally differs.
The same fixture SHA/sections/schedule plus model identity make pairing auditable.

Record input copy, run and read/sync separately, normal/profile wall time,
callback overhead, target Mel hashes, thermal state and sampled memory. Callback
summaries retain top30 operators/top20 types, plus complete type-based category
totals: Scale, layout, convolution/linear, MatMul, normalization, Softmax,
elementwise and other. Totals are non-overlapping; an op's nested packing,
dynamic quantization and GEMM are not individually measured by these callbacks.
`kernelDispatchVerified=false` explicitly prevents claiming a particular ARM
kernel/fallback from a Convolution label alone. No intermediate tensors copied.

Only one Flow/session alive per route. Keep its model buffer while changing
shapes. Existing startup2900MB, after-resize1100MB and 20ms-sampled800MB floor
remain. Each profiled two-step run is bounded to60seconds, metadata to4096 names.
Failure retains completed rows; diagnostic completion is not whole-chain RTF,
AWQ quality approval, minute-long residency or token-streaming acceptance.

Next optimization is chosen only after the paired report: Scale fusion if its
measured cost matters; layout removal if conversions dominate; backend/kernel
instrumentation if convolution/linear remains dominant. Do not attribute the
entire AWQ/W8 time difference to Scale without evidence.
