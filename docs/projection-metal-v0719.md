# v0.7.19 / build33 — current projection W8, Metal High feasibility

Explicit isolated module experiment, not a production GPU switch. Current
110-layer projection W8 (FFN/Q/V/out W8/g64; K preserved), not original FP16,
old44-layer FFN-W8, AWQ or the CPU-attention callback hybrid. Reuse already
imported graph/weight; verify both content SHA256 against the bundled manifest.
No converter, model parameter, CPU generation, streaming or residency changes.

## Test sequence

Cover-install build33; do not uninstall or re-download weights. Run one button
per cool/cold foreground launch, in this order:

1. CPU reference256 — fixed25 real PC speech tokens, target1.00s,224 effective
   frames padded to256; only CPU Flow High/Memory Low,4threads.
2. Metal High256 — same input/model/schedule, Metal High/Memory Low/NoTune.
   If stopped for memory, NaN, mismatch or error, share the unified report now;
   do not proceed to384.
3. CPU reference384 — fixed70 real PC speech tokens,2.80s,314 effective frames.
4. Metal High384 — requires a successful same-build/hash256 probe plus matching
   CPU384 reference; otherwise returns before loading GPU.

Fixed speech-token prefixes are capacity/numeric probes, not complete natural
sentences, generated speech WAVs, LLM decode, HiFT or whole-chain RTF.

Each route owns one Flow, no CPU/GPU model overlap. Conditioner runs briefly on
CPU and is destroyed before Flow. Three runs, each resetting initial latent and
running two steps; upload all six inputs each step. Run+read/sync counts as wall
time; async runSession alone is not GPU inference latency. Record first/warm
times, per-step input/run/read, initialization/resize, thermal state and20ms
sampled memory. Main backend must match CPU/Metal; this does not prove every op
or weight stays in that backend, and CPU fallback remains declared.

## Reference integrity and gates

- Reference requires same app version/build, selected model and manifest SHA,
  actual padded input SHA (including lengths), exact schedule, model/input file
  stamps, shape and finite three-run repeat. Store complete target Mel+SHA.
- A successful CPU reference is cached atomically; failed retries do not replace
  its successful cache. GPU validates cached Mel bytes/SHA before its load,
  checks finite values after decoding, compares only after GPU session release.
- CPU/GPU repeat relative Mel L2 <=1e-5. CPU-vs-Metal module screening <=1%, as
  in the earlier quantized Metal High probe; strict <=1e-3 is separately recorded.
  Passing1% is NOT strict numeric or listening acceptance. CPU dynamic-integer
  activations and Metal kernels need not agree bit-exactly. No zeroing NaNs,
  logit clipping, relaxed thresholds after a failure, or automatic promotion.
- Startup2900MB, after-resize1100MB planning and sampled/current800MB floor stay
  unchanged. Record footprint/available immediately after resize and after raw
  graph release, plus sampled route peak. A protection stop is not proven OOM;
  sampling cannot guarantee capturing every transient allocation.

All routes/results/source logs join the one `cosyvoice-test-report.json` export.
Build32 W8/AWQ CPU hotspot routes are archived after two consistent diagnostic
runs (callback Mel L2=0; W8 K projection vs AWQ FFN hotspots; Scale~2.5%). Their
code and old results remain. New buttons do not restore concluded oldGPU tests.

Next decision uses actual safe forwards and CPU differences, not resize memory
alone. If current projectionW8 High still cannot fit, stop and research GPU-only
FP32 QK/Softmax with low-precision trunk; no reintroduction of per-layer CPU
roundtrips or unverified causal Flash Attention into noncausal Flow.
