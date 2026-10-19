# v0.7.0 / build 14: isolated acceleration routes

No new model files. ACE project is not changed. Phone tests are pending; building
an IPA and passing Windows regression do not establish iPhone speed or memory.

## Routes and acceptance

| Route | Controlled variables | Evidence | Does not establish |
| --- | --- | --- | --- |
| Flow buckets | Same 70 real tokens / 314 effective frames, High4; 384→352→320 | First and two warmed two-step forwards, same sliced140-frame Mel, L2≤1e-3, peak/min | 320 capacity for105 tokens; audio quality beyond fixture |
| Flow threads | Same384 graph/input/High; 4→2→3 threads | Same Mel gate, setup/first/warmed timings, peak/min | All components should use winner |
| LLM reuse | Same penalty→greedy request, three independent instances vs one instance reset before each of3 requests | EOS, legal tokens, exact repeated token equality, load/prefill/decode/wall, peak/min | Production stochastic-sampler equality, cross-sentence KV or all-model residency |
| LLM threads | Same penalty→greedy request, 4→2→3 threads; independent instances | Three requests each, same token gate/timing | Different text/stochastic sampler is quality-certified |
| HiFT threads | Same148-frame Mel and source STFT, coreLow;4→2→3 | Spectrum L2≤1e-3, first/warmed/setup, peak/min | F0/DSP/whole-chain acceleration |
| Flow BF16 | Same384 fixture/4threads;High vs explicit Precision_Low_BF16 | Compile enabled, requested precision, actual main backend, finite/Mel L2, timing/memory | All operators BF16; ARM hardware acceleration; whole-chain quality |

Each acoustic variant destroys its sole session BEFORE the next recovery gate.
BF16 backend other than CPU Extension13 is reported unsupported/fallback, not a
BF16 success. Backend13 is still not proof of every operator's precision.
Finite values alone do not pass the numerical gate. Candidates with divergence
remain reported; no automatic production promotion. Flow W8 is BLOCKED on a valid
original student checkpoint/export chain, not substituted by renaming FP16.

Every route writes `last-experiment-<route>.log` and `.json`. The runner writes
`last-experiment-summary.log`. Routes may be run singly (prefer cold start), or
queued. Failure/insufficient recovery is recorded, not silently skipped as passed.
Stop waits for current native route to return. This is not a hard interrupt.

The first PC check exposed that `sampler_type=greedy` bypasses repetition
penalties and repeated one token until the cap (correctly rejected, no forced
EOS). The isolated LLM routes therefore explicitly use mixed `[penalty, greedy]`
with penalty1.15, leaving production `[penalty, topK, topP, temperature]` intact.

## Adaptive generation and memory

Observed third LLM load grew by821 MB; previous1500 MB preload gate could leave
less than800 MB. New resident preload plan:900 MB load allowance +800 MB floor
+100 MB margin =1800 MB. It is conservative, NOT an iOS guarantee.

If the preload plan fails, release Flow before loading LLM and switch this batch
permanently to serial. If a resident call returns an explicitly typed memory
failure, preserve diagnostics, release Flow and retry ONLY that sentence once.
NaN, model/path error, illegal token or missing EOS are NOT retried as memory
errors. Serial recovery requires2800 MB. No second resident pool/recreate loop.
Initial Flow setup itself still must pass; if it cannot, batch stops safely.

The already-resident Flow's post-LLM gate is1200 MB (800 MB floor +400 MB workspace
allowance), distinct from the1800 MB LLM-load planning gate. This is a candidate
policy needing phone validation. Current AND20ms sampled minimum800 MB checks
remain. A protected failed attempt is retained in min-memory and retry records;
successful fallback does not erase it. Wall/compute RTF include setup, waits,
failed attempts and retries. Known1738 MB after Flow initialization will likely
trigger immediate serial fallback: that is honest safety behavior, not a claim
that residency now accelerates that device. Audio is emitted only once after a
successful final sentence result.

## Removed from main UI

Conditioner load-only, playback tone, fixed25-token sample, Metal/NoTune load-only,
known-NaN FlowLow, accepted HiFTHigh/Low pair and short2-decode residency probe.
Original source/previous logs are preserved. High remains available as a HiFT
generation toggle; new Low thread sweep is a different question.

## Primary references

- Android LLM persistent benchmark: https://github.com/Nian27/CosyVoice3-MNN/blob/main/mnn-jni/CosyVoiceLlmPersistentBenchmark.cpp
- MNN3.6.1 CPU precision dispatch: https://github.com/alibaba/MNN/blob/3.6.1/source/backend/cpu/CPUBackend.cpp
- Precision enum: https://github.com/alibaba/MNN/blob/3.6.1/include/MNN/MNNForwardType.h

Borrowed Android lifetime/reset structure, not its OpenCL/Hexagon backend or RAM
assumptions. Three repeated requests are a microbenchmark, not a minute-long
resident stress acceptance. No claim of RTF<1 or true token-streaming is made.

## Local regression evidence (2026-10-19)

- `pc_v070_routes_check.py`: same70 tokens, Mel L2 vs384 =3.03195e-5
  (352),2.61980e-5 (320); six independent/reset requests produced identical42
  speech tokens with EOS. Windows does NOT test ARM BF16/FP16 or phone memory.
- `pc_resident_sentence_check.py --hift-low`: three real-text sentences ended
  normally (15/44/26 tokens this stochastic run); finite non-silent PCM;
  reused first condition after all sentences and fresh-session output L2 both0.
- Cloud tests cover sentence boundaries/capacity, DSP, numerical metrics and
  budget boundaries/typed one-time retry policy. IPA build receipt is kept in
  ignored `builds/`, separate from unmeasured phone performance.
