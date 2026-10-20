# v0.7.14 / build28 — reduced-head reuse, Flow profiling, W8 Metal High

## Measured basis, not promised gains

Report14 build27: matched five-sentence 370 speech tokens, normal EOS, full vs
reduced compute RTF 1.421 vs 1.247. Reduced run: 18.461s for 14.8s audio;
LLM forward 4.304s, Flow acoustic breakdown 7.652s, HiFT core 2.261s.
The completed pair ran in the same process and is not a cold-start ranking.
The last separate full-head retry was interrupted, not a proven OOM.

ACE was Metal High; its 0.26s run plus 1.58s read/sync is about 1.84s/step,
not 0.26s synchronous inference. Eight steps for 30s is roughly Flow-stage
RTF .49; CosyVoice's current CPU Flow breakdown is about .52. Different graphs,
precision, backends and timings mean this is only an approximate comparison.

## Test order

No new model files. Keep original assets, W8 Flow/fixture and reduced head.
All tests still share one `cosyvoice-test-report.json`; no automatic promotion.

1. Cold start / cool: **B reduced fresh** five sentences.
2. Cold start / cool: **C reduced batch reuse** the same five sentences.
   The LLM is loaded once per batch of at most three, expected reuse flags
   false/true/true/false/true. Each sentence clears KV, re-prefills and resets
   MNN's mixed sampler with seed 2307+request index. No cross-sentence KV or
   LLM/Flow/HiFT co-residency. CPU Flow High/Memory Low stays two steps.
   Compare full complete token lists before attributing timing gains:
   `python scripts/compare_head_e2e.py REPORT.json --reuse`.
   Default A/B fresh routes stay for regression. Each WAV is bounded to its
   latest successful run; C adds `end-to-end-head-reuse.wav`.
3. Cold start: **CPU five-case hotspots**. Each frozen held-out input runs
   first ordinary, warm ordinary, then two steps with MNN CPU op callbacks.
   Only top30 named operators and top20 types are saved, no intermediate
   tensor copies. Callback-vs-ordinary target Mel L2 must be <=1e-5.
   Record callback overhead, thermal state, shapes, setup and normal timings.
   Per-case metadata <=4096 distinct ops, 60s diagnostic run bound, memory
   sample floor800MB / after-resize working-space gate1100MB unchanged.
   This also saves a same-version/build, hash-checked CPU W8 reference for
   the existing natural `zh-question` 384-frame case (3.64s target audio).
4. Cold start: **W8 Metal High384**. One Metal High/Memory Low/NoTune session;
   no CPU session, LLM or HiFT alive. Reject non-Metal main backend. If resize
   leaves <1100MB, record `memory-gate-no-forward`, no unsafe forced forward.
   If safe, three runs/two steps, finite checks, repeat L2 <=1e-5 and CPU
   W8 High/Memory Low target Mel L2 <=1%. CPU dynamic-integer and Metal paths
   differ; passing this probe is NOT listening or end-to-end acceptance.
   Record run+read/sync, not async run alone. Reuse unchanged model files.

MNN3.6.1 exact-anchor engine patch extends `Llm::reset()` only for opt-in head
experiments: recreate the small sampler with latest seed, retain module/weights.
Ordinary generation without the keys stays upstream. App calls set_config(seed)
before reset. `generate(vector)` does not invoke generate_init(), so the reset
hook must not be installed only at generate_init(). No C++ exceptions in engine.

Useful upstream APIs checked: pinned MNN `include/MNN/Interpreter.hpp`
(`runSessionWithCallBackInfo`, Session_Debug) and
`transformers/llm/engine/src/llm.cpp` (`reset`, `set_config`, vector generate).

Not implemented: full-model residency, token streaming, five-minute residency
acceptance, further W4 weights, a guarantee of RTF .85, or default GPU switching.

Safety fix found during API review: MNN BACKENDS session info requires an int
array of length >=1+config count. The old probes passed a scalar pointer; all
current single-config bridge sites now use a two-int buffer, with a regression
check. This fixes an out-of-bounds-write risk, not proof of the earlier crash's
cause. No memory gate has been reduced.
