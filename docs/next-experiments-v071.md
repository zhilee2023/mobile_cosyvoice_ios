# v0.7.1 / build15: one report, isolated threads, next experiments

## What changed in this package

- One user-facing `cosyvoice-test-report.json` export. Module and generation
  results are snapshotted per run, with UUID, app version/build, process identity,
  monotonic duration, state and matching log text. One report retains up to20 runs
  /8 MB; old runs are pruned with a count, source logs are never deleted.
- Logs older than the run start are marked stale, not attached as new evidence.
  On first upgrade, existing files are imported once as uncorrelated historical
  material with unknown original version. Interrupted runs are recovered only
  after a new process launch; never labelled completed or OOM automatically.
- Per-module buttons stay. Flow4/2/3 are separate cold-start requests so4's
  residual allocation cannot block2 inside the same route. Each contains only one
  session, three repeated two-step forwards. Planning gates and800 MB floor stay.
- Independent-thread rows pass finite/replay checks ONLY. The report includes
  raw140-frame Mel and fixture/file stamps; `inspect_test_report.py` compares
  matching4/2/3 results across runs, rejecting mismatched build/model stamps or
  legacy evidence. Stamps are NOT cryptographic hashes. Audio quality unproven.
- Prefill/decode metrics now use per-request before/after counter deltas,
  including engines that reset counters. BF16 first-step failure now records an
  attempt time/0 completed steps, NOT a valid two-step time. Absent warmed runs
  and unavailable Mel L2 are null, not zero. No production model precision change.

## Speed estimate, not a benchmark claim

Latest phone3-sentence run:8.48s audio,15.754s computation,RTF1.858. The three
native-generation intervals sum14.519s (RTF1.712). The difference1.235s includes
discarded resident startup, wrapper recovery and other wrapper overhead; removing
it is a counterfactual calculation, not an already-measured new baseline.

The latest LLM loads were.388/.366/.380s. If one LLM survives and later two loads
are avoided, a rough additional.746s can be saved, giving(14.519-.746)/8.48=1.624.
Cross-module residency/workspace, new waits, thermal drift, sampler/EOS length and
cancel cleanup can erase some savings. Thus1.5–1.7 is a conditional near-term
engineering target for THIS short-sentence workload, not a guaranteed fastest RTF.
If both sessions could safely remain and two recurring Flow resizes(.780+.786s)
also vanished, naive additive arithmetic reaches~1.44, but this combined residency
has NOT passed full-EOS/HiFT/long-text memory checks. Do not promiseRTF<1.
Earlier5.96s single-sentenceRTF1.314 used different text/length/config; it cannot
prove the latest short-sentence pipeline has already reached that rate.

## Prioritized next trials (not implemented in this reporting-only iteration)

1. **Full-chain LLM-only retention, acoustics scheduled sequentially.** Compare
   against serial baseline using the same fixed speech tokens where applicable,
   then real texts with declared sampler/EOS. Reset per-requestKV, retain weights;
   create/destroy Flow, then HiFT, without parallel model compute. First validate
   actual co-resident workspace with full-EOS before minute-long tests; sampled
   floor800 MB, cancellation/failed attempts included. This is still residency:
   serial compute does not imply models cannot overlap in RAM. Android's
   persistent LLM benchmark already reuses a model and calls reset per request.
2. **Small unique Flow bucket vs stage swap.** After independent2/3/4 results,
   compare sole320/352 session reuse with real whole-sentence bounds, against
   recreate-per-sentence.320 allows73 speech tokens/2.92s,352 allows89/3.56s,
   384 allows105/4.20s with prompt174. Never truncate absentEOS. Since measured
   peak barely changed between buckets, do NOT assume smaller time dimension
   makes fullLLM+Flow+HiFT residency safe. Android caches session shape/config;
   reuse its lifecycle idea, not MemoryNormal or OpenCL/Hexagon defaults.
3. **Chunk duration/latency tradeoff.** Short first sentence followed by bounded
   3–4s sentences can amortize load/prompt work. Measure actual speech duration,
   TTFA, queue underruns and pronunciation completeness, not character count
   alone; do not concatenate/truncate token strings to manufacture completion.
4. **True streaming export research.** Official CosyVoice3 has25-token hop,
   prompt-hop alignment, lookahead, token_offset and finalize; its HiFT also has
   explicit cached-Mel/finalize behavior. Export matching streaming semantics
   before comparing first-chunk PCM, join quality and final flush. Current
   offline-prefix PC experiment failed the acoustic equivalence gate; toggling
   playback or splitting WAVs is not a fix. Official current10-stepFlow is not
   the exact public2-step student we ship, so source matching matters.
5. **Lower priority: mixed precision/W8.** BF16 all30720 outputs NaN on first
   forward despite >2.3 GB available; locate first bad op before mixed precision.
   Keep norms/reductions/attention-sensitive operations high only after operator
   evidence, not blanket guessing. W8 needs a valid matched student source/export
   chain; current FP16 conversion attempt was ineffective. No unvalidated W8 or
   BF16 promotion based on a smaller failed-run footprint.

## Primary source inspection (2026-10-19)

- Android LLM reuse/reset and request timing deltas:
  https://github.com/Nian27/CosyVoice3-MNN/blob/main/mnn-jni/CosyVoiceLlmPersistentBenchmark.cpp
- Android cached Flow session/config/shape lifecycle:
  https://github.com/Nian27/CosyVoice3-MNN/blob/main/mnn-jni/CosyVoiceFlowPersistentBenchmark.cpp
- Official CosyVoice3 hop/cache/finalize implementation:
  https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/cli/model.py
- Official lookahead/streaming conditioner andFlow dispatch:
  https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/flow/flow.py

Public pages are mutable. Their mechanisms motivate tests; their benchmarks or
Android hardware-specific speedups are NOT transplanted as iPhone results.
