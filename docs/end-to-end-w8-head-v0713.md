# v0.7.13 / build27 — Flow W8 integration + output-head end-to-end tests

## What changes

- The main long-text and short-WAV UI defaults to FFN W8/block64 Flow,
  CPU Precision High / Memory Low. A toggle and original staged route retain
  baseline generation. The original 17 model assets and ACE project are kept.
- LLM Transformer remains the released mixed W4, default output head W8/full.
  Ordinary text generation does not activate the head adapter or fixed seed.
- Optional full/reduced-head end-to-end buttons both run real LLM, Conditioner,
  W8 Flow two steps, F0 High, selected HiFT core and complete WAV output.
- Previous Flow-only/GPU probes with conclusions are archived from runnable UI.
  Failed Flow-reference retries no longer overwrite the last-success cache.

## Files

Already imported W8 Flow files are reused without another download. Full
generation only needs the two candidate model files, not the frozen fixture BIN.
For the head test, add only these **two optional files** beside the originals:

- `llm-speech-head.mnn`
- `llm-speech-head.mnn.weight`

Local pack: `C:/cosyvoice-ios/builds/v0.7.13-head-import/`, 208,410,842 bytes.
Do not replace original `llm.mnn`, weight, input embedding, tokenizer or config.
The reduced head has 6,563 rows: 6,561 speech, one masked reserved slot, EOS.
The separate approximately 284 MB BF16 input embedding is NOT pruned.

## Fair head pairing

1. Cool the phone, cold-start and run **① 输出头配对A · 完整头整链五句**.
2. Cool/cold-start, run **② 输出头配对B · 裁剪头整链五句**.
3. Share one `cosyvoice-test-report.json`. Audition both saved WAVs if required.

Buttons use the same fixed five texts, all-generated-before-playback mode,
same per-request seed, original MNN mixed penalty/topK/topP/temperature sampler
and repetition penalty 1.15. A fresh LLM is used per request so constructor RNG
seeds reset at the corresponding sentence. Both arms restrict support to the
same speech/EOS set and restore global IDs before the unchanged MNN sampler,
embedding lookup, history penalty and EOS logic. The reserved ID is masked.

This fresh-LLM test schedule differs from the default within-batch LLM reuse:
compare A vs B, not B vs default to attribute pruning gains. W8 Flow and HiFT
configuration stay the same. Natural EOS is required; >105 speech tokens trigger
bounded sentence splitting or explicit failure, not successful truncation.

The unified report includes every complete local speech-token list, text, seed,
EOS/illegal token count, load/prefill/decode times, selected asset identities,
Flow/HiFT stage times, wall and compute RTF, thermal state and 20ms memory.
`scripts/compare_head_e2e.py REPORT.json` only compares matching complete pairs;
token/seed/asset/version mismatches cannot be promoted as equivalent output.

Four bounded latest-success WAVs remain for listening: original Flow,
Flow W8, paired full head, paired reduced head. Copies count in compute time.
No token streaming or RTF 0.85 guarantee; no automatic head promotion.

## Engine/build verification

The patch script validates exact pinned MNN 3.6.1 anchors. It adds an opt-in
global-ID adapter and experiment-only RNG seed. Without the keys the upstream
sampler path is unchanged; default native generation explicitly removes them.
Unexpected logits shape/nonfinite data stops the test, with generate exceptions
reported as failures instead of claiming a completed WAV. CI compiles/tests
the mapping helper, Swift route/planner checks and the unsigned iOS app.

MNN iOS disables C++ exceptions. The engine adapter therefore returns an
explicit failure, sets `LlmStatus::INTERNAL_ERROR`, and returns global ID0
(not EOS). The native caller rejects that status before accepting tokens.
The mapping helper is compiled/tested with `-fno-exceptions` as well.
