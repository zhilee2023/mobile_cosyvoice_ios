# v0.7.16 / build30: controlled five-sentence A/B/C

Report19 build29 module pair: five matching inputs, 8.06418 -> 6.62064s warm
two-step total, sampled footprint1213.35 ->1004.90MB, max paired Mel L2 .015152.
Both runs in the same process; not independent cold-start statistics or quality
acceptance. Complete pruned-head reuse run is still OLD FFN W8:19.08449s/14.8s,
RTF1.28949. New whole-chain benefit is not measured yet.

## Phone protocol

Cover install build30 (keep app/models). No new model imports: use existing
FFN W8 pair, projection QVO W8 pair, speech-head pair and originals.
Cold start/cool before EACH route; fixed five sentences,4threads,HiFT Low,
pruned speech head, per-request seed2307+i, reset KV/sampler, batch<=3 tokens,
two-step CPU High/Memory Low Flow384. Playback complete-only for all arms.

- A: old FFN W8 + pruned head batch reuse + fresh HiFT per sentence.
- B: same, only Flow becomes FFN/Q/V/out W8; K preserved.
- C: same as B, additionally one F0/core pair per acoustic batch.

No global pool, LLM/Flow/HiFT full residency, or token streaming. C releases
HiFT and Flow before the next LLM batch, on failure, close/cancel, dealloc.
Dynamic HiFT lengths resize only when changed. Raw HiFT model retained until
batch end (MNN releaseModel disallows future resize); memory cost measured.
Inputs recopied every call; source random state/phase still restart identically
to fresh baseline. No hidden fresh retry.800MB sampled floor unchanged.

Completed module tests removed from runnable UI, implementations/history kept.
Unified JSON includes per-sentence token/Mel/float-PCM SHA256, model hashes,
stage times, F0/core creates/resizes/reuses and final batch lifecycle. Save
separate A/B/C successful WAV archives for listening. Not MOS/quality approval.

`scripts/compare_projection_e2e.py REPORT.json` requires same-version/build,
text,parameters,model identities,all5 EOS+seeds+token hashes and B/C equal
Mel+PCM hashes. Reject incomplete/cancelled/mismatched comparisons. Small
floating differences produce review-required, not automatic promotion.

PC dynamic reuse replay: six calls (grow/shrink/repeat/return length), same
saved projection Mels, F0High/coreLow: all F0/spectral/PCM byte-exact to fresh
Windows sessions. This is not iPhone Low-kernel or memory/RTF validation.

## Additional user-requested W4 PTQ study

PC-only RTN vs NumPy HQQ proximal-v2 adaptation, block64. Separate FFN W4
(attention original) and Q/V/out W4 (FFN W8) so no K quantization or layer-group
confounding. Frozen20 calibration +5 held-out replay, no activation fitting;
HQQ optimizes weights only, not GPTQ/AWQ. Native MNN IDST packing audited;
unselected bytes and all biases preserved. Not added to phone default or this
IPA until numerical and listening screening.20+5 one-voice corpus is limited.
References: [MNN IDSTEncoder](https://github.com/alibaba/MNN/blob/3.6.1/source/core/IDSTEncoder.hpp),
[HQQ proximal implementation](https://github.com/mobiusml/hqq/blob/master/hqq/core/optimize.py).
