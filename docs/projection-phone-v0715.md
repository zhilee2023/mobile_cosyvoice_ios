# v0.7.15 / build29 — attention projection W8 phone module pair

## Local evidence (not phone speed or audio acceptance)

Frozen20 calibration +5 validation texts, real complete speech-token outputs,
two Flow steps, original model identities and NPZ/WAV hashes rechecked. All
recipes keep44 FFN W8/block64 linears. CPU High/Memory Low has additional
dynamic-integer error beyond ordinary-memory weight-only evaluation.

| Additional W8 projections | Held-out mean Mel L2, Memory Low | Pilot screen |
| --- | ---: | --- |
| none (FFN control) | 0.803% | pass |
| Q | 0.895% | pass |
| K | 2.188% | fail |
| V | 1.230% | pass |
| output | 0.816% | pass |
| QKV | 2.400% | fail |
| QKVO | 2.556% | fail |
| Q + output | 0.901% | pass |
| Q + V + output | 1.292% | pass |

Screens were fixed before replay: max teacher velocity L2<=3%, max closed-loop
Mel L2<=3%, mean Mel L2<=1.5%. Calibration replay also must pass; finite alone
does not pass. K calibration max Mel5.20%; QKVO6.07%. These are exploratory
numerical limits on one prompt voice, not MOS or unseen final quality tests.

The follow-up excluded K based on calibration results. It reused byte-identical
FFN numerical replay but measured new matched control timings. QVO ordinary-
memory validation mean Mel L2 .391%; Memory Low1.292%. Three-run repeat L2=0.
Follow-up Windows warm total over five cases: FFN7.542s, Q+output7.838s,
QVO7.823s. No demonstrated PC acceleration; do not claim phone RTF gains.
Median allocator estimate1056.65->867.23MB is not OS process footprint.

All unselected external weight regions and every bias verified byte-exact;
original model hashes unchanged. No activation-scale fit or new static W8A8.

## Selected optional phone candidate

FFN + Q/V/output =110 W8 linears; K22 plus59 other convolution/linear operators
remain original FP16-stored weights. Runtime CPU precision remains High; only
Memory Low enables the supported dynamic integer path where applicable.

- `flow-projection-ffn-qvout-w8-g64.mnn`:1,346,768 bytes,
  SHA256 `79b0b28e341aabab0198ef60da4402181f1b4dc9c9ff062e20b60d2712221815`
- `.mnn.weight`:521,986,956 bytes,
  SHA256 `884b43e1fae71b55e2ce31aaf199434da9fb64429047a8eb771e34f903ea9205`
- Import pack `C:/cosyvoice-ios/builds/v0.7.15-projection-import/`, two model
  files total523,333,724 bytes. The JSON is bundled in IPA; do not import it.
- Keep existing FFN W8 graph/weight, `flow-w8-fixtures.f32.bin`, all17 originals
  and both output-head candidates. No replacement or default promotion.

1. Cover-install build29, no app deletion. Add only the two new candidate files.
2. Cool/cold start: **① 投影W8对照 · 原FFN五条参考**.
3. Cool/cold start: **② 注意力投影W8候选 · 五条**.
4. Share only `cosyvoice-test-report.json`.

Both routes are isolated CPU High/Memory Low four-thread two-step Flow modules,
not songs/TTS/whole-chain RTF. One session per route reused over five shapes,
first and two warm runs per case.800MB sampled floor/1100MB post-resize gate
unchanged. Control is the already-tested FFN W8, NOT the original FP16 graph.
Save only a fully passed five-case control. Candidate consumes same-version/
build, same model/fixture hashes and target-Mel SHA256. Record both PC-original
and independent-phone-control errors (<=3%) and repeat L2<=1e-5.
`scripts/compare_projection_phone.py REPORT.json` rejects mismatched/incomplete
pairs; figures are Flow-only. GPU High, CPU hotspot and output-head A/B with
conclusions are archived from buttons; old implementations/results remain.
The existing pruned-head five-sentence end-to-end baseline still uses the OLD
FFN W8 and does not silently use this candidate.

PC listening comparisons render saved Mels with original HiFT High and fixed
excitation, same tokens/text/duration, no level normalization. They isolate Flow
changes; they are not iPhone HiFT Low acceptance. See `listening-checked/`.
