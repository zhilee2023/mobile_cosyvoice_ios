# Flow attention-projection W8 sensitivity

Based on phone build28 profile: Convolution (includes converted linears) about
75% of observed op time; 149/150 top30 entries across five inputs are attention
Q/K/V/output projections. This does not say attention projections alone are75%
or that accelerating Softmax would address the bottleneck.

## Isolated experiment

- Preserve `C:/ace-step-mnn-ios` and the dirty primary `C:/cosyvoice-ios` checkout.
  Work in existing `builds/phone-w8-v0712` worktree. Read original model/corpus;
  new candidates go in `builds/projection-w8-20261020` without replacement.
- Original Flow graph/weight SHA256 pinned by the repacker. Recreated FFN W8
  must be byte-identical to the current phone control before replay starts.
- All candidates retain44 FFN W8/block64 operators and independently add
  Q22, K22, V22, output22, QKV66 or QKVO88 projections. Preserve all other
  convolution/normalization/embedding weights and every bias exactly.
- Native MNN IDST dense asymmetric min/max W8. Not HQQ, static W8A8, SmoothQuant,
  or an activation-scale fit. The frozen corpus is used for replay/sensitivity.
- Twenty calibration texts, five held-out texts, real complete-EOS LLM speech
  tokens, two original Flow trajectories each. Recheck NPZ/audio hashes,
  original identities, finite values and exact step2/final reference Mel.
- CPU High, four threads, binding memory enum NORMAL0/LOW2 (strings are ignored
  for memory in this MNN Python binding). Ordinary-memory held-out5 isolates
  weight error; low-memory replay20+5 additionally tests dynamic integer error.
- Record teacher-forced velocity error at both times and closed-loop Mel error
  after two real solver steps; copy all six inputs before every forward.
- Fixed pilot screens: max velocity L2<=3%, max Mel L2<=3%, mean Mel L2<=1.5%.
  These are numerical screening thresholds, NOT perceptual/listening acceptance.
- Separate matched timing: five held-out cases, recipe order rotated by case,
  fresh session per case/recipe, first plus two warm closed-loop runs. No teacher
  forward between trials. Windows lacks ARM int8 dot instructions: timing is not
  a phone prediction; allocator MB is not OS footprint. No automatic promotion.
- Optional listening decodes saved held-out Mels through unchanged HiFT High,
  deterministic excitation and original clipping convention; save unnormalized
  WAVs and spectrum/envelope proxies, never label proxies as MOS.

## Commands (approved runtime with existing read-only MNN/numpy site-packages)

```
python scripts/flow_projection_matrix.py --assets C:/cosyvoice-ios/models/CosyVoice3 --corpus C:/cosyvoice-ios/builds/flow-calibration-20261020 --output C:/cosyvoice-ios/builds/projection-w8-20261020
python scripts/audit_projection_candidates.py --root C:/cosyvoice-ios/builds/projection-w8-20261020 --assets C:/cosyvoice-ios/models/CosyVoice3
python scripts/summarize_projection_matrix.py C:/cosyvoice-ios/builds/projection-w8-20261020/projection-report.json
```

Original files and ACE are not modified. Phone UI/config/quantized models are
not switched by a PC numerical pass. Phone candidate packaging requires its own
asset identity manifest, cold-start memory/finite/repeat tests and listening.

## Follow-up design, after individual-group calibration replay

K Low calibration max Mel L2 reached5.20%, whereas Q and output individually
passed the fixed gates. Therefore a separate follow-up keeps K FP16 and tests
Q+output and Q+V+output on top of the same FFN W8 control. It uses a new output
directory and separate matched control timings, not merged cross-sweep speedups.
This remains exploratory on the same small pilot corpus, not an unseen final
quality evaluation or a fitted/calibrated activation quantizer.

```
python scripts/flow_projection_matrix.py --assets C:/cosyvoice-ios/models/CosyVoice3 --corpus C:/cosyvoice-ios/builds/flow-calibration-20261020 --output C:/cosyvoice-ios/builds/projection-w8-followup-20261020 --recipes ffn-qout-w8 ffn-qvout-w8
```
