# CosyVoice Flow AWQ CPU experiment — 2026-10-21

## Result and deployment boundary

The first local AWQ study is complete. It is a disclosed NumPy/MNN adaptation
of AWQ's activation-aware duo scale search, followed by activation-aware group
clipping. It is **not** the original LLM CLI, a bit-exact upstream reproduction,
an official ready-made CosyVoice AWQ integration, or Q-DiT.

No candidate passed the unchanged quality screen. No phone version was built,
no baseline weights were overwritten, and normal generation stays on W8.

Unified generated report:
`C:/cosyvoice-ios/builds/flow-awq-20261021/cosyvoice-awq-report.json`

All table entries below are the five held-out cases, CPU High / Memory Low.
These are Mel/velocity relative L2 errors, not perceptual quality percentages.

| Recipe | Mean closed-loop Mel L2 | Maximum Mel L2 | Maximum teacher velocity L2 |
| --- | ---: | ---: | ---: |
| Unscaled W4 g32, same AWQ codec | 2.9179% | 3.2566% | 5.6673% |
| AWQ scaling g32, clipping off | 3.6436% | 4.4500% | 5.5332% |
| Unscaled W4 g64, same AWQ codec | 3.7714% | 4.4535% | 6.6902% |
| AWQ scaling g64, clipping off | 3.9905% | 4.5901% | 6.2705% |
| AWQ scaling + clipping g32 | 2.7647% | 3.3613% | 5.2933% |

The existing gates remain mean Mel <=1.5%, maximum Mel <=3%, maximum teacher
velocity <=3%. Clipping improves the g32 mean compared with its unscaled
control, but that is not an acceptance pass. Calibration-layer improvements
do not ensure closed-loop Flow quality. The Normal-memory g32 clipped replay
also fails (mean Mel 2.6017%, maximum Mel 3.0438%, maximum velocity 4.8173%).

Five experimental clipped-g32 WAVs are in the report's `audio` list. They use
the same original High HiFT path as the frozen reference to isolate Flow;
they are not phone HiFT Low acceptance or subjective listening scores.

## Data and reproducibility

- Original graph/weight identity is pinned and checked by SHA256.
- Reuse the frozen 20 calibration texts and five screening texts. The latter
  have been used repeatedly for recipe screening; use a new independent set
  before final acceptance. This corpus covers only one reference voice.
- Capture both true teacher Flow steps for each of the 20 calibration cases.
  Each of 44 FFN linears receives 320 sampled real input rows (eight valid
  positions per step, 40 steps). Padding and held-out inputs never fit scales.
- Callback forward results must match original stored teacher output within
  1e-5 relative L2. Every input is copied again for every run.
- Scaling uses 20 duo-scaling ratios and an additional unit-scale control.
  Choose scales by **activation output reconstruction loss**, not weight MSE.
- Clipping uses the public AWQ per-output/per-group objective, a 20-point
  grid and .5 maximum shrink. Compute its exact group loss using covariance
  to avoid large output-channel/token/group broadcast tensors. A unit test
  compares this with direct activation-output loss. This is not an approximate
  full-layer GPTQ Hessian claim.
- Native IDST W4 payloads use the AWQ integer-zero-point asymmetric codec.
  The toy codec test checks packing/dequantization, and the actual graph is
  replayed through MNN. No CUDA dependency or security-policy change is needed.

## Graph compensation and integrity

Each selected weight becomes `W * s`. Its activation edge is independently
rerouted through an explicit native MNN Scale operation with `1 / s`.
No scaling is incorrectly folded across GELU, adaptive normalization or
residual branches. The first adaptation intentionally does not attempt to
fuse these extra 44 Scale operations; their target-device overhead is unknown.

Keep the verified phone projection-W8 model as the base. Replace only the
44 FFN linears with W4; 66 Q/V/output projection linears stay W8. K and the
remaining layers retain their original encoding. The independent audit checks:

- 192 nonselected external parameter regions are byte-exact;
- all 44 selected biases are byte-exact;
- all 44 FFN payloads declare native W4;
- all inverse scale values and rerouted input edges are exact;
- original operator order, types and output edges remain unchanged, and
  unselected consumers retain their original input edges.

A separate floating-weight scale-only model checks compensation itself.
Its five held-out cases have maximum teacher velocity error 8.7004e-5 and
maximum Mel error 5.0340e-5 (0.005034%). It passes the explicit 1e-4 teacher
control gate; it is not reported as bit-exact or zero error. The tiny native
unit fixture tests one batch; real variable-frame Flow is tested by replay.

The clipped-g32 graph plus weights totals 489,346,380 bytes, compared with
523,333,724 bytes for the projection-W8 base. File size is not runtime memory.
CPU High controls floating compute precision; Memory Low may use dynamic
activation quantization on eligible integer kernels. This experiment does
not claim universal W4A16 semantics. Concurrent PC runs are correctness tests,
not timing rankings, phone RTF measurements or proof of an RTF 0.85 result.

## Tools

Run in the isolated worktree, with the approved Python and existing MNN
site-packages. Use fresh output directories; builders refuse overwrites.

1. `collect_flow_awq_inputs.py`: verified bounded calibration-only capture.
2. `prepare_flow_awq.py fit`: g32/g64 activation-aware scale search.
3. `prepare_flow_awq.py clip`: add clipping using the exact same calibration.
4. `prepare_flow_awq.py build`: native candidate, unit control or floating
   compensation control (`--unit` / `--floating`).
5. `audit_flow_awq.py`: independent parameter and graph edge verification.
6. `evaluate_flow_quant.py`: Low-memory full 20+5 replay; Normal-memory five
   held-out cases. Reports include actual settings and both true Flow steps.
7. `summarize_flow_awq.py --audio`: one report and five explicitly experimental
   listening fixtures. No automatic import or deployment.

Next scope is module-level reconstruction / calibration-guided selective W4
with W8 fallback, and target-kernel-aware activation scaling. Do not loosen
gates, label this a complete Q-DiT reproduction, or promise speed gains.

## Algorithm/schema sources

- https://github.com/alibaba/MNN/blob/3.6.1/transformers/llm/export/utils/awq_quantizer.py
- https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py
- https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_clip.py
- https://github.com/alibaba/MNN/blob/3.6.1/schema/default/MNN.fbs
- https://github.com/alibaba/MNN/blob/3.6.1/schema/default/CaffeOp.fbs
