# v0.7.12 / build26: FFN W8 iPhone probes

CosyVoice3 Fun-CosyVoice3-0.5B-2512, distilled two-step Flow. This build is
isolated from the unbuilt v0.7.11 speech-head experiment and does not patch MNN.
Default generation and the original 17 model files remain unchanged.

## Install and import

Install the unsigned IPA through the same Sideloadly app identity as before;
do not delete the existing app or its model files. Windows import folder:
`C:/cosyvoice-ios/builds/flow-quant-20261020/phone-import-v0712/`.
Add these **three** files to the same Documents/model directory as the original
files (about 587.4 MB additional space):

1. `flow-ffn-w8-g64.mnn`
2. `flow-ffn-w8-g64.mnn.weight`
3. `flow-w8-fixtures.f32.bin`

The small identity manifest is bundled in the IPA. Do not overwrite the old
Flow, import ZIP archives as models, or import the separate LLM speech-head pack.
All five source/candidate/fixture files are size/SHA256 checked before loading.
If the phone has a different original Flow graph, the probe stops explicitly;
do not bypass identity checks or replace the original blindly.

## Three cold-start tests

Keep the app foreground, avoid background music, and let the phone cool between
runs. Force-close/reopen between each button; preserve app documents.

1. `① W8测试 · 原版五条参考`: one original CPU High/Memory Normal session.
   Five frozen independent held-out cases, first + two warm runs each. Save
   independently hashed target Mel files for the same-build iPhone reference.
2. `② FFN W8/block64 · 普通内存`: same inputs, one W8 CPU High/Memory Normal
   session. Requires successful same-version/build/hash reference from step1.
3. `③ FFN W8/block64 · 低内存整数`: still Precision High, Memory Low dynamic
   integer path. This is **not** the old failing whole-Flow Precision Low route.

Each step retains just one model. Later cases resize that session; the small
graph buffer stays valid until all resizes finish. There is no second Flow,
hidden session pool, LLM or HiFT. The 20ms sampler keeps the 800 MB floor and
the pre-forward 1100 MB planning gate. A memory stop is not a numeric pass.

## Share one report

After all three, share only `cosyvoice-test-report.json`. The existing bounded
archive records version/build/process, file and fixture identities, case/shape,
first/warm two-step times, input/run/read split, finite/repeat errors, PC and
independent-phone Mel differences, thermal states, peak footprint and minimum
available memory. Interrupted runs remain marked incomplete on next launch.

Hash verification and model setup are separate from forward timing. Only one
session's load/create time is recorded on each case, not incurred five times;
do not sum that repeated metadata. Flow-only RTF is never whole-chain RTF.

Numeric exploratory ceilings are baseline-vs-PC 0.1%, W8 Normal-vs-phone 1%,
W8 Low-vs-phone 3%, repeated-run 0.1%. These are not MOS/listening acceptance.
No automatic switch to W8 follows a pass. Next gate is whole-chain audible
generation with unchanged LLM and HiFT, then five-sentence memory/RTF testing.
