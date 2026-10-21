# v0.7.17 / build31 — explicit AWQ phone deployment experiment

The user listened to the 4.24-second Chinese original/AWQ pair and explicitly
approved deploying this candidate for a phone trial. Preserve the fact that
the unchanged PC numerical screen FAILED. Listening approval is not a
retrospective numeric pass, full perceptual acceptance, or iPhone RTF/memory
acceptance. The app labels AWQ experimental, and normal generation stays W8.

## Model import

The optional pair is in `C:/cosyvoice-ios/builds/v0.7.17-awq-import/`:

- `flow-awq-ffn-w4-g32-clipped.mnn` — 1,972,992 bytes;
- `flow-awq-ffn-w4-g32-clipped.mnn.weight` — 487,373,388 bytes.

Import **only those two files** into the existing model directory in the
app's file-sharing area. They total 489,346,380 bytes (~489 MB decimal).
Do not overwrite original/W8 files, import a ZIP, uninstall the old app or
delete its Documents. The optional manifest is bundled in the new IPA; the
copy beside the model pair is for inspection, not another required import.
Refresh files: AWQ should show 2/2. Native code verifies full SHA256 before
loading. Wrong/missing files stop clearly; there is no silent W8 fallback.

## Implementation

- Selected Flow: 44 FFN W4/group32 using AWQ duo scaling and activation-aware
  clipping, with 44 explicit inverse Scale operations. Q/V/output projection
  linears (66) stay W8/group64; K and other layers preserve their encoding.
- The listened graph/weight hashes are pinned in `flow-awq-manifest.json`.
  It carries `pcNumericScreenPassed=false`, `userListeningReviewed=true`,
  `phoneValidated=false`, `experimentalOnly=true`, `automaticPromotion=false`.
- CPU High/Memory Low, four threads, unchanged two-step Flow. No GPU change.
- Use the existing pruned W8 LLM output head and per-request seed 2307+i,
  resetting KV/sampler each request. LLM prepares at most three complete-EOS
  token sequences, then is destroyed before Flow/HiFT loading.
- One Flow384 per acoustic batch, with one F0/core pair reused within that
  batch. Release both before the next LLM batch, on failure/cancel/close.
- Existing 800 MB sampled floor, 1,200 MB Flow-stage planning floor and all
  other memory gates remain unchanged. W4 file size does not establish lower
  runtime footprint. Every input is recopied and nonfinite outputs stop.
- AWQ uses the verified recipe through the existing legacy `flowW8` compressed
  Flow flag. Logs/metadata explicitly say W4, not W8, and retain numeric review.
- Completed A/B old W8 routes are archived. W8+HiFT reuse remains available
  as the controlled baseline. Historical implementations/WAVs remain intact.

## Phone protocol

1. Cover-install build31 using the existing signing workflow; do not uninstall.
2. Import the two additional files, refresh, confirm AWQ 2/2.
3. Cool/cold-start and run **W8对照 · 固定五句**.
4. Cool/cold-start again and run **AWQ W4 · 固定五句（实验）**.
5. Optionally edit the text and use **AWQ W4 · 生成当前文字（实验）**.
   This entry preserves the actual editor text, with the same bounded
   sentence planner. Fixed five-sentence buttons intentionally ignore edits.
6. Share the ONE `cosyvoice-test-report.json`; compare same-version, identical
   input/configuration/complete token sequences, lifecycle and stage timings.
   Do not attribute speed differences if speech tokens differ.

Both AWQ entries force HiFT Low and complete-then-play for this first trial;
they do not claim token streaming. The successful AWQ WAV is separately
archived as `end-to-end-awq.wav`, leaving the W8 comparison WAV untouched.
The user's PC listening clip used High HiFT: check HiFT Low phone listening
again. User acceptance of that clip cannot establish all-text stability.

The report includes actual model hashes, 20 ms sampled memory, per-sentence
Flow/HiFT times, LLM prefill/decode, Mel/PCM/token hashes, and failed-PC-screen
review fields. Neither RTF 0.85 nor successful phone deployment is claimed
until real device results arrive. The IPA contains manifests/code, not weights.
