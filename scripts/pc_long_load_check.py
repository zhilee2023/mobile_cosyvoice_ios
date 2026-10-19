"""CPU shape/finite regression, not an iPhone or GPU benchmark."""
from __future__ import annotations
import gc
import json
from pathlib import Path
import time
import numpy as np
from pc_flow_hift_probe import conditioner, csv_tokens, flow

ROOT = Path(__file__).resolve().parents[1]

def main():
    model = ROOT / "models" / "CosyVoice3"
    prompt = csv_tokens(model / "prompt-speech-tokens.csv")
    real = csv_tokens(ROOT / "testdata" / "pc-real-74-speech-tokens.csv")
    assert len(prompt) == 87 and len(real) == 74
    rows = []
    for frames, target in ((384, real[:70]), (512, real * 2)):
        begin = time.perf_counter()
        mu, condition_seconds = conditioner(model, prompt, target)
        mel, forward_seconds = flow(model, mu, runtime_length=frames, precision="high", memory="low")
        assert mel.shape == (80, 2 * len(target)) and np.isfinite(mel).all()
        rows.append(dict(frames=frames, speechTokens=len(target), shape=list(mel.shape), finite=True,
                         peak=float(np.max(np.abs(mel))), conditionerSeconds=condition_seconds,
                         forwardSeconds=forward_seconds, wallSeconds=time.perf_counter()-begin,
                         repeatedCapacityFixture=frames==512))
        print(json.dumps(rows[-1]), flush=True)
        gc.collect()
    output = dict(scope="Windows CPU High finite/shape regression only; not GPU speed, phone memory, long-text audio or streaming acceptance", rows=rows)
    (ROOT / "reports" / "pc-v079-long-load.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
