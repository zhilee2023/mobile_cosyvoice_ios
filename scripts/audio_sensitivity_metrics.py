"""Phase-tolerant audio proxies for locally generated quantization WAVs.

These are objective screening metrics, not listening scores or MOS.
"""

from __future__ import annotations

import json
from pathlib import Path
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "builds" / "local-sensitivity"


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError(f"Expected mono 16-bit WAV: {path}")
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float64) / 32768


def spectrum(waveform: np.ndarray, size: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    frames = np.lib.stride_tricks.sliding_window_view(waveform, size)[::hop]
    window = np.hanning(size)
    magnitude = np.abs(np.fft.rfft(frames * window, axis=1))
    energy = np.sqrt(np.mean(frames * frames, axis=1))
    return magnitude, energy


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        raise ValueError("WAV length mismatch")
    output = {}
    for size, hop in ((512, 128), (1024, 256), (2048, 512)):
        x, xe = spectrum(a, size, hop)
        y, ye = spectrum(b, size, hop)
        log_x = np.log1p(x)
        log_y = np.log1p(y)
        output[f"fft_{size}"] = {
            "log_magnitude_correlation": correlation(log_x, log_y),
            "magnitude_relative_l2": float(np.linalg.norm(y - x) / max(np.linalg.norm(x), 1e-12)),
            "frame_rms_correlation": correlation(xe, ye),
        }
    return output


def main() -> int:
    report = {"scope": "Phase-tolerant STFT and envelope screening; subjective audio remains required.",
              "samples": {}}
    for sample in ("zh_short", "zh_long", "en_short"):
        baseline = read_wav(DIRECTORY / f"{sample}-baseline.wav")
        cases = {}
        for label in ("conditioner-w8", "conditioner-w4", "f0-w8", "f0-w4",
                      "core-w8", "core-w4", "core-w4_block32"):
            cases[label] = compare(baseline, read_wav(DIRECTORY / f"{sample}-{label}.wav"))
        report["samples"][sample] = cases
    (DIRECTORY / "audio-metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for sample, cases in report["samples"].items():
        print(sample)
        for label, result in cases.items():
            score = result["fft_1024"]
            print(f"  {label}: logmag_corr={score['log_magnitude_correlation']:.4f} "
                  f"mag_l2={score['magnitude_relative_l2']:.4f} "
                  f"rms_corr={score['frame_rms_correlation']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
