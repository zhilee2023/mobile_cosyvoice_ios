"""Windows CPU MNN Conditioner → distilled Flow → HiFT smoke pipeline.

Ported for verification from the public CosyVoice3-MNN Android benchmark math.
An input speech-token CSV may be synthetic, so a WAV alone does not establish
text-to-speech quality or genuine streaming.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import wave

import MNN
import numpy as np


PROMPT_FRAMES = 174
SAMPLE_RATE = 24000
MEL_HOP = 480
FFT_SIZE = 16
FFT_HOP = 4
HARMONICS = 9


def csv_tokens(path: Path) -> list[int]:
    with path.open(newline="", encoding="utf-8") as source:
        return [int(item) for row in csv.reader(source) for item in row if item.strip()]


def model_session(path: Path, precision: str = "high", memory: str = "normal"):
    network = MNN.Interpreter(str(path))
    session = network.createSession({"backend": "CPU", "precision": precision, "memory": memory, "numThread": 4})
    return network, session


def put(network, session, name: str, value: np.ndarray, data_type=MNN.Halide_Type_Float) -> None:
    value = np.ascontiguousarray(value)
    target = network.getSessionInput(session, name)
    host = MNN.Tensor(value.shape, data_type, value, MNN.Tensor_DimensionType_Caffe)
    if not target.copyFrom(host):
        raise RuntimeError(f"MNN input copy failed: {name}")


def get(network, session, name: str) -> np.ndarray:
    return np.asarray(network.getSessionOutput(session, name).getNumpyData(), dtype=np.float32).copy()


def finite(name: str, data: np.ndarray) -> None:
    if not np.isfinite(data).all():
        raise RuntimeError(f"Non-finite {name}")


def conditioner(model_dir: Path, prompt: list[int], target: list[int],
                model_path: Path | None = None) -> tuple[np.ndarray, float]:
    sequence = np.asarray([prompt + target], dtype=np.int32)
    network, session = model_session(model_path or model_dir / "flow-conditioner.fp32.mnn")
    network.resizeTensor(network.getSessionInput(session, "tokens"), tuple(sequence.shape))
    network.resizeSession(session)
    put(network, session, "tokens", sequence, MNN.Halide_Type_Int)
    start = time.perf_counter()
    network.runSession(session)
    mu = get(network, session, "mu").reshape(80, 2 * sequence.size)
    duration = time.perf_counter() - start
    finite("Conditioner mu", mu)
    return mu, duration


def flow(model_dir: Path, mu: np.ndarray, runtime_length: int | None = None,
         model_path: Path | None = None, memory: str = "normal") -> tuple[np.ndarray, float]:
    sequence_length = mu.shape[1]
    if runtime_length is None:
        buckets = (512, 768, 1024, 1280, 1536, 2048)
        runtime_length = next((bucket for bucket in buckets if bucket >= sequence_length), None)
    if runtime_length is None or runtime_length < sequence_length:
        raise RuntimeError(f"Sequence too long for the public Flow buckets: {sequence_length}")
    noise = np.fromfile(model_dir / "rand-noise.bin", dtype="<f4").reshape(80, 15000)
    prompt_cond = np.fromfile(model_dir / "prompt-cond.bin", dtype="<f4").reshape(80, PROMPT_FRAMES)
    spks = np.fromfile(model_dir / "spks.bin", dtype="<f4").reshape(1, 80)
    x = np.zeros((1, 80, runtime_length), dtype=np.float32)
    x[0, :, :sequence_length] = noise[:, :sequence_length]
    mask = np.zeros((1, 1, runtime_length), dtype=np.float32)
    mask[:, :, :sequence_length] = 1
    mu_padded = np.zeros_like(x)
    mu_padded[0, :, :sequence_length] = mu
    cond = np.zeros_like(x)
    cond[0, :, :PROMPT_FRAMES] = prompt_cond
    network, session = model_session(
        model_path or model_dir / "flow.cfg-student-2step.batch1.fp16.mnn", "normal", memory)
    shapes = {"x": x.shape, "mask": mask.shape, "mu": mu_padded.shape, "spks": spks.shape,
              "cond": cond.shape, "t": (1,)}
    for name, shape in shapes.items():
        network.resizeTensor(network.getSessionInput(session, name), tuple(shape))
    network.resizeSession(session)
    midpoint = 1.0 - np.cos(np.pi / 4.0)
    start = time.perf_counter()
    for at_time, delta in ((0.0, midpoint), (midpoint, 1.0 - midpoint)):
        # Refeed every input each step. MNN may overwrite input storage after runSession.
        for name, value in (("x", x), ("mask", mask), ("mu", mu_padded), ("spks", spks), ("cond", cond)):
            put(network, session, name, value)
        put(network, session, "t", np.asarray([at_time], dtype=np.float32))
        network.runSession(session)
        velocity = get(network, session, "estimator_out").reshape(x.shape)
        finite("Flow velocity", velocity)
        x = x + np.float32(delta) * velocity
    duration = time.perf_counter() - start
    finite("Flow mel", x)
    target_mel = x[0, :, PROMPT_FRAMES:sequence_length].copy()
    return target_mel, duration


def noise_table(count: int) -> np.ndarray:
    state = 1986
    values = np.empty(count, dtype=np.float32)
    for index in range(count):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        values[index] = (state >> 8) / 16777216.0
    return values.reshape(-1, HARMONICS)


def excitation(f0: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    f0 = f0.reshape(-1)
    samples = len(f0) * MEL_HOP
    noise = noise_table(samples * HARMONICS)
    source = np.empty(samples, dtype=np.float32)
    phase = np.zeros(HARMONICS, dtype=np.float64)
    harmonics = np.arange(1, HARMONICS + 1, dtype=np.float64)
    for frame, frequency in enumerate(f0):
        voiced = frequency > 10.0
        phase += float(frequency) * harmonics / SAMPLE_RATE * (2 * np.pi * MEL_HOP)
        sine = (0.1 * np.sin(phase)).astype(np.float32) if voiced else np.zeros(HARMONICS, dtype=np.float32)
        amplitude = 0.003 if voiced else 0.1 / 3.0
        segment = noise[frame * MEL_HOP:(frame + 1) * MEL_HOP]
        source[frame * MEL_HOP:(frame + 1) * MEL_HOP] = np.tanh(bias + (segment * amplitude + sine) @ weights)
    return source


def stft(source: np.ndarray) -> np.ndarray:
    window = np.hanning(FFT_SIZE + 1)[:-1].astype(np.float32)
    padded = np.pad(source, (FFT_SIZE // 2, FFT_SIZE // 2), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, FFT_SIZE)[::FFT_HOP]
    spectrum = np.fft.rfft(windows * window, n=FFT_SIZE, axis=1)
    return np.concatenate((spectrum.real.T, spectrum.imag.T)).astype(np.float32)


def istft(spectral: np.ndarray) -> np.ndarray:
    frames = spectral.shape[1]
    magnitude = np.minimum(np.exp(np.clip(spectral[:9], -80, 80)), 100.0)
    phase = np.sin(spectral[9:])
    frequency = magnitude * (np.cos(phase) + 1j * np.sin(phase))
    windows = np.fft.irfft(frequency.T, n=FFT_SIZE, axis=1)
    window = np.hanning(FFT_SIZE + 1)[:-1]
    length = (frames - 1) * FFT_HOP + FFT_SIZE
    accumulated = np.zeros(length, dtype=np.float64)
    envelope = np.zeros(length, dtype=np.float64)
    for frame in range(frames):
        start = frame * FFT_HOP
        accumulated[start:start + FFT_SIZE] += windows[frame] * window
        envelope[start:start + FFT_SIZE] += window * window
    good = envelope > 1e-11
    accumulated[good] /= envelope[good]
    return np.clip(accumulated[FFT_SIZE // 2:FFT_SIZE // 2 + (frames - 1) * FFT_HOP], -0.99, 0.99).astype(np.float32)


def hift(model_dir: Path, mel: np.ndarray, core_model_path: Path | None = None,
         f0_model_path: Path | None = None) -> tuple[np.ndarray, dict]:
    frames = mel.shape[1]
    shaped_mel = mel.reshape(1, 80, frames)
    f0_network, f0_session = model_session(f0_model_path or model_dir / "hift-f0.fp32.mnn")
    f0_network.resizeTensor(f0_network.getSessionInput(f0_session, "mel"), tuple(shaped_mel.shape))
    f0_network.resizeSession(f0_session)
    put(f0_network, f0_session, "mel", shaped_mel)
    start = time.perf_counter()
    f0_network.runSession(f0_session)
    f0 = get(f0_network, f0_session, "f0")
    f0_seconds = time.perf_counter() - start
    finite("HiFT f0", f0)

    weight = np.fromfile(model_dir / "source-linear-weight.bin", dtype="<f4")
    bias = np.fromfile(model_dir / "source-linear-bias.bin", dtype="<f4")
    start = time.perf_counter()
    source = excitation(f0, weight, float(bias[0]))
    source_stft = stft(source).reshape(1, 18, frames * 120 + 1)
    stft_seconds = time.perf_counter() - start
    finite("HiFT source STFT", source_stft)

    core_network, core_session = model_session(core_model_path or model_dir / "hift-core.fp32.mnn")
    for name, shape in (("mel", shaped_mel.shape), ("source_stft", source_stft.shape)):
        core_network.resizeTensor(core_network.getSessionInput(core_session, name), tuple(shape))
    core_network.resizeSession(core_session)
    put(core_network, core_session, "mel", shaped_mel)
    put(core_network, core_session, "source_stft", source_stft)
    start = time.perf_counter()
    core_network.runSession(core_session)
    spectral = get(core_network, core_session, "spectral").reshape(18, frames * 120 + 1)
    core_seconds = time.perf_counter() - start
    finite("HiFT spectral", spectral)
    start = time.perf_counter()
    pcm = istft(spectral)[:frames * MEL_HOP]
    istft_seconds = time.perf_counter() - start
    finite("HiFT PCM", pcm)
    return pcm, {"f0_seconds": f0_seconds, "source_stft_seconds": stft_seconds,
                 "core_seconds": core_seconds, "istft_seconds": istft_seconds}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("target_tokens_csv", type=Path)
    parser.add_argument("output_wav", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--hift-core-model", type=Path,
                        help="Optional HiFT core candidate; the default is the published FP32 model")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    prompt = csv_tokens(model_dir / "prompt-speech-tokens.csv")
    target = csv_tokens(args.target_tokens_csv)
    if not target:
        parser.error("Target speech-token CSV is empty")
    if len(target) > 500:
        parser.error("Start with <=500 target tokens for the PC smoke test")
    started = time.perf_counter()
    mu, conditioner_seconds = conditioner(model_dir, prompt, target)
    print(f"Conditioner: {conditioner_seconds:.3f}s; frames={mu.shape[1]}", flush=True)
    mel, flow_seconds = flow(model_dir, mu)
    print(f"Flow: {flow_seconds:.3f}s; target frames={mel.shape[1]}", flush=True)
    core_model = args.hift_core_model.resolve() if args.hift_core_model else None
    if core_model and not core_model.is_file():
        parser.error(f"HiFT core model does not exist: {core_model}")
    pcm, hift_times = hift(model_dir, mel, core_model_path=core_model)
    duration = len(pcm) / SAMPLE_RATE
    peak = float(np.max(np.abs(pcm)))
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    if rms <= 0.001 or peak > 4.0:
        raise RuntimeError(f"Unhealthy PCM: rms={rms} peak={peak}")
    if peak > 0.98:
        pcm *= np.float32(0.98 / peak)
    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output_wav), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(SAMPLE_RATE)
        destination.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    report = {"test": "pc-cosyvoice-flow-hift", "target_speech_tokens": len(target), "audio_seconds": duration,
              "conditioner_seconds": conditioner_seconds, "flow_seconds": flow_seconds, **hift_times,
              "total_seconds": time.perf_counter() - started, "pcm_peak_before_normalization": peak,
              "pcm_rms": rms, "output_wav": str(args.output_wav.resolve()),
              "hift_core_model": str(core_model or (model_dir / "hift-core.fp32.mnn")),
              "scope_note": "Token source determines whether this is real TTS; not yet a streaming test"}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
