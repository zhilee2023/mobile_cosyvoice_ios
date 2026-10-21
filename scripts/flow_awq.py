"""CPU AWQ scale-search adaptation for CosyVoice FFN linears.

Algorithm: MNN 3.6.1 AwqQuantizer._compute_best_scale (duo scaling), and
mit-han-lab/llm-awq. This NumPy per-linear adaptation is not the upstream
LLM CLI or a bit-exact reproduction. Clipping is deliberately OFF for the
first scale-search ablation. Real activation rows, not weight-only MSE, fit
scales; inverse activation scaling must also be present in the MNN graph.
"""
from __future__ import annotations
import struct
import numpy as np


def quantize_awq_weight(w, block):
    """AWQ integer zero-point asymmetric W4, with constant groups exact."""
    w = np.asarray(w, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] % block or block not in (32, 64):
        raise ValueError('Expected divisible pointwise-linear weight')
    g = w.reshape(-1, block)
    lo, hi = g.min(axis=1, keepdims=True), g.max(axis=1, keepdims=True)
    step = (hi - lo) / np.float32(15)
    good = step > np.float32(1e-6)
    safe = np.where(good, step, np.float32(1))
    zero = np.rint(lo / safe)
    q = np.clip(np.rint(g / safe) - zero, 0, 15)
    q = np.where(good, q, 0).astype(np.uint8)
    minimum = np.where(good, zero * step, lo).astype(np.float32)
    step = np.where(good, step, 0).astype(np.float32)
    # Match the native IDST decoder's operation order, not merely the ideal
    # (q * scale + minimum) real-valued expression.
    decoded = (q.astype(np.float32) - 8) * step + (minimum + 8 * step)
    return decoded.reshape(w.shape), q, minimum, step


def encode_awq_weight(w, block):
    decoded, q, minimum, step = quantize_awq_weight(w, block)
    flat = q.ravel()
    packed = ((flat[::2] << 4) | flat[1::2]).tobytes()
    header = bytes([2]) + struct.pack('<II', len(q), block) + bytes([16]) + bytes(range(248, 256)) + bytes(range(8))
    alpha = np.concatenate((minimum, step), axis=1).astype('<f4').tobytes()
    return header + packed, alpha, len(q), decoded


def search_scale(w, x, block, grid=20):
    w, x = np.asarray(w, np.float32), np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != w.shape[1] or len(x) < 40:
        raise ValueError('Need bounded real activation rows from both steps')
    if not np.isfinite(w).all() or not np.isfinite(x).all() or grid != 20:
        raise ValueError('Nonfinite input or unreviewed search grid')
    groups = np.abs(w).reshape(-1, block)
    normalized = groups / (groups.max(axis=1, keepdims=True) + np.float32(1e-6))
    w_mean = normalized.reshape(w.shape).mean(axis=0)
    x_mean = np.abs(x).mean(axis=0)
    reference = x @ w.T
    denominator = max(1e-30, float(np.mean(reference.astype(np.float64) ** 2)))

    def loss(scale):
        dq = quantize_awq_weight(w * scale[None, :], block)[0]
        actual = (x / scale[None, :]) @ dq.T
        return float(np.mean((actual - reference).astype(np.float64) ** 2))

    # An explicit unit-scale candidate prevents scale search from doing worse
    # than the same codec's unscaled W4 on the calibration reconstruction loss.
    best = np.ones(w.shape[1], np.float32)
    base_error = best_error = loss(best)
    best_ratio = None
    history = [{'ratio': None, 'loss': base_error}]
    for index in range(grid):
        ratio = index / grid
        scale = np.maximum(
            x_mean ** np.float32(ratio) / (w_mean ** np.float32(1 - ratio) + np.float32(1e-4)),
            np.float32(1e-4))
        scale /= np.sqrt(scale.max() * scale.min())
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Invalid AWQ channel scale')
        error = loss(scale)
        history.append({'ratio': ratio, 'loss': error})
        if error < best_error:
            best, best_error, best_ratio = scale.copy(), error, ratio
    return best, {'block': block, 'ratio': best_ratio,
                  'unscaled_calibration_loss': base_error,
                  'awq_calibration_loss': best_error,
                  'calibration_relative_output_l2': (best_error / denominator) ** .5,
                  'scale_min': float(best.min()), 'scale_max': float(best.max()),
                  'search': history, 'clipping': False, 'unit_scale_control_in_search': True}


def search_clip(w, x, block):
    """AWQ's per-output/per-group activation-output clipping objective.

    Equivalent to mean((Xg @ (Q(clamp(Wg))-Wg))**2), computed using the
    exact group covariance to avoid materializing output-channel/token/group
    broadcast tensors. No cross-group Hessian approximation to a full-layer
    objective is claimed: the upstream clipping objective itself is per group.
    Grid/shrink match the public AWQ implementation: 20 points, max shrink .5.
    """
    w, x = np.asarray(w, np.float32), np.asarray(x, np.float32)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1] or w.shape[1] % block:
        raise ValueError('Invalid AWQ clipping dimensions')
    grouped = w.reshape(w.shape[0], -1, block)
    xg = x.reshape(len(x), -1, block).astype(np.float64)
    covariance = np.einsum('ngb,ngc->gbc', xg, xg, optimize=True) / len(x)
    max_abs = np.max(np.abs(grouped), axis=-1, keepdims=True)

    def losses(candidate):
        dq = quantize_awq_weight(candidate.reshape(w.shape), block)[0].reshape(grouped.shape)
        d = (dq - grouped).astype(np.float64)
        return np.maximum(0, np.einsum('ogb,gbc,ogc->og', d, covariance, d, optimize=True))

    base = best_loss = losses(grouped)
    best_max = max_abs.copy()
    for index in range(1, 10):
        bound = max_abs * np.float32(1 - index / 20)
        candidate_loss = losses(np.clip(grouped, -bound, bound))
        choose = candidate_loss < best_loss
        best_loss = np.where(choose, candidate_loss, best_loss)
        best_max = np.where(choose[..., None], bound, best_max)
    if not np.isfinite(best_max).all():
        raise ValueError('Nonfinite clipping threshold')
    return best_max, {'unclipped_group_output_loss': float(base.mean()),
                      'clipped_group_output_loss': float(best_loss.mean()),
                      'groups_clipped': int(np.sum(best_max < max_abs)),
                      'groups_total': int(best_max.size), 'clip_grid': 20, 'max_shrink': .5}
