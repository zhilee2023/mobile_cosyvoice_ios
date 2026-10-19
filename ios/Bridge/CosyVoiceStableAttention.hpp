#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace CosyVoiceAttention {
// Q and K already carry the export's scaling factors. K is [heads,d,n],
// not [heads,n,d]. Keep scores in FP32 until row-wise normalization finishes.
inline bool probabilities(const std::vector<float>& q, const std::vector<float>& k,
                          const std::vector<float>& mask, int heads, int n, int d,
                          std::vector<float>& output) {
    if (heads <= 0 || heads > 32 || n <= 0 || n > 1024 || d <= 0 || d > 256) return false;
    const size_t matrix = size_t(n) * n, features = size_t(heads) * n * d;
    if (q.size() != features || k.size() != features ||
        (mask.size() != matrix && mask.size() != matrix * heads)) return false;
    auto finite = [](const std::vector<float>& values) {
        return std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
    };
    if (!finite(q) || !finite(k) || !finite(mask)) return false;
    output.resize(matrix * heads);
    for (int head = 0; head < heads; ++head) {
        const float *qh = q.data() + size_t(head) * n * d;
        const float *kh = k.data() + size_t(head) * d * n;
        float *scores = output.data() + size_t(head) * matrix;
#if defined(__APPLE__)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, d, 1.0f, qh, d, kh, n, 0.0f, scores, n);
#else
        for (int row = 0; row < n; ++row) for (int column = 0; column < n; ++column) {
            float sum = 0;
            for (int inner = 0; inner < d; ++inner) sum += qh[row * d + inner] * kh[inner * n + column];
            scores[row * n + column] = sum;
        }
#endif
        const float *mh = mask.data() + (mask.size() == matrix ? 0 : size_t(head) * matrix);
        for (int row = 0; row < n; ++row) {
            float *values = scores + row * n;
            float maximum = -std::numeric_limits<float>::infinity();
            for (int column = 0; column < n; ++column) {
                values[column] += mh[row * n + column];
                if (!std::isfinite(values[column])) return false;
                maximum = std::max(maximum, values[column]);
            }
            double sum = 0;
            for (int column = 0; column < n; ++column) {
                values[column] = std::exp(values[column] - maximum);
                sum += values[column];
            }
            if (!std::isfinite(sum) || sum <= 0) return false;
            for (int column = 0; column < n; ++column) values[column] = float(values[column] / sum);
        }
    }
    return true;
}
}
