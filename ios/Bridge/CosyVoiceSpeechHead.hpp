#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace CosyVoiceSpeechHead {
constexpr int vocabulary = 158720;
constexpr int offset = 151924;
constexpr int speechCount = 6561;
constexpr int reducedWidth = 6563;
constexpr int eos = offset + 6562;
constexpr float excluded = -1.0e30f;

// Restore GLOBAL vocabulary coordinates before the unmodified MNN sampling
// pipeline. History penalties, RNG selection, embedding and EOS retain IDs.
// Both A/B arms restrict support identically. Reserved slot is never enabled.
inline void restore(const float* input, size_t size, int width, bool reduced,
                    std::vector<float>& output) {
    const int expected = reduced ? reducedWidth : vocabulary;
    if (!input || width != expected || size != size_t(expected))
        throw std::runtime_error("Speech-head experiment requires exactly one matching logits row");
    for (size_t i = 0; i < size; ++i)
        if (!std::isfinite(input[i])) throw std::runtime_error("Speech-head logits are not finite");
    output.assign(vocabulary, excluded);
    const int start = reduced ? 0 : offset;
    for (int i = 0; i < speechCount; ++i) output[offset + i] = input[start + i];
    output[eos] = input[start + 6562];
}
}
