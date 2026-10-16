#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace CosyVoiceDSP {
constexpr double pi = 3.14159265358979323846;
struct Tables {
    double window[16], cosine[9][16], sine[9][16];
    Tables() {
        for (int i = 0; i < 16; ++i) {
            window[i] = 0.5 - 0.5 * std::cos(2 * pi * i / 16);
            for (int b = 0; b < 9; ++b) {
                cosine[b][i] = std::cos(2 * pi * b * i / 16);
                sine[b][i] = std::sin(2 * pi * b * i / 16);
            }
        }
    }
};
inline const Tables& tables() { static const Tables value; return value; }
inline int reflect(int i, int n) {
    while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
    return i;
}
inline std::vector<float> stft(const std::vector<float>& source, int frames) {
    const auto& t = tables();
    std::vector<float> out(18 * frames);
    for (int f = 0; f < frames; ++f) {
        double windowed[16];
        for (int i = 0; i < 16; ++i) windowed[i] = source[reflect(f * 4 + i - 8, int(source.size()))] * t.window[i];
        for (int b = 0; b < 9; ++b) {
            double re = 0, im = 0;
            for (int i = 0; i < 16; ++i) { re += windowed[i] * t.cosine[b][i]; im -= windowed[i] * t.sine[b][i]; }
            out[b * frames + f] = float(re);
            out[(b + 9) * frames + f] = float(im);
        }
    }
    return out;
}
inline std::vector<float> istft(const std::vector<float>& spectral, int frames, int samples) {
    const auto& t = tables();
    std::vector<double> accumulated((frames - 1) * 4 + 16, 0), envelope(accumulated.size(), 0);
    for (int f = 0; f < frames; ++f) {
        double re[9], im[9];
        for (int b = 0; b < 9; ++b) {
            double magnitude = std::min(std::exp(std::clamp(double(spectral[b * frames + f]), -80.0, 80.0)), 100.0);
            double phase = std::sin(double(spectral[(b + 9) * frames + f]));
            re[b] = magnitude * std::cos(phase); im[b] = magnitude * std::sin(phase);
        }
        for (int i = 0; i < 16; ++i) {
            double value = 0;
            for (int b = 0; b < 9; ++b) {
                double contribution = re[b] * t.cosine[b][i] - im[b] * t.sine[b][i];
                value += (b == 0 || b == 8) ? contribution : 2 * contribution;
            }
            const auto position = f * 4 + i;
            accumulated[position] += value / 16 * t.window[i];
            envelope[position] += t.window[i] * t.window[i];
        }
    }
    std::vector<float> pcm(samples);
    for (int i = 0; i < samples; ++i) {
        int p = i + 8;
        double value = envelope[p] > 1e-11 ? accumulated[p] / envelope[p] : accumulated[p];
        pcm[i] = float(std::clamp(value, -0.99, 0.99));
    }
    return pcm;
}
}
