#include "CosyVoiceDSP.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>

std::vector<float> referenceStft(const std::vector<float>& source, int frames) {
    std::vector<float> out(18 * frames);
    for (int f = 0; f < frames; ++f) for (int b = 0; b < 9; ++b) {
        double re = 0, im = 0;
        for (int i = 0; i < 16; ++i) {
            double window = 0.5 - 0.5 * std::cos(2.0 * CosyVoiceDSP::pi * i / 16);
            double sample = source[CosyVoiceDSP::reflect(f * 4 + i - 8, int(source.size()))] * window;
            double angle = -2.0 * CosyVoiceDSP::pi * b * i / 16;
            re += sample * std::cos(angle); im += sample * std::sin(angle);
        }
        out[b * frames + f] = float(re); out[(b + 9) * frames + f] = float(im);
    }
    return out;
}
std::vector<float> referenceIstft(const std::vector<float>& spectral, int frames, int samples) {
    std::vector<double> sum((frames - 1) * 4 + 16, 0), envelope(sum.size(), 0);
    for (int f = 0; f < frames; ++f) for (int i = 0; i < 16; ++i) {
        double value = 0;
        for (int b = 0; b < 9; ++b) {
            double magnitude = std::min(std::exp(std::clamp(double(spectral[b * frames + f]), -80.0, 80.0)), 100.0);
            double phase = std::sin(double(spectral[(b + 9) * frames + f]));
            double re = magnitude * std::cos(phase), im = magnitude * std::sin(phase);
            double angle = 2.0 * CosyVoiceDSP::pi * b * i / 16;
            double v = re * std::cos(angle) - im * std::sin(angle);
            value += (b == 0 || b == 8) ? v : 2 * v;
        }
        double window = 0.5 - 0.5 * std::cos(2.0 * CosyVoiceDSP::pi * i / 16);
        sum[f * 4 + i] += value / 16 * window;
        envelope[f * 4 + i] += window * window;
    }
    std::vector<float> out(samples);
    for (int i = 0; i < samples; ++i) {
        int p = i + 8;
        out[i] = float(std::clamp(envelope[p] > 1e-11 ? sum[p] / envelope[p] : sum[p], -0.99, 0.99));
    }
    return out;
}
void compare(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::runtime_error("shape mismatch");
    double maxError = 0, squared = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(b[i])) throw std::runtime_error("non-finite DSP output");
        double e = double(a[i]) - b[i]; maxError = std::max(maxError, std::abs(e)); squared += e * e;
    }
    if (maxError > 1e-5 || std::sqrt(squared / a.size()) > 1e-7) throw std::runtime_error("DSP difference exceeds tolerance");
    std::cout << " samples=" << a.size() << " maxError=" << maxError << " rmsError=" << std::sqrt(squared / a.size()) << '\n';
}
int main() {
    std::mt19937 random(1986); std::uniform_real_distribution<float> uniform(-0.2f, 0.2f);
    for (int samples : {24000, 71520, 143040}) {
        int frames = samples / 4 + 1;
        std::vector<float> source(samples), spectral(18 * frames);
        for (auto& x : source) x = uniform(random);
        for (auto& x : spectral) x = uniform(random);
        auto begin = std::chrono::steady_clock::now();
        auto oldStft = referenceStft(source, frames);
        auto oldPcm = referenceIstft(spectral, frames, samples);
        auto middle = std::chrono::steady_clock::now();
        auto newStft = CosyVoiceDSP::stft(source, frames);
        auto newPcm = CosyVoiceDSP::istft(spectral, frames, samples);
        auto end = std::chrono::steady_clock::now();
        compare(oldStft, newStft); compare(oldPcm, newPcm);
        std::cout << "audioSeconds=" << samples / 24000.0
                  << " baselineDSP=" << std::chrono::duration<double>(middle - begin).count()
                  << " optimizedDSP=" << std::chrono::duration<double>(end - middle).count() << '\n';
    }
    std::cout << "PASS: production DSP header matches v0.4.1 formula; host timings are NOT iPhone timings\n";
}
