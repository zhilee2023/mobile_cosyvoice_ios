#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
namespace CosyVoiceProbe {
struct Comparison {
    bool valid = false;
    double relativeL2 = 0, correlation = 0, gainDB = 0, alignedL2 = 0;
    double referenceRms = 0, candidateRms = 0, clippedFraction = 0;
};
inline Comparison compare(const std::vector<float>& reference, const std::vector<float>& candidate) {
    Comparison result;
    if (reference.empty() || reference.size() != candidate.size()) return result;
    double rr = 0, cc = 0, rc = 0, difference = 0, rsum = 0, csum = 0;
    size_t clipped = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        double r = reference[i], c = candidate[i];
        if (!std::isfinite(r) || !std::isfinite(c)) return result;
        rr += r * r; cc += c * c; rc += r * c;
        difference += (r-c)*(r-c); rsum += r; csum += c;
        clipped += std::abs(c) >= 0.9899;
    }
    if (rr <= 1e-20 || cc <= 1e-20) return result;
    const double n = double(reference.size());
    result.referenceRms = std::sqrt(rr/n); result.candidateRms = std::sqrt(cc/n);
    result.relativeL2 = std::sqrt(difference/rr);
    result.gainDB = 10 * std::log10(cc/rr);
    double centeredR = rr - rsum*rsum/n, centeredC = cc - csum*csum/n;
    result.correlation = centeredR > 1e-20 && centeredC > 1e-20
        ? std::clamp((rc-rsum*csum/n)/std::sqrt(centeredR*centeredC), -1.0, 1.0) : 0;
    // Residual after fitting candidate ~= scale * reference. Gain is reported
    // separately; the saved listening WAV is NOT gain-normalized.
    double residual = 0, scale = rc/rr;
    for (size_t i=0;i<reference.size();++i) {
        double delta = candidate[i] - scale*reference[i]; residual += delta*delta;
    }
    result.alignedL2 = std::sqrt(residual/cc);
    result.clippedFraction = double(clipped)/n;
    result.valid = true;
    return result;
}
}
