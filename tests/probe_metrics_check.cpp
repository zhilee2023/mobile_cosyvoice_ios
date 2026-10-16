#include "CosyVoiceProbeMetrics.hpp"
#include <iostream>
#include <stdexcept>
void require(bool pass) { if (!pass) throw std::runtime_error("probe metric regression"); }
int main() {
    std::vector<float> reference = {-0.2f,0.1f,0.3f,-0.1f,0.2f}, scaled=reference;
    auto same=CosyVoiceProbe::compare(reference,reference);
    require(same.valid && same.relativeL2<1e-12 && same.correlation>0.999999 && std::abs(same.gainDB)<1e-12);
    for(auto& x:scaled)x*=1.3f;
    auto gain=CosyVoiceProbe::compare(reference,scaled);
    require(gain.valid && gain.relativeL2>0.29 && gain.relativeL2<0.31 && gain.alignedL2<1e-6);
    require(std::abs(gain.gainDB-20*std::log10(1.3))<1e-5 && gain.correlation>0.999999);
    scaled[0]=std::numeric_limits<float>::quiet_NaN();
    require(!CosyVoiceProbe::compare(reference,scaled).valid);
    require(!CosyVoiceProbe::compare(reference,{}).valid);
    require(!CosyVoiceProbe::compare({0,0},{0,0}).valid);
    auto clipped=CosyVoiceProbe::compare({-0.99f,0.1f,0.99f},{-0.99f,0.1f,0.99f});
    require(clipped.valid && std::abs(clipped.clippedFraction-2.0/3.0)<1e-12);
    std::cout << "PASS: production comparison metrics detect gain, residual, NaN, silence, shape and clipping\n";
}
