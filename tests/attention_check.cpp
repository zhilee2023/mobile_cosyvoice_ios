#include "CosyVoiceStableAttention.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
void require(bool ok) { if (!ok) throw std::runtime_error("FP32 attention regression"); }
int main() {
    // Both logits exceed FP16's 65504, with a small, meaningful difference.
    // A softmax applied only AFTER half overflow cannot recover this ratio.
    std::vector<float> q = {256,1,256,1, 256,1,256,1};
    std::vector<float> k = {256,256,0,2, 256,256,2,0};
    std::vector<float> mask(4,0), out;
    require(CosyVoiceAttention::probabilities(q,k,mask,2,2,2,out));
    const float small = 1.f / (1.f + std::exp(2.f));
    require(out.size()==8 && std::abs(out[0]-small)<1e-6 && std::abs(out[1]-(1-small))<1e-6);
    require(std::abs(out[4]-(1-small))<1e-6 && std::abs(out[5]-small)<1e-6);
    for (size_t i=0;i<out.size();i+=2) require(std::isfinite(out[i]) && std::abs(out[i]+out[i+1]-1)<1e-6);
    mask={0,-10000,0,-10000};
    require(CosyVoiceAttention::probabilities(q,k,mask,2,2,2,out));
    for (size_t i=0;i<out.size();i+=2) require(out[i]==1 && out[i+1]==0);
    // Independent per-head masks, zeros and invalid-input handling.
    mask={0,-10000,0,-10000,-10000,0,-10000,0};
    require(CosyVoiceAttention::probabilities(q,k,mask,2,2,2,out));
    require(out[0]==1 && out[4]==0 && out[5]==1);
    require(CosyVoiceAttention::probabilities(std::vector<float>(4,0),std::vector<float>(4,0),std::vector<float>(4,0),1,2,2,out));
    require(out[0]==0.5f && out[1]==0.5f);
    q[0]=std::numeric_limits<float>::quiet_NaN();
    require(!CosyVoiceAttention::probabilities(q,k,mask,2,2,2,out));
    require(!CosyVoiceAttention::probabilities({},k,mask,2,2,2,out));
    require(CosyVoiceAttention::probabilities(std::vector<float>(512*2,0),std::vector<float>(512*2,0),std::vector<float>(512*512,0),1,512,2,out));
    require(out.size()==512*512 && out.front()==1.f/512 && out.back()==1.f/512);
    require(!CosyVoiceAttention::probabilities({}, {}, {}, 16, 1025, 64, out));
    std::cout << "PASS: FP32 large-logit attention, stable normalization, layout and mask broadcasting\n";
}
