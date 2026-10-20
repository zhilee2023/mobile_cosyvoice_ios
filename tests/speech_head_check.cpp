#include "CosyVoiceSpeechHead.hpp"
#include <cassert>
#include <limits>
#include <iostream>

int main() {
    using namespace CosyVoiceSpeechHead;
    std::vector<float> full(vocabulary, 123.f), small(reducedWidth), a, b;
    for (int i = 0; i < reducedWidth; ++i) full[offset + i] = small[i] = float(i) / 1000;
    restore(full.data(),full.size(),vocabulary,false,a);
    restore(small.data(),small.size(),reducedWidth,true,b);
    assert(a == b && b.size() == vocabulary);
    assert(b[0] == excluded && b[offset-1] == excluded && b[eos+1] == excluded);
    assert(b[offset] == small[0] && b[offset+speechCount-1] == small[speechCount-1]);
    assert(b[offset+speechCount] == excluded && b[eos] == small.back());
    auto rejects = [&](size_t size,int width,bool reduced) {
        bool failed=false;
        try { restore(small.data(),size,width,reduced,b); } catch(const std::runtime_error&) { failed=true; }
        assert(failed);
    };
    rejects(small.size(),vocabulary,true); rejects(small.size()-1,reducedWidth,true);
    rejects(small.size()*2,reducedWidth,true);
    small[0]=std::numeric_limits<float>::quiet_NaN(); rejects(small.size(),reducedWidth,true);
    small[0]=std::numeric_limits<float>::infinity(); rejects(small.size(),reducedWidth,true);
    std::cout << "Speech-head global-ID / EOS / reserved / shape / finite checks PASS\n";
}
