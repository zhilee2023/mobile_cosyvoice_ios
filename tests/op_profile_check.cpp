#include "CosyVoiceOpProfile.hpp"
#include <cassert>
#include <limits>
int main() {
    using CosyVoiceOpProfile::Summary;
    Summary s;assert(s.add("MatMul","a",.3));assert(s.add("MatMul","b",.2));assert(s.add("MatMul","a",.1));assert(s.add("Softmax","c",.05));
    assert(s.calls==4);auto ops=s.ranked(false,2);assert(ops.size()==2&&ops[0].name=="a"&&ops[0].calls==2);
    auto types=s.ranked(true,20);assert(types.size()==2&&types[0].type=="MatMul"&&types[0].calls==3);
    assert(std::abs(types[0].seconds-.6)<1e-12);assert(s.ranked(false,0).empty());
    assert(!s.add("bad","bad",std::numeric_limits<double>::quiet_NaN()));assert(!s.valid);
    Summary bounded;for(size_t i=0;i<Summary::limit;++i)assert(bounded.add("type",std::to_string(i),0));
    assert(!bounded.add("type","overflow",0));
    Summary negative;assert(!negative.add("x","y",-1));
}
