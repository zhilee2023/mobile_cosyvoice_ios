#pragma once
#include <cstdint>
#include <cmath>
namespace CosyVoiceFlowW8 {
inline bool validShape(int frames,int effective) {
    return frames>=256 && frames<=512 && frames%32==0 && effective>174 && effective<=frames;
}
inline bool validSection(uint64_t offset,uint64_t count,uint64_t bytes,uint64_t expected) {
    return count==expected && count>0 && offset%4==0 && offset<=bytes && count<=(bytes-offset)/4;
}
inline bool validSchedule(double t0,double t1,double d0,double d1) {
    const double midpoint=1-std::cos(3.14159265358979323846/4);
    return std::isfinite(t0)&&std::isfinite(t1)&&std::isfinite(d0)&&std::isfinite(d1)&&
        std::abs(t0)<1e-8&&std::abs(t1-midpoint)<1e-7&&std::abs(d0-midpoint)<1e-7&&std::abs(d1-(1-midpoint))<1e-7;
}
}
