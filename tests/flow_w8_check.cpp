#include "CosyVoiceFlowW8.hpp"
#include <cassert>
#include <limits>
int main() {
    using namespace CosyVoiceFlowW8;
    assert(validShape(384,314));assert(!validShape(383,314));assert(!validShape(384,385));assert(!validShape(384,174));
    assert(validSection(4,8,36,8));assert(!validSection(4,9,36,8));assert(!validSection(3,8,36,8));
    assert(!validSection(std::numeric_limits<uint64_t>::max(),8,36,8));
    assert(!validSection(0,std::numeric_limits<uint64_t>::max(),36,8));
    const double midpoint=1-std::cos(3.14159265358979323846/4);
    assert(validSchedule(0,midpoint,midpoint,1-midpoint));assert(!validSchedule(0,.5,.5,.5));
    assert(!validSchedule(0,std::numeric_limits<double>::quiet_NaN(),midpoint,1-midpoint));
}
