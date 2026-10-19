#include "CosyVoiceBudget.hpp"
#include <iostream>
#include <stdexcept>

void require(bool value) { if(!value)throw std::runtime_error("budget/capacity regression"); }
int main() {
    using namespace CosyVoiceBudget;
    require(accepts(889000000,889000000)); // Latest short-KV probe's observed minimum.
    require(accepts(800000000,800000000));
    require(!accepts(799999999,900000000));
    require(!accepts(900000000,799999999)); // Recovery does not erase a sampled breach.
    require(!accepts(0,0)); // Unknown/zero availability fails closed.
    require(shouldReleaseResident(1738000000));
    require(shouldReleaseResident(1586000000));
    require(!shouldReleaseResident(1800000000));
    require(residentFlowBeforeRun > minimumAvailable);
    require(mayRetrySerial(false,true,true));
    require(!mayRetrySerial(false,false,true)); // NaN/path/EOS errors never qualify.
    require(!mayRetrySerial(true,true,true));
    require(!mayRetrySerial(false,true,false)); // No serial retry loop.
    require(!fitsSpeech(0) && !fitsSpeech(-1));
    require(fitsSpeech(1) && fitsSpeech(105) && !fitsSpeech(106));
    require(promptFrames+2*105==residentFrames);
    require(25+((25-87%25)%25)+3==41); // Current prompt alignment, not naive 28.
    std::cout << "PASS: production 800MB floor, sampled dips, unknown memory and 105-token capacity\n";
}
