#pragma once
#include <cstdint>

// Available memory is dynamic, not a universal iOS per-process RAM limit.
namespace CosyVoiceBudget {
constexpr std::uint64_t minimumAvailable = 800000000ULL;
constexpr int promptFrames = 174;
constexpr int residentFrames = 384;
constexpr int speechCapacity = (residentFrames-promptFrames)/2;
constexpr bool accepts(std::uint64_t current, std::uint64_t sampledMinimum) {
    return current >= minimumAvailable && sampledMinimum >= minimumAvailable;
}
constexpr bool fitsSpeech(int tokens) {
    return tokens > 0 && tokens <= speechCapacity;
}
static_assert(speechCapacity == 105 && (residentFrames-promptFrames)%2 == 0);
}
