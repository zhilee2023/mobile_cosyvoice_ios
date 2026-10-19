#pragma once
#include <cstdint>

// Available memory is dynamic, not a universal iOS per-process RAM limit.
namespace CosyVoiceBudget {
constexpr std::uint64_t minimumAvailable = 800000000ULL;
// Observed LLM load growth reached 821 MB. Reserve 900 MB for loading,
// plus the unchanged 800 MB floor and 100 MB planning margin.
constexpr std::uint64_t residentLlmBeforeLoad = 1800000000ULL;
constexpr std::uint64_t residentFlowBeforeRun = 1200000000ULL;
constexpr std::uint64_t reusableLlmBeforeLoad = 1700000000ULL;
// Flow High observed resize growth ~1.3GB; reserve 1.5GB plus 800MB floor.
// Planning only: final acceptance still requires actual 20ms measurements.
constexpr std::uint64_t flowWithLlmBeforeResize = 2300000000ULL;
constexpr bool shouldReleaseResident(std::uint64_t available) {
    return available < residentLlmBeforeLoad;
}
constexpr bool mayRetrySerial(bool succeeded, bool memoryFailure, bool hasResident) {
    return !succeeded && memoryFailure && hasResident;
}
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
