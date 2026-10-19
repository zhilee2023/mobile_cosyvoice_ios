#pragma once
#include <CommonCrypto/CommonDigest.h>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace CosyVoiceReference {
// Hash every actual padded input, including lengths. Model stamps separately
// guard local file replacement; they are identity hints, not weight hashes.
inline std::string digest(const std::vector<float>& x, const std::vector<float>& mu,
                          const std::vector<float>& cond, const std::vector<float>& mask,
                          const std::vector<float>& spks) {
    CC_SHA256_CTX context; CC_SHA256_Init(&context);
    for (const auto* input : {&x, &mu, &cond, &mask, &spks}) {
        const uint64_t bytes = input->size() * sizeof(float);
        CC_SHA256_Update(&context, &bytes, sizeof(bytes));
        CC_SHA256_Update(&context, input->data(), CC_LONG(bytes));
    }
    unsigned char hash[CC_SHA256_DIGEST_LENGTH]; CC_SHA256_Final(hash, &context);
    const char* digits = "0123456789abcdef";
    std::string output;
    for (unsigned char value : hash) { output += digits[value >> 4]; output += digits[value & 15]; }
    return output;
}
}
