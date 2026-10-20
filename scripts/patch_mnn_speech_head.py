"""Opt-in MNN 3.6.1 sampler adapter; default production configs are unchanged.

Patch exact anchors or fail. The adapter restores global coordinates before
MNN's real mixed sampler; it does not implement an alternative sampler.
"""
from pathlib import Path
import argparse
import shutil

SAMPLE_ANCHOR = "    auto token_id = mSampler->sample(logits);"
SAMPLE_PATCH = '''    // COSYVOICE_SPEECH_HEAD_EXPERIMENT_V1: opt-in only.
    const auto speechHeadMode = mConfig->config_.value("cosyvoice_head_experiment", std::string(""));
    if (!speechHeadMode.empty()) {
        if (speechHeadMode != "full" && speechHeadMode != "reduced") {
            MNN_ERROR("Invalid speech-head experiment mode\\n");
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return 0; // valid global embedding ID, NOT EOS; status stops generation
        }
        const auto info = logits->getInfo();
        std::vector<float> globalLogits;
        if (!info || info->dim.empty() || !CosyVoiceSpeechHead::restore(
                logits->readMap<float>(), info->size, info->dim.back(),
                speechHeadMode == "reduced", globalLogits)) {
            MNN_ERROR("Speech-head shape/finite validation failed\\n");
            mContext->status = LlmStatus::INTERNAL_ERROR;
            return 0;
        }
        logits = _Const(globalLogits.data(), {CosyVoiceSpeechHead::vocabulary}, NHWC, halide_type_of<float>());
    }
''' + SAMPLE_ANCHOR

SEED_ANCHOR = "    mConfig.max_all_tokens = config->max_all_tokens();"
SEED_PATCH = '''    // Isolated A/B uses the same seed. No key => original random_device RNG.
    if (config->config_.contains("cosyvoice_experiment_seed")) {
        mRng.seed(config->config_.value("cosyvoice_experiment_seed", 2307));
    }
''' + SEED_ANCHOR

# reset() is the explicit NEW-request boundary used by our bridge. Rebuild
# only the lightweight sampler; the loaded transformer/module stays alive.
# generate(vector) does not call generate_init(), so patching that is wrong.
RESET_ANCHOR = "    mCachedPromptText.clear();\n}"
RESET_PATCH = RESET_ANCHOR[:-1] + '''
    // COSYVOICE_HEAD_REQUEST_RESET_V1: opt-in; refresh RNG/pipeline, not weights.
    if (!mConfig->config_.value("cosyvoice_head_experiment", std::string("")).empty()
            && mConfig->config_.contains("cosyvoice_experiment_seed")) {
        mSampler.reset(Sampler::createSampler(mContext, mConfig));
    }
}'''


def patch_text(text: str, anchor: str, replacement: str) -> str:
    if text.count(anchor) != 1 or replacement in text:
        raise ValueError("MNN source changed or already patched; refusing ambiguous patch")
    return text.replace(anchor, replacement, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mnn", type=Path)
    args = parser.parse_args()
    src = args.mnn / "transformers/llm/engine/src"
    llm_path, sampler_path = src / "llm.cpp", src / "sampler.cpp"
    llm = patch_text(llm_path.read_text(encoding="utf-8"), SAMPLE_ANCHOR, SAMPLE_PATCH)
    llm = patch_text(llm, RESET_ANCHOR, RESET_PATCH)
    llm = '#include "CosyVoiceSpeechHead.hpp"\n' + llm
    sampler = patch_text(sampler_path.read_text(encoding="utf-8"), SEED_ANCHOR, SEED_PATCH)
    # No writes until every anchor has been validated.
    shutil.copyfile(Path(__file__).resolve().parents[1] / "ios/Bridge/CosyVoiceSpeechHead.hpp",
                    src / "CosyVoiceSpeechHead.hpp")
    llm_path.write_text(llm, encoding="utf-8")
    sampler_path.write_text(sampler, encoding="utf-8")
    print("Applied opt-in speech-head global-ID adapter and experiment-only RNG seed")


if __name__ == "__main__":
    main()
