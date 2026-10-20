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
        if (speechHeadMode != "full" && speechHeadMode != "reduced")
            throw std::runtime_error("Unknown speech-head experiment mode");
        const auto info = logits->getInfo();
        if (!info || info->dim.empty()) throw std::runtime_error("Missing speech-head shape");
        std::vector<float> globalLogits;
        CosyVoiceSpeechHead::restore(logits->readMap<float>(), info->size,
            info->dim.back(), speechHeadMode == "reduced", globalLogits);
        logits = _Const(globalLogits.data(), {CosyVoiceSpeechHead::vocabulary}, NHWC, halide_type_of<float>());
    }
''' + SAMPLE_ANCHOR

SEED_ANCHOR = "    mConfig.max_all_tokens = config->max_all_tokens();"
SEED_PATCH = '''    // Isolated A/B uses the same seed. No key => original random_device RNG.
    if (config->config_.contains("cosyvoice_experiment_seed")) {
        mRng.seed(config->config_.value("cosyvoice_experiment_seed", 2307));
    }
''' + SEED_ANCHOR


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
