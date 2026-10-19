// A bounded experimental bridge: Metal Low keeps model weights; only scaled
// QK + mask + softmax are recomputed on CPU FP32, then bounded probabilities
// are uploaded. Never attempt to recover lost logits from an Inf tensor.
bool runMetalWithFP32Attention(Graph &flow, Trace &trace, NSMutableDictionary *metrics) {
    const std::string qkSuffix = "/attn/MatMul_output_0", maskSuffix = "/attn/Add_2_output_0", softmaxSuffix = "/attn/Softmax_output_0";
    auto suffix = [](const std::string &name, const std::string &ending) {
        return name.size() >= ending.size() && name.compare(name.size()-ending.size(), ending.size(), ending)==0;
    };
    std::vector<float> q, k, mask, probs;
    std::string block;
    int heads = 0, frames = 0, dimension = 0, captures = 0, masks = 0, replacements = 0;
    bool haveQK = false, haveMask = false;
    __strong NSString *failure = nil;
    double reads = 0, compute = 0, syncWrite = 0;
    auto read = [&](MNN::Tensor *tensor, std::vector<float> &values) -> bool {
        if (!tensor || tensor->getType().code != halide_type_float || tensor->getType().bits != 32 ||
            tensor->elementSize() <= 0 || uint64_t(tensor->elementSize()) * 4 > 8 * 1024 * 1024) return false;
        MNN::Tensor host(tensor, MNN::Tensor::CAFFE, true);
        const size_t count = tensor->elementSize();
        if (host.usize() != count*sizeof(float) || !host.host<float>() || !tensor->copyToHostTensor(&host)) return false;
        values.assign(host.host<float>(), host.host<float>() + count);
        return finite(values);
    };
    auto before = [&](const std::vector<MNN::Tensor *> &inputs, const MNN::OperatorInfo *info) -> bool {
        @autoreleasepool {
            if (failure || !info) return true; // End callback stops the session.
            const auto &name = info->name();
            if (suffix(name,qkSuffix)) {
                if (haveQK || haveMask || inputs.size()!=2) { failure=@"注意力QK回调顺序或数量不符"; return true; }
                const auto qs=inputs[0]->shape(), ks=inputs[1]->shape();
                if (qs.size()!=4 || ks.size()!=4 || qs[0]!=1 || ks[0]!=1 || qs[1]!=16 || ks[1]!=16 ||
                    (qs[2]!=256 && qs[2]!=384 && qs[2]!=512) || qs[3]!=64 || ks[2]!=64 || ks[3]!=qs[2]) {
                    failure=@"Q/K形状不符合本固定实验，未猜测布局"; return true;
                }
                heads=qs[1]; frames=qs[2]; dimension=qs[3];
                auto started=std::chrono::steady_clock::now();
                if (!read(inputs[0],q) || !read(inputs[1],k)) { failure=@"Q/K读取失败或上游已有非有限值"; return true; }
                reads+=secondsSince(started); ++captures; haveQK=true;
                block=name.substr(0,name.size()-qkSuffix.size());
            } else if (suffix(name,maskSuffix)) {
                if (!haveQK || haveMask || inputs.size()!=2 || name.substr(0,name.size()-maskSuffix.size())!=block) {
                    failure=@"注意力mask未匹配到同一层QK"; return true;
                }
                const auto ms=inputs[1]->shape();
                if (ms.size()!=4 || ms[0]!=1 || (ms[1]!=1 && ms[1]!=heads) || ms[2]!=frames || ms[3]!=frames) {
                    failure=@"注意力mask形状不匹配"; return true;
                }
                auto started=std::chrono::steady_clock::now();
                if (!read(inputs[1],mask)) { failure=@"mask读取失败或含非有限值"; return true; }
                reads+=secondsSince(started); ++masks; haveMask=true;
            }
            if ((haveQK || haveMask) && !sampledBudgetOK(trace)) failure=@"FP32注意力读取阶段低于800MB";
            return true;
        }
    };
    auto after = [&](const std::vector<MNN::Tensor *> &outputs, const MNN::OperatorInfo *info) -> bool {
        @autoreleasepool {
            if (failure) return false;
            if (!info || !suffix(info->name(),softmaxSuffix)) return true;
            const auto &name=info->name();
            if (!haveQK || !haveMask || outputs.size()!=1 || name.substr(0,name.size()-softmaxSuffix.size())!=block) {
                failure=@"Softmax未匹配到同一层完整QK与mask"; return false;
            }
            if (!sampledBudgetOK(trace)) { failure=@"FP32注意力计算前低于800MB"; return false; }
            auto started=std::chrono::steady_clock::now();
            if (!CosyVoiceAttention::probabilities(q,k,mask,heads,frames,dimension,probs)) {
                failure=@"FP32注意力计算失败，未剪裁或替换为零"; return false;
            }
            compute+=secondsSince(started);
            MNN::Tensor *output=outputs[0];
            if (output->shape()!=std::vector<int>({1,heads,frames,frames}) || output->getType().code!=halide_type_float || output->getType().bits!=32) {
                failure=@"Softmax逻辑输出形状或类型不匹配"; return false;
            }
            MNN::Tensor host(output,MNN::Tensor::CAFFE,true);
            if (!host.host<float>() || host.usize()!=probs.size()*sizeof(float)) { failure=@"概率主机缓冲不匹配"; return false; }
            started=std::chrono::steady_clock::now();
            // First synchronize the queued original Softmax. Its invalid data
            // is discarded; only then overwrite its output for downstream V.
            if (!output->copyToHostTensor(&host)) { failure=@"原Softmax同步失败"; return false; }
            std::memcpy(host.host<float>(),probs.data(),probs.size()*sizeof(float));
            if (!output->copyFromHostTensor(&host)) { failure=@"FP32概率回传失败"; return false; }
            syncWrite+=secondsSince(started); ++replacements; haveQK=false; haveMask=false;
            if (replacements==1 || replacements==22)
                trace.record([NSString stringWithFormat:@"局部FP32注意力已回传%d/22层；层%@；CPU计算累计%.4f秒，读取%.4f秒，同步回传%.4f秒",replacements,[NSString stringWithUTF8String:block.c_str()],compute,reads,syncWrite]);
            return sampledBudgetOK(trace);
        }
    };
    const auto started=std::chrono::steady_clock::now();
    const auto code=flow.net->runSessionWithCallBackInfo(flow.session,before,after,false);
    metrics[@"mode"]=@"Metal Low trunk + CPU FP32 QK-mask-softmax";
    metrics[@"qkCaptures"]=@(captures); metrics[@"maskCaptures"]=@(masks); metrics[@"softmaxReplacements"]=@(replacements);
    metrics[@"readQKMaskSeconds"]=@(reads); metrics[@"fp32ComputeSeconds"]=@(compute); metrics[@"syncWriteSeconds"]=@(syncWrite);
    metrics[@"callbackRunSeconds"]=@(secondsSince(started)); metrics[@"runReturnCode"]=@(int(code));
    metrics[@"blasThreads"]=@"Accelerate system-managed";
    if (failure) { metrics[@"failure"]=failure; trace.record(failure); }
    const bool ok=!failure && code==MNN::NO_ERROR && captures==22 && masks==22 && replacements==22 && !haveQK && !haveMask && sampledBudgetOK(trace);
    metrics[@"completedAll22Layers"]=@(ok);
    return ok;
}
