// Included in runExperiment after its report, sampler and prompt are ready.
// New buttons own ONE backend; GPU reads a matched cold-start CPU report.
// Historical raw-GPU routes retain their old sequential-reference behavior.
if ([route hasPrefix:@"flow-metal-low-"] || [route hasPrefix:@"flow-metal-hybrid-"] || [route hasPrefix:@"flow-cpu-reference-"] || [route isEqual:@"flow-metal-nan"] ||
    [route hasPrefix:@"flow-projection-metal-high-"] || [route hasPrefix:@"flow-projection-cpu-reference-"]) {
    const bool projection=[route hasPrefix:@"flow-projection-"];
    const bool diagnose = [route isEqual:@"flow-metal-nan"];
    const bool hybrid = [route hasPrefix:@"flow-metal-hybrid-"];
    const bool cpuOnly = [route hasPrefix:@"flow-cpu-reference-"]||[route hasPrefix:@"flow-projection-cpu-reference-"];
    const bool splitReference = cpuOnly || hybrid || projection;
    NSString *identityFailure=nil;
    NSDictionary *projectionIdentity=projection?verifyFlowProjection(directory,trace,&identityFailure):nil;
    if(projection&&!projectionIdentity)return finish(false,identityFailure?:@"投影W8身份校验失败；不回退旧模型");
    const int frames = [route hasSuffix:@"512"] ? 512 : (diagnose || [route hasSuffix:@"256"] ? 256 : 384);
    const int speechCount = frames == 512 ? 148 : (frames == 256 ? 25 : 70);
    const int targetFrames = 2 * speechCount;
    const int effective = kPromptFrames + targetFrames;
    if(projection&&!cpuOnly&&frames==384) {
        NSData *priorData=[NSData dataWithContentsOfFile:documentPath(@"last-success-flow-projection-metal-high-256.json")];
        id parsed=priorData?[NSJSONSerialization JSONObjectWithData:priorData options:0 error:nil]:nil;
        NSDictionary *prior=[parsed isKindOfClass:NSDictionary.class]?parsed:nil;
        NSArray *priorRows=[prior[@"rows"] isKindOfClass:NSArray.class]?prior[@"rows"]:@[];
        NSDictionary *priorFlow=priorRows.count==2&&[priorRows[0] isKindOfClass:NSDictionary.class]?priorRows[0]:nil;
        NSDictionary *priorComparison=priorRows.count==2&&[priorRows[1] isKindOfClass:NSDictionary.class]?priorRows[1]:nil;
        if(![prior[@"success"] boolValue]||![prior[@"version"] isEqual:[NSBundle.mainBundle objectForInfoDictionaryKey:@"CFBundleShortVersionString"]]||
           ![prior[@"build"] isEqual:[NSBundle.mainBundle objectForInfoDictionaryKey:@"CFBundleVersion"]]||
           ![priorFlow[@"modelIdentity"][@"files"] isEqual:projectionIdentity[@"files"]]||
           ![priorFlow[@"modelIdentity"][@"manifestSHA256"] isEqual:projectionIdentity[@"manifestSHA256"]]||
           [priorFlow[@"completedRuns"] intValue]!=3||[priorFlow[@"backend"] intValue]!=MNN_FORWARD_METAL||
           ![priorFlow[@"requestedPrecision"] isEqual:@"High"]||![priorComparison[@"numericProbePassed"] boolValue])
            return finish(false,@"请先完成本版本投影W8 Metal High 256帧；256失败时不继续384，未加载GPU");
    }
    NSString *flowFile=projection?projectionIdentity[@"files"][@"candidateGraph"][@"file"]:@"flow.cfg-student-2step.batch1.fp16.mnn";
    NSString *flowPath = [directory stringByAppendingPathComponent:flowFile];
    NSDictionary *weights = [NSFileManager.defaultManager attributesOfItemAtPath:[flowPath stringByAppendingString:@".weight"] error:nil];
    if ([weights[NSFileSize] unsignedLongLongValue] == 0)
        return finish(false, @"Flow外部权重缺失或为空，未运行GPU");

    std::vector<int> tokens = prompt;
    // 512 is a REPEATED-token capacity fixture, not a natural long utterance.
    // The full-text sentence routes below are the audible long-text tests.
    for (int i = 0; i < speechCount; ++i) tokens.push_back(kProbeSpeechTokens[i % std::size(kProbeSpeechTokens)]);
    std::vector<float> mu, noise, promptCond, spks;
    {
        Graph conditioner(4, true);
        if (!conditioner.open([directory stringByAppendingPathComponent:@"flow-conditioner.fp32.mnn"], false, trace, @"GPU固定Conditioner") ||
            !conditioner.shape("tokens", {1, int(tokens.size())})) return finish(false, @"Conditioner会话失败");
        conditioner.net->resizeSession(conditioner.session);
        conditioner.releaseModel();
        if (!conditioner.inputInt("tokens", tokens) || !conditioner.run() ||
            !conditioner.output("mu", 80 * effective, mu)) return finish(false, @"Conditioner输出失败");
    }
    if (!readFloats(directory, @"rand-noise.bin", 80 * 15000, noise) ||
        !readFloats(directory, @"prompt-cond.bin", 80 * kPromptFrames, promptCond) ||
        !readFloats(directory, @"spks.bin", 80, spks)) return finish(false, @"固定输入缺失");
    std::vector<float> initial(80 * frames, 0), paddedMu(initial.size(), 0), cond(initial.size(), 0), mask(frames, 0);
    for (int channel = 0; channel < 80; ++channel) {
        for (int frame = 0; frame < effective; ++frame) {
            initial[channel * frames + frame] = noise[channel * 15000 + frame];
            paddedMu[channel * frames + frame] = mu[channel * effective + frame];
        }
        std::copy(promptCond.begin() + channel * kPromptFrames, promptCond.begin() + (channel + 1) * kPromptFrames,
                  cond.begin() + channel * frames);
    }
    std::fill(mask.begin(), mask.begin() + effective, 1.0f);
    noise.clear(); noise.shrink_to_fit(); mu.clear(); mu.shrink_to_fit();
    trace.record([NSString stringWithFormat:@"固定%d个speech token；有效%d帧→%d帧；本次仅一个后端、3轮两步；不运行LLM或HiFT；512帧为74-token重复两次的容量负载，非长句音质验收", speechCount, effective, frames]);
    if (hybrid) trace.record(@"混合候选：Metal Low主干；每层QK+mask+Softmax由CPU FP32重算后回传概率；复制/同步/BLAS均计入两步wall");
    if(projection)trace.record(@"新版110层投影W8：CPU High/Memory Low独立参考，或Metal High/Memory Low/NoTune；非AWQ，无注意力回调，无LLM/HiFT；正式CPU生成不变");

    NSMutableDictionary *stamps = [NSMutableDictionary dictionary];
    for (NSString *name in @[flowFile, [flowFile stringByAppendingString:@".weight"], @"flow-conditioner.fp32.mnn", @"prompt-speech-tokens.csv", @"rand-noise.bin", @"prompt-cond.bin", @"spks.bin"]) {
        NSDictionary *attributes = [NSFileManager.defaultManager attributesOfItemAtPath:[directory stringByAppendingPathComponent:name] error:nil];
        if ([attributes[NSFileSize] unsignedLongLongValue] == 0) return finish(false, @"参考身份检查：模型或输入文件缺失");
        stamps[name] = @{@"bytes": attributes[NSFileSize], @"modifiedEpoch": @([attributes[NSFileModificationDate] timeIntervalSince1970])};
    }
    const auto inputIdentity = CosyVoiceReference::digest(initial, paddedMu, cond, mask, spks);
    NSString *digest = [NSString stringWithUTF8String:inputIdentity.c_str()];
    NSString *fixture = [NSString stringWithFormat:@"%@%d-prompt87-effective%d-noise-v1-f32-native", frames == 512 ? @"repeated74x2-" : @"real", speechCount, effective];
    const float scheduleMidpoint=float(1.0-std::cos(kPi/4.0));
    NSDictionary *schedule=@{@"times":@[@0,@(scheduleMidpoint)],@"deltas":@[@(scheduleMidpoint),@(1.f-scheduleMidpoint)]};
    NSDictionary *savedRow = nil;
    NSString *referenceRoute = [NSString stringWithFormat:projection?@"flow-projection-cpu-reference-%d":@"flow-cpu-reference-%d", frames];
    NSString *referenceFile=[NSString stringWithFormat:projection?@"last-success-%@.json":@"last-experiment-%@.json",referenceRoute];
    if (!cpuOnly && splitReference) {
        NSData *data = [NSData dataWithContentsOfFile:documentPath(referenceFile)];
        id parsed = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
        NSDictionary *saved = [parsed isKindOfClass:NSDictionary.class] ? parsed : nil;
        NSArray *savedRows = [saved[@"rows"] isKindOfClass:NSArray.class] ? saved[@"rows"] : @[];
        if ([saved[@"success"] boolValue] && [saved[@"version"] isEqual:[NSBundle.mainBundle objectForInfoDictionaryKey:@"CFBundleShortVersionString"]] &&
            [saved[@"build"] isEqual:[NSBundle.mainBundle objectForInfoDictionaryKey:@"CFBundleVersion"]] && savedRows.count == 1) {
            NSDictionary *candidate = [savedRows[0] isKindOfClass:NSDictionary.class] ? savedRows[0] : nil;
            if ([candidate[@"fixtureID"] isEqual:fixture] && [candidate[@"inputSHA256"] isEqual:digest] &&
                [candidate[@"modelFileStamps"] isEqual:stamps] && [candidate[@"frames"] intValue] == frames &&
                [candidate[@"backend"] intValue] == 0 && [candidate[@"completedRuns"] intValue] == 3 &&
                [candidate[@"status"] isEqual:@"finite-repeat-pass"] &&
                (!projection||([candidate[@"modelIdentity"][@"files"] isEqual:projectionIdentity[@"files"]]&&
                    [candidate[@"modelIdentity"][@"manifestSHA256"] isEqual:projectionIdentity[@"manifestSHA256"]]&&
                    [candidate[@"schedule"] isEqual:schedule]&&[candidate[@"requestedPrecision"] isEqual:@"High"]&&
                    [candidate[@"memoryMode"] isEqual:@"Low"]&&[candidate[@"cpuThreads"] intValue]==4&&
                    [candidate[@"repeatRelativeL2"] doubleValue]<=1e-5)) &&
                [candidate[@"melShape"] isEqual:@[@80, @(targetFrames)]] &&
                [candidate[@"byteOrder"] isEqual:@"little-endian"] &&
                [candidate[@"melF32Base64"] isKindOfClass:NSString.class]) savedRow = candidate;
        }
        if (!savedRow) return finish(false, [NSString stringWithFormat:@"请先冷启动完成CPU参考%d帧，再冷启动测本GPU项；要求同版本/输入SHA256/文件时间与大小一致，不复用旧参考", frames]);
        if(projection){
            NSData *verifiedMel=[[NSData alloc]initWithBase64EncodedString:savedRow[@"melF32Base64"] options:0];
            if(verifiedMel.length!=80*targetFrames*sizeof(float)||![savedRow[@"melSHA256"] isEqual:dataSHA256(verifiedMel)])
                return finish(false,@"CPU参考Mel长度/哈希不符，未加载GPU");
        }
    }

    std::vector<float> gpuMel, cpuMel;
    auto runBackend = [&](bool metal, std::vector<float> &mel) -> bool {
        NSMutableDictionary *row = [@{@"backendRequested": metal ? @"Metal" : @"CPU", @"requestedPrecision": metal&&!projection ? @"Low" : @"High",
            @"frames": @(frames), @"speechTokens": @(speechCount), @"audioSeconds": @(speechCount / 25.0),
            @"wholeChainRTFMeasured": @NO, @"status": @"started", @"completedRuns": @0,
            @"fixtureID": fixture, @"inputSHA256": digest, @"modelFileStamps": stamps,
            @"fixtureIsRepeatedCapacityLoad": @(frames == 512),
            @"attentionMode": metal && hybrid ? @"CPU FP32 QK-mask-softmax" : @"original graph",
            @"runIncludesAttentionCallbacks": @(metal && hybrid),
            @"thermalStart": @(NSProcessInfo.processInfo.thermalState)} mutableCopy];
        if(projection){row[@"modelIdentity"]=projectionIdentity;row[@"schedule"]=schedule;
            row[@"memoryMode"]=@"Low";row[@"cpuThreads"]=@4;row[@"tuning"]=metal?@"None":@"not-applicable";
            row[@"scope"]=@"fixed speech-token Flow only; not whole-text generation or streaming";
            row[@"numericProbeLimit"]=@0.01;row[@"strictRelativeL2Limit"]=@0.001;row[@"listeningValidated"]=@NO;}
        NSMutableArray *runs = [NSMutableArray array]; row[@"runs"] = runs; [rows addObject:row];
        MemorySampler memory;
        auto finalizeRow = [&](bool ok) {
            memory.stop(trace);
            row[@"peakFootprintMB"] = @(memory.peak.load() / 1e6);
            row[@"minAvailableMB"] = @(memory.available.load() / 1e6);
            row[@"thermalEnd"] = @(NSProcessInfo.processInfo.thermalState);
            return ok;
        };
        // Only stage measurements live here; graph ownership ends before caller
        // writes the final report or starts the other backend.
        bool ok = [&]() -> bool {
            Graph flow(4, true);
            NSString *label = projection?(metal?@"投影W8 Metal High NoTune":@"投影W8 CPU High参考"):(metal ? @"Flow Metal Low NoTune" : @"Flow CPU High参考");
            auto started = std::chrono::steady_clock::now();
            if (!flow.open(flowPath, metal, trace, label, metal&&!projection, false, metal)) { row[@"status"] = @"session-failed"; return false; }
            row[@"loadCreateSeconds"] = @(secondsSince(started));
            int backend = -1;
            backend=flow.mainBackend();
            row[@"backend"] = @(backend);
            trace.record([NSString stringWithFormat:@"%@ 主backend=%d（Metal=1，CPU=0；逐算子可能回退）", label, backend]);
            if ((metal && backend != MNN_FORWARD_METAL) || (!metal && backend != MNN_FORWARD_CPU)) {
                row[@"status"] = @"unexpected-backend"; return false;
            }
            if (!sampledBudgetOK(trace)) { row[@"status"] = @"memory-before-resize"; return false; }
            if (!(flow.shape("x", {1, 80, frames}) && flow.shape("mask", {1, 1, frames}) &&
                  flow.shape("mu", {1, 80, frames}) && flow.shape("spks", {1, 80}) &&
                  flow.shape("cond", {1, 80, frames}) && flow.shape("t", {1}))) { row[@"status"] = @"input-shape-failed"; return false; }
            started = std::chrono::steady_clock::now();
            flow.net->resizeSession(flow.session);
            row[@"footprintAfterResizeMB"]=@(footprintBytes()/1e6);row[@"availableAfterResizeMB"]=@(os_proc_available_memory()/1e6);
            flow.releaseModel();
            row[@"resizeSeconds"] = @(secondsSince(started));
            row[@"footprintAfterReleaseModelMB"]=@(footprintBytes()/1e6);row[@"availableAfterReleaseModelMB"]=@(os_proc_available_memory()/1e6);
            trace.record([NSString stringWithFormat:@"%@ %d帧 resize %.3f秒；前向前规划余量至少1100 MB", label, frames, [row[@"resizeSeconds"] doubleValue]]);
            if (!sampledBudgetOK(trace) || os_proc_available_memory() < 1100000000ULL) {
                row[@"status"] = @"memory-after-resize-no-forward"; trace.memoryFailure = true; return false;
            }
            std::vector<float> firstMel;
            double repeatL2 = 0, warmSeconds = 0;
            const float midpoint = float(1.0 - std::cos(kPi / 4.0));
            for (int run = 0; run < 3; ++run) {
                NSMutableDictionary *runRow = [@{@"index": @(run + 1), @"completedSteps": @0, @"status": @"started"} mutableCopy];
                NSMutableArray *steps = [NSMutableArray array]; runRow[@"steps"] = steps; [runs addObject:runRow];
                auto x = initial;
                auto runStart = std::chrono::steady_clock::now();
                for (int step = 0; step < 2; ++step) {
                    if (!sampledBudgetOK(trace)) { row[@"status"] = @"memory-before-step"; return false; }
                    trace.record([NSString stringWithFormat:@"%@ 第%d轮 %d/2开始；六个输入逐步重传", label, run + 1, step + 1]);
                    auto stepStart = std::chrono::steady_clock::now();
                    std::vector<float> time = {step ? midpoint : 0.f}, velocity;
                    if (!(flow.inputFloat("x", x) && flow.inputFloat("mask", mask) && flow.inputFloat("mu", paddedMu) &&
                          flow.inputFloat("spks", spks) && flow.inputFloat("cond", cond) && flow.inputFloat("t", time))) {
                        row[@"status"] = @"input-copy-failed"; return false;
                    }
                    const double inputSeconds = secondsSince(stepStart);
                    if (!sampledBudgetOK(trace)) { row[@"status"] = @"memory-after-input"; return false; }
                    auto phaseStart = std::chrono::steady_clock::now();
                    if (metal && diagnose) {
                        NSDictionary *diagnostic = diagnoseMetalNaN(flow, trace);
                        row[@"operatorDiagnosis"] = diagnostic;
                        row[@"status"] = diagnostic[@"stopReason"];
                        runRow[@"status"] = diagnostic[@"stopReason"];
                        return false;
                    }
                    NSMutableDictionary *attentionMetrics = [NSMutableDictionary dictionary];
                    bool ran;
                    if (metal && hybrid) {
                        ran = runMetalWithFP32Attention(flow, trace, attentionMetrics);
                        row[@"lastAttentionAttempt"] = attentionMetrics;
                    } else ran = flow.run();
                    if (!ran) { row[@"status"] = metal && hybrid ? @"hybrid-attention-run-failed" : @"run-failed"; return false; }
                    const double runSeconds = secondsSince(phaseStart);
                    phaseStart = std::chrono::steady_clock::now();
                    const bool valid = flow.output("estimator_out", x.size(), velocity);
                    const double readSeconds = secondsSince(phaseStart);
                    NSMutableDictionary *stepRow = [@{@"step": @(step + 1), @"inputSeconds": @(inputSeconds),
                        @"runSeconds": @(runSeconds), @"readSyncSeconds": @(readSeconds), @"finite": @(valid),
                        @"wallSeconds": @(secondsSince(stepStart))} mutableCopy];
                    if (metal && hybrid) stepRow[@"fp32Attention"] = attentionMetrics;
                    [steps addObject:stepRow];
                    trace.record([NSString stringWithFormat:@"%@ 第%d轮 %d/2：input %.4f run %.4f read/sync %.4f秒；有效=%@", label, run + 1, step + 1, inputSeconds, runSeconds, readSeconds, valid ? @"是" : @"否"]);
                    if (!valid) { row[@"status"] = @"nonfinite-or-output-failed"; return false; }
                    const float dt = step ? 1.f - midpoint : midpoint;
                    for (size_t i = 0; i < x.size(); ++i) x[i] += dt * velocity[i];
                    if (!finite(x)) { row[@"status"] = @"nonfinite-latent"; return false; }
                    runRow[@"completedSteps"] = @(step + 1);
                    if (!sampledBudgetOK(trace)) { row[@"status"] = @"memory-after-step"; return false; }
                }
                const double elapsed = secondsSince(runStart);
                mel.resize(80 * targetFrames);
                for (int channel = 0; channel < 80; ++channel)
                    std::copy(x.begin() + channel * frames + kPromptFrames, x.begin() + channel * frames + effective,
                              mel.begin() + channel * targetFrames);
                if (!run) { firstMel = mel; row[@"firstTwoStepsSeconds"] = @(elapsed); }
                else warmSeconds += elapsed;
                const auto replay = CosyVoiceProbe::compare(firstMel, mel);
                if (!replay.valid) { row[@"status"] = @"invalid-or-zero-mel"; return false; }
                repeatL2 = std::max(repeatL2, replay.relativeL2);
                runRow[@"twoStepsSeconds"] = @(elapsed); runRow[@"status"] = @"finite-completed";
                row[@"completedRuns"] = @(run + 1);
                trace.record([NSString stringWithFormat:@"%@ 第%d轮两步完成 %.3f秒；Mel峰值 %.6f；重复L2 %.8g", label, run + 1, elapsed, peak(mel), replay.relativeL2]);
            }
            row[@"warmTwoStepsMeanSeconds"] = @(warmSeconds / 2);
            row[@"repeatRelativeL2"] = @(repeatL2);
            row[@"melShape"] = @[@80, @(targetFrames)];
            row[@"byteOrder"] = @"little-endian";
            row[@"melF32Base64"] = [[NSData dataWithBytes:mel.data() length:mel.size() * sizeof(float)] base64EncodedStringWithOptions:0];
            if(projection)row[@"melSHA256"]=dataSHA256([NSData dataWithBytes:mel.data() length:mel.size()*sizeof(float)]);
            const double repeatLimit=projection?1e-5:1e-3;
            row[@"repeatRelativeL2Limit"]=@(repeatLimit);
            row[@"status"] = repeatL2 <= repeatLimit ? @"finite-repeat-pass" : @"repeat-divergence";
            const double duration = speechCount / 25.0;
            row[@"warmFlowOnlyRTF"] = @(warmSeconds / 2 / duration);
            row[@"coldFlowOnlyRTF"] = @(([row[@"loadCreateSeconds"] doubleValue] + [row[@"resizeSeconds"] doubleValue] + [row[@"firstTwoStepsSeconds"] doubleValue]) / duration);
            return repeatL2 <= repeatLimit;
        }();
        trace.record(metal ? @"本路线唯一Metal会话已释放；无LLM/HiFT共驻" : @"CPU参考会话已释放");
        return finalizeRow(ok);
    };
    if (cpuOnly) {
        const bool ok = runBackend(false, cpuMel);
        return finish(ok, ok ? @"独立CPU参考完成；输入SHA256/文件身份/完整目标Mel与首次预热耗时已保存；请冷启动再测相同帧数GPU，不必共享模型内存" : @"CPU参考未通过；不供GPU数值对照");
    }
    if (!waitAvailable(2900000000ULL, trace, @"Metal独立会话前")) return finish(false, @"请冷启动单独测；未加载GPU");
    const bool gpuOK = runBackend(true, gpuMel);
    if (diagnose) {
        NSDictionary *diagnostic = [rows.lastObject objectForKey:@"operatorDiagnosis"];
        return finish([diagnostic[@"completed"] boolValue], [NSString stringWithFormat:@"GPU算子定位：%@；仅诊断完成状态，不表示GPU数值通过；请分享统一报告", diagnostic[@"stopReason"] ?: @"内存或加载门槛停止"]);
    }
    if (!gpuOK) return finish(false, @"Metal前向、数值或内存门槛未通过；详见本路线分项，无自动CPU替代");
    if (splitReference) {
        NSData *data = [[NSData alloc] initWithBase64EncodedString:savedRow[@"melF32Base64"] options:0];
        if (data.length != 80 * targetFrames * sizeof(float)) return finish(false, @"独立CPU参考Mel长度无效，未放宽检查");
        cpuMel.resize(80 * targetFrames); std::memcpy(cpuMel.data(), data.bytes, data.length);
        if(projection && (![savedRow[@"melSHA256"] isEqual:dataSHA256(data)]||!finite(cpuMel)))
            return finish(false,@"CPU参考Mel哈希或有限值校验失败，未作为数值对照");
        const auto compared = CosyVoiceProbe::compare(cpuMel, gpuMel);
        const bool strict = compared.valid && compared.relativeL2 <= 1e-3;
        const double probeLimit=projection?0.01:0.001;
        const bool probePassed=compared.valid&&compared.relativeL2<=probeLimit;
        NSDictionary *gpuRow = rows.firstObject;
        [rows addObject:@{@"status": strict ? @"strict-numeric-pass" : (projection&&probePassed?@"numeric-probe-pass-quality-pending":@"needs-quality-review"), @"comparison": projection?@"independent same-build projection W8 CPU High/Low-memory vs Metal High/Low-memory":@"independent-cold-start CPU High vs Metal hybrid, target Mel",
            @"inputSHA256": digest, @"referenceRoute": referenceRoute, @"melRelativeL2": compared.valid ? @(compared.relativeL2) : NSNull.null,
            @"melCorrelation": compared.valid ? @(compared.correlation) : NSNull.null, @"melGainDB": compared.valid ? @(compared.gainDB) : NSNull.null,
            @"cpuWarmTwoStepsSeconds": savedRow[@"warmTwoStepsMeanSeconds"], @"gpuWarmTwoStepsSeconds": gpuRow[@"warmTwoStepsMeanSeconds"],
            @"strictRelativeL2Limit": @0.001,@"strictNumericPassed":@(strict),@"numericProbeLimit":@(probeLimit),@"numericProbePassed":@(probePassed), @"listeningValidated": @NO}];
        trace.record([NSString stringWithFormat:@"独立CPU对照L2 %.8g；GPU预热%.3f秒，CPU预热%.3f秒；无需在释放Metal后加载第二套模型；非整链RTF", compared.relativeL2,[gpuRow[@"warmTwoStepsMeanSeconds"] doubleValue],[savedRow[@"warmTwoStepsMeanSeconds"] doubleValue]]);
        if(projection)return finish(probePassed,probePassed?@"投影W8 Metal High模块1%筛查通过；严格1e-3另列，不等于音质/整链通过；正式生成仍CPU":@"投影W8 GPU输出有限，但CPU相对Mel偏差超1%筛查线；未切换正式生成");
        return finish(strict, strict ? @"GPU有限值/重复/独立CPU数值对照通过；模块首次/预热分列，未接入正式生成或验收音质" : @"GPU输出有限，但相对独立CPU参考偏差未过1e-3；需复核音质，未切正式生成");
    }
    // Never overlap CPU and GPU model lifetimes. Residual Metal allocations may
    // take time to retire; preserve GPU data if the reference cannot start.
    if (!waitAvailable(2500000000ULL, trace, @"释放Metal后CPU参考前")) return finish(false, @"GPU有限前向已完成，CPU参考因回收余量不足未运行；尚未完成数值对照");
    if (!runBackend(false, cpuMel)) return finish(false, @"CPU参考失败，GPU不能计为数值对照通过");
    const auto compared = CosyVoiceProbe::compare(cpuMel, gpuMel);
    const bool strict = compared.valid && compared.relativeL2 <= 1e-3;
    [rows addObject:@{@"status": strict ? @"strict-numeric-pass" : @"needs-quality-review", @"comparison": @"Metal Low vs CPU High, target Mel only",
        @"melRelativeL2": compared.valid ? @(compared.relativeL2) : NSNull.null,
        @"melCorrelation": compared.valid ? @(compared.correlation) : NSNull.null,
        @"melGainDB": compared.valid ? @(compared.gainDB) : NSNull.null,
        @"strictRelativeL2Limit": @0.001, @"listeningValidated": @NO}];
    trace.record([NSString stringWithFormat:@"Metal Low相对CPU High目标Mel：L2 %.8g correlation %.8g gain %.5f dB；未运行HiFT、未试听、非整链RTF", compared.relativeL2, compared.correlation, compared.gainDB]);
    return finish(strict, strict ? (hybrid ? @"Metal Low主干+CPU FP32注意力：两步/重复/CPU数值对照/800MB门槛通过；尚未测整链RTF和试听" : @"GPU两步、重复一致性、CPU数值对照与800 MB内存门槛通过；整链速度和音质仍待验证") : @"有限前向完成，但与CPU的Mel差异需复核；未切换正式生成路线");
}
