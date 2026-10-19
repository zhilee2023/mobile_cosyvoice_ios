// Included inside CosyVoiceGenerate's implementation. One route at a time;
// every acoustic variant has a fresh session, never a hidden session pool.
+ (NSDictionary<NSString *, id> *)runExperiment:(NSString *)route atDirectory:(NSString *)directory {
    @autoreleasepool {
        NSSet *allowed = [NSSet setWithArray:@[@"flow-buckets",@"flow-threads",@"flow-thread-4",@"flow-thread-2",@"flow-thread-3",@"flow-bf16",@"llm-reuse",@"llm-threads",@"hift-threads",@"flow-metal-low-256",@"flow-metal-low-384",@"flow-metal-nan",@"flow-metal-hybrid-256",@"flow-metal-hybrid-384"]];
        if (![allowed containsObject:route]) return @{@"success":@NO,@"message":@"未知试验路线"};
        Trace trace([NSString stringWithFormat:@"last-experiment-%@.log",route]);
        MemorySampler allMemory; trace.sampler=&allMemory;
        NSMutableArray *rows=[NSMutableArray array];
        trace.record([NSString stringWithFormat:@"路线 %@；v%@；20ms采样；最低余量800 MB；相同输入，首次/预热分开；不等于整链RTF",route,[NSBundle.mainBundle objectForInfoDictionaryKey:@"CFBundleShortVersionString"]],true);
        bool singleThread = [route hasPrefix:@"flow-thread-"];
        auto finish = [&](bool ok, NSString *message) -> NSDictionary * {
            trace.record(message);
            NSMutableDictionary *summary=[result(ok,message,trace) mutableCopy];
            summary[@"route"]=route; summary[@"rows"]=rows;
            summary[@"wholeChainRTFMeasured"]=@NO;
            summary[@"automaticPromotion"]=@NO;
            summary[@"diagnosticOnly"]=@([route isEqual:@"flow-metal-nan"]);
            if ([route isEqual:@"flow-metal-nan"]) summary[@"numericalInferencePassed"]=@NO;
            summary[@"version"]=[NSBundle.mainBundle objectForInfoDictionaryKey:@"CFBundleShortVersionString"]?:@"unknown";
            summary[@"build"]=[NSBundle.mainBundle objectForInfoDictionaryKey:@"CFBundleVersion"]?:@"unknown";
            summary[@"crossThreadComparisonPending"]=@(singleThread);
            summary[@"wallSeconds"]=@(secondsSince(trace.start));
            NSData *json=[NSJSONSerialization dataWithJSONObject:summary options:NSJSONWritingPrettyPrinted error:nil];
            NSString *path=documentPath([NSString stringWithFormat:@"last-experiment-%@.json",route]);
            if (!json || ![json writeToFile:path atomically:YES]) {
                summary[@"success"]=@NO; summary[@"message"]=@"试验已返回，但JSON报告写入失败，请分享文本日志";
            }
            return summary;
        };
        if(!waitAvailable(2900000000ULL,trace,@"独立路线启动前"))return finish(false,@"初始余量不足，请冷启动后单独测本路线");
        std::vector<int> prompt;
        NSString *csv=[NSString stringWithContentsOfFile:[directory stringByAppendingPathComponent:@"prompt-speech-tokens.csv"] encoding:NSUTF8StringEncoding error:nil];
        for(NSString *item in [csv componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@",\n\r "]])if(item.length)prompt.push_back(item.intValue);
        if(prompt.size()!=87)return finish(false,@"提示token必须为87个");

#include "CosyVoiceMetalExperiment.inc"

        if([route hasPrefix:@"llm-"]) {
            // Controlled penalty + greedy benchmark, NOT production sampling.
            // Same short request repeated; exact token equality checks reset/KV
            // isolation. This does not validate all-model residency or audio.
            bool reuse=[route isEqual:@"llm-reuse"];
            std::vector<int> reference;
            double freshTotal=0, reusedTotal=0;
            std::vector<int> modes=reuse?std::vector<int>{0,1}:std::vector<int>{4,2,3};
            for(int mode:modes) {
                if(!waitAvailable(2800000000ULL,trace,@"LLM组前"))return finish(false,@"组间内存未恢复，后续组未运行");
                LlmOwner retained(nullptr,&Llm::destroy);
                int threads=reuse?4:mode;
                for(int run=0;run<3;++run) {
                    MemorySampler memory;
                    auto started=std::chrono::steady_clock::now();
                    NSString *failure=nil; bool limitReached=false;
                    std::vector<int> speech;
                    NSMutableDictionary *row=[NSMutableDictionary dictionary];
                    row[@"label"]=[NSString stringWithFormat:@"LLM %d线程 %@ 第%d次",threads,(reuse&&mode)?@"保留实例/reset KV":@"独立实例",run+1];
                    trace.record(row[@"label"]);
                    bool ok=generateSpeechTokens(directory,@"今天天气很好。",prompt,trace,speech,&failure,threads,&limitReached,
                        false,{},false,105,(reuse&&mode)?&retained:nullptr,row,true);
                    memory.stop(trace);
                    row[@"wallSeconds"]=@(secondsSince(started));
                    row[@"speechTokens"]=@(speech.size());
                    row[@"peakFootprintMB"]=@(memory.peak.load()/1e6);
                    row[@"minAvailableMB"]=@(memory.available.load()/1e6);
                    row[@"sampler"]=@"mixed [penalty, greedy] / repetition_penalty 1.15";
                    row[@"eosValidated"]=@(ok);
                    if(reference.empty()&&ok)reference=speech;
                    bool equal=ok&&speech==reference;
                    row[@"sameTokensAsReference"]=@(equal);
                    row[@"status"]=ok?(equal?@"pass":@"token-divergence"):@"failed";
                    if(failure)row[@"failure"]=failure;
                    [rows addObject:row];
                    if(reuse) { if(mode)reusedTotal += [row[@"wallSeconds"] doubleValue];else freshTotal += [row[@"wallSeconds"] doubleValue]; }
                    trace.record([NSString stringWithFormat:@"%@；%@；load=%.3f prefill=%.3f decode=%.3f wall=%.3f秒；speech=%zu；相同token=%@",
                        row[@"label"],row[@"status"],[row[@"loadSeconds"] doubleValue],[row[@"prefillSeconds"] doubleValue],
                        [row[@"decodeSeconds"] doubleValue],[row[@"wallSeconds"] doubleValue],speech.size(),equal?@"是":@"否"]);
                    if(!ok)return finish(false,failure?:@"LLM未正常EOS或内存保护；未截断冒充完成");
                    if(!sampledBudgetOK(trace))return finish(false,@"LLM组内曾低于800 MB，停止路线");
                }
                retained.reset();
                trace.record(@"本组LLM已销毁；无跨组实例或KV；下一组先等待内存恢复");
            }
            if(reuse)trace.record([NSString stringWithFormat:@"三次请求总wall：独立%.3f秒，复用%.3f秒（含首载；独立包含释放后回收等待）；仅LLM，不能当整链加速比",freshTotal,reusedTotal]);
            bool matched=true;for(NSDictionary *row in rows)matched=matched&&[row[@"sameTokensAsReference"] boolValue];
            return finish(matched,matched?@"LLM路线完成：EOS/合法token/reset一致性通过；尚未接入共驻生成":@"LLM输出不一致，未建议切换；请分享报告");
        }

        if([route isEqual:@"hift-threads"]) {
            NSDictionary *prepared=[self runAtDirectory:directory text:nil lowPrecisionFlow:NO optimized:YES cpuThreads:4 prepareHiFT:YES];
            if(![prepared[@"success"] boolValue])return finish(false,prepared[@"message"]?:@"HiFT输入准备失败");
            int frames=[prepared[@"frames"] intValue], stftFrames=frames*120+1;
            NSData *melData=prepared[@"mel"],*stftData=prepared[@"stft"];
            if(frames!=148||melData.length!=80*frames*sizeof(float)||stftData.length!=18*stftFrames*sizeof(float))return finish(false,@"HiFT固定输入长度不符");
            std::vector<float> mel(80*frames),source(18*stftFrames),reference;
            std::memcpy(mel.data(),melData.bytes,melData.length);std::memcpy(source.data(),stftData.bytes,stftData.length);
            for(int threads:{4,2,3}) {
                if(!waitAvailable(2500000000ULL,trace,@"HiFT线程组前"))return finish(false,@"组间内存未恢复");
                MemorySampler memory;
                std::vector<float> spectrum;
                double first=0,warm=0,setup=0;
                int actualThreads=-1;
                long thermalStart=NSProcessInfo.processInfo.thermalState;
                {
                    auto started=std::chrono::steady_clock::now();Graph core(threads,true);
                    if(!core.open([directory stringByAppendingPathComponent:@"hift-core.fp32.mnn"],false,trace,@"HiFT core Low",true)||
                       !core.shape("mel",{1,80,frames})||!core.shape("source_stft",{1,18,stftFrames}))return finish(false,@"HiFT线程会话失败");
                    core.net->resizeSession(core.session);core.releaseModel();setup=secondsSince(started);
                    core.net->getSessionInfo(core.session,MNN::Interpreter::THREAD_NUMBER,&actualThreads);
                    for(int run=0;run<3;++run) {
                        if(!sampledBudgetOK(trace))return finish(false,@"HiFT前向前内存保护");
                        started=std::chrono::steady_clock::now();
                        if(!core.inputFloat("mel",mel)||!core.inputFloat("source_stft",source)||!core.run()||!core.output("spectral",source.size(),spectrum))return finish(false,@"HiFT线程输出无效");
                        double elapsed=secondsSince(started);if(!run)first=elapsed;else warm+=elapsed/2;
                    }
                }
                memory.stop(trace);
                if(reference.empty())reference=spectrum;
                auto compared=CosyVoiceProbe::compare(reference,spectrum);
                bool equivalent=compared.valid&&compared.relativeL2<=1e-3;
                [rows addObject:@{@"threads":@(threads),@"actualThreads":@(actualThreads),@"thermalStart":@(thermalStart),@"thermalEnd":@(NSProcessInfo.processInfo.thermalState),@"setupSeconds":@(setup),@"firstSeconds":@(first),@"warmMeanSeconds":@(warm),
                    @"relativeL2":@(compared.relativeL2),@"status":equivalent?@"pass":@"numerical-divergence",
                    @"peakFootprintMB":@(memory.peak.load()/1e6),@"minAvailableMB":@(memory.available.load()/1e6)}];
                trace.record([NSString stringWithFormat:@"HiFT Low %d线程：setup %.3f 首次%.3f 预热%.3f秒；频谱L2 %.8g（仅core，不含F0/DSP/LLM/Flow）",threads,setup,first,warm,compared.relativeL2]);
                if(!equivalent)return finish(false,@"HiFT线程候选偏差超过1e-3，未提升默认配置");
            }
            return finish(true,@"HiFT线程路线完成；默认仍4线程；请按预热和端到端结果选择");
        }

        // SAME 70 real speech tokens => effective 314 frames, 2.8 seconds.
        // Reference384, candidates352/320; mask/noise/conditioner values match.
        constexpr int effective=314;
        std::vector<int> tokens=prompt;tokens.insert(tokens.end(),std::begin(kProbeSpeechTokens),std::begin(kProbeSpeechTokens)+70);
        std::vector<float> mu,noise,promptCond,spks;
        {
            Graph conditioner(4,true);
            if(!conditioner.open([directory stringByAppendingPathComponent:@"flow-conditioner.fp32.mnn"],false,trace,@"固定Conditioner")||!conditioner.shape("tokens",{1,int(tokens.size())}))return finish(false,@"Conditioner失败");
            conditioner.net->resizeSession(conditioner.session);conditioner.releaseModel();
            if(!conditioner.inputInt("tokens",tokens)||!conditioner.run()||!conditioner.output("mu",80*effective,mu))return finish(false,@"Conditioner输出失败");
        }
        if(!readFloats(directory,@"rand-noise.bin",80*15000,noise)||!readFloats(directory,@"prompt-cond.bin",80*kPromptFrames,promptCond)||!readFloats(directory,@"spks.bin",80,spks))return finish(false,@"固定条件缺失");
        struct Variant { int frames,threads;bool bf16; };
        std::vector<Variant> variants;
        if([route isEqual:@"flow-buckets"])variants={{384,4,false},{352,4,false},{320,4,false}};
        else if([route isEqual:@"flow-threads"])variants={{384,4,false},{384,2,false},{384,3,false}};
        else if(singleThread)variants={{384,[[route substringFromIndex:route.length-1] intValue],false}};
        else variants={{384,4,false},{384,4,true}};
        std::vector<float> reference;
        bool allEquivalent=true,allSupported=true;
        for(auto variant:variants) {
            if(!waitAvailable(2800000000ULL,trace,@"独立Flow组前"))return finish(false,@"Flow组间内存未恢复；已完成结果保留，未运行后续组");
            MemorySampler memory;
            int n=variant.frames,backend=-1,actualThreads=-1;
            long thermalStart=NSProcessInfo.processInfo.thermalState;
            std::vector<float> initial(80*n,0),paddedMu(80*n,0),cond(80*n,0),mask(n,0),mel,firstMel;
            for(int c=0;c<80;++c){
                for(int f=0;f<effective;++f){initial[c*n+f]=noise[c*15000+f];paddedMu[c*n+f]=mu[c*effective+f];}
                for(int f=0;f<kPromptFrames;++f)cond[c*n+f]=promptCond[c*kPromptFrames+f];
            }
            std::fill(mask.begin(),mask.begin()+effective,1.0f);
            auto started=std::chrono::steady_clock::now();
            double setup=0,first=0,warm=0,replayL2=0;
            int completedRuns=0,firstCompletedSteps=0;
            bool valid=true;
            {
                Graph flow(variant.threads,true);
                if(!flow.open([directory stringByAppendingPathComponent:@"flow.cfg-student-2step.batch1.fp16.mnn"],false,trace,variant.bf16?@"Flow请求BF16":@"Flow High",false,variant.bf16)||
                    !(flow.shape("x",{1,80,n})&&flow.shape("mu",{1,80,n})&&flow.shape("cond",{1,80,n})&&flow.shape("mask",{1,1,n})&&flow.shape("spks",{1,80})&&flow.shape("t",{1})))return finish(false,@"Flow输入会话失败");
                flow.net->resizeSession(flow.session);flow.releaseModel();setup=secondsSince(started);
                flow.net->getSessionInfo(flow.session,MNN::Interpreter::BACKENDS,&backend);
                flow.net->getSessionInfo(flow.session,MNN::Interpreter::THREAD_NUMBER,&actualThreads);
                trace.record([NSString stringWithFormat:@"Flow %d帧 %d线程 %@；setup %.3f秒；主backend=%d；BF16编译启用，但逐算子可回退，不能声称全图BF16",n,variant.threads,variant.bf16?@"BF16请求":@"High",setup,backend]);
                if(variant.bf16&&backend!=13)allSupported=false;
                for(int run=0;run<3&&valid;++run) {
                    if(!sampledBudgetOK(trace))return finish(false,@"Flow运行前内存保护");
                    auto x=initial;started=std::chrono::steady_clock::now();
                    const float midpoint=float(1.0-std::cos(kPi/4.0));
                    for(int step=0;step<2&&valid;++step) {
                        std::vector<float> t={step?midpoint:0.f},v;
                        valid=flow.inputFloat("x",x)&&flow.inputFloat("mask",mask)&&flow.inputFloat("mu",paddedMu)&&flow.inputFloat("spks",spks)&&flow.inputFloat("cond",cond)&&flow.inputFloat("t",t)&&flow.run()&&flow.output("estimator_out",x.size(),v);
                        if(valid){float dt=step?1.f-midpoint:midpoint;for(size_t i=0;i<x.size();++i)x[i]+=dt*v[i];valid=finite(x);}
                        if(!run&&valid)firstCompletedSteps=step+1;
                        if(!sampledBudgetOK(trace))return finish(false,@"Flow前向期间曾低于800 MB，停止路线");
                    }
                    double elapsed=secondsSince(started);if(!run)first=elapsed;else if(valid)warm+=elapsed;
                    if(valid)++completedRuns;
                    if(valid){
                        mel.resize(80*140);for(int c=0;c<80;++c)std::copy(x.begin()+c*n+kPromptFrames,x.begin()+c*n+effective,mel.begin()+c*140);
                        if(!run)firstMel=mel;
                        auto replay=CosyVoiceProbe::compare(firstMel,mel);
                        if(!replay.valid)valid=false;else replayL2=std::max(replayL2,replay.relativeL2);
                    }
                }
            } // Destroy BEFORE next variant's memory gate.
            memory.stop(trace);
            if(reference.empty()) {
                if(!valid)return finish(false,@"High参考组失败，未把其他候选冒充参考");
                reference=mel;
            }
            auto compared=CosyVoiceProbe::compare(reference,mel);
            bool equivalent=valid&&compared.valid&&compared.relativeL2<=1e-3&&replayL2<=1e-3;
            allEquivalent=allEquivalent&&equivalent;
            double warmMean=completedRuns>1?warm/(completedRuns-1):0;
            NSString *status=!valid?@"nonfinite-or-run-failure":(variant.bf16&&backend!=13?@"unsupported-or-fallback":(equivalent?(singleThread?@"finite-repeat-pass-cross-thread-pending":@"pass"):@"needs-quality-review"));
            NSMutableDictionary *row=[@{@"frames":@(n),@"threads":@(variant.threads),@"actualThreads":@(actualThreads),@"thermalStart":@(thermalStart),@"thermalEnd":@(NSProcessInfo.processInfo.thermalState),@"requestedPrecision":variant.bf16?@"BF16":@"High",
                @"backend":@(backend),@"setupSeconds":@(setup),@"firstAttemptSeconds":@(first),@"firstCompletedSteps":@(firstCompletedSteps),@"completedRuns":@(completedRuns),
                @"firstTwoStepsSeconds":firstCompletedSteps==2?@(first):NSNull.null,@"warmTwoStepsMeanSeconds":completedRuns>1?@(warmMean):NSNull.null,
                @"finite":@(valid),@"melRelativeL2":!singleThread&&compared.valid?@(compared.relativeL2):NSNull.null,@"repeatRelativeL2":@(replayL2),@"status":status,
                @"peakFootprintMB":@(memory.peak.load()/1e6),@"minAvailableMB":@(memory.available.load()/1e6),@"audioSeconds":@2.8} mutableCopy];
            if(singleThread&&valid) {
                row[@"fixtureID"]=@"real70-prompt87-effective314-noise-v1-f32-native";
                row[@"melShape"]=@[@80,@140];
                row[@"melF32Base64"]=[[NSData dataWithBytes:mel.data() length:mel.size()*sizeof(float)] base64EncodedStringWithOptions:0];
                row[@"byteOrder"]=@"little-endian";
                NSMutableDictionary *stamps=[NSMutableDictionary dictionary];
                for(NSString *name in @[@"flow.cfg-student-2step.batch1.fp16.mnn",@"flow.cfg-student-2step.batch1.fp16.mnn.weight",@"flow-conditioner.fp32.mnn",@"prompt-speech-tokens.csv",@"rand-noise.bin",@"prompt-cond.bin",@"spks.bin"]) {
                    NSDictionary *a=[NSFileManager.defaultManager attributesOfItemAtPath:[directory stringByAppendingPathComponent:name] error:nil];
                    stamps[name]=@{@"bytes":a[NSFileSize]?:@0,@"modifiedEpoch":@([a[NSFileModificationDate] timeIntervalSince1970])};
                }
                row[@"modelFileStamps"]=stamps; // File identity hints, NOT cryptographic hashes.
            }
            [rows addObject:row];
            trace.record([NSString stringWithFormat:@"%@：%d帧/%d线程；首轮尝试%.3f秒，完成%d/2步；完整前向%d轮；预热两步均值=%@；Mel跨组L2=%@；峰值%.0f MB，最低余量%.0f MB；非整链RTF",status,n,variant.threads,first,firstCompletedSteps,completedRuns,row[@"warmTwoStepsMeanSeconds"],row[@"melRelativeL2"],memory.peak.load()/1e6,memory.available.load()/1e6]);
            // Numerical failure is reported, not silently replaced with High.
            if(!valid)trace.record(@"本组失败；会话已释放；只在内存恢复后继续其他组，不生成候选WAV");
        }
        return finish(allEquivalent&&allSupported,allEquivalent&&allSupported?(singleThread?@"独立线程完成：有限值/重复一致性通过；原始Mel已存统一报告，跨线程L2仍待复核":@"Flow路线完成：有限值与Mel一致性门槛通过；未测端到端音质/长文"):@"路线完成但候选未全部通过；偏差/回退均保留报告，默认仍High4线程");
    }
}
