// Included inside @implementation CosyVoiceGenerate. Kept separate so the
// already-validated generation path remains readable. UI serializes all calls.
+ (NSDictionary<NSString *, id> *)probeHiFTPrecisionAtDirectory:(NSString *)directory {
    @autoreleasepool {
        Trace trace(@"last-hift-precision.log");
        MemorySampler memory;
        auto finish = [&](bool ok, NSString *message) {
            memory.stop(trace);
            trace.record([NSString stringWithFormat:@"%@；20ms采样峰值 %.0f MB，最低可用 %.0f MB；不是端到端RTF或音质认证",
                          message, trace.peakMB, trace.minAvailableMB]);
            return result(ok, message, trace);
        };
        NSDictionary *prepared = [self runAtDirectory:directory text:nil lowPrecisionFlow:NO
                                           optimized:YES cpuThreads:4 prepareHiFT:YES];
        trace.peakMB = std::max(trace.peakMB, [prepared[@"peakFootprintMB"] doubleValue]);
        if (![prepared[@"success"] boolValue]) return finish(false, prepared[@"message"] ?: @"固定HiFT输入准备失败");
        const int frames = [prepared[@"frames"] intValue], stftFrames = frames*120+1;
        NSData *melData = prepared[@"mel"], *stftData = prepared[@"stft"];
        if (frames != 148 || melData.length != 80*frames*sizeof(float) || stftData.length != 18*stftFrames*sizeof(float))
            return finish(false, @"固定HiFT输入形状不符，未运行精度对照");
        std::vector<float> mel(80*frames), sourceStft(18*stftFrames);
        std::memcpy(mel.data(), melData.bytes, melData.length);
        std::memcpy(sourceStft.data(), stftData.bytes, stftData.length);
        trace.record(@"开始配对：相同148帧Mel、相同F0与source-STFT；4线程；仅core High→Low；每组独立会话，3次前向");
        std::vector<float> referencePcm, candidatePcm, referenceSpectrum, candidateSpectrum;
        double warmed[2] = {}, first[2] = {};
        for (int mode=0; mode<2; ++mode) {
            if (os_proc_available_memory() < 1800000000ULL) return finish(false, @"HiFT对照组加载前可用内存不足1800 MB");
            std::vector<float> spectrum;
            {
                Graph core(4,true);
                NSString *label = mode ? @"HiFT core CPU Low 候选" : @"HiFT core CPU High 参考";
                if (!core.open([directory stringByAppendingPathComponent:@"hift-core.fp32.mnn"], false, trace, label, mode==1))
                    return finish(false, @"HiFT core对照会话创建失败");
                if (!(core.shape("mel",{1,80,frames}) && core.shape("source_stft",{1,18,stftFrames})))
                    return finish(false, @"HiFT core对照输入形状失败");
                auto begin = std::chrono::steady_clock::now();
                core.net->resizeSession(core.session); core.releaseModel();
                int backend=-1; core.net->getSessionInfo(core.session, MNN::Interpreter::BACKENDS, &backend);
                trace.record([NSString stringWithFormat:@"%@ resize %.3f 秒；主backend=%d（Low可能为CPU Extension）", label, secondsSince(begin), backend]);
                if (os_proc_available_memory() < 1000000000ULL) return finish(false, @"HiFT resize后可用内存不足1000 MB");
                for (int run=0;run<3;++run) {
                    if (os_proc_available_memory() < 800000000ULL) return finish(false, @"HiFT前向前可用内存不足800 MB");
                    begin=std::chrono::steady_clock::now();
                    if (!core.inputFloat("mel",mel) || !core.inputFloat("source_stft",sourceStft) || !core.run() ||
                        !core.output("spectral",18*stftFrames,spectrum))
                        return finish(false, mode ? @"HiFT Low输出无效，未保存本轮候选WAV，默认仍High" : @"HiFT High参考输出无效");
                    double elapsed=secondsSince(begin);
                    if (run==0) first[mode]=elapsed; else warmed[mode]+=elapsed/2;
                    trace.record([NSString stringWithFormat:@"%@ 第%d次 core前向+同步 %.3f 秒",label,run+1,elapsed]);
                    if (memory.available.load() < 800000000ULL) return finish(false, @"20ms采样曾低于800 MB，停止HiFT试验");
                }
            }
            auto pcm=CosyVoiceDSP::istft(spectrum,stftFrames,frames*kHop);
            auto stats=CosyVoiceProbe::compare(pcm,pcm);
            if (!stats.valid || stats.referenceRms<0.001 || !writeWav(pcm, mode ? @"hift-low-candidate.wav" : @"hift-high-reference.wav"))
                return finish(false, @"HiFT PCM非有限、静音或WAV写入失败");
            trace.record([NSString stringWithFormat:@"%@ WAV已保存；RMS %.6f；峰值 %.6f；接近削波比例 %.6f",
                          mode ? @"Low候选" : @"High参考",stats.referenceRms,peak(pcm),stats.clippedFraction]);
            if (mode==0) { referencePcm=std::move(pcm); referenceSpectrum=std::move(spectrum); }
            else { candidatePcm=std::move(pcm); candidateSpectrum=std::move(spectrum); }
        }
        auto pcm=CosyVoiceProbe::compare(referencePcm,candidatePcm);
        auto spectrum=CosyVoiceProbe::compare(referenceSpectrum,candidateSpectrum);
        if (!pcm.valid || !spectrum.valid) return finish(false, @"配对指标无法计算");
        trace.record([NSString stringWithFormat:@"对照：频谱L2 %.8g；PCM相对L2 %.8g；相关 %.8g；RMS增益差 %.4f dB；最佳标量对齐残差 %.8g；候选接近削波 %.6f",
                      spectrum.relativeL2,pcm.relativeL2,pcm.correlation,pcm.gainDB,pcm.alignedL2,pcm.clippedFraction]);
        trace.record([NSString stringWithFormat:@"core计时：High首次 %.3f 秒、预热后均值 %.3f 秒；Low首次 %.3f 秒、预热后均值 %.3f 秒；core加速 %.3fx。音频2.96秒；不包括LLM/Flow/加载/resize/DSP",
                      first[0],warmed[0],first[1],warmed[1],warmed[0]/std::max(warmed[1],1e-9)]);
        return finish(true,@"HiFT配对完成；请盲听两段2.96秒WAV；未自动改动默认High精度");
    }
}

+ (NSDictionary<NSString *, id> *)probeResidentFlowAtDirectory:(NSString *)directory {
    @autoreleasepool {
        Trace trace(@"last-resident-flow.log");
        MemorySampler memory;
        trace.record(@"开始共驻探针：唯一Flow CPU High 384帧；先基线→保留Flow加载LLM/prefill/2 decode→LLM仍存活时Flow两步→销毁LLM→再次复用Flow；不生成音频",true);
        auto finish = [&](bool ok, NSString *message) {
            memory.stop(trace);
            trace.record([NSString stringWithFormat:@"%@；20ms采样峰值 %.0f MB，最低可用 %.0f MB；非整句/长文共驻验收",
                          message,trace.peakMB,trace.minAvailableMB]);
            return result(ok,message,trace);
        };
        if (os_proc_available_memory()<2900000000ULL) return finish(false,@"共驻试验初始可用内存不足2900 MB，请冷启动后单独测");
        std::vector<int> prompt;
        NSString *csv=[NSString stringWithContentsOfFile:[directory stringByAppendingPathComponent:@"prompt-speech-tokens.csv"] encoding:NSUTF8StringEncoding error:nil];
        for (NSString *item in [csv componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@",\n\r "]]) if(item.length)prompt.push_back(item.intValue);
        if(prompt.size()!=87)return finish(false,@"提示token不是87，停止探针");
        auto tokens=prompt;
        tokens.insert(tokens.end(),std::begin(kProbeSpeechTokens),std::end(kProbeSpeechTokens));
        constexpr int frames=384, effective=322;
        std::vector<float> mu;
        {
            Graph conditioner(4,true);
            if(!conditioner.open([directory stringByAppendingPathComponent:@"flow-conditioner.fp32.mnn"],false,trace,@"Conditioner"))
                return finish(false,@"共驻Conditioner加载失败");
            if(!conditioner.shape("tokens",{1,int(tokens.size())}))return finish(false,@"共驻Conditioner形状失败");
            conditioner.net->resizeSession(conditioner.session);conditioner.releaseModel();
            if(!conditioner.inputInt("tokens",tokens)||!conditioner.run()||!conditioner.output("mu",80*effective,mu))
                return finish(false,@"共驻Conditioner输出失败");
        }
        std::vector<float> noise,promptCond,spks;
        if(!readFloats(directory,@"rand-noise.bin",80*15000,noise)||!readFloats(directory,@"prompt-cond.bin",80*kPromptFrames,promptCond)||!readFloats(directory,@"spks.bin",80,spks))
            return finish(false,@"共驻固定条件缺失");
        std::vector<float> initial(80*frames,0), paddedMu(initial.size(),0),cond(initial.size(),0),mask(frames,0);
        for(int c=0;c<80;++c) {
            for(int f=0;f<effective;++f){ initial[c*frames+f]=noise[c*15000+f];paddedMu[c*frames+f]=mu[c*effective+f]; }
            for(int f=0;f<kPromptFrames;++f)cond[c*frames+f]=promptCond[c*kPromptFrames+f];
        }
        std::fill(mask.begin(),mask.begin()+effective,1.0f);
        noise.clear();noise.shrink_to_fit();mu.clear();mu.shrink_to_fit();
        std::unique_ptr<Graph> flow(new Graph(4,true));
        if(!flow->open([directory stringByAppendingPathComponent:@"flow.cfg-student-2step.batch1.fp16.mnn"],false,trace,@"常驻Flow CPU High"))
            return finish(false,@"共驻Flow会话失败");
        if(!(flow->shape("x",{1,80,frames})&&flow->shape("mask",{1,1,frames})&&flow->shape("mu",{1,80,frames})&&flow->shape("spks",{1,80})&&flow->shape("cond",{1,80,frames})&&flow->shape("t",{1})))
            return finish(false,@"共驻Flow形状失败");
        auto begin=std::chrono::steady_clock::now();
        flow->net->resizeSession(flow->session);flow->releaseModel();
        trace.record([NSString stringWithFormat:@"唯一Flow resize %.3f 秒；后续三轮不再resize，不创建其他Flow会话",secondsSince(begin)]);
        if(os_proc_available_memory()<1500000000ULL)return finish(false,@"Flow后可用内存不足1500 MB，未加载LLM");
        std::vector<float> reference,last;
        auto runFlow = [&](NSString *phase) {
            if(os_proc_available_memory()<800000000ULL||memory.available.load()<800000000ULL){trace.record(@"共驻阶段余量或20ms采样低于800 MB，未开始本轮Flow");return false;}
            auto x=initial;
            auto started=std::chrono::steady_clock::now();
            trace.record([phase stringByAppendingString:@"开始：同一会话、同一输入，resize=0"]);
            const float midpoint=float(1.0-std::cos(kPi/4.0));
            for(int step=0;step<2;++step){
                if(os_proc_available_memory()<800000000ULL){trace.record(@"共驻Flow前向前不足800 MB，停止");return false;}
                std::vector<float> t={step?midpoint:0.0f},velocity;
                if(!(flow->inputFloat("x",x)&&flow->inputFloat("mask",mask)&&flow->inputFloat("mu",paddedMu)&&flow->inputFloat("spks",spks)&&flow->inputFloat("cond",cond)&&flow->inputFloat("t",t)&&flow->run()&&flow->output("estimator_out",x.size(),velocity)))return false;
                float delta=step?1.0f-midpoint:midpoint;
                for(size_t i=0;i<x.size();++i)x[i]+=delta*velocity[i];
                if(!finite(x)||memory.available.load()<800000000ULL){trace.record(@"共驻输出非有限或20ms采样低于800 MB");return false;}
                trace.record([NSString stringWithFormat:@"%@ Flow %d/2完成，latent峰值 %.6f",phase,step+1,peak(x)]);
            }
            last=std::move(x);
            trace.record([NSString stringWithFormat:@"%@两步完成 %.3f 秒",phase,secondsSince(started)]);
            return true;
        };
        if(!runFlow(@"独立基线"))return finish(false,@"独立Flow基线失败，未加载LLM");
        reference=last;
        if(os_proc_available_memory()<1500000000ULL)return finish(false,@"基线后余量不足1500 MB，未加载LLM");
        NSString *failure=nil;bool limitReached=false;std::vector<int> speech;
        bool llmOk=generateSpeechTokens(directory,@"你好，欢迎来到离线语音测试。",prompt,trace,speech,&failure,4,&limitReached,true,[&]{return runFlow(@"LLM仍存活的共驻阶段");});
        if(!llmOk)return finish(false,failure?:@"共驻LLM失败");
        auto compared=CosyVoiceProbe::compare(reference,last);
        trace.record([NSString stringWithFormat:@"共驻Flow相对独立基线L2 %.8g",compared.relativeL2]);
        if(!compared.valid||compared.relativeL2>1e-3)return finish(false,@"共驻Flow与基线不一致");
        if(!runFlow(@"LLM销毁后复用"))return finish(false,@"LLM销毁后Flow复用失败");
        compared=CosyVoiceProbe::compare(reference,last);
        if(!compared.valid||compared.relativeL2>1e-3)return finish(false,@"第三轮Flow输出与基线不一致");
        flow.reset();
        trace.record(@"唯一Flow与LLM均已释放；没有保留隐藏会话池");
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        return finish(true,@"短KV共驻与唯一Flow三轮复用通过；未运行HiFT，未验证整句或分钟级常驻");
    }
}
