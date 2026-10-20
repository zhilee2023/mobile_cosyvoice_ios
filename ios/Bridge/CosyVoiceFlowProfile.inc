// CPU-only observational profile: no intermediate tensor reads, no new weights.
if([route isEqual:@"flow-w8-profile"]||[route isEqual:@"flow-w8-metal-high"]) {
    const bool gpu=[route isEqual:@"flow-w8-metal-high"];
    NSString *failure=nil;
    NSDictionary *identity=verifySelectedAssets(directory,@"flow-w8-manifest",@"cosyvoice3-flow-ffn-w8-g64-v1",
        @[@"candidateGraph",@"candidateWeight",@"fixtures"],true,trace,&failure);
    if(!identity)return finish(false,failure?:@"身份校验失败");
    NSData *data=[NSData dataWithContentsOfURL:[NSBundle.mainBundle URLForResource:@"flow-w8-manifest" withExtension:@"json"]];
    NSDictionary *manifest=data?[NSJSONSerialization JSONObjectWithData:data options:0 error:nil]:nil;
    NSData *fixture=[NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:identity[@"files"][@"fixtures"][@"file"]]];
    NSArray *cases=manifest[@"cases"];
    if(!fixture||![cases isKindOfClass:NSArray.class]||cases.count!=5||![manifest[@"byteOrder"] isEqual:@"little-endian"])
        return finish(false,@"需要已校验的五条固定输入");
    auto section=[&](NSDictionary *entry,NSString *key,size_t count,std::vector<float>& out)->bool {
        NSDictionary *spec=entry[@"sections"][key];
        if(![spec isKindOfClass:NSDictionary.class]||![spec[@"offset"] isKindOfClass:NSNumber.class]||![spec[@"count"] isKindOfClass:NSNumber.class])return false;
        const uint64_t offset=[spec[@"offset"] unsignedLongLongValue];
        if(!CosyVoiceFlowW8::validSection(offset,[spec[@"count"] unsignedLongLongValue],fixture.length,count))return false;
        out.resize(count);std::memcpy(out.data(),static_cast<const char*>(fixture.bytes)+offset,count*sizeof(float));return finite(out);
    };
    if(gpu) {
        NSDictionary *entry=nil;for(NSDictionary *item in cases)if([item[@"id"] isEqual:@"zh-question"])entry=item;
        if(!entry||[entry[@"frames"] intValue]!=384)return finish(false,@"384帧固定输入缺失");
        NSData *savedData=[NSData dataWithContentsOfFile:documentPath(@"flow-w8-metal-reference.json")];
        NSDictionary *saved=savedData?[NSJSONSerialization JSONObjectWithData:savedData options:0 error:nil]:nil;
        NSDictionary *savedIdentity=saved[@"identity"];
        NSData *melData=[NSData dataWithContentsOfFile:documentPath(@"flow-w8-metal-reference.f32")];
        const int n=384,effective=[entry[@"effectiveFrames"] intValue],target=effective-kPromptFrames;
        if(!CosyVoiceFlowW8::validShape(n,effective)||![saved[@"success"] boolValue]||
           ![saved[@"version"] isEqual:[NSBundle.mainBundle objectForInfoDictionaryKey:@"CFBundleShortVersionString"]]||
           ![saved[@"build"] isEqual:[NSBundle.mainBundle objectForInfoDictionaryKey:@"CFBundleVersion"]]||
           ![saved[@"caseID"] isEqual:entry[@"id"]]||![savedIdentity[@"files"] isEqual:identity[@"files"]]||
           ![savedIdentity[@"manifestSHA256"] isEqual:identity[@"manifestSHA256"]]||
           melData.length!=size_t(80*target)*sizeof(float)||![dataSHA256(melData) isEqual:saved[@"melSHA256"]])
            return finish(false,@"需要先完成本版本④CPU热点参考，再冷启动测Metal；不接受旧版本/错输入/损坏参考");
        std::vector<float> reference(80*target);std::memcpy(reference.data(),melData.bytes,melData.length);
        std::vector<float> initial,mask,mu,spks,cond;
        if(!finite(reference)||!section(entry,@"x",80*n,initial)||!section(entry,@"mask",n,mask)||!section(entry,@"mu",80*n,mu)||!section(entry,@"spks",80,spks)||!section(entry,@"cond",80*n,cond))return finish(false,@"GPU固定输入/CPU参考无效");
        NSMutableArray *runs=[NSMutableArray array];
        NSMutableDictionary *row=[@{@"caseID":entry[@"id"],@"identity":identity,@"frames":@384,@"effectiveFrames":@(effective),
            @"audioSeconds":entry[@"audioSeconds"],@"precision":@"High",@"memory":@"Low",@"tuning":@"None",
            @"status":@"started",@"runs":runs,@"completedRuns":@0,@"wholeChainRTFMeasured":@NO,
            @"thermalStart":@(NSProcessInfo.processInfo.thermalState),@"comparison":@"independent same-build CPU W8 High/Memory Low",
            @"numericProbeLimit":@0.01,@"listeningValidated":@NO} mutableCopy];[rows addObject:row];
        bool passed=[&]()->bool {
            Graph flow(4,true);auto setup=std::chrono::steady_clock::now();
            if(!flow.open([directory stringByAppendingPathComponent:identity[@"files"][@"candidateGraph"][@"file"]],true,trace,@"W8 Metal High/NoTune",false,false,true)) {failure=@"Metal会话创建失败";return false;}
            if(!(flow.shape("x",{1,80,n})&&flow.shape("mu",{1,80,n})&&flow.shape("cond",{1,80,n})&&flow.shape("mask",{1,1,n})&&flow.shape("spks",{1,80})&&flow.shape("t",{1}))) {failure=@"Metal输入shape失败";return false;}
            auto resize=std::chrono::steady_clock::now();flow.net->resizeSession(flow.session);
            row[@"resizeSeconds"]=@(secondsSince(resize));row[@"setupSeconds"]=@(secondsSince(setup));
            const int backend=flow.mainBackend();row[@"mainBackend"]=@(backend);
            row[@"footprintAfterResizeMB"]=@(footprintBytes()/1e6);row[@"availableAfterResizeMB"]=@(os_proc_available_memory()/1e6);
            flow.releaseModel();
            // Do not relax the working-space or sampled floor to force a run.
            if(backend!=MNN_FORWARD_METAL){failure=@"主后端非Metal，不能冒充GPU测速";row[@"status"]=@"unsupported-backend";return false;}
            if(!sampledBudgetOK(trace)||os_proc_available_memory()<1100000000ULL){trace.memoryFailure=true;failure=@"Metal resize后余量不足1100MB，安全停止、未前向";row[@"status"]=@"memory-gate-no-forward";return false;}
            std::vector<float> first;
            for(int run=0;run<3;++run) {
                auto x=initial;auto start=std::chrono::steady_clock::now();double runTotal=0,readTotal=0;
                for(int step=0;step<2;++step) {
                    if(!sampledBudgetOK(trace)){failure=@"Metal前向前800MB采样保护";return false;}
                    if(!(flow.inputFloat("x",x)&&flow.inputFloat("mask",mask)&&flow.inputFloat("mu",mu)&&flow.inputFloat("spks",spks)&&flow.inputFloat("cond",cond)&&flow.inputFloat("t",{[entry[@"times"][step] floatValue]}))) {failure=@"Metal六输入复制失败";return false;}
                    auto clock=std::chrono::steady_clock::now();if(!flow.run()){failure=@"Metal runSession失败";return false;}runTotal+=secondsSince(clock);
                    clock=std::chrono::steady_clock::now();std::vector<float> velocity;
                    if(!flow.output("estimator_out",x.size(),velocity)){failure=@"Metal输出形状/NaN/Inf失败";row[@"status"]=@"numeric-failure";return false;}readTotal+=secondsSince(clock);
                    for(size_t i=0;i<x.size();++i)x[i]+=[entry[@"deltas"][step] floatValue]*velocity[i];
                    if(!finite(x)||!sampledBudgetOK(trace)){failure=@"Metal latent有限值/800MB保护失败";return false;}
                }
                const double elapsed=secondsSince(start);std::vector<float> mel(80*target);
                for(int c=0;c<80;++c)std::copy(x.begin()+c*n+kPromptFrames,x.begin()+c*n+effective,mel.begin()+c*target);
                if(run==0)first=mel;
                const auto paired=CosyVoiceProbe::compare(reference,mel),repeat=CosyVoiceProbe::compare(first,mel);
                [runs addObject:@{@"index":@(run+1),@"twoStepsRunPlusReadSeconds":@(elapsed),@"runSessionSeconds":@(runTotal),
                    @"readSyncSeconds":@(readTotal),@"cpuMelL2":paired.valid?@(paired.relativeL2):NSNull.null,
                    @"repeatMelL2":repeat.valid?@(repeat.relativeL2):NSNull.null,@"finite":@YES,
                    @"flowOnlyRTF":@(elapsed/[entry[@"audioSeconds"] doubleValue])}];row[@"completedRuns"]=@(run+1);
                if(!paired.valid||!repeat.valid||paired.relativeL2>.01||repeat.relativeL2>1e-5){failure=@"Metal前向有限，但CPU偏差/重复偏差超限，未通过数值门槛";row[@"status"]=@"needs-quality-review";return false;}
            }
            row[@"status"]=@"numeric-probe-pass";return true;
        }();
        row[@"thermalEnd"]=@(NSProcessInfo.processInfo.thermalState);
        return finish(passed,passed?@"W8 Metal High三轮两步通过模块门槛；不等于整链可用/音质验收，默认仍CPU":(failure?:@"GPU实验停止"));
    }
    bool ok=[&]()->bool {
        Graph flow(4,true);
        if(!flow.open([directory stringByAppendingPathComponent:identity[@"files"][@"candidateGraph"][@"file"]],false,trace,@"Flow W8 CPU热点",false,false,false,false,true)) {
            failure=@"Flow创建失败";return false;
        }
        // Keep graph buffer until all variable-shape cases finish, one session.
        for(NSDictionary *entry in cases) {
            const int n=[entry[@"frames"] intValue],effective=[entry[@"effectiveFrames"] intValue];
            NSArray *times=entry[@"times"],*deltas=entry[@"deltas"];
            if(!CosyVoiceFlowW8::validShape(n,effective)||times.count!=2||deltas.count!=2||
               !CosyVoiceFlowW8::validSchedule([times[0] doubleValue],[times[1] doubleValue],[deltas[0] doubleValue],[deltas[1] doubleValue])) {failure=@"输入shape/时间表无效";return false;}
            std::vector<float> initial,mask,mu,spks,cond;
            if(!section(entry,@"x",80*n,initial)||!section(entry,@"mask",n,mask)||!section(entry,@"mu",80*n,mu)||!section(entry,@"spks",80,spks)||!section(entry,@"cond",80*n,cond)) {failure=@"固定输入切片失败";return false;}
            NSMutableDictionary *row=[@{@"caseID":entry[@"id"],@"frames":@(n),@"effectiveFrames":@(effective),@"identity":identity,
                @"diagnosticOnly":@YES,@"timingIsWholeChainRTF":@NO,@"precision":@"High",@"memory":@"Low",@"threads":@4,
                @"thermalStart":@(NSProcessInfo.processInfo.thermalState),@"status":@"started"} mutableCopy];[rows addObject:row];
            auto setup=std::chrono::steady_clock::now();
            if(!(flow.shape("x",{1,80,n})&&flow.shape("mu",{1,80,n})&&flow.shape("cond",{1,80,n})&&flow.shape("mask",{1,1,n})&&flow.shape("spks",{1,80})&&flow.shape("t",{1}))) {failure=@"Flow形状设置失败";return false;}
            flow.net->resizeSession(flow.session);row[@"resizeSeconds"]=@(secondsSince(setup));
            const int backend=flow.mainBackend();row[@"mainBackend"]=@(backend);
            if(!sampledBudgetOK(trace)||os_proc_available_memory()<1100000000ULL){failure=@"resize后余量不足1100MB，未前向";return false;}
            std::vector<float> reference;
            // First and warm ordinary runs, THEN one profiled two-step run.
            for(int run=0;run<3;++run) {
                const bool profiled=run==2;auto x=initial;CosyVoiceOpProfile::Summary stats;
                const auto start=std::chrono::steady_clock::now();
                for(int step=0;step<2;++step) {
                    if(!sampledBudgetOK(trace)){failure=@"800MB采样保护停止";return false;}
                    if(!(flow.inputFloat("x",x)&&flow.inputFloat("mask",mask)&&flow.inputFloat("mu",mu)&&flow.inputFloat("spks",spks)&&flow.inputFloat("cond",cond)&&flow.inputFloat("t",{[times[step] floatValue]}))) {failure=@"Flow六输入复制失败";return false;}
                    if(profiled) {
                        const size_t callsBefore=stats.calls;
                        auto opStart=std::chrono::steady_clock::now();bool active=false,stopped=false;
                        auto before=[&](const std::vector<MNN::Tensor*>&,const MNN::OperatorInfo*)->bool {active=true;opStart=std::chrono::steady_clock::now();return true;};
                        auto after=[&](const std::vector<MNN::Tensor*>&,const MNN::OperatorInfo *info)->bool {
                            const double elapsed=secondsSince(opStart);
                            if(!active||!info||!stats.add(info->type(),info->name(),elapsed)||!sampledBudgetOK(trace)||secondsSince(start)>60) {stopped=true;return false;}
                            active=false;return true;
                        };
                        const auto code=flow.net->runSessionWithCallBackInfo(flow.session,before,after,true);
                        if(code!=MNN::NO_ERROR||stopped||active||stats.calls<=callsBefore){failure=@"回调未执行/超时/元数据上限/800MB保护，未计成功";return false;}
                    } else if(!flow.run()){failure=@"普通Flow前向失败";return false;}
                    std::vector<float> velocity;
                    if(!flow.output("estimator_out",x.size(),velocity)){failure=@"输出形状或非有限值失败";return false;}
                    for(size_t i=0;i<x.size();++i)x[i]+=[deltas[step] floatValue]*velocity[i];
                    if(!finite(x)||!sampledBudgetOK(trace)){failure=@"latent有限值/800MB门槛失败";return false;}
                }
                const double wall=secondsSince(start);
                std::vector<float> mel(80*(effective-kPromptFrames));
                for(int c=0;c<80;++c)std::copy(x.begin()+c*n+kPromptFrames,x.begin()+c*n+effective,mel.begin()+c*(effective-kPromptFrames));
                if(run==0)row[@"firstTwoStepsSeconds"]=@(wall);
                if(run==1){reference=mel;row[@"warmOrdinaryTwoStepsSeconds"]=@(wall);}
                if(profiled) {
                    const auto comparison=CosyVoiceProbe::compare(reference,mel);
                    row[@"profiledTwoStepsSeconds"]=@(wall);row[@"profileVsOrdinaryMelL2"]=comparison.valid?@(comparison.relativeL2):NSNull.null;
                    row[@"callbackObservedSeconds"]=@(stats.seconds);row[@"callbackCalls"]=@(stats.calls);
                    row[@"profileOverheadRatio"]=@(wall/[row[@"warmOrdinaryTwoStepsSeconds"] doubleValue]);
                    auto serialize=[&](bool byType)->NSArray* {
                        NSMutableArray *items=[NSMutableArray array];
                        for(const auto& op:stats.ranked(byType,byType?20:30))
                            [items addObject:@{@"type":[NSString stringWithUTF8String:op.type.c_str()]?:@"invalid-name",
                                @"name":[NSString stringWithUTF8String:op.name.c_str()]?:@"invalid-name",@"calls":@(op.calls),
                                @"seconds":@(op.seconds),@"maxSeconds":@(op.maxSeconds),@"fractionOfObservedOpTime":@(stats.seconds>0?op.seconds/stats.seconds:0)}];
                        return items;
                    };
                    row[@"topTypes"]=serialize(true);row[@"topOperators"]=serialize(false);
                    if(!stats.valid||!comparison.valid||comparison.relativeL2>1e-5){failure=@"回调与普通运行Mel偏差超限，需复核";return false;}
                }
            }
            row[@"thermalEnd"]=@(NSProcessInfo.processInfo.thermalState);row[@"status"]=@"profile-completed";
            if([entry[@"id"] isEqual:@"zh-question"]) {
                NSData *mel=[NSData dataWithBytes:reference.data() length:reference.size()*sizeof(float)];
                NSDictionary *cache=@{@"success":@YES,@"caseID":entry[@"id"],@"identity":identity,@"melSHA256":dataSHA256(mel),
                    @"version":[NSBundle.mainBundle objectForInfoDictionaryKey:@"CFBundleShortVersionString"],
                    @"build":[NSBundle.mainBundle objectForInfoDictionaryKey:@"CFBundleVersion"]};
                NSData *json=[NSJSONSerialization dataWithJSONObject:cache options:0 error:nil];
                if(!json||![mel writeToFile:documentPath(@"flow-w8-metal-reference.f32") atomically:YES]||
                   ![json writeToFile:documentPath(@"flow-w8-metal-reference.json") atomically:YES]) {failure=@"CPU参考写入失败；GPU不允许使用未校验缓存";return false;}
                row[@"savedMetalCPUReference"]=@YES;
            }
            trace.record([NSString stringWithFormat:@"%@：普通预热两步%.3f秒，回调诊断%.3f秒；只保留top30算子/top20类型，不作整链速度结论",entry[@"id"],[row[@"warmOrdinaryTwoStepsSeconds"] doubleValue],[row[@"profiledTwoStepsSeconds"] doubleValue]]);
        }
        return true;
    }();
    return finish(ok,ok?@"五条CPU Flow热点分析完成；回调耗时非生产基准、非整链RTF；只需分享统一报告":(failure?:@"热点分析提前停止"));
}
