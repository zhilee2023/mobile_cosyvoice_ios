// Isolated CPU High Flow test. One model lifetime, five frozen held-out cases.
// Never touches production LLM/HiFT/model paths, never declares whole-chain RTF.
if ([@[@"flow-w8-reference",@"flow-w8-normal",@"flow-w8-low"] containsObject:route]) {
    const bool baseline=[route isEqual:@"flow-w8-reference"], low=[route isEqual:@"flow-w8-low"];
    NSURL *url=[NSBundle.mainBundle URLForResource:@"flow-w8-manifest" withExtension:@"json"];
    NSData *manifestData=url?[NSData dataWithContentsOfURL:url]:nil;
    id parsed=manifestData?[NSJSONSerialization JSONObjectWithData:manifestData options:0 error:nil]:nil;
    NSDictionary *manifest=[parsed isKindOfClass:NSDictionary.class]?parsed:nil;
    if (![manifest[@"recipe"] isEqual:@"cosyvoice3-flow-ffn-w8-g64-v1"] || ![manifest[@"byteOrder"] isEqual:@"little-endian"])
        return finish(false,@"W8测试身份清单缺失或不兼容");
    auto hashData=[](NSData *data)->NSString * {
        unsigned char value[CC_SHA256_DIGEST_LENGTH];CC_SHA256(data.bytes,CC_LONG(data.length),value);
        NSMutableString *hex=[NSMutableString string];for(unsigned char b:value)[hex appendFormat:@"%02x",b];return hex;
    };
    auto fileHash=[](NSString *path)->NSString * {
        FILE *file=std::fopen(path.fileSystemRepresentation,"rb");if(!file)return nil;
        CC_SHA256_CTX hash;CC_SHA256_Init(&hash);std::vector<unsigned char> buffer(1024*1024);size_t n;
        while((n=std::fread(buffer.data(),1,buffer.size(),file)))CC_SHA256_Update(&hash,buffer.data(),CC_LONG(n));
        bool ok=!std::ferror(file);std::fclose(file);if(!ok)return nil;
        unsigned char value[CC_SHA256_DIGEST_LENGTH];CC_SHA256_Final(value,&hash);
        NSMutableString *hex=[NSMutableString string];for(unsigned char b:value)[hex appendFormat:@"%02x",b];return hex;
    };
    const auto verifyStart=std::chrono::steady_clock::now();
    NSMutableDictionary *identities=[NSMutableDictionary dictionary];
    for(NSString *key in @[@"sourceGraph",@"sourceWeight",@"candidateGraph",@"candidateWeight",@"fixtures"]) {
        NSDictionary *entry=manifest[@"files"][key];NSString *file=entry[@"file"];
        if(![entry isKindOfClass:NSDictionary.class] || ![file isKindOfClass:NSString.class] || !file.length ||
           ![file.lastPathComponent isEqual:file] || [file hasPrefix:@"."])
            return finish(false,@"W8清单文件名不合法");
        NSString *path=[directory stringByAppendingPathComponent:file];
        NSDictionary *attrs=[NSFileManager.defaultManager attributesOfItemAtPath:path error:nil];
        if(![attrs[NSFileType] isEqual:NSFileTypeRegular] || [attrs[NSFileSize] unsignedLongLongValue]!=[entry[@"bytes"] unsignedLongLongValue])
            return finish(false,[NSString stringWithFormat:@"缺少或大小不符：%@。请导入W8测试包的三个文件；原17个文件保留。",file]);
        NSString *actual=fileHash(path);
        if(!actual || ![actual isEqual:entry[@"sha256"]])return finish(false,[NSString stringWithFormat:@"%@ SHA256不匹配，未加载Flow；不自动替换原模型",file]);
        identities[key]=actual;
    }
    const double verifySeconds=secondsSince(verifyStart);
    NSString *manifestSHA=hashData(manifestData);
    trace.record([NSString stringWithFormat:@"W8全部模型/输入SHA256通过，%.3f秒；校验单独计时，不混入模块前向。High精度，Memory=%@；不运行LLM/HiFT",verifySeconds,low?@"Low动态整数":@"Normal"]);
    NSData *fixture=[NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:manifest[@"files"][@"fixtures"][@"file"]]];
    if(!fixture)return finish(false,@"W8固定输入读取失败");
    NSArray *cases=[manifest[@"cases"] isKindOfClass:NSArray.class]?manifest[@"cases"]:nil;
    if(cases.count!=5)return finish(false,@"W8固定输入必须为5条独立验证样本");
    auto section=[&](NSDictionary *entry,NSString *name,size_t expected,std::vector<float>& result)->bool {
        NSDictionary *spec=entry[@"sections"][name];
        if(![spec isKindOfClass:NSDictionary.class] || ![spec[@"offset"] isKindOfClass:NSNumber.class] || ![spec[@"count"] isKindOfClass:NSNumber.class])return false;
        uint64_t offset=[spec[@"offset"] unsignedLongLongValue], count=[spec[@"count"] unsignedLongLongValue];
        if(!CosyVoiceFlowW8::validSection(offset,count,fixture.length,expected))return false;
        result.resize(expected);std::memcpy(result.data(),static_cast<const char *>(fixture.bytes)+offset,expected*sizeof(float));return finite(result);
    };
    NSDictionary *saved=nil;
    if(!baseline) {
        NSData *data=[NSData dataWithContentsOfFile:documentPath(@"last-experiment-flow-w8-reference.json")];
        id item=data?[NSJSONSerialization JSONObjectWithData:data options:0 error:nil]:nil;
        saved=[item isKindOfClass:NSDictionary.class]?item:nil;
        if(![saved[@"success"] boolValue] || ![saved[@"version"] isEqual:[NSBundle.mainBundle objectForInfoDictionaryKey:@"CFBundleShortVersionString"]] ||
           ![saved[@"build"] isEqual:[NSBundle.mainBundle objectForInfoDictionaryKey:@"CFBundleVersion"]] || [saved[@"rows"] count]!=5)
            return finish(false,@"先冷启动完成①W8原版参考，再冷启动测试②或③；不接受旧版本或未完成的参考");
    }
    NSString *model=[directory stringByAppendingPathComponent:manifest[@"files"][baseline?@"sourceGraph":@"candidateGraph"][@"file"]];
    NSString *failure=nil;
    bool ok=[&]()->bool {
        Graph flow(4,true);
        const auto started=std::chrono::steady_clock::now();
        if(!flow.open(model,false,trace,baseline?@"原版Flow High":@"FFN W8/block64 High",false,false,false,!low)) {failure=@"W8会话创建失败";return false;}
        const double loadSeconds=secondsSince(started);
        if(!sampledBudgetOK(trace)){failure=@"会话创建后触发800MB内存保护";return false;}
        int backend=-1,threads=-1;
        flow.net->getSessionInfo(flow.session,MNN::Interpreter::BACKENDS,&backend);
        flow.net->getSessionInfo(flow.session,MNN::Interpreter::THREAD_NUMBER,&threads);
        if(backend!=MNN_FORWARD_CPU || threads!=4){failure=@"CPU后端或4线程设置未生效";return false;}
        for(NSDictionary *entry in cases) {
            NSString *caseID=entry[@"id"];const int n=[entry[@"frames"] intValue], effective=[entry[@"effectiveFrames"] intValue], target=effective-kPromptFrames;
            NSArray *times=entry[@"times"],*deltas=entry[@"deltas"];
            if(![caseID isKindOfClass:NSString.class] || ![caseID.lastPathComponent isEqual:caseID] ||
               !CosyVoiceFlowW8::validShape(n,effective) || times.count!=2 || deltas.count!=2 ||
               !CosyVoiceFlowW8::validSchedule([times[0] doubleValue],[times[1] doubleValue],[deltas[0] doubleValue],[deltas[1] doubleValue])){failure=@"固定输入形状或两步时间表不合法";return false;}
            std::vector<float> initial,mask,mu,spks,cond,pcMel;
            if(!section(entry,@"x",80*n,initial)||!section(entry,@"mask",n,mask)||!section(entry,@"mu",80*n,mu)||
               !section(entry,@"spks",80,spks)||!section(entry,@"cond",80*n,cond)||!section(entry,@"referenceMel",80*target,pcMel)){failure=@"固定输入长度/有限值检查失败";return false;}
            std::vector<float> phoneMel;
            NSString *referenceFile=[NSString stringWithFormat:@"flow-w8-cpu-%@.f32",caseID];
            if(!baseline) {
                NSDictionary *matched=nil;
                for(NSDictionary *row in saved[@"rows"])if([row[@"caseID"] isEqual:caseID])matched=row;
                if(!matched || ![matched[@"manifestSHA256"] isEqual:manifestSHA] || ![matched[@"fileSHA256"] isEqual:identities] || ![matched[@"status"] isEqual:@"numeric-probe-pass"]){failure=@"CPU参考输入/权重/清单身份不匹配";return false;}
                NSData *data=[NSData dataWithContentsOfFile:documentPath(referenceFile)];
                if(data.length!=80*target*sizeof(float) || ![hashData(data) isEqual:matched[@"targetMelSHA256"]]){failure=@"独立CPU参考Mel损坏或未保存";return false;}
                phoneMel.resize(80*target);std::memcpy(phoneMel.data(),data.bytes,data.length);
            }
            NSMutableDictionary *row=[@{@"caseID":caseID,@"text":entry[@"text"],@"frames":@(n),@"effectiveFrames":@(effective),
                @"audioSeconds":entry[@"audioSeconds"],@"status":@"started",@"precision":@"High",@"memory":low?@"Low":@"Normal",
                @"requestedMemoryEnum":@(low?2:0),@"backend":@(backend),@"actualThreads":@(threads),
                @"fileSHA256":identities,@"manifestSHA256":manifestSHA,@"hashVerificationSeconds":@(verifySeconds),
                @"sessionLoadCreateSeconds":@(loadSeconds),@"sessionCreationCount":@1,@"thermalStart":@(NSProcessInfo.processInfo.thermalState),
                @"wholeChainRTFMeasured":@NO,@"listeningValidated":@NO,@"completedRuns":@0} mutableCopy];
            [rows addObject:row];NSMutableArray *runs=[NSMutableArray array];row[@"runs"]=runs;
            const auto resizeStart=std::chrono::steady_clock::now();
            if(!(flow.shape("x",{1,80,n})&&flow.shape("mu",{1,80,n})&&flow.shape("cond",{1,80,n})&&flow.shape("mask",{1,1,n})&&flow.shape("spks",{1,80})&&flow.shape("t",{1}))){failure=@"Flow输入shape设置失败";return false;}
            // Keep the small graph buffer: later held-out cases have different
            // shapes. Do not releaseModel before all resizes have completed.
            flow.net->resizeSession(flow.session);row[@"resizeSeconds"]=@(secondsSince(resizeStart));
            float allocatorMB=0;flow.net->getSessionInfo(flow.session,MNN::Interpreter::MEMORY,&allocatorMB);
            row[@"sessionAllocatorMB"]=@(allocatorMB);row[@"footprintAfterResizeMB"]=@(footprintBytes()/1e6);
            row[@"availableAfterResizeMB"]=@(os_proc_available_memory()/1e6);
            if(!sampledBudgetOK(trace) || os_proc_available_memory()<1100000000ULL){trace.memoryFailure=true;failure=@"resize后余量不足1100MB，未前向；800MB保护不放宽";return false;}
            std::vector<float> firstMel,mel;
            double warm=0,repeatL2=0;
            for(int run=0;run<3;++run) {
                auto x=initial;const auto runStart=std::chrono::steady_clock::now();NSMutableArray *steps=[NSMutableArray array];
                NSMutableDictionary *runRow=[@{@"index":@(run+1),@"steps":steps,@"completedSteps":@0,@"status":@"started"} mutableCopy];[runs addObject:runRow];
                for(int step=0;step<2;++step) {
                    if(!sampledBudgetOK(trace)){failure=@"前向前800MB内存保护";return false;}
                    std::vector<float> velocity,timeValue={[times[step] floatValue]};const auto stepStart=std::chrono::steady_clock::now();
                    if(!(flow.inputFloat("x",x)&&flow.inputFloat("mask",mask)&&flow.inputFloat("mu",mu)&&flow.inputFloat("spks",spks)&&flow.inputFloat("cond",cond)&&flow.inputFloat("t",timeValue))){failure=@"六输入复制失败";return false;}
                    const double inputSeconds=secondsSince(stepStart);const auto forwardStart=std::chrono::steady_clock::now();
                    if(!flow.run()){failure=@"Flow runSession失败";return false;}
                    const double forwardSeconds=secondsSince(forwardStart);const auto readStart=std::chrono::steady_clock::now();
                    if(!flow.output("estimator_out",x.size(),velocity)){failure=@"Flow输出形状/NaN/Inf检查失败";return false;}
                    const double readSeconds=secondsSince(readStart);const float dt=[deltas[step] floatValue];
                    for(size_t i=0;i<x.size();++i)x[i]+=dt*velocity[i];
                    if(!finite(x)||!sampledBudgetOK(trace)){failure=@"latent有限值或800MB内存门槛失败";return false;}
                    [steps addObject:@{@"step":@(step+1),@"inputSeconds":@(inputSeconds),@"runSeconds":@(forwardSeconds),@"readSyncSeconds":@(readSeconds),@"finite":@YES}];
                    runRow[@"completedSteps"]=@(step+1);
                }
                const double elapsed=secondsSince(runStart);mel.resize(80*target);
                for(int c=0;c<80;++c)std::copy(x.begin()+c*n+kPromptFrames,x.begin()+c*n+effective,mel.begin()+c*target);
                if(run==0){firstMel=mel;row[@"firstTwoStepsSeconds"]=@(elapsed);}else warm+=elapsed/2;
                const auto replay=CosyVoiceProbe::compare(firstMel,mel);
                if(!replay.valid){failure=@"Mel比较无效";return false;}repeatL2=std::max(repeatL2,replay.relativeL2);
                runRow[@"twoStepsSeconds"]=@(elapsed);runRow[@"status"]=@"finite-completed";row[@"completedRuns"]=@(run+1);
                trace.record([NSString stringWithFormat:@"%@ 第%d轮两步 %.3f秒；重复L2 %.8g；非整链RTF",caseID,run+1,elapsed,replay.relativeL2]);
            }
            const auto pc=CosyVoiceProbe::compare(pcMel,mel);
            const auto paired=CosyVoiceProbe::compare(baseline?pcMel:phoneMel,mel);
            const double limit=baseline?0.001:(low?0.03:0.01);
            const bool passed=pc.valid&&paired.valid&&paired.relativeL2<=limit&&repeatL2<=0.001;
            row[@"pcBaselineMelRelativeL2"]=pc.valid?@(pc.relativeL2):NSNull.null;
            row[@"comparisonReference"]=baseline?@"PC original CPU High":@"same-build independent iPhone CPU High";
            row[@"pairedMelRelativeL2"]=paired.valid?@(paired.relativeL2):NSNull.null;
            row[@"numericProbeLimit"]=@(limit);row[@"repeatRelativeL2"]=@(repeatL2);
            row[@"warmTwoStepsMeanSeconds"]=@(warm);row[@"warmFlowOnlyRTF"]=@(warm/[entry[@"audioSeconds"] doubleValue]);
            row[@"thermalEnd"]=@(NSProcessInfo.processInfo.thermalState);row[@"status"]=passed?@"numeric-probe-pass":@"needs-quality-review";
            if(baseline&&passed) {
                NSData *data=[NSData dataWithBytes:mel.data() length:mel.size()*sizeof(float)];
                if(![data writeToFile:documentPath(referenceFile) atomically:YES]){failure=@"CPU参考Mel写入失败";return false;}
                row[@"targetMelSHA256"]=hashData(data);row[@"referenceMelFile"]=referenceFile;
            }
            trace.record([NSString stringWithFormat:@"%@；%@；参考Mel L2 %.6g，PC L2 %.6g；预热两步%.3f秒；当前峰值%.0fMB、最低余量%.0fMB；非听感验收",caseID,row[@"status"],paired.relativeL2,pc.relativeL2,warm,allMemory.peak.load()/1e6,allMemory.available.load()/1e6]);
            if(!passed){failure=@"有限前向已完成，但数值差异需复核，未自动切换默认模型";return false;}
        }
        return true;
    }();
    trace.record(@"本次唯一Flow已销毁；没有LLM/HiFT或隐藏会话池");
    return finish(ok,ok?@"五条独立固定输入的两步/重复/数值探针完成；非整链RTF、非音质验收；请分享统一测试报告":(failure?:@"W8探针提前停止，已完成结果保留"));
}
