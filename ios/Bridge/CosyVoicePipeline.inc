// Same translation unit as Graph/LlmOwner; no process-global model pool.
struct PipelineLlmOwner : LlmOwner {
    PipelineLlmOwner() : LlmOwner(nullptr,&Llm::destroy) {}
};
@implementation CosyVoicePipeline {
    NSString *_directory;
    NSString *_mode;
    BOOL _useW8;
    BOOL _projection;
    BOOL _awq;
    BOOL _reuseHiFT;
    NSString *_headMode;
    BOOL _reuseHead;
    NSDictionary *_flowIdentity;
    NSDictionary *_headIdentity;
    PipelineLlmOwner _llm;
    std::unique_ptr<Graph> _flow;
    std::unique_ptr<HiFTBatchOwner> _hift;
    std::vector<std::vector<int>> _prepared;
    NSArray<NSString *> *_texts;
    size_t _at;
    std::mutex _lock;
    BOOL _closed;
    NSUInteger _requests;
    NSUInteger _reusedRequests;
    NSUInteger _flowCreates;
    NSUInteger _flowReuses;
    NSUInteger _budgetReleases;
}
+ (NSDictionary<NSString *,id> *)beginAtDirectory:(NSString *)directory mode:(NSString *)mode {
    @autoreleasepool {
        Trace trace(@"last-pipeline-lifecycle.log"); MemorySampler sampler; trace.sampler=&sampler;
        trace.record([NSString stringWithFormat:@"新整链对照 %@；4线程/Flow High两步/20ms采样；未获长文验收",mode],true);
        if (![@[@"llm-chain-reuse",@"two-sentence-staged",@"three-sentence-staged",@"six-sentence-staged",@"three-sentence-staged-w8",@"three-sentence-staged-w8-head-full",@"three-sentence-staged-w8-head-reduced",@"three-sentence-staged-w8-head-reduced-reuse",@"three-sentence-staged-projection-head-reuse",@"three-sentence-staged-projection-head-reuse-hift-reuse",@"three-sentence-staged-awq-head-reuse-hift-reuse"] containsObject:mode])
            return result(false,@"未知完整链路实验",trace);
        if (!waitAvailable(2900000000ULL,trace,@"对照启动前")) return result(false,@"初始余量不足；未加载候选模型",trace);
        CosyVoicePipeline *owner=[CosyVoicePipeline new];
        owner->_directory=[directory copy]; owner->_mode=[mode copy];
        owner->_projection=[mode hasPrefix:@"three-sentence-staged-projection-"];
        owner->_awq=[mode isEqual:@"three-sentence-staged-awq-head-reuse-hift-reuse"];
        owner->_reuseHiFT=[mode isEqual:@"three-sentence-staged-projection-head-reuse-hift-reuse"]||owner->_awq;
        // Legacy flag means compressed Flow. Its verified recipe distinguishes
        // W8 from AWQ W4; no global/default model selection changes.
        owner->_useW8=[mode hasPrefix:@"three-sentence-staged-w8"]||owner->_projection||owner->_awq;
        owner->_reuseHead=[mode isEqual:@"three-sentence-staged-w8-head-reduced-reuse"]||owner->_projection||owner->_awq;
        owner->_headMode=[mode hasSuffix:@"-head-full"]?@"full":([mode hasSuffix:@"-head-reduced"]||owner->_reuseHead?@"reduced":nil);
        NSString *failure=nil;
        if(owner->_useW8) {
            owner->_flowIdentity=owner->_awq?verifyFlowAwq(directory,trace,&failure):(owner->_projection?verifyFlowProjection(directory,trace,&failure):verifyFlowW8(directory,trace,&failure));
            if(!owner->_flowIdentity)return result(false,failure?:@"W8身份检查失败",trace);
        }
        if(owner->_headMode) {
            owner->_headIdentity=verifySpeechHead(directory,trace,&failure);
            if(!owner->_headIdentity)return result(false,failure?:@"词表裁剪候选身份检查失败",trace);
            trace.record(owner->_reuseHead?@"裁剪头批内复用：每句清KV并重置相同seed采样器；每批最多3句token，释放LLM后才加载Flow；无全模型共驻":@"输出头端到端配对：每句独立LLM、逐句相同seed；完整头/裁剪头均限制speech/EOS；input embedding/tokenizer不裁剪；不与默认mixed路线直接归因比较");
        }
        NSMutableDictionary *out=[result(true,@"本批调度器已创建；模型按阶段加载，无隐藏会话池",trace) mutableCopy];
        out[@"session"]=owner; return out;
    }
}

- (NSDictionary<NSString *,id> *)generateText:(NSString *)text lowPrecisionHiFT:(BOOL)low {
    std::lock_guard<std::mutex> guard(_lock);
    if (_closed || ![_mode isEqual:@"llm-chain-reuse"])
        return @{@"success":@NO,@"message":@"本批已关闭或调度模式不匹配"};
    if(_llm && os_proc_available_memory()<1000000000ULL) { _llm.reset(); ++_budgetReleases; }
    bool hadLlm=bool(_llm); ++_requests; if(hadLlm)++_reusedRequests;
    Trace lifecycle(@"last-pipeline-lifecycle.log");
    lifecycle.record([NSString stringWithFormat:@"请求%lu；已有LLM=%@；每句reset KV；本次声学仅一套Flow，随后释放",(unsigned long)_requests,hadLlm?@"是":@"否"]);
    NSMutableDictionary *out=[[CosyVoiceGenerate runAtDirectory:_directory text:text lowPrecisionFlow:NO
        optimized:YES cpuThreads:4 prepareHiFT:NO resident:nullptr lowPrecisionHiFT:low preparedSpeech:nullptr reuseLlm:&_llm] mutableCopy];
    if ([out[@"llmMetrics"][@"releasedBeforeFlowForBudget"] boolValue] || [out[@"llmMetrics"][@"releasedAfterResizeForBudget"] boolValue]) ++_budgetReleases;
    out[@"executionMode"]=_mode; out[@"llmPresentBeforeRequest"]=@(hadLlm);
    out[@"llmRetainedAfterRequest"]=@(bool(_llm)); out[@"reusedLlmRequests"]=@(_reusedRequests);
    out[@"llmBudgetReleases"]=@(_budgetReleases);
    if (![out[@"success"] boolValue]) { _llm.reset(); lifecycle.record(@"请求未成功：LLM已释放；若超限由上层有限拆句重试"); }
    else lifecycle.record([NSString stringWithFormat:@"请求完成；LLM仍保留=%@，累计复用%lu，规划释放%lu",_llm?@"是":@"否",(unsigned long)_reusedRequests,(unsigned long)_budgetReleases]);
    return out;
}

- (NSDictionary<NSString *,id> *)prepareTexts:(NSArray<NSString *> *)texts {
    std::lock_guard<std::mutex> guard(_lock);
    @autoreleasepool {
        Trace trace(@"last-generation.log"); MemorySampler sampler; trace.sampler=&sampler;
        const BOOL three=[_mode isEqual:@"three-sentence-staged"]||_useW8;
        const BOOL six=[_mode isEqual:@"six-sentence-staged"];
        const NSUInteger batchLimit=six?6:(three?3:2);
        trace.record([NSString stringWithFormat:@"最多%lu句分阶段：同一LLM逐句reset KV生成完整EOS token，再释放LLM；此阶段不加载Flow/HiFT",(unsigned long)batchLimit],true);
        if(_at<_prepared.size())return result(false,@"小批仍有未消费句段，不能覆盖token队列",trace);
        _hift.reset(); _flow.reset(); _llm.reset(); _prepared.clear(); _texts=nil; _at=0;
        if (_closed || (!six && !three && ![_mode isEqual:@"two-sentence-staged"]) || texts.count<1 || texts.count>batchLimit)
            return result(false,[NSString stringWithFormat:@"只允许本批1–%lu个完整句段",(unsigned long)batchLimit],trace);
        for(id text in texts) if(![text isKindOfClass:NSString.class] || [text length]<1 || [text length]>40)
            return result(false,@"句段长度或类型无效",trace);
        std::vector<int> prompt;
        NSString *csv=[NSString stringWithContentsOfFile:[_directory stringByAppendingPathComponent:@"prompt-speech-tokens.csv"] encoding:NSUTF8StringEncoding error:nil];
        for(NSString *item in [csv componentsSeparatedByCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@",\n\r "]]) if(item.length)prompt.push_back(item.intValue);
        if(prompt.size()!=87)return result(false,@"音色提示token无效",trace);
        NSMutableArray *llmRows = [NSMutableArray array];
        for(NSUInteger index=0;index<texts.count;++index) {
            std::vector<int> speech; NSString *failure=nil; bool limit=false;
            NSMutableDictionary *llmMetrics = [NSMutableDictionary dictionary];
            const bool reusedLlm=bool(_llm);
            ++_requests; if(reusedLlm)++_reusedRequests;
            trace.record([NSString stringWithFormat:@"本小批句%lu/%lu；至多105 speech token，必须正常EOS",(unsigned long)index+1,(unsigned long)texts.count]);
            // Fresh A/B stays unchanged. Reuse C refreshes the sampler at the
            // explicit reset boundary, preserving per-request seed semantics.
            const int seed=2307+int(_requests)-1;
            if(!generateSpeechTokens(_directory,texts[index],prompt,trace,speech,&failure,4,&limit,false,{},false,105,(_headMode&&!_reuseHead)?nullptr:&_llm,llmMetrics,false,_headMode,seed)) {
                // Strong out-parameter survives the helper's inner pool.
                trace.record([@"停止：" stringByAppendingString:failure?:@"LLM失败"]);
                _llm.reset(); _prepared.clear();
                NSMutableDictionary *out=[result(false,failure?:@"LLM失败",trace) mutableCopy];
                out[@"retrySplit"]=@(limit); out[@"failedTextIndex"]=@(index); return out;
            }
            llmMetrics[@"speechTokens"] = @(speech.size());
            llmMetrics[@"text"]=texts[index];llmMetrics[@"requestIndex"]=@(_requests);
            llmMetrics[@"speechTokenSHA256"]=dataSHA256([NSData dataWithBytes:speech.data() length:speech.size()*sizeof(int)]);
            if(_headMode) {
                NSMutableArray *ids=[NSMutableArray array];for(int token:speech)[ids addObject:@(token)];
                llmMetrics[@"tokens"]=ids;llmMetrics[@"eosValidated"]=@YES;
                llmMetrics[@"sampling"]=@"MNN mixed penalty/topK/topP/temperature, repetition_penalty=1.15";
                llmMetrics[@"llmReused"]=@(reusedLlm);
                llmMetrics[@"samplerResetPerRequest"]=@YES;
            }
            [llmRows addObject:llmMetrics];
            _prepared.push_back(std::move(speech));
        }
        _texts=[texts copy]; _llm.reset();
        trace.record([NSString stringWithFormat:@"本批完整token已就绪；LLM已销毁；尚未创建Flow；token队列最多%lu句",(unsigned long)batchLimit]);
        if(!waitAvailable(2500000000ULL,trace,@"分阶段释放LLM后")) { _prepared.clear(); _texts=nil; return result(false,@"LLM释放后余量未恢复",trace); }
        NSMutableDictionary *out=[result(true,[NSString stringWithFormat:@"本批1–%lu句完整EOS token已就绪；无活跃LLM/KV",(unsigned long)batchLimit],trace) mutableCopy];
        out[@"preparedCount"]=@(_prepared.size()); out[@"executionMode"]=_mode;
        out[@"llmRequests"] = llmRows;
        out[@"flowQuantization"]=_awq?@"AWQ FFN W4/g32 scaling+clipping; Q/V/out W8/g64; K preserved":(_projection?@"FFN+Q/V/out W8/block64; K preserved":(_useW8?@"FFN W8/block64":@"original FP16-stored"));
        out[@"hiFTBatchReuse"]=@(_reuseHiFT);
        out[@"headMode"]=_headMode?:@"default-full";
        out[@"headWeightReuse"] = @(_reuseHead);
        Trace life(@"last-pipeline-lifecycle.log");life.record([NSString stringWithFormat:@"已准备%zu句完整token；LLM已释放，Flow尚未加载；累计LLM请求%lu/复用%lu",_prepared.size(),(unsigned long)_requests,(unsigned long)_reusedRequests]);
        return out;
    }
}

- (NSDictionary<NSString *,id> *)generatePreparedWithLowPrecisionHiFT:(BOOL)low {
    std::lock_guard<std::mutex> guard(_lock);
    @autoreleasepool {
        Trace trace(@"last-generation.log");MemorySampler sampler;trace.sampler=&sampler;
        trace.record(@"分阶段声学：LLM已释放；本小批唯一Flow384，HiFT逐句运行；只预存有界token队列，不预存整批WAV",true);
        if(_closed || (!_useW8 && ![_mode isEqual:@"two-sentence-staged"] && ![_mode isEqual:@"three-sentence-staged"] && ![_mode isEqual:@"six-sentence-staged"]) || _at>=_prepared.size() || _llm)
            return result(false,@"没有已准备的完整句段或错误的模型共驻状态",trace);
        bool reused=bool(_flow);
        if(!_flow) {
            if(!waitAvailable(2500000000ULL,trace,@"唯一Flow创建前"))return result(false,@"Flow加载前规划余量不足",trace);
            _flow.reset(new Graph(4,true));Graph &f=*_flow;
            f.flowW8=_useW8;f.flowIdentity=_flowIdentity;
            NSString *flowFile=_useW8?_flowIdentity[@"files"][@"candidateGraph"][@"file"]:@"flow.cfg-student-2step.batch1.fp16.mnn";
            if(!f.open([_directory stringByAppendingPathComponent:flowFile],false,trace,_awq?@"分阶段唯一Flow AWQ W4 High/Memory Low":(_useW8?@"分阶段唯一Flow FFN W8 High/Memory Low":@"分阶段唯一Flow")) ||
               !(f.shape("x",{1,80,384}) && f.shape("mu",{1,80,384}) && f.shape("cond",{1,80,384}) &&
                 f.shape("mask",{1,1,384}) && f.shape("spks",{1,80}) && f.shape("t",{1}))) {
                _flow.reset();return result(false,@"分阶段Flow创建/形状失败",trace);
            }
            auto start=std::chrono::steady_clock::now();f.net->resizeSession(f.session);f.releaseModel();++_flowCreates;
            trace.record([NSString stringWithFormat:@"本小批唯一Flow resize %.3f秒；下一句不再resize",secondsSince(start)]);
            if(!sampledBudgetOK(trace) || os_proc_available_memory()<1200000000ULL) {
                _flow.reset();return memoryResult(@"分阶段Flow初始化后余量不足",trace);
            }
            // Persist setup evidence before runAtDirectory starts its fresh log.
            Trace life(@"last-pipeline-lifecycle.log");
            NSString *setup=[NSString stringWithContentsOfFile:documentPath(@"last-generation.log") encoding:NSUTF8StringEncoding error:nil];
            if(setup)life.record(setup);
        } else ++_flowReuses;
        if(!sampledBudgetOK(trace) || os_proc_available_memory()<1200000000ULL) {
            _flow.reset();return memoryResult(@"复用Flow前余量不足，未执行声学",trace);
        }
        size_t index=_at;
        if(_reuseHiFT&&!_hift)_hift.reset(new HiFTBatchOwner());
        NSMutableDictionary *out=[[CosyVoiceGenerate runAtDirectory:_directory text:_texts[index] lowPrecisionFlow:NO
            optimized:YES cpuThreads:4 prepareHiFT:NO resident:_flow.get() lowPrecisionHiFT:low preparedSpeech:&_prepared[index] reuseLlm:nullptr flowW8:NO reuseHiFT:_hift.get()] mutableCopy];
        _flow->traceLog=nullptr;
        // Include setup and every intermediate phase; never clear sampled dips.
        sampler.stop(trace);
        out[@"peakFootprintMB"]=@(std::max(trace.peakMB,[out[@"peakFootprintMB"] doubleValue]));
        out[@"minAvailableMB"]=@(std::min(trace.minAvailableMB,[out[@"minAvailableMB"] doubleValue]));
        if(!sampledBudgetOK(trace)) {out[@"success"]=@NO;out[@"failureKind"]=@"memory";out[@"message"]=@"整次声学调用含低于800 MB的采样，不计成功";}
        out[@"executionMode"]=_mode;out[@"flowReusedWithinPair"]=@(reused);
        out[@"flowReusedWithinBatch"]=@(reused);out[@"batchSize"]=@(_prepared.size());
        out[@"flowCreateCount"]=@(_flowCreates);out[@"flowReuseCount"]=@(_flowReuses);
        out[@"headMode"]=_headMode?:@"default-full";out[@"llmHeadPruned"]=@([_headMode isEqual:@"reduced"]);
        out[@"headWeightReuse"] = @(_reuseHead);
        ++_at;
        bool done=![out[@"success"] boolValue] || _at>=_prepared.size();
        Trace life(@"last-pipeline-lifecycle.log");
        life.record([NSString stringWithFormat:@"声学句%zu；复用Flow=%@；本批创建%lu/复用%lu；小批结束=%@",index+1,reused?@"是":@"否",(unsigned long)_flowCreates,(unsigned long)_flowReuses,done?@"是":@"否"]);
        if(done) {
            if(_hift)out[@"hiFTBatchFinalLifecycle"]=_hift->metrics();
            _hift.reset();_flow.reset();_prepared.clear();_texts=nil;_at=0;
            life.record(@"小批声学结束：唯一Flow、HiFT会话与token队列已释放；下一批LLM前没有HiFT常驻");
        }
        return out;
    }
}

- (NSDictionary<NSString *,id> *)close {
    std::lock_guard<std::mutex> guard(_lock);
    Trace trace(@"last-pipeline-lifecycle.log");MemorySampler sampler;trace.sampler=&sampler;
    _hift.reset();_flow.reset();_llm.reset();_prepared.clear();_texts=nil;_closed=YES;
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    trace.record([NSString stringWithFormat:@"关闭%@；全部会话/跨句KV/token队列已释放；LLM请求%lu/复用%lu，Flow创建%lu/复用%lu，LLM规划释放%lu",_mode,(unsigned long)_requests,(unsigned long)_reusedRequests,(unsigned long)_flowCreates,(unsigned long)_flowReuses,(unsigned long)_budgetReleases]);
    NSMutableDictionary *out=[result(true,@"本批完整链路候选已关闭",trace) mutableCopy];
    out[@"llmRequests"]=@(_requests);out[@"llmReusedRequests"]=@(_reusedRequests);
    out[@"flowCreates"]=@(_flowCreates);out[@"flowReuses"]=@(_flowReuses);out[@"llmBudgetReleases"]=@(_budgetReleases);
    out[@"flowQuantization"]=_awq?@"AWQ FFN W4/g32 scaling+clipping; Q/V/out W8/g64; K preserved":(_projection?@"FFN+Q/V/out W8/block64; K preserved":(_useW8?@"FFN W8/block64":@"original FP16-stored"));
    out[@"hiFTBatchReuse"]=@(_reuseHiFT);
    out[@"headMode"]=_headMode?:@"default-full";out[@"llmHeadPruned"]=@([_headMode isEqual:@"reduced"]);
    out[@"headWeightReuse"] = @(_reuseHead);
    if(_flowIdentity)out[@"flowIdentity"]=_flowIdentity;if(_headIdentity)out[@"headIdentity"]=_headIdentity;
    return out;
}
- (void)dealloc { _hift.reset();_flow.reset();_llm.reset(); }
@end
