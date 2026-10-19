// Included after CosyVoiceGenerate's implementation in the same translation unit.
// This object belongs to ONE sentence-stream batch; no static session pool.
@implementation CosyVoiceResidentFlow {
    std::unique_ptr<Graph> _flow;
    NSString *_directory;
    std::mutex _lock;
    NSUInteger _calls;
    BOOL _closed;
    NSUInteger _fallbacks;
}

+ (NSDictionary<NSString *, id> *)beginAtDirectory:(NSString *)directory {
    @autoreleasepool {
        Trace trace(@"last-resident-lifecycle.log");
        MemorySampler sampler;
        trace.sampler = &sampler;
        trace.record(@"开始本批唯一Flow：CPU High / 384帧 / 4线程；完整语音链路试验，非长文已验收", true);
        if (!waitAvailable(2900000000ULL, trace, @"常驻初始化前"))
            return result(false, @"初始可用内存未恢复到2900 MB；请冷启动后重试", trace);
        CosyVoiceResidentFlow *owner = [CosyVoiceResidentFlow new];
        owner->_directory = [directory copy];
        owner->_flow.reset(new Graph(4, true));
        Graph &flow = *owner->_flow;
        if (!flow.open([directory stringByAppendingPathComponent:@"flow.cfg-student-2step.batch1.fp16.mnn"],
                       false, trace, @"本批唯一常驻Flow"))
            return result(false, @"常驻Flow创建失败", trace);
        if (!(flow.shape("x", {1,80,kResidentFlowFrames}) && flow.shape("mu", {1,80,kResidentFlowFrames}) &&
              flow.shape("cond", {1,80,kResidentFlowFrames}) && flow.shape("mask", {1,1,kResidentFlowFrames}) &&
              flow.shape("spks", {1,80}) && flow.shape("t", {1})))
            return result(false, @"常驻Flow输入形状失败", trace);
        auto started = std::chrono::steady_clock::now();
        flow.net->resizeSession(flow.session);
        flow.releaseModel();
        trace.record([NSString stringWithFormat:@"唯一Flow resize完成 %.3f秒；本批后续不resize、不创建第二套Flow",secondsSince(started)]);
        if (!sampledBudgetOK(trace) || !waitAvailable(1500000000ULL, trace, @"Flow初始化后"))
            return result(false, @"常驻Flow初始化后内存保护未通过", trace);
        flow.traceLog = nullptr;
        NSMutableDictionary *ready = [result(true, @"本批384帧Flow已就绪；真实整句/HiFT共驻仍需真机验证", trace) mutableCopy];
        ready[@"session"] = owner;
        ready[@"startupSeconds"] = @(secondsSince(trace.start));
        return ready;
    }
}

- (NSDictionary<NSString *, id> *)generateText:(NSString *)text {
    return [self generateText:text lowPrecisionHiFT:NO];
}

- (NSDictionary<NSString *, id> *)generateText:(NSString *)text lowPrecisionHiFT:(BOOL)low {
    std::lock_guard<std::mutex> guard(_lock);
    if (_closed) return @{@"success": @NO, @"message": @"本批已关闭，不能继续生成"};
    ++_calls;
    Trace lifecycle(@"last-resident-lifecycle.log");
    auto started = std::chrono::steady_clock::now();
    bool releasedForBudget = false;
    if (_flow && CosyVoiceBudget::shouldReleaseResident(os_proc_available_memory())) {
        lifecycle.record(@"LLM加载前不足1800 MB：先释放唯一Flow，切换本批剩余句为串行；不下调800 MB底线");
        _flow.reset(); ++_fallbacks; releasedForBudget = true;
    }
    auto runSerial = [&]() -> NSDictionary * {
        if (!waitAvailable(2800000000ULL,lifecycle,@"串行回退前"))
            return result(false,@"Flow已释放，但内存未恢复；本句未生成",lifecycle);
        return [CosyVoiceGenerate runAtDirectory:_directory text:text lowPrecisionFlow:NO
            optimized:YES cpuThreads:4 prepareHiFT:NO resident:nullptr lowPrecisionHiFT:low];
    };
    NSDictionary *output;
    if (!_flow) output = runSerial();
    else {
        output = [CosyVoiceGenerate runAtDirectory:_directory text:text lowPrecisionFlow:NO
            optimized:YES cpuThreads:4 prepareHiFT:NO resident:_flow.get() lowPrecisionHiFT:low];
        _flow->traceLog = nullptr;
        // Only explicit memory failures qualify. Never conceal NaNs, illegal
        // tokens, absent EOS, model/path failures or arbitrary native errors.
        if (CosyVoiceBudget::mayRetrySerial([output[@"success"] boolValue], [output[@"failureKind"] isEqual:@"memory"], bool(_flow))) {
            lifecycle.record(@"常驻调用触发内存保护；保存失败诊断，释放Flow后只重试当前句一次");
            NSString *failed = [NSString stringWithContentsOfFile:documentPath(@"last-generation.log") encoding:NSUTF8StringEncoding error:nil];
            if (failed) lifecycle.record([@"失败尝试（计入耗时）：\n" stringByAppendingString:failed]);
            double firstPeak = [output[@"peakFootprintMB"] doubleValue];
            double firstMin = [output[@"minAvailableMB"] doubleValue];
            _flow.reset(); ++_fallbacks; releasedForBudget = true;
            NSMutableDictionary *retry = [runSerial() mutableCopy];
            retry[@"peakFootprintMB"] = @(std::max(firstPeak,[retry[@"peakFootprintMB"] doubleValue]));
            retry[@"minAvailableMB"] = @(std::min(firstMin,[retry[@"minAvailableMB"] doubleValue]));
            retry[@"hadProtectedAttempt"] = @YES;
            output = retry;
        }
    }
    NSMutableDictionary *merged = [output mutableCopy];
    merged[@"executionMode"] = _flow ? @"resident384" : @"serial-fallback";
    merged[@"fallbackCount"] = @(_fallbacks);
    merged[@"generationIncludingRetrySeconds"] = @(secondsSince(started));
    lifecycle.record([NSString stringWithFormat:@"调用%lu：%@；本次回退=%@，累计%lu；耗时%.3f秒（含等待/失败/重试）",
        (unsigned long)_calls,merged[@"executionMode"],releasedForBudget?@"是":@"否",(unsigned long)_fallbacks,secondsSince(started)]);
    return merged;
}

- (NSDictionary<NSString *, id> *)close {
    std::lock_guard<std::mutex> guard(_lock);
    Trace trace(@"last-resident-lifecycle.log");
    MemorySampler sampler;
    trace.sampler = &sampler;
    trace.record([NSString stringWithFormat:@"关闭本批常驻Flow；调用%lu次（包括失败重试）；%@",
                  (unsigned long)_calls, _flow ? @"开始释放唯一会话" : @"已关闭，无需重复释放"]);
    _flow.reset();
    _closed = YES;
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    trace.record(@"本批Flow已释放；没有隐藏Flow池、跨句LLM或KV；内存仍可能异步回收");
    return result(true, @"常驻Flow已关闭", trace);
}

- (void)dealloc {
    _flow.reset(); // Last-resort ownership cleanup if a caller fails to close explicitly.
}
@end
