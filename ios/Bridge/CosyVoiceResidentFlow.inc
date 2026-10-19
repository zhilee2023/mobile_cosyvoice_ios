// Included after CosyVoiceGenerate's implementation in the same translation unit.
// This object belongs to ONE sentence-stream batch; no static session pool.
@implementation CosyVoiceResidentFlow {
    std::unique_ptr<Graph> _flow;
    NSString *_directory;
    std::mutex _lock;
    NSUInteger _calls;
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
    std::lock_guard<std::mutex> guard(_lock);
    if (!_flow) return @{@"success": @NO, @"message": @"常驻Flow已关闭，不能继续生成"};
    ++_calls;
    NSDictionary *output = [CosyVoiceGenerate runAtDirectory:_directory text:text lowPrecisionFlow:NO
        optimized:YES cpuThreads:4 prepareHiFT:NO resident:_flow.get()];
    _flow->traceLog = nullptr; // Never retain a pointer to the returned call's stack Trace.
    return output;
}

- (NSDictionary<NSString *, id> *)close {
    std::lock_guard<std::mutex> guard(_lock);
    Trace trace(@"last-resident-lifecycle.log");
    MemorySampler sampler;
    trace.sampler = &sampler;
    trace.record([NSString stringWithFormat:@"关闭本批常驻Flow；调用%lu次（包括失败重试）；%@",
                  (unsigned long)_calls, _flow ? @"开始释放唯一会话" : @"已关闭，无需重复释放"]);
    _flow.reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    trace.record(@"本批Flow已释放；没有隐藏Flow池、跨句LLM或KV；内存仍可能异步回收");
    return result(true, @"常驻Flow已关闭", trace);
}

- (void)dealloc {
    _flow.reset(); // Last-resort ownership cleanup if a caller fails to close explicitly.
}
@end
