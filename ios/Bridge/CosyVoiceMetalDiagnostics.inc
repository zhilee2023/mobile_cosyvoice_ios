// Observational diagnostic: callbacks synchronize and alter timing. Keep at
// most eight op summaries; no retained intermediate tensors or tensor dumps.
NSDictionary *metalTensorStats(MNN::Tensor *tensor) {
    if (!tensor) return @{@"status": @"missing", @"invalid": @0};
    const auto type = tensor->getType();
    NSMutableArray *shape = [NSMutableArray array];
    for (int dimension : tensor->shape()) [shape addObject:@(dimension)];
    NSMutableDictionary *stats = [@{@"shape": shape, @"bits": @(type.bits), @"code": @(type.code), @"invalid": @0} mutableCopy];
    if (type.code != halide_type_float || (type.bits != 32 && type.bits != 16)) {
        stats[@"status"] = @"non-float-skipped"; return stats;
    }
    const int count = tensor->elementSize();
    if (count <= 0 || uint64_t(count) * (type.bits / 8) > 8 * 1024 * 1024) {
        stats[@"status"] = @"size-limit-skipped"; return stats;
    }
    // CAFFE host layout strips physical channel padding; never scan padded,
    // uninitialized NC4HW4 lanes as though they were logical model outputs.
    MNN::Tensor host(tensor, MNN::Tensor::CAFFE, true);
    if (host.usize() < size_t(count) * (type.bits / 8) || !host.host<uint8_t>() || !tensor->copyToHostTensor(&host)) {
        stats[@"status"] = @"copy-failed"; return stats;
    }
    size_t invalid = 0, nanCount = 0, infCount = 0, first = 0;
    double maximum = 0;
    for (int index = 0; index < count; ++index) {
        double value;
        if (type.bits == 32) value = host.host<float>()[index];
        else {
            const uint16_t bits = host.host<uint16_t>()[index];
            const int exponent = (bits >> 10) & 31, mantissa = bits & 1023;
            if (exponent == 31) value = mantissa ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
            else value = exponent ? std::ldexp(double(1024 + mantissa), exponent - 25) : std::ldexp(double(mantissa), -24);
            if (bits & 0x8000) value = -value;
        }
        if (!std::isfinite(value)) {
            if (!invalid) first = index;
            ++invalid; nanCount += std::isnan(value); infCount += std::isinf(value);
        } else maximum = std::max(maximum, std::abs(value));
    }
    stats[@"status"] = @"read"; stats[@"elements"] = @(count); stats[@"invalid"] = @(invalid);
    stats[@"nanCount"] = @(nanCount); stats[@"infCount"] = @(infCount); stats[@"finitePeak"] = @(maximum);
    if (invalid) stats[@"firstInvalidIndex"] = @(first);
    return stats;
}

NSDictionary *diagnoseMetalNaN(Graph &flow, Trace &trace) {
    auto start = std::chrono::steady_clock::now();
    NSMutableDictionary *diagnostic = [@{@"diagnosticOnly": @YES, @"timingIsBenchmark": @NO,
        @"foundNonFinite": @NO, @"completed": @NO} mutableCopy];
    NSMutableArray *recent = [NSMutableArray array];
    __strong NSArray *inputsBefore = @[];
    size_t visited = 0, inspected = 0, skipped = 0;
    auto snapshots = [&](const std::vector<MNN::Tensor *> &tensors) -> NSArray * {
        NSMutableArray *items = [NSMutableArray array];
        for (MNN::Tensor *tensor : tensors) {
            NSDictionary *item = metalTensorStats(tensor); [items addObject:item];
            if ([item[@"status"] isEqual:@"read"]) ++inspected;
            else if (![item[@"status"] isEqual:@"non-float-skipped"]) ++skipped;
        }
        return items;
    };
    auto before = [&](const std::vector<MNN::Tensor *> &tensors, const MNN::OperatorInfo *) -> bool {
        inputsBefore = snapshots(tensors); return true;
    };
    auto after = [&](const std::vector<MNN::Tensor *> &tensors, const MNN::OperatorInfo *info) -> bool {
        ++visited;
        NSString *name = info ? [NSString stringWithUTF8String:info->name().c_str()] : @"unknown";
        NSString *type = info ? [NSString stringWithUTF8String:info->type().c_str()] : @"unknown";
        if (!name) name = @"invalid-UTF8-name";
        if (!type) type = @"invalid-UTF8-type";
        if (!sampledBudgetOK(trace)) { diagnostic[@"stopReason"] = @"memory-below-800MB"; return false; }
        if (secondsSince(start) > 60) { diagnostic[@"stopReason"] = @"60-second-time-limit"; return false; }
        NSArray *outputs = snapshots(tensors);
        NSDictionary *op = @{@"index": @(visited), @"name": name, @"type": type,
                              @"inputsBefore": inputsBefore, @"outputsAfter": outputs};
        [recent addObject:op]; if (recent.count > 8) [recent removeObjectAtIndex:0];
        bool inputBad = false, outputBad = false, copyFailed = false;
        for (NSDictionary *item in inputsBefore) { inputBad |= [item[@"invalid"] unsignedLongLongValue] != 0; copyFailed |= [item[@"status"] isEqual:@"copy-failed"]; }
        for (NSDictionary *item in outputs) { outputBad |= [item[@"invalid"] unsignedLongLongValue] != 0; copyFailed |= [item[@"status"] isEqual:@"copy-failed"]; }
        if (inputBad || outputBad) {
            diagnostic[@"foundNonFinite"] = @YES; diagnostic[@"completed"] = @YES;
            diagnostic[@"firstObservedOp"] = op; diagnostic[@"inputsAlreadyNonFinite"] = @(inputBad);
            diagnostic[@"stopReason"] = @"first-observed-nonfinite";
            trace.record([NSString stringWithFormat:@"首个观测到非有限值：第%zu个算子 %@ / %@；输入已有异常=%@；停止后续算子", visited, name, type, inputBad ? @"是" : @"否"]);
            return false;
        }
        if (copyFailed) { diagnostic[@"stopReason"] = @"tensor-copy-failed"; return false; }
        if (visited == 1 || visited % 32 == 0)
            trace.record([NSString stringWithFormat:@"逐算子定位进度 %zu：%@ / %@；已读取%zu个浮点tensor，跳过%zu个", visited, name, type, inspected, skipped]);
        return sampledBudgetOK(trace);
    };
    trace.record(@"开始逐算子同步诊断：每次最多读取8MB逻辑tensor，最多60秒；只定位第一步，不作为速度数据");
    const auto code = flow.net->runSessionWithCallBackInfo(flow.session, before, after, true);
    diagnostic[@"runReturnCode"] = @(int(code)); diagnostic[@"visitedOps"] = @(visited);
    diagnostic[@"inspectedFloatTensors"] = @(inspected); diagnostic[@"skippedTensors"] = @(skipped);
    diagnostic[@"recentOps"] = recent; diagnostic[@"diagnosticWallSeconds"] = @(secondsSince(start));
    if (code == MNN::NO_ERROR && visited && !diagnostic[@"stopReason"]) {
        diagnostic[@"completed"] = @YES; diagnostic[@"stopReason"] = @"full-first-step-no-observed-nonfinite";
    } else if (!diagnostic[@"stopReason"]) diagnostic[@"stopReason"] = visited ? @"runtime-error" : @"callbacks-not-observed";
    trace.record([NSString stringWithFormat:@"逐算子诊断返回：%@；算子%zu；读取%zu，跳过%zu；code=%d", diagnostic[@"stopReason"], visited, inspected, skipped, int(code)]);
    return diagnostic;
}
