// A single bounded acoustic batch owns these sessions; never a global pool.
// Keep raw model metadata until destruction: later lengths may need resize.
struct HiFTBatchOwner {
    std::unique_ptr<Graph> f0, core;
    int f0Frames=0, coreFrames=0;
    bool coreLow=true;
    NSUInteger f0Creates=0, coreCreates=0, f0Resizes=0, coreResizes=0, f0Reuses=0, coreReuses=0;
    Graph *prepare(bool isCore,int frames,bool low,int threads,bool optimized,NSString *directory,Trace &trace) {
        if(frames<1 || frames>210 || threads!=4 || !optimized) return nullptr;
        auto &slot=isCore?core:f0; int &previous=isCore?coreFrames:f0Frames;
        if(isCore && slot && coreLow!=low) {trace.record(@"HiFT批内精度改变，停止；不隐藏重建会话");return nullptr;}
        if(!slot) {
            slot.reset(new Graph(threads,optimized));
            NSString *file=isCore?@"hift-core.fp32.mnn":@"hift-f0.fp32.mnn";
            if(!slot->open([directory stringByAppendingPathComponent:file],false,trace,isCore?@"批内唯一HiFT core":@"批内唯一HiFT F0",isCore&&low)) {slot.reset();return nullptr;}
            if(isCore){++coreCreates;coreLow=low;}else ++f0Creates;
        } else {if(isCore)++coreReuses;else ++f0Reuses;}
        slot->traceLog=&trace;
        if(previous!=frames) {
            const auto started=std::chrono::steady_clock::now();
            if(!slot->shape("mel",{1,80,frames}) || (isCore&&!slot->shape("source_stft",{1,18,frames*kHop/4+1}))) return nullptr;
            slot->net->resizeSession(slot->session);
            int resizeStatus=-1;
            if(!slot->net->getSessionInfo(slot->session,MNN::Interpreter::RESIZE_STATUS,&resizeStatus)||resizeStatus!=0)return nullptr;
            previous=frames; if(isCore)++coreResizes;else ++f0Resizes;
            trace.record([NSString stringWithFormat:@"批内HiFT %@ resize %d帧 %.3f秒；保留模型以支持下一长度",isCore?@"core":@"F0",frames,secondsSince(started)]);
        } else trace.record(@"HiFT同长度复用，resize=0；每次仍重传全部输入");
        return slot.get();
    }
    void unbind() {if(f0)f0->traceLog=nullptr;if(core)core->traceLog=nullptr;}
    NSDictionary *metrics() const {
        return @{@"scope":@"current acoustic batch only; destroyed before next LLM load",
          @"f0Creates":@(f0Creates),@"coreCreates":@(coreCreates),@"f0Resizes":@(f0Resizes),@"coreResizes":@(coreResizes),
          @"f0Reuses":@(f0Reuses),@"coreReuses":@(coreReuses),@"activeSessions":@((f0?1:0)+(core?1:0))};
    }
};
struct HiFTTraceBinding {
    HiFTBatchOwner *owner;
    ~HiFTTraceBinding(){if(owner)owner->unbind();}
};
