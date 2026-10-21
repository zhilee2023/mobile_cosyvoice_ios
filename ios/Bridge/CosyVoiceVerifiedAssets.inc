// Included inside the existing helper namespace. No global verification cache;
// every generation/pipeline owner validates its own selected assets once.
NSString *dataSHA256(NSData *data) {
    unsigned char bytes[CC_SHA256_DIGEST_LENGTH];CC_SHA256(data.bytes,CC_LONG(data.length),bytes);
    NSMutableString *hex=[NSMutableString string];for(unsigned char b:bytes)[hex appendFormat:@"%02x",b];return hex;
}
NSString *streamSHA256(NSString *path) {
    FILE *file=std::fopen(path.fileSystemRepresentation,"rb");if(!file)return nil;
    CC_SHA256_CTX hash;CC_SHA256_Init(&hash);std::vector<unsigned char> buffer(1024*1024);size_t n;
    while((n=std::fread(buffer.data(),1,buffer.size(),file)))CC_SHA256_Update(&hash,buffer.data(),CC_LONG(n));
    bool ok=!std::ferror(file);std::fclose(file);if(!ok)return nil;
    unsigned char bytes[CC_SHA256_DIGEST_LENGTH];CC_SHA256_Final(bytes,&hash);
    NSMutableString *hex=[NSMutableString string];for(unsigned char b:bytes)[hex appendFormat:@"%02x",b];return hex;
}
NSDictionary *verifySelectedAssets(NSString *directory,NSString *resource,NSString *recipe,
                                  NSArray<NSString *> *keys,bool nested,Trace &trace,CosyVoiceFailureOutput failure) {
    const auto start=std::chrono::steady_clock::now();
    NSURL *url=[NSBundle.mainBundle URLForResource:resource withExtension:@"json"];
    NSData *data=url?[NSData dataWithContentsOfURL:url]:nil;
    id parsed=data?[NSJSONSerialization JSONObjectWithData:data options:0 error:nil]:nil;
    NSDictionary *manifest=[parsed isKindOfClass:NSDictionary.class]?parsed:nil;
    if(![manifest[@"recipe"] isEqual:recipe]){*failure=@"包内模型身份清单缺失或不兼容";return nil;}
    NSDictionary *entries=nested?manifest[@"files"]:manifest;
    NSMutableDictionary *verified=[NSMutableDictionary dictionary];
    for(NSString *key in keys) {
        NSDictionary *entry=entries[key];NSString *file=entry[@"file"];
        if(![entry isKindOfClass:NSDictionary.class]||![file isKindOfClass:NSString.class]||!file.length||
           ![file.lastPathComponent isEqual:file]||[file hasPrefix:@"."]){*failure=@"模型清单文件名无效";return nil;}
        NSString *path=[directory stringByAppendingPathComponent:file];
        NSDictionary *attrs=[NSFileManager.defaultManager attributesOfItemAtPath:path error:nil];
        if(![attrs[NSFileType] isEqual:NSFileTypeRegular]||[attrs[NSFileSize] unsignedLongLongValue]!=[entry[@"bytes"] unsignedLongLongValue]) {
            *failure=[NSString stringWithFormat:@"模型缺失或大小不符：%@；原文件不替换，请导入相应候选文件",file];return nil;
        }
        NSString *actual=streamSHA256(path);
        if(!actual||![actual isEqual:entry[@"sha256"]]){*failure=[NSString stringWithFormat:@"%@ SHA256不匹配；没有自动回退原模型",file];return nil;}
        verified[key]=entry;
    }
    trace.record([NSString stringWithFormat:@"%@ 所选文件SHA256通过，%.3f秒；只在本次拥有者初始化时校验；耗时计入整链",recipe,secondsSince(start)]);
    return @{@"recipe":recipe,@"files":verified,@"manifestSHA256":dataSHA256(data),@"verificationSeconds":@(secondsSince(start))};
}
NSDictionary *verifyFlowW8(NSString *directory,Trace &trace,CosyVoiceFailureOutput failure) {
    return verifySelectedAssets(directory,@"flow-w8-manifest",@"cosyvoice3-flow-ffn-w8-g64-v1",
                                @[@"candidateGraph",@"candidateWeight"],true,trace,failure);
}
NSDictionary *verifySpeechHead(NSString *directory,Trace &trace,CosyVoiceFailureOutput failure) {
    return verifySelectedAssets(directory,@"speech-head-manifest",@"cosyvoice3-w8-speech-head-v1",
                                @[@"sourceGraph",@"sourceWeight",@"candidateGraph",@"candidateWeight"],false,trace,failure);
}
NSDictionary *verifyFlowProjection(NSString *directory,Trace &trace,CosyVoiceFailureOutput failure) {
    return verifySelectedAssets(directory,@"flow-projection-manifest",@"cosyvoice3-flow-projection-w8-g64-v1",
                                @[@"candidateGraph",@"candidateWeight"],true,trace,failure);
}
