#pragma once
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>
namespace CosyVoiceOpProfile {
struct Entry {
    std::string type, name;
    size_t calls = 0;
    double seconds = 0, maxSeconds = 0;
};
// Operator-type buckets, not claims about the selected GEMM microkernel.
inline std::string category(const std::string& type) {
    if(type=="Scale")return "scale";
    if(type.find("Convolution")!=std::string::npos)return "linear-or-convolution";
    if(type.find("MatMul")!=std::string::npos)return "matmul";
    if(type=="Raster"||type.find("Transpose")!=std::string::npos||type=="ConvertTensor"||type=="Reshape"||type=="Permute")return "layout";
    if(type.find("Norm")!=std::string::npos)return "normalization";
    if(type.find("Softmax")!=std::string::npos)return "softmax";
    if(type=="BinaryOp"||type=="UnaryOp"||type=="Eltwise"||type=="ReLU"||type=="ReLU6")return "elementwise";
    return "other";
}
// Bounded scalar metadata only. No tensors, model pointers or hidden sessions.
class Summary {
    std::map<std::pair<std::string,std::string>,Entry> entries;
public:
    static constexpr size_t limit = 4096;
    bool valid = true;
    size_t calls = 0;
    double seconds = 0;
    bool add(const std::string& type,const std::string& name,double elapsed) {
        if (!valid || !std::isfinite(elapsed) || elapsed < 0 || type.size()>256 || name.size()>2048) return valid=false;
        auto key=std::make_pair(type,name);auto found=entries.find(key);
        if(found==entries.end()) {
            if(entries.size()>=limit)return valid=false;
            found=entries.emplace(key,Entry{type,name}).first;
        }
        auto& row=found->second;++row.calls;row.seconds+=elapsed;row.maxSeconds=std::max(row.maxSeconds,elapsed);
        ++calls;seconds+=elapsed;return true;
    }
    std::vector<Entry> ranked(bool byType,size_t count) const {
        std::vector<Entry> rows;
        if(byType) {
            std::map<std::string,Entry> types;
            for(const auto& item:entries) {
                const auto& e=item.second;auto& row=types[e.type];row.type=e.type;
                row.calls+=e.calls;row.seconds+=e.seconds;row.maxSeconds=std::max(row.maxSeconds,e.maxSeconds);
            }
            for(const auto& item:types)rows.push_back(item.second);
        } else for(const auto& item:entries)rows.push_back(item.second);
        std::sort(rows.begin(),rows.end(),[](const Entry& a,const Entry& b){return a.seconds==b.seconds?std::make_pair(a.type,a.name)<std::make_pair(b.type,b.name):a.seconds>b.seconds;});
        if(rows.size()>count)rows.resize(count);return rows;
    }
    std::vector<Entry> categories() const {
        std::map<std::string,Entry> groups;
        for(const auto& item:entries) {
            const auto& e=item.second;auto key=category(e.type);auto& row=groups[key];row.type=key;
            row.calls+=e.calls;row.seconds+=e.seconds;row.maxSeconds=std::max(row.maxSeconds,e.maxSeconds);
        }
        std::vector<Entry> rows;for(const auto& item:groups)rows.push_back(item.second);
        return rows;
    }
};
}
