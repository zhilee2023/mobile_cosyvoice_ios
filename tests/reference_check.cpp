#include "CosyVoiceReference.hpp"
#include <cassert>
#include <iostream>
int main() {
    std::vector<float> x{1,2}, mu{3,4}, cond{5,6}, mask{1}, spks{0};
    const auto first = CosyVoiceReference::digest(x,mu,cond,mask,spks);
    assert(first.size()==64 && first==CosyVoiceReference::digest(x,mu,cond,mask,spks));
    mu[0] += 1; assert(first!=CosyVoiceReference::digest(x,mu,cond,mask,spks)); mu[0] -= 1;
    assert(first!=CosyVoiceReference::digest(mu,x,cond,mask,spks));
    x.push_back(0); assert(first!=CosyVoiceReference::digest(x,mu,cond,mask,spks));
    std::cout << "PASS: SHA256 fixture identity includes values, ordering and lengths\n";
}
