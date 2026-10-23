//go:build ignore
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
struct Slice { uint64_t first,last; Slice(uint64_t a,uint64_t b):first(a),last(b){} };
namespace at {
struct Tensor {
    std::vector<uint16_t> values;
    std::vector<size_t> sizes() const { return {values.size()}; }
    Tensor index(std::initializer_list<Slice> intervals) const {
        if(intervals.size()!=1) throw std::runtime_error("non-unit tensor slice");
        auto s=*intervals.begin();
        auto start=std::min(s.first,uint64_t(values.size()));
        auto end=std::max(start,std::min(s.last,uint64_t(values.size())));
        return {std::vector<uint16_t>(values.begin()+start,values.begin()+end)};
    }
};
}
namespace dorado::utils {
namespace detail { [[noreturn]] void trim_throw(size_t,const std::pair<int,int>&) { throw std::runtime_error("unsupported mods branch"); } }
size_t find_rna_polya(const std::string&) { throw std::runtime_error("RNA outside oracle domain"); }
template<class... A> void trace_log(A&&...) { throw std::runtime_error("RNA outside oracle domain"); }
#include "vector.slice.cpp"
#include "string.slice.cpp"
#include "move.slice.cpp"
#include "mean.slice.cpp"
}
namespace dorado {
struct ReadCommon {
    std::string seq,qstring,read_id;
    std::vector<uint8_t> moves,base_mod_probs,base_mod_simplex_motif_hits;
    struct ModInfo { std::string alphabet; };
    std::shared_ptr<ModInfo> mod_base_info;
    struct Attributes { int model_stride; } attributes;
    at::Tensor raw_data;
    uint64_t num_trimmed_samples{},num_minknow_events{},pre_trim_seq_length{};
    size_t mean_qscore_start_pos{};
    bool is_rna_model{},is_mux{};
    float calculate_mean_qscore() const;
};
struct SimplexRead { ReadCommon read_common; };
struct Trimmer { static void trim_sequence(SimplexRead&,std::pair<int,int>,bool); };
#include "tensor.slice.cpp"
#include "trim.slice.cpp"
#include "quality.slice.cpp"
bool filtered(const ReadCommon& read_common,float m_min_qscore) {
    const size_t m_min_read_length=5;
    const std::unordered_set<std::string> m_read_ids_to_filter;
    return (
#include "filter.slice.cpp"
    );
}
}
template<class T> T get() { uint64_t value=0; for(size_t i=0;i<sizeof(T);i++){int c=std::cin.get();if(c==EOF)throw std::runtime_error("truncated input");value|=uint64_t(uint8_t(c))<<(8*i);}return T(value); }
template<class T> void put(T value) { for(size_t i=0;i<sizeof(T);i++)std::cout.put(char((uint64_t(value)>>(8*i))&255)); }
std::string get_string(size_t n) { std::string s(n,'\0');if(!std::cin.read(s.data(),n))throw std::runtime_error("truncated string");return s; }
void slice_check() {
    at::Tensor t{{0,0x8000,1,0x7bff,0xfbff}};
    if(t.index({Slice(1,4)}).values!=std::vector<uint16_t>{0x8000,1,0x7bff}||t.index({Slice(2,99)}).values!=std::vector<uint16_t>{1,0x7bff,0xfbff}||!t.index({Slice(9,99)}).values.empty()||!t.index({Slice(2,2)}).values.empty())throw std::runtime_error("slice shim failed");
}
int main() {
    try {
        static_assert(sizeof(int)==4 && sizeof(size_t)==8);
        slice_check();
        while(std::cin.peek()!=EOF) {
            uint8_t mode=get<uint8_t>();
            dorado::SimplexRead read; auto& r=read.read_common;
            auto n=get<uint32_t>();r.seq=get_string(n);r.qstring=get_string(n);
            if(mode==1) {
                r.mean_qscore_start_pos=get<uint64_t>();float threshold=std::bit_cast<float>(get<uint32_t>());
                put<uint8_t>(!dorado::filtered(r,threshold));put<uint32_t>(std::bit_cast<uint32_t>(r.calculate_mean_qscore()));
                continue;
            }
            auto start=get<uint32_t>();auto end=get<uint32_t>();
            r.num_trimmed_samples=get<uint64_t>();r.num_minknow_events=get<uint64_t>();r.pre_trim_seq_length=get<uint64_t>();r.is_mux=get<uint8_t>();r.attributes.model_stride=int(get<uint32_t>());
            auto m=get<uint32_t>();r.moves.resize(m);for(auto& v:r.moves)v=get<uint8_t>();
            auto s=get<uint32_t>();r.raw_data.values.resize(s);for(auto& v:r.raw_data.values)v=get<uint16_t>();
            dorado::Trimmer::trim_sequence(read,{int(start),int(end)},false);
            put<uint32_t>(r.seq.size());std::cout.write(r.seq.data(),r.seq.size());std::cout.write(r.qstring.data(),r.qstring.size());
            put<uint32_t>(r.moves.size());for(auto v:r.moves)put<uint8_t>(v);
            put<uint32_t>(r.raw_data.values.size());for(auto v:r.raw_data.values)put<uint16_t>(v);
            put<uint64_t>(r.num_trimmed_samples);put<uint64_t>(r.raw_data.values.size()+r.num_trimmed_samples);put<uint64_t>(r.num_minknow_events);put<uint64_t>(r.pre_trim_seq_length);put<uint8_t>(r.is_mux);put<uint32_t>(r.attributes.model_stride);
        }
        if(!std::cout)throw std::runtime_error("stdout failure");
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
