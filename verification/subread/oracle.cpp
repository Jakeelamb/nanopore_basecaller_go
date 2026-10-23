//go:build ignore

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "utils/uuid_utils.h"
#include "utils/time_utils.h"

namespace at::indexing {
struct Slice { uint64_t start, end; Slice(uint64_t a, uint64_t b): start(a), end(b) {} };
}
struct FlatSignal {
    std::shared_ptr<std::vector<uint16_t>> data = std::make_shared<std::vector<uint16_t>>();
    uint64_t first=0, last=0;
    int64_t size(int dimension) const { if (dimension!=0) throw std::runtime_error("non-flat signal"); return int64_t(last-first); }
    FlatSignal index(std::initializer_list<at::indexing::Slice> slices) const {
        if (slices.size()!=1) throw std::runtime_error("non-flat slice");
        const auto s=*slices.begin();
        if(s.start>s.end || s.end>last-first) throw std::runtime_error("slice outside storage");
        auto view=*this; view.first=first+s.start; view.last=first+s.end; return view;
    }
};
namespace dorado {
struct Attributes {
    uint32_t mux=UINT32_MAX;
    int32_t read_number=-1, channel_number=-1;
    std::string start_time, filename;
    bool is_end_reason_mux_change=false;
    std::string end_reason="unknown", pore_type="not_set";
    uint64_t sample_rate=0, num_samples=0;
    int model_stride=-1;
};
struct ReadCommon {
    FlatSignal raw_data;
    std::string read_id, seq, qstring;
    std::vector<uint8_t> moves, base_mod_probs, base_mod_simplex_motif_hits;
    std::string run_id, flow_cell_product_code, sequencing_kit, flowcell_id, position_id,
        acquisition_id, experiment_id, model_name, sample_id;
    int64_t protocol_start_time_ms=0;
    uint64_t num_minknow_events=0;
    Attributes attributes;
    uint64_t start_time_ms=0;
    size_t pre_trim_seq_length=0;
    uint64_t read_tag=0, client_info=0;
    uint32_t mean_qscore_start_pos=0;
    float shift=0, scale=0;
    std::string scaling_method, parent_read_id;
    std::shared_ptr<int> mod_base_info;
    uint64_t num_trimmed_samples=0;
    bool is_duplex=false, is_rna_model=false;
    size_t subread_id=0, split_count=1;
    uint32_t split_point=0;
    size_t get_raw_data_samples() const { if(is_duplex) throw std::runtime_error("duplex outside oracle domain"); return raw_data.size(0); }
};
struct SimplexRead {
    ReadCommon read_common;
    float digitisation=0, range=0, offset=0, open_pore_level=0, scaling=0;
    uint64_t start_sample=0, end_sample=0, run_acquisition_start_time_ms=0;
    std::string prev_read, next_read;
};
using SimplexReadPtr=std::unique_ptr<SimplexRead>;
namespace utils {
template <typename... Args> void trace_log(const char*, const Args&...) {}
#define NVTX3_FUNC_RANGE() ((void)0)
#include "moves_to_map.slice.cpp"
#include "shallow_copy_read.slice.cpp"
}
namespace splitter {
using PosRange=std::pair<uint64_t,uint64_t>;
using PosRanges=std::vector<PosRange>;
namespace {
#include "assign_subread_parent_id.slice.cpp"
}
#include "subread.slice.cpp"
class DuplexReadSplitter {
public:
    struct ExtRead { SimplexReadPtr read; };
    explicit DuplexReadSplitter(PosRanges spacers): spacers_(std::move(spacers)) {}
    ExtRead create_ext_read(SimplexReadPtr r) const { return {std::move(r)}; }
    std::vector<SimplexReadPtr> subreads(SimplexReadPtr, const PosRanges&) const;
    template<typename SplitFinder> void apply_split_finder(std::vector<ExtRead>&, const char*, const SplitFinder&) const;
    std::vector<ExtRead> apply_split_finders(ExtRead) const;
    std::vector<SimplexReadPtr> split(SimplexReadPtr) const;
private:
    PosRanges spacers_;
};
#include "subreads.slice.cpp"
#include "apply_split_finder.slice.cpp"
std::vector<DuplexReadSplitter::ExtRead> DuplexReadSplitter::apply_split_finders(ExtRead read) const {
    std::vector<ExtRead> reads;
    reads.push_back(std::move(read));
    apply_split_finder(reads,"SUPPLIED_SPACERS",[this](const ExtRead&) { return spacers_; });
    return reads;
}
#include "split.slice.cpp"
}
}

constexpr uint32_t max_count=1U<<24;
struct Input {
    template<class T> void operator()(T& value) const {
        if (!std::cin.read(reinterpret_cast<char*>(&value),sizeof(value))) throw std::runtime_error("truncated scalar");
    }
    void operator()(bool& value) const { uint8_t v=0; (*this)(v); if(v>1) throw std::runtime_error("invalid bool"); value=v; }
    void operator()(std::string& value) const {
        uint32_t n=0; (*this)(n); if(n>max_count) throw std::runtime_error("string too large");
        value.resize(n); if(!std::cin.read(value.data(),n)) throw std::runtime_error("truncated string");
    }
    template<class T> void operator()(std::vector<T>& value) const {
        uint32_t n=0; (*this)(n); if(n>max_count) throw std::runtime_error("vector too large");
        value.resize(n); for(auto& v:value) (*this)(v);
    }
};
struct Output {
    template<class T> void operator()(T& value) const { std::cout.write(reinterpret_cast<const char*>(&value),sizeof(value)); }
    void operator()(bool& value) const { uint8_t v=value; (*this)(v); }
    void operator()(std::string& value) const { uint32_t n=value.size(); (*this)(n); std::cout.write(value.data(),n); }
    template<class T> void operator()(std::vector<T>& value) const { uint32_t n=value.size(); (*this)(n); for(auto& v:value) (*this)(v); }
};
template<class Wire> void transfer(dorado::SimplexRead& read, Wire wire, bool input) {
    auto& r=read.read_common; auto& a=r.attributes;
    std::vector<uint16_t> signal;
    if(!input) signal.assign(r.raw_data.data->begin()+r.raw_data.first,r.raw_data.data->begin()+r.raw_data.last);
    wire(signal);
    if(input) { r.raw_data.data=std::make_shared<std::vector<uint16_t>>(std::move(signal)); r.raw_data.first=0; r.raw_data.last=r.raw_data.data->size(); }
    wire(r.seq); wire(r.qstring); wire(r.moves);
    uint64_t stride=uint64_t(a.model_stride); wire(stride); if(input) { if(stride==0 || stride>INT32_MAX) throw std::runtime_error("invalid stride"); a.model_stride=int(stride); }
    wire(r.read_id); wire(r.parent_read_id); wire(read.prev_read); wire(read.next_read);
    wire(read.start_sample); wire(read.end_sample); wire(read.run_acquisition_start_time_ms); wire(a.sample_rate);
    wire(r.start_time_ms); wire(a.start_time); wire(r.num_trimmed_samples); wire(r.split_point);
    wire(r.subread_id); wire(r.split_count);
    int64_t number=a.read_number; wire(number); if(input) { if(number<INT32_MIN || number>INT32_MAX) throw std::runtime_error("invalid read number"); a.read_number=int32_t(number); }
    wire(a.num_samples); wire(r.num_minknow_events); wire(r.pre_trim_seq_length); wire(r.mean_qscore_start_pos); wire(a.end_reason); wire(a.is_end_reason_mux_change);
    wire(read.digitisation); wire(read.range); wire(read.offset); wire(read.open_pore_level);
    wire(r.shift); wire(r.scale); wire(read.scaling); wire(r.scaling_method);
    wire(r.run_id); wire(r.flow_cell_product_code); wire(r.sequencing_kit); wire(r.flowcell_id); wire(r.position_id);
    wire(r.acquisition_id); wire(r.experiment_id); wire(r.model_name); wire(r.sample_id); wire(r.protocol_start_time_ms);
    wire(r.read_tag); wire(r.client_info); wire(a.mux); wire(a.channel_number); wire(a.filename); wire(a.pore_type);
}
int main(int argc, char** argv) {
    static_assert(sizeof(size_t)==8 && sizeof(int)==4);
    static_assert(std::endian::native==std::endian::little);
    static_assert(std::chrono::system_clock::duration::period::num==1 &&
                  std::chrono::system_clock::duration::period::den==1000000000 &&
                  std::numeric_limits<std::chrono::system_clock::rep>::is_signed &&
                  std::numeric_limits<std::chrono::system_clock::rep>::digits==63);
    try {
        if(argc!=1) {
            if(argc!=4 || std::string(argv[1])!="--clock-boundary") throw std::runtime_error("invalid clock probe arguments");
            auto parse=[](const std::string& s) {
                if(s.empty() || s.find_first_not_of("0123456789")!=std::string::npos) throw std::runtime_error("invalid clock probe integer");
                return std::stoull(s);
            };
            const uint64_t acquisition=parse(argv[2]), rounded_offset=parse(argv[3]);
            constexpr auto max_ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::duration::max()).count();
            if(acquisition>uint64_t(max_ms) || rounded_offset>uint64_t(max_ms)-acquisition) throw std::runtime_error("timestamp outside native signed-nanosecond clock domain");
            std::cout<<dorado::utils::get_string_timestamp_from_unix_time_ms(int64_t(acquisition+rounded_offset))<<'\n';
            return std::cout ? 0 : 2;
        }
        std::string magic(8,'\0'); std::cin.read(magic.data(),8);
        if(magic!="SUBREAD1") throw std::runtime_error("invalid protocol");
        uint32_t n=0; Input{}(n); if(n>10000) throw std::runtime_error("too many reads"); Output{}(n);
        for(uint32_t i=0;i<n;++i) {
            auto read=std::make_unique<dorado::SimplexRead>(); transfer(*read,Input{},true);
            uint32_t count=0; Input{}(count); if(count>max_count) throw std::runtime_error("too many spacers");
            dorado::splitter::PosRanges spacers;
            uint64_t previous=0;
            for(uint32_t j=0;j<count;++j) { uint64_t a=0,b=0; Input{}(a); Input{}(b); if(a>b || b>read->read_common.seq.size() || (j>0 && a<previous)) throw std::runtime_error("invalid spacer"); spacers.emplace_back(a,b); previous=b; }
            dorado::splitter::DuplexReadSplitter splitter(std::move(spacers));
            auto results=splitter.split(std::move(read)); count=results.size(); Output{}(count);
            for(auto& result:results) transfer(*result,Output{},false);
        }
        uint32_t ids=0; Input{}(ids); if(ids>10000) throw std::runtime_error("too many UUID probes"); Output{}(ids);
        for(uint32_t i=0;i<ids;++i) { std::string parent; uint64_t index=0; Input{}(parent); Input{}(index); auto id=dorado::utils::derive_uuid(parent,std::to_string(index)); Output{}(id); }
        if(std::cin.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("trailing protocol bytes");
        if(!std::cout) throw std::runtime_error("output failed");
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 2; }
}
