//go:build ignore

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <htslib/hts.h>
#include <htslib/sam.h>
#include "utils/time_utils.h"

namespace dorado {
namespace utils {
#include "mean.slice.cpp"
size_t find_rna_polya(const std::string&) { throw std::runtime_error("RNA is outside the oracle domain"); }
template<typename... T> void trace_log(T&&...) {}
}
const std::string UNCLASSIFIED = "unclassified", UNCLASSIFIED_STR = "unclassified";
enum class StrandOrientation { UNKNOWN };
char to_char(StrandOrientation) { throw std::runtime_error("primer classification is outside the oracle domain"); }
struct ReadCommon {
    struct {
        uint64_t sample_rate{};
        int mux{}, channel_number{}, read_number{}, model_stride{};
        std::string start_time, filename, pore_type, end_reason;
    } attributes;
    struct { StrandOrientation orientation{StrandOrientation::UNKNOWN}; std::string umi_tag_sequence; } primer_classification;
    std::string run_id, model_name, barcode, seq, qstring, scaling_method, parent_read_id;
    bool is_rna_model{};
    uint32_t mean_qscore_start_pos{};
    uint64_t num_trimmed_samples{}, num_minknow_events{}, raw_samples{};
    float shift{}, scale{};
    int split_point{}, rna_poly_tail_length{-2}, poly_tail_signal_anchor{};
    std::array<std::pair<int,int>,2> poly_tail_signal_boundaries{};
    std::vector<uint8_t> moves;
    static constexpr int POLY_TAIL_NOT_ENABLED = -2;
    size_t get_raw_data_samples() const { return raw_samples; }
    std::string generate_read_group() const;
    float calculate_mean_qscore() const;
    void generate_read_tags(bam1_t*,bool,bool) const;
};
#include "group.slice.cpp"
#include "quality.slice.cpp"
#include "tags.slice.cpp"
struct TrimFlags {
    bool has_adapter() const { return false; }
    bool has_primer() const { return false; }
    bool has_barcode() const { return false; }
};
#include "trim.slice.cpp"
struct Barcode { std::string barcode_name, normalized_barcode_name, alias; };
struct HtsData {
    std::unique_ptr<bam1_t, decltype(&bam_destroy1)> bam_ptr{nullptr, bam_destroy1};
    struct { std::string flowcell_id; int64_t protocol_start_time_ms{}; TrimFlags trim_flags; } read_attrs;
    std::shared_ptr<Barcode> barcoding_result;
};
namespace hts_writer {
enum class OutputMode { FASTQ };
struct HtsFileWriter {
    OutputMode m_mode{OutputMode::FASTQ};
    std::string m_gpu_names;
    void prepare_item(HtsData&) const;
};
#include "writer.slice.cpp"
}
}
#include "selected.slice.cpp"

std::string unhex(const std::string& value) {
    if (value == "-") return {};
    if (value.size()%2) throw std::runtime_error("odd oracle input hex length");
    std::string out;
    for (size_t i=0;i<value.size();i+=2) out.push_back(static_cast<char>(std::stoul(value.substr(i,2),nullptr,16)));
    return out;
}
int main() {
    try {
        std::cerr << "htslib=" << hts_version() << '\n';
        auto* output = hts_open("-", "wf");
        if (!output) throw std::runtime_error("cannot open native FASTQ stdout");
        for (const auto* tag : fastq_aux_tags) {
            if (hts_set_opt(output, FASTQ_OPT_AUX, tag) < 0) throw std::runtime_error("cannot set FASTQ tag");
        }
        std::string id, seq, quality, run, model, flow;
        uint64_t sample, run_sample_rate;
        int64_t run_acquisition_start_time_ms, protocol;
        unsigned channel, qstart;
        auto* header = sam_hdr_init();
        while (std::cin >> id >> seq >> quality >> run >> model >> flow >> channel >> sample >> run_sample_rate >> run_acquisition_start_time_ms >> protocol >> qstart) {
            if (!run_sample_rate) throw std::runtime_error("zero sample rate");
            dorado::ReadCommon read;
            read.seq=unhex(seq); read.qstring=unhex(quality); read.run_id=unhex(run); read.model_name=unhex(model);
            read.attributes.channel_number=channel; read.attributes.sample_rate=run_sample_rate;
            read.mean_qscore_start_pos=qstart;
            struct { uint64_t start_sample; } read_data{sample};
            using namespace dorado;
#include "clock.slice.cpp"
            read.attributes.start_time=start_time;
            HtsData data;
            data.bam_ptr.reset(bam_init1());
            std::string phred=read.qstring;
            for (auto& c: phred) c-=33;
            if (bam_set1(data.bam_ptr.get(),id.size(),id.c_str(),BAM_FUNMAP,-1,-1,0,0,nullptr,-1,-1,0,read.seq.size(),read.seq.c_str(),phred.c_str(),0)<0) throw std::runtime_error("cannot make oracle BAM");
            read.generate_read_tags(data.bam_ptr.get(),false,false);
            data.read_attrs.flowcell_id=unhex(flow); data.read_attrs.protocol_start_time_ms=protocol;
            hts_writer::HtsFileWriter writer;
            writer.prepare_item(data);
            std::cerr << "mean_bits=" << std::bit_cast<uint32_t>(read.calculate_mean_qscore()) << '\n';
            if (sam_write1(output,header,data.bam_ptr.get())<0) throw std::runtime_error("native FASTQ write failed");
        }
        if (!std::cin.eof()) throw std::runtime_error("incomplete oracle input");
        sam_hdr_destroy(header);
        if (hts_close(output)<0) throw std::runtime_error("native FASTQ close failed");
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
