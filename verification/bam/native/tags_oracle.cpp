// Native tag oracle for the unaligned BAM writer.
//
// This file is a template. Each placeholder comment below is replaced by
// verification/bam/tags_oracle.go with a byte-exact slice of the pinned Dorado
// source, and the whole-file source hashes are pinned in that file. Do not edit
// the marked regions here: they are overwritten.
//
// The oracle links real htslib and calls the pinned C++ tag emitters on
// synthetic ReadCommon values. It is not a full Dorado build: the class body,
// get_raw_data_samples, generate_modbase_tags and trace_log are shims, and the
// shims are listed in docs/bam-writer.md.
#include <htslib/sam.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dorado {

inline const std::string UNCLASSIFIED_STR = "unclassified";

//__STRAND__
//__TO_CHAR__
//__BARCODE__
//__PRIMER__

struct BamDestructor {
    void operator()(bam1_t* b) const { bam_destroy1(b); }
};
using BamPtr = std::unique_ptr<bam1_t, BamDestructor>;

inline const std::string UNCLASSIFIED = "unclassified";

namespace details {
//__ATTRIBUTES__
}  // namespace details

namespace utils {
// The pinned trace_log is a logging helper. It is a no-op shim here: it has no
// effect on the emitted tag bytes.
template <typename... Args>
void trace_log(Args&&...) {}
size_t find_rna_polya(const std::string& seq);
float mean_qscore_from_qstring(std::string_view qstring);
}  // namespace utils

class ReadCommon {
public:
    static constexpr int POLY_TAIL_NOT_FOUND = -1;
    static constexpr int POLY_TAIL_NOT_ENABLED = -2;

    std::string read_id;
    std::string seq;
    std::string qstring;
    std::vector<uint8_t> moves;
    std::string run_id;
    std::string model_name;
    dorado::details::Attributes attributes;
    std::shared_ptr<BarcodeScoreResult> barcoding_result;
    PrimerClassification primer_classification{};
    uint32_t mean_qscore_start_pos = 0;
    std::string barcode{};
    float shift = 0;
    float scale = 0;
    std::string scaling_method;
    std::string parent_read_id;
    uint64_t num_trimmed_samples = 0;
    bool is_duplex{false};
    bool is_rna_model{false};
    int rna_poly_tail_length{POLY_TAIL_NOT_ENABLED};
    int poly_tail_signal_anchor{POLY_TAIL_NOT_ENABLED};
    std::array<std::pair<int, int>, 2> poly_tail_signal_boundaries;
    uint32_t split_point{0};
    uint64_t num_minknow_events{0};

    // SHIM for at::Tensor raw_data: the pinned accessor reads a tensor
    // dimension, so the oracle stores the same sample count directly.
    int64_t shim_raw_data_samples{0};
    size_t get_raw_data_samples() const { return size_t(shim_raw_data_samples); }

    std::vector<BamPtr> extract_sam_lines(bool emit_moves,
                                          std::optional<uint8_t> modbase_threshold,
                                          bool is_duplex_parent) const;
    float calculate_mean_qscore() const;

private:
    void generate_duplex_read_tags(bam1_t*) const;
    void generate_read_tags(bam1_t* aln, bool emit_moves, bool is_duplex_parent) const;
    // SHIM: the pinned body returns early whenever mod_base_info is null, which
    // is every case this oracle builds. Modified-base tags are a separate unit.
    void generate_modbase_tags(bam1_t*, std::optional<uint8_t>) const {}
    std::string generate_read_group() const;
};

//__READ_GROUP__
//__READ_TAGS__
//__DUPLEX_TAGS__
//__MEAN_QSCORE__
//__EXTRACT__

}  // namespace dorado

namespace dorado::utils {
//__FIND_POLYA__
//__MEAN_QSTRING__
}  // namespace dorado::utils

namespace {

struct Reader {
    const std::vector<uint8_t>& b;
    size_t p = 0;
    void need(size_t n) {
        if (p + n > b.size()) throw std::runtime_error("truncated fixture");
    }
    uint8_t u8() {
        need(1);
        return b[p++];
    }
    uint32_t u32() {
        need(4);
        uint32_t v = uint32_t(b[p]) | uint32_t(b[p + 1]) << 8 | uint32_t(b[p + 2]) << 16 |
                     uint32_t(b[p + 3]) << 24;
        p += 4;
        return v;
    }
    int32_t i32() { return int32_t(u32()); }
    uint64_t u64() {
        uint64_t lo = u32();
        return lo | (uint64_t(u32()) << 32);
    }
    float f32() {
        uint32_t bits = u32();
        float v;
        std::memcpy(&v, &bits, 4);
        return v;
    }
    std::string str() {
        uint32_t n = u32();
        need(n);
        std::string s(reinterpret_cast<const char*>(&b[p]), n);
        p += n;
        return s;
    }
};

void write_u32(std::ostream& out, uint32_t v) {
    unsigned char b[4] = {static_cast<unsigned char>(v), static_cast<unsigned char>(v >> 8),
                          static_cast<unsigned char>(v >> 16), static_cast<unsigned char>(v >> 24)};
    out.write(reinterpret_cast<const char*>(b), 4);
}

dorado::ReadCommon read_case(Reader& r) {
    dorado::ReadCommon c;
    c.read_id = r.str();
    c.seq = r.str();
    c.qstring = r.str();
    c.num_trimmed_samples = r.u64();
    c.shim_raw_data_samples = int64_t(r.u64());
    c.attributes.sample_rate = r.u64();
    c.attributes.mux = r.u32();
    c.attributes.channel_number = r.i32();
    c.attributes.start_time = r.str();
    c.attributes.read_number = r.i32();
    c.attributes.filename = r.str();
    c.attributes.model_stride = r.i32();
    c.attributes.pore_type = r.str();
    c.attributes.end_reason = r.str();
    c.num_minknow_events = r.u64();
    if (r.u8() != 0) {
        uint32_t n = r.u32();
        c.moves.resize(n);
        for (uint32_t i = 0; i < n; ++i) c.moves[i] = r.u8();
    }
    c.shift = r.f32();
    c.scale = r.f32();
    c.scaling_method = r.str();
    c.run_id = r.str();
    c.model_name = r.str();
    c.barcode = r.str();
    if (r.u8() != 0) {
        auto result = std::make_shared<dorado::BarcodeScoreResult>();
        result->variant = r.str();
        result->barcode_score = r.f32();
        result->top_barcode_pos.first = r.i32();
        result->top_barcode_pos.second = r.i32();
        result->top_barcode_score = r.f32();
        result->bottom_barcode_pos.first = r.i32();
        result->bottom_barcode_pos.second = r.i32();
        result->bottom_barcode_score = r.f32();
        c.barcoding_result = result;
    }
    c.primer_classification.orientation = dorado::StrandOrientation(int8_t(r.u8()));
    c.primer_classification.umi_tag_sequence = r.str();
    c.parent_read_id = r.str();
    c.split_point = r.u32();
    if (r.u8() != 0) {
        c.rna_poly_tail_length = r.i32();
        c.poly_tail_signal_anchor = r.i32();
        c.poly_tail_signal_boundaries[0].first = r.i32();
        c.poly_tail_signal_boundaries[0].second = r.i32();
        c.poly_tail_signal_boundaries[1].first = r.i32();
        c.poly_tail_signal_boundaries[1].second = r.i32();
    }
    c.is_duplex = r.u8() != 0;
    c.is_rna_model = r.u8() != 0;
    c.mean_qscore_start_pos = r.u32();
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: tags_oracle FIXTURE OUT\n";
        return 2;
    }
    try {
        std::FILE* f = std::fopen(argv[1], "rb");
        if (!f) throw std::runtime_error("cannot open fixture");
        std::vector<uint8_t> input;
        unsigned char buf[65536];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) input.insert(input.end(), buf, buf + n);
        std::fclose(f);
        Reader r{input};
        const char magic[8] = {'B', 'T', 'A', 'G', 'F', '0', '0', '1'};
        for (char ch : magic) {
            if (char(r.u8()) != ch) throw std::runtime_error("bad fixture magic");
        }
        uint32_t cases = r.u32();
        if (cases == 0 || cases > 4096) throw std::runtime_error("invalid case count");

        std::ofstream out(argv[2], std::ios::binary);
        if (!out) throw std::runtime_error("cannot open output");
        const char omagic[8] = {'B', 'T', 'A', 'G', 'R', '0', '0', '1'};
        out.write(omagic, 8);
        write_u32(out, cases);

        for (uint32_t i = 0; i < cases; ++i) {
            auto c = read_case(r);
            bool is_duplex_parent = r.u8() != 0;
            bool emit_moves = r.u8() != 0;
            auto alns = c.extract_sam_lines(emit_moves, std::nullopt, is_duplex_parent);
            if (alns.size() != 1) throw std::runtime_error("expected one alignment");
            bam1_t* aln = alns[0].get();
            const uint8_t* aux = bam_get_aux(aln);
            uint32_t len = uint32_t(bam_get_l_aux(aln));
            write_u32(out, len);
            out.write(reinterpret_cast<const char*>(aux), len);
        }
        if (r.p != input.size()) throw std::runtime_error("trailing fixture bytes");
        out.flush();
        if (!out) throw std::runtime_error("cannot write output");
        std::cout << "emitted " << cases << " native tag cases\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
