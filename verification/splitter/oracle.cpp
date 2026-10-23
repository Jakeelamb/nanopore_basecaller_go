//go:build ignore

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <edlib.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#if defined(__SSE2__)
#include <immintrin.h>
#elif defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#include "source/dorado/splitter/include/splitter/ReadSplitter.h"

namespace dorado {
struct SimplexRead {
    struct {
        at::Tensor raw_data;
        std::vector<uint8_t> moves;
        std::string seq, qstring, read_id;
        struct { int model_stride; } attributes;
    } read_common;
};
namespace utils {
template <typename... Args> void trace_log(const char*, const Args&...) {}
#include "move_cum_sums.slice.cpp"
}
namespace splitter {
#include "filter_ranges.slice.cpp"
namespace detail {
#include "fast_half_comparable.slice.cpp"
}
#include "detect_pore_signal.slice.cpp"
namespace {
#include "qscore_mean.slice.cpp"
#include "find_best_adapter_match.slice.cpp"
}
class DuplexReadSplitter {
public:
    explicit DuplexReadSplitter(DuplexSplitSettings settings) : m_settings(std::move(settings)) {}
    struct ExtRead;
    ExtRead create_ext_read(SimplexReadPtr) const;
    PosRanges possible_pore_regions(const ExtRead&) const;
    bool check_nearby_adapter(const SimplexRead&, PosRange, int) const;
    const DuplexSplitSettings m_settings;
};
#include "ExtRead.slice.cpp"
#include "create_ext_read.slice.cpp"
#include "possible_pore_regions.slice.cpp"
#include "check_nearby_adapter.slice.cpp"
}
}

namespace {
constexpr uint32_t max_elements = 1U << 26;
constexpr uint32_t max_reads = 100000;
template <typename T> T read_scalar() {
    static_assert(std::endian::native == std::endian::little);
    T value{};
    if (!std::cin.read(reinterpret_cast<char*>(&value), sizeof(value))) {
        throw std::runtime_error("truncated scalar");
    }
    return value;
}
uint32_t count() {
    auto value = read_scalar<uint32_t>();
    if (value > max_elements) throw std::runtime_error("count exceeds oracle limit");
    return value;
}
template <typename T> void write_scalar(T value) {
    std::cout.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
void read_bytes(char* data, size_t length) {
    if (!std::cin.read(data, static_cast<std::streamsize>(length))) {
        throw std::runtime_error("truncated payload");
    }
}
void ranges(const dorado::splitter::PosRanges& rs) {
    write_scalar<uint32_t>(static_cast<uint32_t>(rs.size()));
    for (const auto& r : rs) {
        write_scalar(r.first);
        write_scalar(r.second);
    }
}
dorado::SimplexReadPtr input() {
    auto read = std::make_unique<dorado::SimplexRead>();
    auto& r = read->read_common;
    auto n = count();
    r.seq.resize(n);
    r.qstring.resize(n);
    read_bytes(r.seq.data(), n);
    read_bytes(r.qstring.data(), n);
    auto stride = read_scalar<uint64_t>();
    if (stride == 0 || stride > INT32_MAX) throw std::runtime_error("invalid stride");
    r.attributes.model_stride = static_cast<int>(stride);
    n = count();
    if (n == 0) throw std::runtime_error("empty move table");
    r.moves.resize(n);
    read_bytes(reinterpret_cast<char*>(r.moves.data()), n);
    uint64_t bases = 0;
    for (auto m : r.moves) {
        if (m > 1) throw std::runtime_error("nonbinary move");
        bases += m;
    }
    if (bases != r.seq.size()) throw std::runtime_error("move base count differs");
    for (size_t i = 0; i < r.seq.size(); i++) {
        if (r.seq[i] != 'A' && r.seq[i] != 'C' && r.seq[i] != 'G' && r.seq[i] != 'T' && r.seq[i] != 'N')
            throw std::runtime_error("invalid DNA base");
        if (r.qstring[i] <= 33 || r.qstring[i] > 126) throw std::runtime_error("invalid quality");
    }
    n = count();
    std::vector<uint16_t> signal(n);
    read_bytes(reinterpret_cast<char*>(signal.data()), size_t(n) * 2);
    const uint64_t floor = n / stride, ceil = floor + (n % stride != 0);
    if (r.moves.size() != floor && r.moves.size() != ceil) throw std::runtime_error("signal/move stride mismatch");
    for (auto s : signal) if ((s & 0x7c00) == 0x7c00) throw std::runtime_error("nonfinite half");
    r.raw_data = at::empty({static_cast<int64_t>(n)}, at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
    if (n != 0) std::memcpy(r.raw_data.data_ptr<c10::Half>(), signal.data(), size_t(n) * 2);
    return read;
}
void run() {
    using namespace dorado::splitter;
    auto n = count();
    if (n > max_reads) throw std::runtime_error("read count exceeds oracle limit");
    write_scalar(n);
    for (uint32_t i = 0; i < n; i++) {
        DuplexSplitSettings settings(true);
        settings.simplex_mode = true;
        DuplexReadSplitter splitter(settings);
        auto read = splitter.create_ext_read(input());
        auto clusters = detect_pore_signal<c10::Half>(read.read->read_common.raw_data, settings.pore_thr,
                settings.pore_cl_dist, settings.expect_pore_prefix, settings.duration_thr);
        write_scalar<uint32_t>(static_cast<uint32_t>(clusters.size()));
        for (const auto& c : clusters) {
            write_scalar(c.start_sample);
            write_scalar(c.end_sample);
            write_scalar(c.argmax_sample);
            write_scalar(c.max_val.x);
        }
        write_scalar<uint32_t>(static_cast<uint32_t>(read.move_sums.size()));
        for (auto v : read.move_sums) write_scalar(v);
        auto all_settings = settings;
        all_settings.top_candidates = INT32_MAX;
        DuplexReadSplitter all_candidates(all_settings);
        const auto raw = all_candidates.possible_pore_regions(read);
        ranges(raw);
        write_scalar<uint32_t>(static_cast<uint32_t>(raw.size()));
        for (auto r : raw) {
            auto mean = qscore_mean(read.read->read_common.qstring, {r.first, r.first + settings.qscore_check_span});
            write_scalar(std::bit_cast<uint32_t>(mean));
        }
        ranges(read.possible_pore_regions);
        write_scalar<uint32_t>(static_cast<uint32_t>(read.possible_pore_regions.size()));
        const auto spacers = filter_ranges(read.possible_pore_regions, [&](PosRange r) {
            return splitter.check_nearby_adapter(*read.read, r, settings.adapter_edist);
        });
        for (auto r : read.possible_pore_regions) {
            const PosRange search{r.first, std::min(r.second + settings.pore_adapter_span,
                                                   uint64_t(read.read->read_common.seq.size()))};
            const auto match = find_best_adapter_match(settings.adapter, read.read->read_common.seq,
                                                       settings.adapter_edist, search);
            const bool found = splitter.check_nearby_adapter(*read.read, r, settings.adapter_edist);
            if (found != match.has_value()) throw std::runtime_error("adapter probe differs");
            write_scalar(r.first); write_scalar(r.second);
            write_scalar(search.first); write_scalar(search.second);
            write_scalar<uint8_t>(found ? 1 : 0);
            write_scalar<uint64_t>(match ? match->first : 0);
            write_scalar<uint64_t>(match ? match->second : 0);
        }
        ranges(spacers);
    }
    if (std::cin.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing input");
}
}
int main(int argc, char** argv) {
    try {
        at::set_num_threads(1);
        if (argc == 2 && std::string(argv[1]) == "--half-table") {
            write_scalar(c10::Half(2.8f).x);
            for (uint32_t bits = 0; bits <= UINT16_MAX; bits++) {
                c10::Half value(static_cast<uint16_t>(bits), c10::Half::from_bits());
                write_scalar(dorado::splitter::detail::fast_half_comparable(value));
            }
        } else if (argc == 1) {
            run();
        } else {
            throw std::runtime_error("unsupported argument");
        }
        if (!std::cout) throw std::runtime_error("output write failed");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
