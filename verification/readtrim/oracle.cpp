//go:build ignore

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct Slice {
    size_t start, stop;
    Slice(size_t first, size_t last) : start(first), stop(last) {}
};
// This shim only implements the one-dimensional positive unit-step slice used
// by the unchanged source. It clamps the stop like ATen and copies uint16 bits.
struct Signal {
    std::vector<uint16_t> values;
    Signal index(std::initializer_list<Slice> slices) const {
        if (slices.size() != 1) throw std::invalid_argument("only one-dimensional slices supported");
        const auto s = *slices.begin();
        const auto stop = std::min(s.stop, values.size());
        const auto start = std::min(s.start, stop);
        return {std::vector<uint16_t>(values.begin() + start, values.begin() + stop)};
    }
};
struct ReadCommon {
    struct Attributes {
        bool is_end_reason_mux_change{};
        int model_stride{};
        uint64_t num_samples{};
    } attributes;
    std::string read_id, seq, qstring;
    std::vector<uint8_t> moves;
    Signal raw_data;
    uint64_t num_minknow_events{};
    size_t pre_trim_seq_length{};
    uint64_t get_raw_data_samples() const { return raw_data.values.size(); }
};
template<class... Args> void trace_log(Args&&...) {}
namespace dorado::utils {
#include "move.slice.cpp"
#include "string.slice.cpp"
#include "index.slice.cpp"
#include "mux.slice.cpp"
}

template<class T> T get() {
    T value = 0;
    for (size_t i=0; i<sizeof(T); ++i) {
        const int c = std::cin.get();
        if (c == EOF) throw std::runtime_error("truncated input");
        value |= T(uint64_t(uint8_t(c)) << (i*8));
    }
    return value;
}
template<class T> void put(T value) {
    for (size_t i=0; i<sizeof(T); ++i) std::cout.put(char((uint64_t(value) >> (i*8)) & 255));
}
std::string get_string(size_t n) {
    std::string value(n, '\0');
    if (!std::cin.read(value.data(), n)) throw std::runtime_error("truncated string");
    return value;
}
void test_slice_shim() {
    Signal s{{0,0x8000,0x0001,0x7bff,0xfbff}};
    if (s.index({Slice(0,0)}).values != std::vector<uint16_t>{} ||
        s.index({Slice(0,3)}).values != std::vector<uint16_t>{0,0x8000,0x0001} ||
        s.index({Slice(0,5)}).values != s.values ||
        s.index({Slice(0,99)}).values != s.values ||
        Signal{}.index({Slice(0,3)}).values != std::vector<uint16_t>{})
        throw std::runtime_error("slice shim contract failure");
    auto copied = s.index({Slice(0,3)});
    copied.values[0] = 42;
    if (s.values[0] != 0) throw std::runtime_error("slice shim changed source storage");
}
int main() {
    try {
        static_assert(sizeof(int)==4 && sizeof(size_t)==8);
        test_slice_shim();
        while (std::cin.peek() != EOF) {
            ReadCommon r;
            r.attributes.is_end_reason_mux_change = get<uint8_t>() != 0;
            r.attributes.model_stride = int(get<uint32_t>());
            r.num_minknow_events = get<uint64_t>();
            r.pre_trim_seq_length = get<uint64_t>();
            auto n = get<uint32_t>();
            r.seq = get_string(n);
            r.qstring = get_string(n);
            auto moves = get<uint32_t>();
            r.moves.resize(moves);
            for (auto& m : r.moves) m = get<uint8_t>();
            auto samples = get<uint32_t>();
            r.raw_data.values.resize(samples);
            for (auto& s : r.raw_data.values) s = get<uint16_t>();
            r.attributes.num_samples = samples;
            dorado::utils::mux_change_trim_read(r);
            put<uint8_t>(r.attributes.is_end_reason_mux_change);
            put<uint32_t>(r.attributes.model_stride);
            put<uint64_t>(r.num_minknow_events);
            put<uint64_t>(r.pre_trim_seq_length);
            put<uint32_t>(r.seq.size());
            std::cout.write(r.seq.data(), r.seq.size());
            std::cout.write(r.qstring.data(), r.qstring.size());
            put<uint32_t>(r.moves.size());
            for (auto m : r.moves) put<uint8_t>(m);
            put<uint32_t>(r.raw_data.values.size());
            for (auto s : r.raw_data.values) put<uint16_t>(s);
            put<uint64_t>(r.attributes.num_samples);
        }
        if (!std::cout) throw std::runtime_error("stdout failure");
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
