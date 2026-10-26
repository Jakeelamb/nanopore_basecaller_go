//go:build ignore

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cmath>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "beam_search.slice.cpp"
}  // namespace dorado::basecall::decode, opened by the verbatim source slice

static float read_float() {
    uint32_t bits;
    if (!(std::cin >> bits)) throw std::runtime_error("truncated float bits");
    return std::bit_cast<float>(bits);
}

static void write_hex(const std::string& value) {
    for (unsigned char c : value) std::cout << std::hex << std::setw(2) << std::setfill('0') << unsigned(c);
    std::cout << std::dec;
}

static size_t read_size(const std::string& token) {
    size_t value;
    auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
    if (error != std::errc{} || end != token.data() + token.size())
        throw std::runtime_error("invalid driver framing");
    return value;
}

static size_t read_size() {
    std::string token;
    if (!(std::cin >> token)) throw std::runtime_error("incomplete driver record");
    return read_size(token);
}

int main() {
    try {
        constexpr size_t max_elements = size_t(1) << 27;
        constexpr auto marker = "nanopore-decode-oracle/v2";
        std::string token;
        if (!(std::cin >> token)) return 0;  // Legacy empty stream.
        const bool v2 = token == marker;
        if (v2) {
            if (!(std::cin >> token)) throw std::runtime_error("incomplete v2 stream");
            std::cout << marker << '\n';
        }
        do {
            const size_t blocks = read_size(token), states_count = read_size(), beam_width = read_size();
            size_t stride = 0, span = 0;
            if (v2) { stride = read_size(); span = read_size(); }
            float cut = read_float(), blank = read_float(), shift = read_float(), scale = read_float();
            if (blocks == 0 || blocks > max_elements || states_count < 4 || states_count > 16384 || beam_width == 0 || beam_width > 256)
                throw std::runtime_error("invalid driver dimensions");
            int bits = 0;
            for (auto n = states_count; n > 1; n >>= 1) ++bits;
            if ((size_t(1) << bits) != states_count || bits % 2 != 0)
                throw std::runtime_error("invalid driver state count");
            const size_t channels = 4 * states_count;
            if (!v2) stride = channels;
            if (stride < channels || stride > max_elements ||
                blocks - 1 > (max_elements - channels) / stride ||
                blocks + 1 > max_elements / states_count || blocks + 1 > max_elements / beam_width)
                throw std::runtime_error("invalid driver span");
            const size_t expected_span = (blocks - 1) * stride + channels;
            if (v2 && span != expected_span) throw std::runtime_error("invalid driver span");
            span = expected_span;
            const float bound = float(std::numeric_limits<float>::max() / (16.0 * double(blocks + 1)));
            if (!std::isfinite(cut) || !std::isfinite(blank) || !std::isfinite(shift) ||
                !std::isfinite(scale) || scale <= 0 || (cut != 0 && cut < 1) || std::abs(blank) > bound)
                throw std::runtime_error("invalid driver options");
            std::vector<float> scores(span), back((blocks + 1) * states_count), posts(back.size());
            for (auto& v : scores) v = read_float();
            for (auto& v : back) v = read_float();
            for (auto& v : posts) v = read_float();
            for (size_t row = 0; row < blocks; ++row)
                for (size_t channel = 0; channel < channels; ++channel) {
                    const float v = scores[row * stride + channel];
                    if (!std::isfinite(v) || std::abs(v) > bound)
                        throw std::runtime_error("invalid logical score");
                }
            for (auto v : back)
                if (!std::isfinite(v) || std::abs(v) > bound) throw std::runtime_error("invalid guide");
            for (auto v : posts)
                if (!std::isfinite(v) || v < 0 || v > 1) throw std::runtime_error("invalid posterior");
            std::vector<int32_t> states(blocks);
            std::vector<uint8_t> moves(blocks);
            std::vector<float> qualities(4 * blocks);
            float score = dorado::basecall::decode::beam_search<float, float>(scores.data(), stride,
                    back.data(), posts.data(), bits, blocks, beam_width, cut, blank, states, moves, qualities, 1.0f, 1.0f);
            auto [sequence, quality] = generate_sequence(moves, states, qualities, shift, scale);
            if (v2) std::cout << stride << ' ' << span << ' ';
            write_hex(sequence); std::cout << ' '; write_hex(quality);
            std::cout << ' ' << blocks;
            for (auto v : moves) std::cout << ' ' << unsigned(v);
            for (auto v : states) std::cout << ' ' << v;
            std::cout << ' ' << std::bit_cast<uint32_t>(score);
            for (auto v : qualities) std::cout << ' ' << std::bit_cast<uint32_t>(v);
            std::cout << '\n';
        } while (std::cin >> token);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
