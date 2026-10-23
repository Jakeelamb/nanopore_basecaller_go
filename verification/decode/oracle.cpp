//go:build ignore

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cmath>
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

int main() {
    try {
        size_t blocks, states_count, beam_width;
        while (std::cin >> blocks >> states_count >> beam_width) {
            float cut = read_float(), blank = read_float(), shift = read_float(), scale = read_float();
            if (blocks == 0 || states_count < 4 || states_count > 16384 || beam_width == 0 || beam_width > 256)
                throw std::runtime_error("invalid driver dimensions");
            int bits = 0;
            for (auto n = states_count; n > 1; n >>= 1) ++bits;
            if ((size_t(1) << bits) != states_count || bits % 2 != 0)
                throw std::runtime_error("invalid driver state count");
            std::vector<float> scores(blocks * 4 * states_count), back((blocks + 1) * states_count), posts(back.size());
            for (auto& v : scores) v = read_float();
            for (auto& v : back) v = read_float();
            for (auto& v : posts) v = read_float();
            std::vector<int32_t> states(blocks);
            std::vector<uint8_t> moves(blocks);
            std::vector<float> qualities(4 * blocks);
            float score = dorado::basecall::decode::beam_search<float, float>(scores.data(), 4 * states_count,
                    back.data(), posts.data(), bits, blocks, beam_width, cut, blank, states, moves, qualities, 1.0f, 1.0f);
            auto [sequence, quality] = generate_sequence(moves, states, qualities, shift, scale);
            write_hex(sequence); std::cout << ' '; write_hex(quality);
            std::cout << ' ' << blocks;
            for (auto v : moves) std::cout << ' ' << unsigned(v);
            for (auto v : states) std::cout << ' ' << v;
            std::cout << ' ' << std::bit_cast<uint32_t>(score);
            for (auto v : qualities) std::cout << ' ' << std::bit_cast<uint32_t>(v);
            std::cout << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
