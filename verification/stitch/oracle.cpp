//go:build ignore

// Protocol adapter only. The assembly algorithm is linked from untouched,
// hash-checked Dorado stitch.cpp, with only its ReadCommon container shimmed.
#include "read_pipeline/base/stitch.h"
#include "read_pipeline/base/messages/ReadCommon.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

std::string unhex(const std::string& value) {
    if (value == "-") return {};
    if (value.size() % 2) throw std::runtime_error("odd hex length");
    std::string result;
    for (std::size_t i = 0; i < value.size(); i += 2) {
        result.push_back(static_cast<char>(std::stoul(value.substr(i, 2), nullptr, 16)));
    }
    return result;
}

void hex(std::string_view value) {
    constexpr char digits[] = "0123456789abcdef";
    if (value.empty()) { std::cout << '-'; return; }
    for (unsigned char byte : value) std::cout << digits[byte >> 4] << digits[byte & 15];
}

int main() {
    try {
        std::size_t samples, stride, count;
        bool variable;
        while (std::cin >> samples >> stride >> variable >> count) {
            dorado::ReadCommon read;
            read.raw_samples = samples;
            read.attributes.model_stride = static_cast<int>(stride);
            std::vector<dorado::utils::Chunk> chunks;
            chunks.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                std::size_t offset, size, granularity;
                std::string sequence, quality, moves;
                if (!(std::cin >> offset >> size >> granularity >> sequence >> quality >> moves)) {
                    throw std::runtime_error("incomplete oracle input");
                }
                chunks.emplace_back(offset, size, granularity);
                chunks.back().seq = unhex(sequence);
                chunks.back().qstring = unhex(quality);
                auto decoded_moves = unhex(moves);
                chunks.back().moves.assign(decoded_moves.begin(), decoded_moves.end());
            }
            std::vector<const dorado::utils::Chunk*> refs;
            for (const auto& chunk : chunks) refs.push_back(&chunk);
            dorado::utils::stitch_chunks(read, refs, variable);
            hex(read.seq); std::cout << ' '; hex(read.qstring); std::cout << ' ';
            hex(std::string_view(reinterpret_cast<const char*>(read.moves.data()), read.moves.size()));
            std::cout << '\n';
        }
        if (!std::cin.eof()) throw std::runtime_error("invalid oracle input");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
