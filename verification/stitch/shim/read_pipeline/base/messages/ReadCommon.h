#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dorado {
// Test-only container shim for untouched stitch.cpp. This is algorithm parity,
// not Dorado ReadCommon ABI, tensor storage, or pipeline/runtime qualification.
class ReadCommon {
public:
    struct {
        int model_stride;
    } attributes;
    std::size_t raw_samples{};
    std::string seq;
    std::string qstring;
    std::vector<uint8_t> moves;
    std::size_t get_raw_data_samples() const { return raw_samples; }
};
}  // namespace dorado
