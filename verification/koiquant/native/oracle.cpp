#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace utils {
// SCALED_TENSOR_BLOCK
// QUANTIZE_BLOCK
}

// RAW_IO_BLOCK

struct HalfStorage { at::Tensor t; };

utils::ScaledTensor untouched(const at::Tensor& input) {
    const int C = 512, H = 8, D = 64;
    HalfStorage wqkv_weights_f16{input.clone()};
    utils::ScaledTensor wqkv_weights_i8;
// CACHE_BLOCK
    return wqkv_weights_i8;
}

std::string dtype(const at::Tensor& t) {
    switch (t.scalar_type()) {
        case at::kHalf: return "binary16";
        case at::kChar: return "int8";
        case at::kFloat: return "float32";
        default: return "unknown-" + std::to_string(static_cast<int>(t.scalar_type()));
    }
}

bool same_bytes(const at::Tensor& left, const at::Tensor& right) {
    auto a = left.to(at::kCPU).contiguous();
    auto b = right.to(at::kCPU).contiguous();
    return a.scalar_type() == b.scalar_type() && a.sizes() == b.sizes() &&
           a.nbytes() == b.nbytes() &&
           std::memcmp(a.const_data_ptr(), b.const_data_ptr(), a.nbytes()) == 0;
}

void observe(const std::string& dir, const std::string& name, const at::Tensor& value,
             std::ofstream& metadata) {
    auto snapshot = value.to(at::kCPU).contiguous();
    save(dir + "/" + name + ".bin", snapshot);
    metadata << name << '\t' << dtype(value) << '\t' << value.device() << '\t';
    for (int i = 0; i < value.dim(); ++i) metadata << (i ? "," : "") << value.size(i);
    metadata << '\t';
    for (int i = 0; i < value.dim(); ++i) metadata << (i ? "," : "") << value.stride(i);
    metadata << '\t' << value.nbytes() << '\n';
    metadata.flush();
    if (!metadata) throw std::runtime_error("stage metadata write failed");
}

void capture(const std::string& root, const std::string& name, at::Device device) {
    const std::string dir = root + "/observations/" + name;
    if (!std::filesystem::create_directory(dir)) throw std::runtime_error("observation already exists");
    std::ofstream metadata(dir + "/stages.tsv");
    if (!metadata) throw std::runtime_error("stage metadata open failed");
    auto host = read_half(root + "/inputs/" + name + "-u16le.bin", {3, 8, 64, 512});
    observe(dir, "input", host, metadata);
    auto input = host.to(device);
    observe(dir, "transfer", input, metadata);
    if (!same_bytes(host, input)) throw std::runtime_error("raw half transfer changed input");

    auto original = untouched(input.clone());
    observe(dir, "untouched-i8tile", original.t, metadata);
    observe(dir, "untouched-freciprocal", original.scale, metadata);

    auto t = input.clone();
    auto absolute = t.abs();
    observe(dir, "abs", absolute, metadata);
    auto fp_range = absolute.amax(-1);
    observe(dir, "amax", fp_range, metadata);
    constexpr int levels = 256;
    auto quant_scale = ((levels / 2) / fp_range);
    observe(dir, "hscale", quant_scale, metadata);
    auto product = t * quant_scale.unsqueeze(-1);
    observe(dir, "hproduct", product, metadata);
    auto rounded = product.round();
    observe(dir, "hround", rounded, metadata);
    auto clipped = rounded.clip(-127, 127);
    observe(dir, "hclip", clipped, metadata);
    auto chars = clipped.to(at::ScalarType::Char);
    observe(dir, "char", chars, metadata);
    auto scale = quant_scale.to(at::ScalarType::Float);
    observe(dir, "fscale", scale, metadata);
    scale.reciprocal_();
    observe(dir, "freciprocal", scale, metadata);
    auto tiled = chars.view({3, 8, 4, 16, 32, 16}).transpose(-3, -2).contiguous();
    observe(dir, "i8tile", tiled, metadata);
    if (!same_bytes(tiled, original.t) || !same_bytes(scale, original.scale))
        throw std::runtime_error("instrumented output differs from untouched source");
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    try {
        const std::string root = argv[1], tier = argv[2], ordinal_text = argv[3];
        if (tier != "cpu" && tier != "cuda") throw std::runtime_error("unknown device tier");
        size_t consumed = 0;
        const int ordinal = std::stoi(ordinal_text, &consumed);
        if (consumed != ordinal_text.size() || ordinal < 0 || (tier == "cpu" && ordinal != 0))
            throw std::runtime_error("invalid ordinal");
        const at::Device device = tier == "cpu" ? at::Device(at::kCPU) : at::Device(at::kCUDA, ordinal);
        at::set_num_threads(1);
        at::globalContext().setDeterministicFillUninitializedMemory(false);
        at::globalContext().setDeterministicAlgorithms(true, false);
        c10::InferenceMode guard;
        if (!std::filesystem::create_directory(root + "/observations"))
            throw std::runtime_error("observations already exists");
        std::ifstream fixtures(root + "/fixture-list.txt");
        if (!fixtures) throw std::runtime_error("fixture list missing");
        std::string name;
        bool failed = false;
        while (std::getline(fixtures, name)) {
            if (name.empty() || name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-") != std::string::npos)
                throw std::runtime_error("invalid fixture name");
            try {
                capture(root, name, device);
                std::cout << "fixture\t" << name << "\tpassed\n";
            } catch (const std::exception& e) {
                std::cerr << "fixture\t" << name << "\terror\t" << e.what() << '\n';
                failed = true;
            }
        }
        std::ifstream maps("/proc/self/maps");
        std::ofstream loaded(root + "/observations/loaded-maps.txt");
        loaded << maps.rdbuf();
        if (!loaded) throw std::runtime_error("loaded maps capture failed");
        return failed ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
