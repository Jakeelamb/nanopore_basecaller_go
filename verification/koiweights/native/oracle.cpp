#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <torch/serialize.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <memory>

// ROUNDING_BLOCK

void save(const std::string& path, const at::Tensor& value) {
    auto t = value.contiguous();
    if (!t.device().is_cpu()) throw std::runtime_error("non-CPU output");
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char*>(t.const_data_ptr()), t.nbytes());
    if (!out) throw std::runtime_error("write failed " + path);
}

at::Tensor read_half(const std::string& path, std::vector<int64_t> shape) {
    auto t = at::empty(shape, at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
    std::ifstream in(path, std::ios::binary);
    in.read(static_cast<char*>(t.mutable_data_ptr()), t.nbytes());
    if (!in || in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("half input byte count");
    return t;
}

struct LinearStorage { at::Tensor weight; };
struct AttentionStorage { std::shared_ptr<LinearStorage> wqkv; };
struct HalfStorage { at::Tensor t; };

void pack(const at::Tensor& weight, const std::string& prefix) {
    const int C = 512, H = 8, D = 64;
    auto self_attn = std::make_shared<AttentionStorage>();
    self_attn->wqkv = std::make_shared<LinearStorage>();
    self_attn->wqkv->weight = weight;
    HalfStorage wqkv_weights_f16;
    // INTERLEAVE_BLOCK
    save(prefix + "-interleaved-u16le.bin", wqkv_weights_f16.t);
    // TILE_BLOCK
    save(prefix + "-tiled-u16le.bin", wqkv_weights_f16.t);
    apply_rounding(wqkv_weights_f16.t, 4);
    save(prefix + "-rounded-u16le.bin", wqkv_weights_f16.t);
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    try {
        at::set_num_threads(1);
        at::globalContext().setDeterministicFillUninitializedMemory(false);
        at::globalContext().setDeterministicAlgorithms(true, false);
        c10::InferenceMode guard;
        const std::string root = argv[1];
        auto all = read_half(root + "/inputs/all-half-u16le.bin", {65536});
        apply_rounding(all, 4);
        save(root + "/expected/all-half-rounded-u16le.bin", all);
        for (const std::string name : {"qkv-axes", "qkv-half-patterns"}) {
            auto raw = read_half(root + "/inputs/" + name + "-u16le.bin", {1536, 512});
            pack(raw, root + "/expected/" + name);
        }
        std::vector<at::Tensor> weights;
        torch::load(weights, argv[2], at::Device(at::kCPU));
        if (weights.size() != 1 || weights[0].scalar_type() != at::kFloat ||
            weights[0].sizes() != at::IntArrayRef({1536, 512}) || !weights[0].device().is_cpu())
            throw std::runtime_error("weight must be one CPU float32 [1536,512] tensor");
        auto weight = weights[0].contiguous();
        save(root + "/inputs/genuine-f32le.bin", weight);
        auto half = weight.to(at::kHalf);
        save(root + "/inputs/genuine-u16le.bin", half);
        pack(half, root + "/expected/genuine");
        std::ifstream maps("/proc/self/maps");
        std::ofstream loaded(root + "/loaded-maps.txt");
        loaded << maps.rdbuf();
        if (!loaded) throw std::runtime_error("maps capture failed");
        std::cout << "CPU half packing and one genuine archive load, one float32-to-half cast\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
