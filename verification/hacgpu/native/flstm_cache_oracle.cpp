// H04 native FLSTM weight-cache oracle.
//
// This program is verification-only. It links the pinned Dorado static libraries
// from the qualified GCC15 source build, loads the untouched HAC graph through the
// stock `load_crf_model` entry point, runs the graph once so the stock
// `FLSTMStackImpl::forward_koi` populates its device weight caches exactly as it
// does on the first real forward pass, and then dumps:
//
//   * the seven device cache tensors per layer that the kernels actually consume,
//     and
//   * the pre-permutation `utils::quantize_tensor` outputs those caches are built
//     from (FLSTMStack.cpp:165-217).
//
// Nothing here is a port. The quantization, the half casts and the permutations
// are all stock code. Before writing anything the oracle re-applies the source's
// own permute expressions to the freshly recomputed intermediates and requires
// the result to equal the live cache byte for byte, so the intermediates it
// publishes are provably the ones the cache was built from.
//
// The input only has to be a valid declared case: the caches are a function of the
// model weights, not of the signal. The capture reuses the admitted H01 input so
// the whole run stays tied to one declared case.

#include "basecall/crf_utils.h"
#include "config/BasecallModelConfig.h"
#include "model/CRFModel.h"
#include "nn/ConvStack.h"
#include "nn/RNNStack.h"
#include "nn/WorkingMemory.h"
#include "torch_utils/tensor_utils.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/nn.h>

// Verification-only access shim: the pinned headers are included with `private`
// rewritten to `public` so the oracle can read the layer list and the device
// caches the stock code keeps private. It does not modify the stock source, does
// not change any object layout, and is confined to this translation unit. It is
// applied only after the torch/ATen headers are already included, because
// rewriting `private` inside the standard library headers breaks them.
#define private public
#include "nn/FLSTMStack.h"
#undef private

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using dorado::nn::FLSTMStackImpl;
using dorado::nn::TensorLayout;
using dorado::nn::WorkingMemory;

struct Options {
    std::string model_dir;
    std::string input;
    std::string out;
    std::string case_name;
    int ordinal = 0;
    int batch = 1;
    int chunk = 0;
};

std::string json_escape(const std::string &in) {
    std::ostringstream out;
    for (unsigned char c : in) {
        switch (c) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c)
                        << std::dec;
                } else {
                    out << char(c);
                }
        }
    }
    return out.str();
}

std::string hex_sha(const void *data, size_t bytes) {
    // Small self-contained SHA-256 so the oracle does not depend on openssl headers.
    static const uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
            0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
            0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
            0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::vector<uint8_t> buf(static_cast<const uint8_t *>(data),
                             static_cast<const uint8_t *>(data) + bytes);
    buf.push_back(0x80);
    while (buf.size() % 64 != 56) {
        buf.push_back(0);
    }
    uint64_t bits = uint64_t(bytes) * 8;
    for (int i = 7; i >= 0; --i) {
        buf.push_back(uint8_t((bits >> (8 * i)) & 0xff));
    }
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t off = 0; off < buf.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(buf[off + 4 * i]) << 24) | (uint32_t(buf[off + 4 * i + 1]) << 16) |
                   (uint32_t(buf[off + 4 * i + 2]) << 8) | uint32_t(buf[off + 4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }
    std::ostringstream out;
    for (uint32_t v : h) {
        out << std::hex << std::setw(8) << std::setfill('0') << v;
    }
    return out.str();
}

std::vector<char> read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    in.seekg(0, std::ios::end);
    auto size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<char> body(static_cast<size_t>(size));
    if (!body.empty() && !in.read(body.data(), size)) {
        throw std::runtime_error("cannot read " + path);
    }
    return body;
}

std::string dump_tensor(const at::Tensor &t, const std::string &path) {
    auto cpu = t.detach().to(at::kCPU).contiguous();
    auto bytes = cpu.numel() * cpu.element_size();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
    out.write(static_cast<const char *>(cpu.data_ptr()), bytes);
    out.close();
    return hex_sha(cpu.data_ptr(), size_t(bytes));
}

std::string shape_json(at::IntArrayRef sizes) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (i) out << ",";
        out << sizes[i];
    }
    out << "]";
    return out.str();
}

std::string dtype_name(at::ScalarType t) { return c10::toString(t); }

// same_bytes compares two tensors by their live bytes. The caches are dense
// tensors the source builds with `.contiguous()`, so there is nothing to mask.
bool same_bytes(const at::Tensor &a, const at::Tensor &b) {
    if (a.scalar_type() != b.scalar_type() || a.sizes() != b.sizes()) {
        return false;
    }
    auto x = a.detach().to(at::kCPU).contiguous();
    auto y = b.detach().to(at::kCPU).contiguous();
    const auto bytes = size_t(x.numel() * x.element_size());
    return bytes == size_t(y.numel() * y.element_size()) &&
           std::memcmp(x.data_ptr(), y.data_ptr(), bytes) == 0;
}

// dump_runtime_paths records exactly which shared objects this process mapped, so
// the orchestrator can hash the files that actually ran rather than the sonames
// the link line asked for.
void dump_runtime_paths(const std::string &path) {
    std::ifstream maps("/proc/self/maps");
    if (!maps) {
        throw std::runtime_error("cannot read /proc/self/maps");
    }
    std::vector<std::string> seen;
    std::string line;
    while (std::getline(maps, line)) {
        std::istringstream fields(line);
        std::vector<std::string> parts;
        std::string token;
        while (fields >> token) {
            parts.push_back(token);
        }
        if (parts.size() < 6 || parts[0].find('-') == std::string::npos) {
            continue;
        }
        std::string file;
        for (size_t i = 5; i < parts.size(); ++i) {
            if (i > 5) file += " ";
            file += parts[i];
        }
        auto slash = file.find_last_of('/');
        std::string name = slash == std::string::npos ? file : file.substr(slash + 1);
        auto prefixed = [&](const char *p) { return name.rfind(p, 0) == 0; };
        bool family = prefixed("libtorch") || prefixed("libc10") || prefixed("libcu") ||
                      prefixed("libnv") || prefixed("libz") || prefixed("libgomp") ||
                      prefixed("libiomp") || prefixed("libstdc++");
        if (!family) {
            continue;
        }
        if (std::find(seen.begin(), seen.end(), file) == seen.end()) {
            seen.push_back(file);
        }
    }
    std::sort(seen.begin(), seen.end());
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot write " + path);
    }
    out << "{\n  \"schema\": \"nanopore-hacgpu-runtime-paths/v1\",\n  \"paths\": [";
    for (size_t i = 0; i < seen.size(); ++i) {
        out << (i ? ",\n    " : "\n    ") << "\"" << json_escape(seen[i]) << "\"";
    }
    out << "\n  ]\n}\n";
}

// Dumped is one published tensor: its role, its file, and the facts a consumer
// must agree with before it compares bytes.
struct Dumped {
    std::string role;
    std::string file;
    std::string sha;
    std::string shape;
    std::string dtype;
    bool contiguous = false;
};

struct LayerDump {
    int layer = -1;
    bool reverse = false;
    std::vector<Dumped> tensors;
    // Per-row quantization domain, measured from the intermediate scale tensor.
    float scale_min = 0;
    float scale_max = 0;
    int zero_rows = 0;
    int nonfinite_rows = 0;
};

Options parse(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (a == "--model-dir") {
            o.model_dir = need("--model-dir");
        } else if (a == "--input") {
            o.input = need("--input");
        } else if (a == "--out") {
            o.out = need("--out");
        } else if (a == "--case") {
            o.case_name = need("--case");
        } else if (a == "--ordinal") {
            o.ordinal = std::stoi(need("--ordinal"));
        } else if (a == "--batch") {
            o.batch = std::stoi(need("--batch"));
        } else if (a == "--chunk") {
            o.chunk = std::stoi(need("--chunk"));
        } else {
            throw std::runtime_error("unknown argument " + a);
        }
    }
    if (o.model_dir.empty() || o.input.empty() || o.out.empty()) {
        throw std::runtime_error("--model-dir, --input and --out are required");
    }
    return o;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options opt = parse(argc, argv);
        const auto config = dorado::config::load_model_config(opt.model_dir);
        if (config.is_tx_model()) {
            throw std::runtime_error("H04 oracle expects the CRF (LSTM/FLSTM) graph, not a TX model");
        }
        if (!config.is_flstm_model()) {
            throw std::runtime_error("H04 oracle expects the pinned FLSTM HAC graph");
        }
        if (!config.lstm_inner_dim.has_value()) {
            throw std::runtime_error("H04 oracle expects a factorised FLSTM (lstm_inner_dim)");
        }

        const auto device = at::Device(at::kCUDA, opt.ordinal);
        const auto opts = at::TensorOptions().dtype(at::kHalf).device(device);
        c10::cuda::CUDAGuard guard(device);

        auto holder = dorado::basecall::load_crf_model(config, opts);
        auto &model = holder->get<dorado::basecall::model::CRFModelImpl>();

        auto *rnns_impl = dynamic_cast<FLSTMStackImpl *>(model.rnns.get());
        if (rnns_impl == nullptr) {
            throw std::runtime_error("H04 oracle expects an FLSTM recurrent stack");
        }
        if (rnns_impl->first_reverse_) {
            // Recorded, not assumed: the reverse parity below comes from this flag.
        }
        const int C = rnns_impl->C_;
        const int K = rnns_impl->K_;
        if (C <= 0 || K <= 0) {
            throw std::runtime_error("H04 oracle read a non-positive FLSTM C_/K_");
        }

        // The declared case's input. Its bytes do not influence the caches, but
        // running the same admitted input keeps this capture tied to the case the
        // downstream replay uses.
        const auto raw = read_file(opt.input);
        const int64_t features = config.num_features;
        int64_t chunk = opt.chunk;
        if (chunk == 0) {
            const int64_t elems = int64_t(raw.size() / 2);
            if (elems % (opt.batch * features) != 0) {
                throw std::runtime_error("input payload does not divide into [batch, features, T]");
            }
            chunk = elems / (opt.batch * features);
        }
        if (int64_t(raw.size()) != int64_t(opt.batch) * features * chunk * 2) {
            throw std::runtime_error("input payload size does not match [batch, features, T]");
        }
        auto input_cpu = at::from_blob(const_cast<char *>(raw.data()),
                                       {opt.batch, features, chunk},
                                       at::TensorOptions().dtype(at::kHalf));
        auto input = input_cpu.to(device, /*non_blocking=*/false, /*copy=*/true).contiguous();

        // Literal replay of CRFModelImpl::run_koi up to and including the recurrent
        // stack (CRFModel.cpp:87-145). `forward_koi` builds one device cache set per
        // layer on the first call (FLSTMStack.cpp:165-217); this is that call.
        WorkingMemory wm(int(input.size(0)));
        wm.next_TC(int(chunk), int(features), TensorLayout::NTC);
        model.convs->reserve_working_memory(wm, nullptr, std::nullopt);
        model.rnns->reserve_working_memory(wm);
        model.linear1->reserve_working_memory(wm, nullptr);
        if (model.linear2) {
            model.linear2->reserve_working_memory(wm, nullptr);
        }
        wm.allocate_backing_tensor(input.device());
        auto wm_in = wm.next_TC(int(chunk), int(features), TensorLayout::NTC);
        wm_in.index({torch::indexing::Slice()}) = input.transpose(1, 2);

        model.convs->run_koi(wm, nullptr);
        model.rnns->run_koi(wm, nullptr);

        const int layers = int(std::ssize(rnns_impl->layers_));
        if (layers != config.lstm_layers) {
            throw std::runtime_error("recurrent stack layer count differs from the config");
        }
        // Every cache vector must hold exactly one entry per layer: `forward_koi`
        // appends when `size() == layer`, so a short vector means a layer never
        // built its cache.
        const std::vector<std::pair<const char *, const std::vector<at::Tensor> *>> caches = {
                {"dn_weights_ih", &rnns_impl->device_dn_weights_ih_},
                {"dn_weights_scale_ih", &rnns_impl->device_dn_weights_scale_ih_},
                {"dn_weights_hh", &rnns_impl->device_dn_weights_hh_},
                {"dn_weights_scale_hh", &rnns_impl->device_dn_weights_scale_hh_},
                {"up_weights", &rnns_impl->device_up_weights_},
                {"up_weights_scale", &rnns_impl->device_up_weights_scale_},
                {"up_bias", &rnns_impl->device_up_bias_},
        };
        for (const auto &c : caches) {
            if (int(c.second->size()) != layers) {
                throw std::runtime_error(std::string("device cache ") + c.first + " holds " +
                                         std::to_string(c.second->size()) + " layers, want " +
                                         std::to_string(layers));
            }
        }

        std::vector<LayerDump> dumps;
        auto emit = [&](LayerDump &d, const std::string &role, const at::Tensor &t) {
            Dumped out;
            out.role = role;
            out.file = "layer" + std::to_string(d.layer) + "-" + role + ".bin";
            out.shape = shape_json(t.sizes());
            out.dtype = dtype_name(t.scalar_type());
            out.contiguous = t.is_contiguous();
            out.sha = dump_tensor(t, opt.out + "/" + out.file);
            d.tensors.push_back(std::move(out));
        };

        const auto f16 = at::TensorOptions().dtype(at::kHalf).device(device);
        for (int layer = 0; layer < layers; ++layer) {
            LayerDump d;
            d.layer = layer;
            d.reverse = rnns_impl->first_reverse_ ^ bool(layer & 1);

            const auto &params = rnns_impl->layers_[layer]->named_parameters();

            // --- down projections: quantize_tensor(w.to(f16), 1) then permute ---
            for (const std::string which : {"ih", "hh"}) {
                const auto w16 = params["dn_weight_" + which].to(f16);
                const auto q = dorado::utils::quantize_tensor(w16, 1);
                emit(d, "dn-" + which + "-quantized", q.t);
                emit(d, "dn-" + which + "-quant-scale", q.scale);
            }
            // --- up projection: cat(ih, hh, dim 1) then quantize_tensor(.., 1) ---
            const auto up16 = torch::cat({params["up_weight_ih"].to(f16),
                                          params["up_weight_hh"].to(f16)},
                                         1);
            const auto up_q = dorado::utils::quantize_tensor(up16, 1);
            emit(d, "up-quantized", up_q.t);
            emit(d, "up-quant-scale", up_q.scale);
            const auto bias16 = params["up_bias_ih"].to(f16);
            emit(d, "up-bias-f16", bias16);

            // --- re-apply the source's own permutes and require the live cache ---
            auto dn_q = [&](const std::string &which) {
                return dorado::utils::quantize_tensor(params["dn_weight_" + which].to(f16), 1);
            };
            for (const std::string which : {"ih", "hh"}) {
                const auto q = dn_q(which);
                const auto want_t = q.t.view({-1, 2, 2, 2, 2, 2, C})
                                            .permute({2, 0, 4, 1, 3, 5, 6})
                                            .contiguous()
                                            .view({K, C});
                const auto want_s = q.scale.view({-1, 2, 2, 2, 2, 2})
                                            .permute({2, 0, 4, 1, 3, 5})
                                            .contiguous()
                                            .view({K})
                                            .to(f16);
                const auto &got_t = which == "ih" ? rnns_impl->device_dn_weights_ih_[layer]
                                                  : rnns_impl->device_dn_weights_hh_[layer];
                const auto &got_s = which == "ih"
                                            ? rnns_impl->device_dn_weights_scale_ih_[layer]
                                            : rnns_impl->device_dn_weights_scale_hh_[layer];
                if (!same_bytes(want_t, got_t) || !same_bytes(want_s, got_s)) {
                    throw std::runtime_error("layer " + std::to_string(layer) + " dn_weight_" +
                                             which +
                                             " cache is not the permuted quantized tensor");
                }
            }
            {
                const auto want_t = up_q.t.view({4, -1, 2, 2, 2, 2, 2 * K})
                                            .permute({1, 5, 0, 2, 3, 4, 6})
                                            .contiguous()
                                            .view({-1, 2 * K});
                const auto want_s = up_q.scale.to(f16)
                                            .view({4, -1, 2, 2, 2, 2})
                                            .permute({1, 5, 2, 3, 4, 0})
                                            .contiguous()
                                            .view({-1});
                const auto want_b = bias16.view({4, -1, 2, 2, 2, 2})
                                            .permute({1, 5, 2, 3, 4, 0})
                                            .contiguous()
                                            .view({-1});
                if (!same_bytes(want_t, rnns_impl->device_up_weights_[layer]) ||
                    !same_bytes(want_s, rnns_impl->device_up_weights_scale_[layer]) ||
                    !same_bytes(want_b, rnns_impl->device_up_bias_[layer])) {
                    throw std::runtime_error("layer " + std::to_string(layer) +
                                             " up-projection cache is not the permuted quantized "
                                             "tensor");
                }
            }

            // --- the seven live caches ---
            emit(d, "dn-ih-cache", rnns_impl->device_dn_weights_ih_[layer]);
            emit(d, "dn-ih-cache-scale", rnns_impl->device_dn_weights_scale_ih_[layer]);
            emit(d, "dn-hh-cache", rnns_impl->device_dn_weights_hh_[layer]);
            emit(d, "dn-hh-cache-scale", rnns_impl->device_dn_weights_scale_hh_[layer]);
            emit(d, "up-cache", rnns_impl->device_up_weights_[layer]);
            emit(d, "up-cache-scale", rnns_impl->device_up_weights_scale_[layer]);
            emit(d, "up-cache-bias", rnns_impl->device_up_bias_[layer]);

            // --- measured quantization domain ---
            //
            // `quant_scale = 128 / amax` (tensor_utils.cpp:294). A zero row makes
            // that division infinite and the subsequent Char cast undefined, so the
            // admitted domain is recorded rather than assumed: the consumer can
            // reject a model whose rows leave it instead of comparing garbage.
            auto dn_scale_ih = dn_q("ih").scale.to(at::kCPU).contiguous();
            auto dn_scale_hh = dn_q("hh").scale.to(at::kCPU).contiguous();
            auto up_scale = up_q.scale.to(at::kCPU).contiguous();
            bool seen = false;
            auto scan = [&](const at::Tensor &s) {
                const float *p = s.data_ptr<float>();
                for (int64_t i = 0; i < s.numel(); ++i) {
                    const float v = p[i];
                    if (!std::isfinite(v)) {
                        ++d.nonfinite_rows;
                        continue;
                    }
                    if (v == 0.0f) {
                        ++d.zero_rows;
                    }
                    if (!seen || v < d.scale_min) d.scale_min = v;
                    if (!seen || v > d.scale_max) d.scale_max = v;
                    seen = true;
                }
            };
            d.scale_min = 0;
            d.scale_max = 0;
            scan(dn_scale_ih);
            scan(dn_scale_hh);
            scan(up_scale);
            dumps.push_back(std::move(d));
        }

        std::ostringstream meta;
        meta << "{\n";
        meta << "  \"schema\": \"nanopore-hacgpu-flstm-cache/v1\",\n";
        meta << "  \"status\": \"captured\",\n";
        meta << "  \"case\": \"" << json_escape(opt.case_name) << "\",\n";
        meta << "  \"model_dir\": \"" << json_escape(opt.model_dir) << "\",\n";
        meta << "  \"model_package\": \"" << json_escape(config.model_name()) << "\",\n";
        meta << "  \"source_commit\": \"8b8fc5d36a9c0baab262a743cb175b5878e38ca6\",\n";
        meta << "  \"ordinal\": " << opt.ordinal << ",\n";
        meta << "  \"batch\": " << opt.batch << ",\n";
        meta << "  \"chunk\": " << chunk << ",\n";
        meta << "  \"features\": " << features << ",\n";
        meta << "  \"lstm_layers\": " << layers << ",\n";
        meta << "  \"lstm_size\": " << config.lstm_size << ",\n";
        meta << "  \"lstm_inner_dim\": " << *config.lstm_inner_dim << ",\n";
        meta << "  \"C\": " << C << ",\n";
        meta << "  \"K\": " << K << ",\n";
        meta << "  \"first_reverse\": " << (rnns_impl->first_reverse_ ? "true" : "false") << ",\n";
        meta << "  \"input_sha256\": \"" << hex_sha(raw.data(), raw.size()) << "\",\n";
        meta << "  \"reservation_bytes\": " << wm.reservation_bytes << ",\n";
        // The quantization contract this capture is evidence for. The constants
        // are the ones `utils::quantize_tensor` uses (tensor_utils.cpp:291-297).
        meta << "  \"quantize\": {\"levels\": 256, \"dim\": 1, \"quant_max\": 127, "
                "\"quant_min\": -127, \"weight_dtype\": \"Char\", \"scale_dtype\": \"Float\", "
                "\"input_dtype\": \"Half\", \"round\": \"half_to_even\"},\n";
        meta << "  \"layers\": [\n";
        for (size_t i = 0; i < dumps.size(); ++i) {
            const auto &d = dumps[i];
            meta << "    {\"layer\": " << d.layer << ", \"reverse\": "
                 << (d.reverse ? "true" : "false") << ", \"scale_min\": " << std::setprecision(9)
                 << d.scale_min << ", \"scale_max\": " << std::setprecision(9) << d.scale_max
                 << ", \"zero_scale_rows\": " << d.zero_rows
                 << ", \"nonfinite_scale_rows\": " << d.nonfinite_rows << ", \"tensors\": [\n";
            for (size_t j = 0; j < d.tensors.size(); ++j) {
                const auto &t = d.tensors[j];
                meta << "      {\"role\": \"" << t.role << "\", \"file\": \"" << t.file
                     << "\", \"shape\": " << t.shape << ", \"dtype\": \"" << t.dtype
                     << "\", \"contiguous\": " << (t.contiguous ? "true" : "false")
                     << ", \"sha256\": \"" << t.sha << "\"}"
                     << (j + 1 == d.tensors.size() ? "\n" : ",\n");
            }
            meta << "    ]}" << (i + 1 == dumps.size() ? "\n" : ",\n");
        }
        meta << "  ]\n";
        meta << "}\n";

        std::ofstream meta_out(opt.out + "/caches.json", std::ios::trunc);
        if (!meta_out) {
            throw std::runtime_error("cannot write caches.json");
        }
        meta_out << meta.str();
        meta_out.close();

        dump_runtime_paths(opt.out + "/runtime-paths.json");

        std::printf("captured %d layers x 14 tensors, C %d K %d, reservation %lld bytes\n", layers,
                    C, K, static_cast<long long>(wm.reservation_bytes));
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "flstm cache oracle: %s\n", e.what());
        return 1;
    }
}
