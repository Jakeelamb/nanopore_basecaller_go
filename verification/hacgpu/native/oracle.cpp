// H01 native HAC stage oracle.
//
// This program is verification-only. It links the pinned Dorado static libraries
// from the qualified GCC15 source build, loads the untouched HAC graph through the
// stock `load_crf_model` entry point, and then replays `CRFModelImpl::run_koi`
// stage by stage while dumping the exact live bytes of every stage output and the
// complete raw parent backing storage.
//
// Nothing here is a port. It is the pinned native oracle every downstream HAC card
// consumes. All arithmetic comes from the stock library; this file only sequences
// the same calls `CRFModel.cpp:87-145` makes and records what it observes.
//
// Stage capture is deliberately "orchestration-only": the sequence of
// `reserve_working_memory` / `allocate_backing_tensor` / `run_koi` calls is copied
// from the pinned source and must stay in lockstep with it.

#include "basecall/crf_utils.h"
#include "config/BasecallModelConfig.h"
#include "model/CRFModel.h"
#include "nn/ConvStack.h"
#include "nn/RNNStack.h"
#include "nn/WorkingMemory.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/nn.h>

// Verification-only access shim: the pinned headers are included with `private`
// rewritten to `public` so the oracle can read the layer lists and the device
// caches the stock code keeps private. It does not modify the stock source, does
// not change any object layout, and is confined to this translation unit.
extern "C" {
#include "koi.h"
}

#define private public
#include "nn/FactorisedLinearLayer.h"
#include "nn/FLSTMStack.h"
#undef private

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using dorado::config::BasecallModelConfig;
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

// dump_runtime_paths records exactly which shared objects this process mapped, so
// the orchestrator can hash the files that actually ran rather than the sonames
// the link line asked for. The oracle only reports paths; hashing is the
// orchestrator's job.
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

struct Stage {
    std::string name;
    std::string shape;
    std::string dtype;
    std::string layout;
    int64_t offset_bytes = 0;
    int64_t bytes = 0;
    std::string data_sha;
    std::string data_file;
    std::string backing_sha;
    std::string backing_file;
    int64_t reservation_bytes = 0;
    int64_t current_bytes = 0;
    int wm_n = 0, wm_t = 0, wm_c = 0;
    bool input_to_rev_lstm = true;
    std::string kind;
    int layer = -1;
    // Unwritten time rows: ranges along the T axis that the pinned source never
    // writes for this stage. Everything outside them is live evidence; these are
    // prospective exclusions.
    std::vector<std::pair<int, int>> unwritten_t;
};

// complement_rows returns the T rows of [0, total) that are not written, given a
// written window and any individually written rows.
std::vector<std::pair<int, int>> complement_rows(int total,
                                                 int window_start,
                                                 int window_count,
                                                 const std::vector<int> &extra_rows) {
    std::vector<bool> written(size_t(total), false);
    for (int i = 0; i < window_count; ++i) {
        const int row = window_start + i;
        if (row >= 0 && row < total) {
            written[size_t(row)] = true;
        }
    }
    for (int row : extra_rows) {
        if (row >= 0 && row < total) {
            written[size_t(row)] = true;
        }
    }
    std::vector<std::pair<int, int>> runs;
    int start = -1;
    for (int i = 0; i < total; ++i) {
        if (!written[size_t(i)]) {
            if (start < 0) start = i;
        } else if (start >= 0) {
            runs.push_back({start, i - start});
            start = -1;
        }
    }
    if (start >= 0) {
        runs.push_back({start, total - start});
    }
    return runs;
}

// tensor_sha digests a live device tensor through a contiguous CPU copy, so the
// digest is of the bytes the graph produced, not of a view description.
std::string tensor_sha(const at::Tensor &t) {
    auto cpu = t.detach().to(at::kCPU).contiguous();
    return hex_sha(cpu.data_ptr(), size_t(cpu.numel() * cpu.element_size()));
}

// stage_kind classifies a stage name for downstream consumers.
std::string stage_kind(const std::string &name) {
    if (name == "input") return "input";
    if (name.rfind("conv", 0) == 0) return "convolution";
    if (name.rfind("flstm", 0) == 0) return "recurrent";
    if (name == "linear1" || name == "linear2") return "projection";
    if (name == "scores") return "scores";
    throw std::runtime_error("unknown stage name " + name);
}

int stage_layer(const std::string &name) {
    for (const char *prefix : {"conv", "flstm"}) {
        if (name.rfind(prefix, 0) == 0) {
            return std::stoi(name.substr(std::string(prefix).size()));
        }
    }
    return -1;
}

// flstm_replay is a literal transcription of FLSTMStackImpl::forward_koi's layer
// loop (FLSTMStack.cpp:125-295), split so the caller can capture the evolving
// in/out buffer after every layer. The device weight caches must already be
// populated by one untouched `rnns->run_koi` call, which is exactly what the
// stock code does on the first forward pass; the replay never re-quantises them
// because that would be a second, different graph.
void flstm_replay(dorado::nn::FLSTMStackImpl &stack,
                  WorkingMemory &wm,
                  const std::function<void(int)> &after_layer) {
    if (wm.layout != TensorLayout::CUTLASS_TNC_I8) {
        throw std::runtime_error("flstm replay expects the CUTLASS_TNC_I8 layout");
    }
    auto inout = wm.current;
    inout[0] = 0;
    inout[wm.T + 1] = 0;
    inout[wm.T + 2] = 1;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    const int64_t dn_bfr_size = 2 * (wm.T + 1) * (int64_t)wm.N * (2 * stack.K_);
    const int64_t dn_scale_bfr_size = 4 * ((wm.T + 1) * (int64_t)wm.N);
    const int64_t workspace_bfr_size = 4 * ((int64_t)wm.N / 2U);

    auto temp_bfr = wm.temp({dn_bfr_size + dn_scale_bfr_size + workspace_bfr_size}, torch::kU8);
    temp_bfr.zero_();

    auto dn_bfr = temp_bfr.narrow(0, 0, dn_bfr_size);
    auto dn_scale_bfr = temp_bfr.narrow(0, dn_bfr_size, dn_scale_bfr_size);
    auto workspace_bfr = temp_bfr.narrow(0, dn_bfr_size + dn_scale_bfr_size, workspace_bfr_size);

    for (int layer = 0; layer < std::ssize(stack.layers_); ++layer) {
        const bool reverse = stack.first_reverse_ ^ (layer & 1);
        if (std::ssize(stack.device_up_weights_) == layer) {
            throw std::runtime_error(
                    "flstm replay requires the stock forward pass to have populated the device "
                    "weight caches");
        }
        const int parity = (1 - (layer % 2U));
        host_factorised_lstm(
                stream, wm.N, wm.T, reverse ? -1 : 1, parity, inout.data_ptr(),
                workspace_bfr.data_ptr(), /*encoding=*/nullptr,
                stack.device_dn_weights_ih_[layer].data_ptr(),
                stack.device_dn_weights_scale_ih_[layer].data_ptr(),
                stack.device_dn_weights_hh_[layer].data_ptr(),
                stack.device_dn_weights_scale_hh_[layer].data_ptr(), dn_bfr.data_ptr(),
                dn_scale_bfr.data_ptr(), stack.device_up_weights_[layer].data_ptr(),
                stack.device_up_weights_scale_[layer].data_ptr(),
                stack.device_up_bias_[layer].data_ptr());
        wm.is_input_to_rev_lstm = !reverse;
        after_layer(layer);
    }
}

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
            throw std::runtime_error("H01 oracle expects the CRF (LSTM/FLSTM) graph, not a TX model");
        }
        if (!config.is_flstm_model()) {
            throw std::runtime_error("H01 oracle expects the pinned FLSTM HAC graph");
        }

        const auto device = at::Device(at::kCUDA, opt.ordinal);
        const auto opts = at::TensorOptions().dtype(at::kHalf).device(device);
        c10::cuda::CUDAGuard guard(device);

        auto holder = dorado::basecall::load_crf_model(config, opts);
        auto &model = holder->get<dorado::basecall::model::CRFModelImpl>();

        // Input payload is raw little-endian F16, [batch, num_features, T].
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

        // --- literal replay of CRFModelImpl::run_koi (CRFModel.cpp:87-145) ---
        //
        // The graph is driven twice. The first run calls the stock stack entry
        // points exactly as `run_koi` does and records the digest of each stack
        // output. The second run replays the same plan one layer at a time and
        // captures every layer; its stack outputs must match the first run byte
        // for byte, so a per-layer replay can never silently diverge from the
        // untouched graph.
        auto *rnns_impl = dynamic_cast<dorado::nn::FLSTMStackImpl *>(model.rnns.get());
        if (rnns_impl == nullptr) {
            throw std::runtime_error("H01 oracle expects an FLSTM recurrent stack");
        }

        auto plan = [&](WorkingMemory &wm) {
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
        };

        std::map<std::string, std::string> stack_sha;
        int64_t reservation_bytes = 0;
        int64_t backing_elems = 0;
        {
            WorkingMemory ref(int(input.size(0)));
            plan(ref);
            model.convs->run_koi(ref, nullptr);
            stack_sha["convs"] = tensor_sha(ref.current);
            model.rnns->run_koi(ref, nullptr);
            stack_sha["rnns"] = tensor_sha(ref.current);
            model.linear1->run_koi(ref, nullptr);
            stack_sha["linear1"] = tensor_sha(ref.current);
            if (model.linear2) {
                model.linear2->run_koi(ref, nullptr);
            }
            stack_sha["scores"] = tensor_sha(ref.current);
            reservation_bytes = ref.reservation_bytes;
            backing_elems = ref.backing_tensor.numel();
        }

        WorkingMemory wm(int(input.size(0)));
        plan(wm);
        if (wm.reservation_bytes != reservation_bytes) {
            throw std::runtime_error("per-layer replay planned a different working memory");
        }
        const auto backing_dtype = wm.backing_tensor.scalar_type();
        const auto backing_ptr = reinterpret_cast<uintptr_t>(wm.backing_tensor.data_ptr());
        if (backing_ptr % 256 != 0) {
            throw std::runtime_error("backing tensor is not 256-byte aligned");
        }

        std::vector<Stage> stages;
        auto capture = [&](const std::string &name,
                           std::vector<std::pair<int, int>> unwritten_t = {}) {
            const auto &cur = wm.current;
            if (!cur.defined()) {
                throw std::runtime_error("stage " + name + " has no current tensor");
            }
            Stage s;
            s.name = name;
            s.kind = stage_kind(name);
            s.layer = stage_layer(name);
            s.unwritten_t = std::move(unwritten_t);
            s.shape = shape_json(cur.sizes());
            s.dtype = dtype_name(cur.scalar_type());
            s.layout = dorado::nn::to_string(wm.layout);
            s.offset_bytes = int64_t(reinterpret_cast<uintptr_t>(cur.data_ptr()) - backing_ptr);
            s.bytes = cur.numel() * cur.element_size();
            s.reservation_bytes = reservation_bytes;
            s.current_bytes = wm.current_bytes;
            s.wm_n = wm.N;
            s.wm_t = wm.T;
            s.wm_c = wm.C;
            s.input_to_rev_lstm = wm.is_input_to_rev_lstm;
            s.data_file = "stage-" + name + ".bin";
            s.backing_file = "stage-" + name + "-backing.bin";
            if (s.offset_bytes < 0 || s.offset_bytes + s.bytes > backing_elems * 2) {
                throw std::runtime_error("stage " + name + " escapes the backing tensor");
            }
            s.data_sha = dump_tensor(cur, opt.out + "/" + s.data_file);
            s.backing_sha = dump_tensor(wm.backing_tensor, opt.out + "/" + s.backing_file);
            stages.push_back(std::move(s));
        };

        capture("input");

        for (size_t i = 0; i < model.convs->layers.size(); ++i) {
            auto &layer = model.convs->layers[i];
            layer.run_koi(wm, nullptr);
            std::vector<std::pair<int, int>> unwritten;
            if (layer.output_layout == TensorLayout::NTC && layer.params.size <= 16 &&
                layer.output_T_padding > 0) {
                // ConvStack.cpp:239-242: the small-conv path allocates
                // `T_out + 2 * output_T_padding` rows and narrows to
                // `[padding, padding + T_out)` before the kernel writes, so the
                // two padding bands are never written.
                const int pad = layer.output_T_padding;
                unwritten.push_back({0, pad});
                unwritten.push_back({pad + (wm.T - 2 * pad), pad});
            } else if (layer.output_layout != TensorLayout::NTC) {
                // ConvStack.cpp:283-288: the CUTLASS path allocates `T_out + 5`
                // rows and writes only the `T`-row NTC view, whose offset is
                // `is_input_to_rev_lstm ? 1 : 2`. Nothing initialises the rest.
                const int start = wm.is_input_to_rev_lstm ? 1 : 2;
                const int total = int(wm.current.size(0));
                unwritten = complement_rows(total, start, wm.T, {});
            }
            capture("conv" + std::to_string(i), unwritten);
        }
        if (tensor_sha(wm.current) != stack_sha["convs"]) {
            throw std::runtime_error(
                    "per-layer convolution replay does not reproduce the untouched conv stack");
        }

        flstm_replay(*rnns_impl, wm, [&](int layer) {
            // FLSTMStack.cpp:139-147 initialises rows 0, T+1 and T+2 explicitly;
            // the kernel writes the T-row window the next reader uses, whose
            // offset is `is_input_to_rev_lstm ? 1 : 2` (WorkingMemory.cpp:65-70).
            // The remaining guard rows are never written.
            const int total = int(wm.current.size(0));
            const int start = wm.is_input_to_rev_lstm ? 1 : 2;
            capture("flstm" + std::to_string(layer),
                    complement_rows(total, start, wm.T, {0, wm.T + 1, wm.T + 2}));
        });
        if (tensor_sha(wm.current) != stack_sha["rnns"]) {
            throw std::runtime_error(
                    "per-layer FLSTM replay does not reproduce the untouched recurrent stack");
        }

        model.linear1->run_koi(wm, nullptr);
        capture("linear1");
        if (model.linear2) {
            model.linear2->run_koi(wm, nullptr);
            capture("linear2");
        }

        if (wm.layout != TensorLayout::NTC) {
            throw std::runtime_error("final working-memory layout is not NTC");
        }
        capture("scores");
        // --- end literal replay ---

        const auto &scores = wm.current;
        auto scores_cpu = scores.detach().to(at::kCPU).contiguous();

        std::ostringstream meta;
        meta << "{\n";
        meta << "  \"schema\": \"nanopore-hacgpu-native-stage-capture/v1\",\n";
        meta << "  \"status\": \"captured\",\n";
        meta << "  \"case\": \"" << json_escape(opt.case_name) << "\",\n";
        meta << "  \"model_dir\": \"" << json_escape(opt.model_dir) << "\",\n";
        meta << "  \"model_package\": \"" << json_escape(config.model_name()) << "\",\n";
        meta << "  \"source_commit\": \"8b8fc5d36a9c0baab262a743cb175b5878e38ca6\",\n";
        meta << "  \"ordinal\": " << opt.ordinal << ",\n";
        meta << "  \"batch\": " << opt.batch << ",\n";
        meta << "  \"chunk\": " << chunk << ",\n";
        meta << "  \"stride\": " << config.stride << ",\n";
        meta << "  \"t_out\": " << (chunk / config.stride) << ",\n";
        meta << "  \"features\": " << features << ",\n";
        meta << "  \"lstm_layers\": " << config.lstm_layers << ",\n";
        meta << "  \"lstm_size\": " << config.lstm_size << ",\n";
        meta << "  \"lstm_inner_dim\": " << config.lstm_inner_dim.value_or(0) << ",\n";
        meta << "  \"convs\": " << config.convs.size() << ",\n";
        meta << "  \"scale\": " << config.scale << ",\n";
        meta << "  \"standardise\": " << (config.signal_norm_params.standardisation.standardise ? "true" : "false")
             << ",\n";
        meta << "  \"standardisation_mean\": " << config.signal_norm_params.standardisation.mean << ",\n";
        meta << "  \"standardisation_stdev\": " << config.signal_norm_params.standardisation.stdev << ",\n";
        meta << "  \"out_features\": " << config.out_features.value_or(0) << ",\n";
        // CRFModel.cpp:51-70: with `can_use_koi && lstm_size == 1024 &&
        // decomposition == 128 && tanh_x5` the decoder is one FactorisedLinearLayer
        // and `linear2` is never constructed; otherwise it is two LinearCRF stages.
        meta << "  \"linear2_present\": " << (model.linear2 ? "true" : "false") << ",\n";
        meta << "  \"linear1_kind\": \""
             << (model.linear2 ? "plain" : "factorised") << "\",\n";
        meta << "  \"reservation_bytes\": " << reservation_bytes << ",\n";
        meta << "  \"backing_elems\": " << backing_elems << ",\n";
        meta << "  \"backing_dtype\": \"" << dtype_name(backing_dtype) << "\",\n";
        meta << "  \"backing_alignment\": 256,\n";
        meta << "  \"scores_shape\": " << shape_json(scores.sizes()) << ",\n";
        meta << "  \"scores_dtype\": \"" << dtype_name(scores.scalar_type()) << "\",\n";
        meta << "  \"scores_contiguous\": " << (scores.is_contiguous() ? "true" : "false")
             << ",\n";
        meta << "  \"scores_sha256\": \""
             << hex_sha(scores_cpu.data_ptr(),
                        size_t(scores_cpu.numel() * scores_cpu.element_size()))
             << "\",\n";
        meta << "  \"stack_sha256\": {\n";
        {
            size_t i = 0;
            for (const auto &kv : stack_sha) {
                meta << "    \"" << kv.first << "\": \"" << kv.second << "\""
                     << (++i == stack_sha.size() ? "\n" : ",\n");
            }
        }
        meta << "  },\n";
        meta << "  \"stages\": [\n";
        for (size_t i = 0; i < stages.size(); ++i) {
            const auto &s = stages[i];
            meta << "    {\"name\": \"" << s.name << "\", \"kind\": \"" << s.kind
                 << "\", \"layer\": " << s.layer << ", \"shape\": " << s.shape
                 << ", \"dtype\": \"" << s.dtype << "\", \"layout\": \"" << s.layout
                 << "\", \"offset_bytes\": " << s.offset_bytes << ", \"bytes\": " << s.bytes
                 << ", \"reservation_bytes\": " << s.reservation_bytes
                 << ", \"current_bytes\": " << s.current_bytes << ", \"N\": " << s.wm_n
                 << ", \"T\": " << s.wm_t << ", \"C\": " << s.wm_c
                 << ", \"is_input_to_rev_lstm\": " << (s.input_to_rev_lstm ? "true" : "false")
                 << ", \"unwritten_t\": [";
            for (size_t u = 0; u < s.unwritten_t.size(); ++u) {
                meta << (u ? ", [" : "[") << s.unwritten_t[u].first << ", "
                     << s.unwritten_t[u].second << "]";
            }
            meta << "], \"data_file\": \"" << s.data_file << "\", \"data_sha256\": \""
                 << s.data_sha << "\", \"backing_file\": \"" << s.backing_file
                 << "\", \"backing_sha256\": \"" << s.backing_sha << "\"}"
                 << (i + 1 == stages.size() ? "\n" : ",\n");
        }
        meta << "  ]\n";
        meta << "}\n";

        std::ofstream meta_out(opt.out + "/capture.json", std::ios::trunc);
        if (!meta_out) {
            throw std::runtime_error("cannot write capture.json");
        }
        meta_out << meta.str();
        meta_out.close();

        dump_runtime_paths(opt.out + "/runtime-paths.json");

        std::printf("captured %zu stages, reservation %lld bytes, scores %s\n", stages.size(),
                    static_cast<long long>(reservation_bytes), shape_json(scores.sizes()).c_str());
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "hacgpu oracle: %s\n", e.what());
        return 1;
    }
}
