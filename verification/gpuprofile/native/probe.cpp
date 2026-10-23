#include <cuda_runtime_api.h>
#include <koi.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <utility>

namespace {
cudaDeviceProp device_prop{};
bool koi_can_use_cutlass_internal(const cudaDeviceProp* const prop);
bool koi_can_run_flstm_internal(const cudaDeviceProp* const prop);
}

struct Params {
    int d_model = 512;
    int nhead = 8;
    std::pair<int, int> attn_window{127, 128};
    int dim_feedforward = 2048;
};

struct Options {
    struct Device { bool is_cuda() const { return true; } };
    Device device() const { return {}; }
};

struct Parser {
    template <typename T> T get(const char*) const { return false; }
};

namespace utils {
template <typename T> T get_dev_opt(const char*, T fallback) { return fallback; }
}

namespace config {
struct BasecallModelConfig {
    int kind = 0;
    int lstm_size = 1024;
    std::optional<int> lstm_inner_dim{128};
    struct Tx { Params tx; };
    std::optional<Tx> tx{Tx{}};
    bool is_lstm_model() const { return kind == 0; }
    bool is_flstm_model() const { return kind == 1; }
    bool is_tx_model() const { return kind == 2; }
};
}

namespace dorado::nn {
bool koi_can_use_cutlass(int) { return koi_can_use_cutlass_internal(&device_prop); }
bool koi_can_run_flstm(int) { return koi_can_run_flstm_internal(&device_prop); }
bool koi_can_run_tx_vcs(int);
}

#include "predicates.h"

namespace dorado::nn {
bool koi_can_run_tx_vcs(int device_id) { return source_tx_vcs(device_id); }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fputs("one maps output path required\n", stderr);
        return 1;
    }
    int count = 0;
    int device = -1;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count != 1 ||
        cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&device_prop, device) != cudaSuccess) {
        std::fputs("device observation failed or visible device count is not one\n", stderr);
        return 1;
    }
    int driver = 0;
    int runtime = 0;
    if (cudaDriverGetVersion(&driver) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime) != cudaSuccess) {
        std::fputs("runtime version observation failed\n", stderr);
        return 1;
    }
    const int f16 = koi_tc_is_available(KOI_F16);
    const int e4m3 = koi_tc_is_available(KOI_E4M3);
    Params params;
    config::BasecallModelConfig hac;
    hac.kind = 1;
    config::BasecallModelConfig sup;
    sup.kind = 2;
    const std::array<int, 1> ids{device};
    const bool hac_fused = koi_can_run_flstm_internal(&device_prop);
    const bool sup_tiled = source_tiled(params);
    const bool fp8 = source_f8();
    const bool int8 = source_i8();
    const bool hac_vcs = dorado::api::check_variable_chunk_sizes_supported(hac, ids);
    const bool sup_vcs = dorado::api::check_variable_chunk_sizes_supported(sup, ids);
    const bool default_vcs = source_default_vcs();
    std::ifstream maps("/proc/self/maps");
    std::ofstream loaded(argv[1]);
    loaded << maps.rdbuf();
    loaded.flush();
    if (!maps || !loaded) {
        std::fputs("runtime mapping capture failed\n", stderr);
        return 1;
    }
    std::printf("{\"device_uuid\":\"");
    for (unsigned char byte : device_prop.uuid.bytes) { std::printf("%02x", byte); }
    std::printf("\",\"device_ordinal\":%d,\"visible_device_count\":%d,"
                "\"compute_major\":%d,\"compute_minor\":%d,\"sm_count\":%d,"
                "\"driver_version\":%d,\"runtime_version\":%d,"
                "\"cuda_visible_devices_present\":%s,\"f16_status\":%d,\"e4m3_status\":%d,"
                "\"predicates\":{\"hac_fused\":%s,\"sup_tiled\":%s,\"fp8\":%s,"
                "\"int8\":%s,\"hac_vcs\":%s,\"sup_vcs\":%s,\"default_vcs\":%s,"
                "\"remove_bits\":%d}}\n",
                device, count, device_prop.major, device_prop.minor,
                device_prop.multiProcessorCount, driver, runtime,
                std::getenv("CUDA_VISIBLE_DEVICES") ? "true" : "false", f16, e4m3,
                hac_fused ? "true" : "false", sup_tiled ? "true" : "false",
                fp8 ? "true" : "false", int8 ? "true" : "false",
                hac_vcs ? "true" : "false", sup_vcs ? "true" : "false",
                default_vcs ? "true" : "false", source_remove_bits());
    return (f16 < KOI_SUCCESS || f16 > KOI_NOT_SUPPORTED ||
            e4m3 < KOI_SUCCESS || e4m3 > KOI_NOT_SUPPORTED) ? 1 : 0;
}
