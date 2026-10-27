#include "capture.h"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
struct Counts {
    unsigned fat_binary;
    unsigned functions;
    unsigned variables;
    unsigned end;
    unsigned unregister;
    unsigned copies;
    unsigned error_strings;
    unsigned unexpected;
};
Counts counts{};
void* token;
char report_path[4096];
unsigned expected_copies;

[[noreturn]] void unexpected(const char* name) {
    ++counts.unexpected;
    std::fprintf(stderr, "unexpected CUDA call: %s\n", name);
    std::fflush(stderr);
    std::abort();
}

void check_handle(void** handle) {
    if (handle != &token) {
        unexpected("registration handle");
    }
}

void final_report() {
    const bool complete = counts.fat_binary == 1 && counts.functions == 6 && counts.variables == 24 &&
                          counts.end == 1 && counts.unregister == 1 && counts.copies == expected_copies &&
                          counts.error_strings == 0 && counts.unexpected == 0 && expected_copies != 0;
    FILE* file = report_path[0] ? std::fopen(report_path, "wx") : nullptr;
    if (!file) {
        std::fprintf(stderr, "transport report open failed\n");
        std::_Exit(91);
    }
    const int written = std::fprintf(file,
        "{\"schema\":\"hac-vcs-aux-transport/v1\",\"fat_binary\":%u,\"functions\":%u,"
        "\"variables\":%u,\"end\":%u,\"unregister\":%u,\"copies\":%u,\"expected_copies\":%u,"
        "\"error_strings\":%u,\"unexpected\":%u,\"complete\":%s}\n",
        counts.fat_binary, counts.functions, counts.variables, counts.end, counts.unregister,
        counts.copies, expected_copies, counts.error_strings, counts.unexpected, complete ? "true" : "false");
    const int close_status = std::fclose(file);
    if (written < 0 || close_status != 0 || !complete) {
        std::_Exit(92);
    }
}
}

void capture_transport_begin(const char* directory, unsigned copies) {
    if (std::strlen(directory) + 22 >= sizeof(report_path) || expected_copies != 0) {
        unexpected("capture initialization");
    }
    std::snprintf(report_path, sizeof(report_path), "%s/transport-final.json", directory);
    expected_copies = copies;
}

extern "C" void** CUDARTAPI __cudaRegisterFatBinary(void* fat_binary) {
    if (!fat_binary || counts.fat_binary != 0 || std::atexit(final_report) != 0) {
        unexpected("RegisterFatBinary");
    }
    ++counts.fat_binary;
    token = fat_binary;
    return &token;
}

extern "C" void CUDARTAPI __cudaRegisterFatBinaryEnd(void** handle) {
    check_handle(handle);
    ++counts.end;
}

extern "C" void CUDARTAPI __cudaUnregisterFatBinary(void** handle) {
    check_handle(handle);
    ++counts.unregister;
}

extern "C" void CUDARTAPI __cudaRegisterFunction(void** handle, const char* host_function,
                                                 char* device_function, const char* device_name,
                                                 int thread_limit, uint3* tid, uint3* bid,
                                                 dim3* block, dim3* grid, int* warp_size) {
    check_handle(handle);
    if (!host_function || !device_function || !device_name || thread_limit != -1 || tid || bid || block || grid || warp_size) {
        unexpected("RegisterFunction metadata");
    }
    ++counts.functions;
}

extern "C" void CUDARTAPI __cudaRegisterVar(void** handle, char* host_variable, char* device_address,
                                            const char* device_name, int external, std::size_t size,
                                            int constant, int global) {
    check_handle(handle);
    if (!host_variable || !device_address || !device_name || external || size != 1 || constant || global) {
        unexpected("RegisterVar metadata");
    }
    ++counts.variables;
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void* destination, const void* source,
                                                std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    ++counts.copies;
    return static_cast<cudaError_t>(capture_h2d(destination, source, bytes, static_cast<int>(kind), stream));
}

extern "C" const char* CUDARTAPI cudaGetErrorString(cudaError_t) {
    ++counts.error_strings;
    return "verification transport rejected upload";
}

extern "C" cudaError_t CUDARTAPI cudaGetDevice(int*) { unexpected("cudaGetDevice"); }
extern "C" cudaError_t CUDARTAPI cudaDeviceGetAttribute(int*, cudaDeviceAttr, int) { unexpected("cudaDeviceGetAttribute"); }
extern "C" cudaError_t CUDARTAPI cudaGetLastError() { unexpected("cudaGetLastError"); }
extern "C" unsigned CUDARTAPI __cudaPushCallConfiguration(dim3, dim3, std::size_t, cudaStream_t) { unexpected("__cudaPushCallConfiguration"); }
extern "C" cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3*, dim3*, std::size_t*, void*) { unexpected("__cudaPopCallConfiguration"); }
extern "C" cudaError_t CUDARTAPI __cudaGetKernel(cudaKernel_t*, const void*) { unexpected("__cudaGetKernel"); }
extern "C" cudaError_t CUDARTAPI __cudaLaunchKernel(cudaKernel_t, dim3, dim3, void**, std::size_t, cudaStream_t) { unexpected("__cudaLaunchKernel"); }
