#ifndef NANOPORE_TENSOR_RUNTIME_H
#define NANOPORE_TENSOR_RUNTIME_H
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>
#include <mutex>
inline void initialize_tensor_runtime() {
    static std::once_flag once;
    std::call_once(once,[]{
        at::set_num_threads(1);
        at::globalContext().setDeterministicFillUninitializedMemory(false);
        at::globalContext().setDeterministicAlgorithms(true,false);
    });
    // CGo can enter on an OS thread that did not establish the global policy.
    static thread_local const bool initialized = [] {
        at::init_num_threads();
        return true;
    }();
    (void)initialized;
}
#endif
