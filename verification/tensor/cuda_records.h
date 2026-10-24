#pragma once
#include <memory>
#include <vector>
#ifdef NT_ENABLE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif
struct CUDAWorkResources {
 std::vector<at::Tensor> tensors;
 std::vector<std::shared_ptr<std::vector<uint16_t>>> staging;
};
struct CUDADeviceRecord {
 uint64_t engine=0,id=0; int32_t ordinal=0; bool poisoned=false;
 std::shared_ptr<CUDAWorkResources> pending;
};
struct CUDAStreamRecord {
 std::shared_ptr<CUDADeviceRecord> device; uint64_t id=0;
#ifdef NT_ENABLE_CUDA
 cudaStream_t stream=nullptr;
#endif
};
struct CUDAHalfRecord {
 std::shared_ptr<CUDADeviceRecord> device;
 uint64_t id=0,allocation=0,offset=0,storage=0,end=0;
 at::Tensor tensor;
};
struct CUDAStorageRecord {
 std::shared_ptr<CUDADeviceRecord> device;
 uint64_t id=0,allocation=0,offset=0,storage=0,end=0;
 uint32_t dtype=0,scale_dtype=0;
 int32_t scale_tag=0;
 uint64_t scale_owner=0;
 int32_t tags[8]{};
 at::Tensor tensor;
};
