#ifndef NANOPORE_TENSOR_ABI_H
#define NANOPORE_TENSOR_ABI_H
#include <stddef.h>
#include <stdint.h>
#define NT_ABI_VERSION 4
#define NT_ERROR_CAP 2048
#ifdef __cplusplus
extern "C" {
#endif
uint32_t nt_version(void);
int nt_open(uint64_t*, char*, size_t);
int nt_close(uint64_t, char*, size_t);
int nt_new(uint64_t, const int64_t*, size_t, const float*, size_t, uint64_t*, char*, size_t);
int nt_new_int64(uint64_t, const int64_t*, size_t, const int64_t*, size_t, uint64_t*, char*, size_t);
int nt_load(uint64_t, const char*, uint64_t*, char*, size_t);
int nt_release(uint64_t, uint64_t, char*, size_t);
int nt_shape(uint64_t, uint64_t, int64_t*, size_t, size_t*, char*, size_t);
int nt_data(uint64_t, uint64_t, float*, size_t, char*, size_t);
int nt_data_int64(uint64_t, uint64_t, int64_t*, size_t, char*, size_t);
/* Operation IDs are unchanged since ABI 3. Each call returns an independently owned handle. */
enum nt_op { NT_ADD=1, NT_MUL, NT_MATMUL, NT_CLAMP, NT_TANH, NT_SIGMOID,
 NT_SILU, NT_TRANSPOSE, NT_FLIP, NT_RESHAPE, NT_NARROW, NT_CONCAT, NT_STACK,
 NT_CONV1D, NT_LINEAR, NT_ADD_SCALAR, NT_MUL_SCALAR, NT_PERMUTE, NT_CONTIGUOUS, NT_ZEROS,
 NT_INDEX_SELECT, NT_LOGSUMEXP, NT_SOFTMAX, NT_UNSQUEEZE, NT_ARGSORT,
 NT_MEAN, NT_RSQRT, NT_RECIPROCAL, NT_SIN, NT_COS, NT_SUB, NT_SQUARE };
int nt_run(uint64_t, int, const uint64_t*, size_t, const int64_t*, size_t,
           const double*, size_t, uint64_t*, char*, size_t);
int nt_sdpa(uint64_t, uint64_t, uint64_t, uint64_t, const int64_t*, size_t,
            const uint8_t*, size_t, uint64_t*, char*, size_t);
#define NT_CUDA_MAX_RANK 8
enum nt_cuda_status { NT_CUDA_OK=0, NT_CUDA_ARGUMENT, NT_CUDA_CLOSED, NT_CUDA_OWNER,
 NT_CUDA_DTYPE, NT_CUDA_DEVICE, NT_CUDA_STREAM, NT_CUDA_LAYOUT, NT_CUDA_UNSUPPORTED,
 NT_CUDA_RUNTIME, NT_CUDA_CUBLAS, NT_CUDA_INTERNAL };
enum nt_scalar_type { NT_SCALAR_NONE=0, NT_SCALAR_F16, NT_SCALAR_F32, NT_SCALAR_I64 };
enum nt_cuda_stage { NT_CUDA_VALIDATE=0, NT_CUDA_ALLOCATE, NT_CUDA_COPY,
 NT_CUDA_ENQUEUE, NT_CUDA_DRAIN, NT_CUDA_DESTROY };
typedef struct nt_cuda_detail {
 uint32_t stage; int32_t native_code; uint64_t handle; uint64_t actual_engine;
 uint64_t expected_device; uint64_t actual_device;
 int32_t expected_ordinal; int32_t actual_ordinal;
 uint32_t expected_dtype; uint32_t actual_dtype;
} nt_cuda_detail;
typedef struct nt_half_meta {
 uint32_t rank; uint32_t dtype; uint64_t device; int32_t ordinal; uint32_t reserved;
 uint64_t offset_elements; uint64_t storage_elements;
 int64_t shape[NT_CUDA_MAX_RANK]; int64_t strides[NT_CUDA_MAX_RANK];
} nt_half_meta;
int nt_cuda_device_open(uint64_t engine, int32_t ordinal, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_device_close(uint64_t engine, uint64_t device, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_stream_open(uint64_t engine, uint64_t device, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_stream_close(uint64_t engine, uint64_t stream, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_half_new(uint64_t engine, uint64_t device, uint64_t stream, const int64_t *shape, size_t rank, const uint16_t *bits, size_t count, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_half_view(uint64_t engine, uint64_t parent, uint64_t offset_elements, const int64_t *shape, const int64_t *strides, size_t rank, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_half_meta(uint64_t engine, uint64_t half, nt_half_meta *out, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_half_bits(uint64_t engine, uint64_t stream, uint64_t half, uint16_t *bits, size_t count, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_half_release(uint64_t engine, uint64_t half, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_gemm_f16(uint64_t engine, uint64_t stream, uint64_t a, uint64_t b, uint64_t out, nt_cuda_detail *detail, char *error, size_t error_cap);
#ifdef __cplusplus
}
#endif
#endif
