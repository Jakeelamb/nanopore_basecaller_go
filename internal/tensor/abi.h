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

/* Typed storage. Additive under ABI 4: a backend without these symbols is
 * rejected by mandatory symbol resolution at library open. Dtypes mirror the
 * pinned Koi type ids used by the HAC/SUP graphs. */
#define NT_STORAGE_MAX_RANK 8
enum nt_dtype { NT_DTYPE_F16=1, NT_DTYPE_I8, NT_DTYPE_U8, NT_DTYPE_I32, NT_DTYPE_F32, NT_DTYPE_E4M3 };
/* Pinned KoiTensor dim tags: the named enum values 0..2 and the single-character
 * tags the source passes literally (TxModules.cpp:534-556). */
enum nt_koi_tag { NT_KOI_TAG_QKV=0, NT_KOI_TAG_GATE_IFGO=1, NT_KOI_TAG_W_U=2 };
typedef struct nt_storage_meta {
 uint32_t rank; uint32_t dtype; uint64_t device; int32_t ordinal;
 uint32_t scale_dtype; uint32_t scale_present; int32_t scale_tag;
 uint64_t allocation; uint64_t offset_elements; uint64_t storage_elements;
 int64_t shape[NT_STORAGE_MAX_RANK]; int64_t strides[NT_STORAGE_MAX_RANK];
 int32_t tags[NT_STORAGE_MAX_RANK];
} nt_storage_meta;
int nt_cuda_storage_new(uint64_t engine, uint64_t device, uint64_t stream, uint32_t dtype,
 const int64_t *shape, size_t rank, const int32_t *tags, const uint8_t *bytes, size_t count,
 uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_storage_view(uint64_t engine, uint64_t parent, uint32_t dtype, uint64_t offset_elements,
 const int64_t *shape, const int64_t *strides, const int32_t *tags, size_t rank, uint64_t *out,
 nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_storage_meta(uint64_t engine, uint64_t storage, nt_storage_meta *out,
 nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_storage_upload(uint64_t engine, uint64_t stream, uint64_t storage, const uint8_t *bytes,
 size_t count, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_storage_download(uint64_t engine, uint64_t stream, uint64_t storage, uint8_t *bytes,
 size_t count, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_storage_copy(uint64_t engine, uint64_t stream, uint64_t dst, uint64_t src,
 nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_storage_fill(uint64_t engine, uint64_t stream, uint64_t storage, const uint8_t *pattern,
 size_t count, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_storage_scale(uint64_t engine, uint64_t storage, uint64_t scale, int32_t scale_tag,
 nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_storage_release(uint64_t engine, uint64_t storage, nt_cuda_detail *detail,
 char *error, size_t error_cap);

/* HAC small convolution. Additive under ABI 4, like typed storage: a backend
 * without these symbols is rejected by mandatory symbol resolution at open.
 * The one caller-facing primitive is a single pinned host_convolution_f16 call
 * (ConvStack.cpp:239-262). The weight-packing symbol is a verification-only
 * oracle: the Go caller packs weights itself and the consumer compares the two
 * byte for byte. It is never part of the caller path. */
enum nt_koi_activation { NT_KOI_IDENTITY=0, NT_KOI_SWISH, NT_KOI_SWISH_CLAMP,
 NT_KOI_TANH, NT_KOI_TANH_X5 };
int nt_cuda_conv_small_f16(uint64_t engine, uint64_t stream,
 uint64_t in, uint64_t weights, uint64_t bias, uint64_t out,
 int32_t n, int32_t c_in, int32_t c_out, int32_t t_in,
 int32_t window, int32_t stride, int32_t padding, int32_t out_n_stride,
 uint32_t activation, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_cuda_weights_permute_cast_f16(uint64_t engine, uint64_t stream,
 uint64_t weight, uint64_t bias, uint64_t packed, uint64_t packed_bias,
 nt_cuda_detail *detail, char *error, size_t error_cap);
#ifdef __cplusplus
}
#endif
#endif
