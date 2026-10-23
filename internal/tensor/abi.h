#ifndef NANOPORE_TENSOR_ABI_H
#define NANOPORE_TENSOR_ABI_H
#include <stddef.h>
#include <stdint.h>
#define NT_ABI_VERSION 2
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
/* Operation IDs are part of ABI 2. Each call returns an independently owned handle. */
enum nt_op { NT_ADD=1, NT_MUL, NT_MATMUL, NT_CLAMP, NT_TANH, NT_SIGMOID,
 NT_SILU, NT_TRANSPOSE, NT_FLIP, NT_RESHAPE, NT_NARROW, NT_CONCAT, NT_STACK,
 NT_CONV1D, NT_LINEAR, NT_ADD_SCALAR, NT_MUL_SCALAR, NT_PERMUTE, NT_CONTIGUOUS, NT_ZEROS,
 NT_INDEX_SELECT, NT_LOGSUMEXP, NT_SOFTMAX, NT_UNSQUEEZE, NT_ARGSORT };
int nt_run(uint64_t, int, const uint64_t*, size_t, const int64_t*, size_t,
           const double*, size_t, uint64_t*, char*, size_t);
#ifdef __cplusplus
}
#endif
#endif
