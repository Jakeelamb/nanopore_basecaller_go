#include "abi.h"
typedef struct nt_library nt_library;
nt_library* nt_library_open(const char*, char*, size_t);
void nt_library_close(nt_library*);
int nt_call_open(nt_library*,uint64_t*,char*,size_t);
int nt_call_close(nt_library*,uint64_t,char*,size_t);
int nt_call_new(nt_library*,uint64_t,const int64_t*,size_t,const float*,size_t,uint64_t*,char*,size_t);
int nt_call_new_int64(nt_library*,uint64_t,const int64_t*,size_t,const int64_t*,size_t,uint64_t*,char*,size_t);
int nt_call_load(nt_library*,uint64_t,const char*,uint64_t*,char*,size_t);
int nt_call_release(nt_library*,uint64_t,uint64_t,char*,size_t);
int nt_call_shape(nt_library*,uint64_t,uint64_t,int64_t*,size_t,size_t*,char*,size_t);
int nt_call_data(nt_library*,uint64_t,uint64_t,float*,size_t,char*,size_t);
int nt_call_data_int64(nt_library*,uint64_t,uint64_t,int64_t*,size_t,char*,size_t);
int nt_call_run(nt_library*,uint64_t,int,const uint64_t*,size_t,const int64_t*,size_t,const double*,size_t,uint64_t*,char*,size_t);
int nt_call_sdpa(nt_library*,uint64_t,uint64_t,uint64_t,uint64_t,const int64_t*,size_t,const uint8_t*,size_t,uint64_t*,char*,size_t);

int nt_call_cuda_device_open(nt_library *library, uint64_t engine, int32_t ordinal, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_device_close(nt_library *library, uint64_t engine, uint64_t device, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_stream_open(nt_library *library, uint64_t engine, uint64_t device, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_stream_close(nt_library *library, uint64_t engine, uint64_t stream, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_half_new(nt_library *library, uint64_t engine, uint64_t device, uint64_t stream, const int64_t *shape, size_t rank, const uint16_t *bits, size_t count, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_half_view(nt_library *library, uint64_t engine, uint64_t parent, uint64_t offset_elements, const int64_t *shape, const int64_t *strides, size_t rank, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_half_meta(nt_library *library, uint64_t engine, uint64_t half, nt_half_meta *out, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_half_bits(nt_library *library, uint64_t engine, uint64_t stream, uint64_t half, uint16_t *bits, size_t count, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_half_release(nt_library *library, uint64_t engine, uint64_t half, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_gemm_f16(nt_library *library, uint64_t engine, uint64_t stream, uint64_t a, uint64_t b, uint64_t out, nt_cuda_detail *detail, char *error, size_t error_cap);

int nt_call_cuda_storage_new(nt_library *library, uint64_t engine, uint64_t device, uint64_t stream, uint32_t dtype, const int64_t *shape, size_t rank, const int32_t *tags, const uint8_t *bytes, size_t count, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_storage_view(nt_library *library, uint64_t engine, uint64_t parent, uint32_t dtype, uint64_t offset_elements, const int64_t *shape, const int64_t *strides, const int32_t *tags, size_t rank, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_storage_meta(nt_library *library, uint64_t engine, uint64_t storage, nt_storage_meta *out, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_storage_upload(nt_library *library, uint64_t engine, uint64_t stream, uint64_t storage, const uint8_t *bytes, size_t count, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_storage_download(nt_library *library, uint64_t engine, uint64_t stream, uint64_t storage, uint8_t *bytes, size_t count, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_storage_copy(nt_library *library, uint64_t engine, uint64_t stream, uint64_t dst, uint64_t src, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_storage_fill(nt_library *library, uint64_t engine, uint64_t stream, uint64_t storage, const uint8_t *pattern, size_t count, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_storage_scale(nt_library *library, uint64_t engine, uint64_t storage, uint64_t scale, int32_t scale_tag, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_storage_release(nt_library *library, uint64_t engine, uint64_t storage, nt_cuda_detail *detail, char *error, size_t error_cap);

int nt_call_cuda_conv_small_f16(nt_library *library, uint64_t engine, uint64_t stream, uint64_t in, uint64_t weights, uint64_t bias, uint64_t out, int32_t n, int32_t c_in, int32_t c_out, int32_t t_in, int32_t window, int32_t stride, int32_t padding, int32_t out_n_stride, uint32_t activation, nt_cuda_detail *detail, char *error, size_t error_cap);
int nt_call_cuda_weights_permute_cast_f16(nt_library *library, uint64_t engine, uint64_t stream, uint64_t weight, uint64_t bias, uint64_t packed, uint64_t packed_bias, nt_cuda_detail *detail, char *error, size_t error_cap);
