// HAC last convolution boundary. Included once from backend.cpp inside the
// single ABI translation unit, after the small-convolution operations.
//
// The pinned source path this mirrors is the `cutlass_conv` branch of
// ConvLayer::run_koi (ConvStack.cpp:264-322):
//
//   in.slice(1, 0, padding) = 0;
//   in.slice(1, -padding, None) = 0;
//   wm.next_TC(T_out, C_out, output_layout);
//   out_ntc = wm.get_current_NTC_view();
//   res = host_convolution(stream, wm.N, T_in, C_in, C_out, winlen, stride,
//                          padding, int(in.stride(0)), int(out_ntc.stride(0)),
//                          int(out_ntc.stride(1)),
//                          in.slice(1, padding, None).data_ptr(),
//                          out_ntc.data_ptr(), out_layout, w_device.data_ptr(),
//                          b_device.data_ptr(), activation, KOI_F16, out_type,
//                          use_f32_accum ? KOI_F32 : KOI_F16);
//   if (res != KOI_SUCCESS)
//       res = host_linear(stream, KOI_F16, activation, out_type, wm.N, T_out,
//                         C_in * winlen, C_out, int(in.stride(0)),
//                         stride * C_in, int(out_ntc.stride(0)),
//                         int(out_ntc.stride(1)), in.data_ptr(),
//                         w_device.data_ptr(), out_ntc.data_ptr(), out_layout,
//                         nullptr, b_device.data_ptr());
//   if (res != KOI_SUCCESS) throw ...;
//
// Everything above the two primitive calls is Go's: the padding zeroing, the
// weight transpose into [C_out, W*C_in], the CUTLASS output view and its two
// strides, the dev-option dispatch and the fallback decision. This file only
// validates the typed handles, makes one named primitive call, and reports the
// primitive's own KoiResult.
//
// `koi_conv` and `koi_conv_f32` are source dev options (ConvStack.cpp:285-288).
// Go carries them as explicit parameters so a control can flip one and observe
// the arithmetic change; the boundary never reads an environment variable.
// koi.h carries no linkage specifier; the pinned Dorado source includes it inside
// `extern "C"` (ConvStack.cpp:12-14) because libkoi exports C symbols.
#ifdef NT_ENABLE_CUDA
extern "C" {
#include "koi.h"
}
#endif

namespace {
#ifdef NT_ENABLE_CUDA
// get_koi_activation (ConvStack.cpp:21-31) can only return these three. IDENTITY
// and TANH_X5 are not reachable from a config activation, so they are rejected
// here instead of being silently treated as something the source could produce.
// This is deliberately a local helper rather than a reuse of the small
// convolution's: this file stays independently includable.
KoiActivation lastconv_activation(uint32_t activation) {
    switch (activation) {
        case NT_KOI_SWISH:
            return KOI_SWISH;
        case NT_KOI_SWISH_CLAMP:
            return KOI_SWISH_CLAMP;
        case NT_KOI_TANH:
            return KOI_TANH;
    }
    throw CUDAFailure(NT_CUDA_UNSUPPORTED, "unknown Koi activation for the last convolution");
}
KoiTypeId lastconv_type(uint32_t type) {
    switch (type) {
        case NT_KOI_I8:
            return KOI_I8;
        case NT_KOI_F16:
            return KOI_F16;
        case NT_KOI_F32:
            return KOI_F32;
    }
    throw CUDAFailure(NT_CUDA_DTYPE, "unsupported Koi type for the last convolution");
}
// The declared Koi type must agree with the storage's own dtype: a request that
// says F16 while holding I8 would ask the primitive to reinterpret bytes.
void lastconv_type_matches(const std::shared_ptr<CUDAStorageRecord>& p, uint32_t type,
                           nt_cuda_detail* d, const char* why) {
    d->handle = p->id;
    d->actual_dtype = p->dtype;
    d->expected_dtype = type == NT_KOI_I8    ? NT_DTYPE_I8
                        : type == NT_KOI_F16 ? NT_DTYPE_F16
                        : type == NT_KOI_F32 ? NT_DTYPE_F32
                                             : NT_DTYPE_F16;
    cuda_require(p->dtype == d->expected_dtype, NT_CUDA_DTYPE, why);
}
#endif
}  // namespace

extern "C" {
int nt_cuda_conv_koi_f16(uint64_t e, uint64_t stream_id, uint64_t in_id, uint64_t weight_id,
                         uint64_t bias_id, uint64_t out_id, int32_t n, int32_t t_in,
                         int32_t c_in, int32_t c_out, int32_t window, int32_t stride,
                         int32_t padding, int32_t in_n_stride, int32_t out_n_stride,
                         int32_t out_t_stride, uint32_t activation, uint32_t in_type,
                         uint32_t out_type, uint32_t accum_type, int32_t* koi_result,
                         nt_cuda_detail* d, char* err, size_t cap) {
    return cuda_boundary(d, err, cap, [&] {
        cuda_engine(e, d);
        cuda_require(koi_result, NT_CUDA_ARGUMENT, "last convolution needs a result out-parameter");
        *koi_result = -1;
#ifdef NT_ENABLE_CUDA
        auto s = cuda_stream(e, stream_id, d);
        auto in = cuda_storage(e, in_id, d);
        auto w = cuda_storage(e, weight_id, d);
        auto b = cuda_storage(e, bias_id, d);
        auto out = cuda_storage(e, out_id, d);
        for (auto p : {in, w, b, out}) {
            d->handle = p->id;
            same_device(s->device, p->device, d);
        }
        lastconv_type_matches(in, in_type, d, "last convolution input type differs from its storage");
        lastconv_type_matches(w, NT_KOI_F16, d, "last convolution weights must be F16");
        lastconv_type_matches(b, NT_KOI_F16, d, "last convolution bias must be F16");
        lastconv_type_matches(out, out_type, d, "last convolution output type differs from its storage");
        cuda_require(accum_type == NT_KOI_F16 || accum_type == NT_KOI_F32, NT_CUDA_DTYPE,
                     "last convolution accumulation must be F16 or F32");
        cuda_require(n > 0 && t_in > 0 && c_in > 0 && c_out > 0 && window > 0 && stride > 0,
                     NT_CUDA_ARGUMENT, "last convolution geometry must be positive");
        // ConvStack.cpp:222-223 passes `winlen / 2` and derives T_in from the
        // padded allocation.
        cuda_require(padding == window / 2, NT_CUDA_ARGUMENT,
                     "last convolution padding must be window/2");
        const int64_t t_out = int64_t(t_in) / int64_t(stride);
        cuda_require(t_out > 0 && t_out * stride == int64_t(t_in), NT_CUDA_ARGUMENT,
                     "last convolution input length is not a stride multiple");

        const auto& it = in->tensor;
        const auto& wt = w->tensor;
        const auto& bt = b->tensor;
        const auto& ot = out->tensor;
        storage_contiguous(in, d, "last convolution input must be the contiguous NTC allocation");
        storage_contiguous(w, d, "last convolution weights must be contiguous [C_out,W*C_in]");
        storage_contiguous(b, d, "last convolution bias must be contiguous [C_out]");
        d->handle = in->id;
        // The source slices `padding` rows off a contiguous [N, T_in + 2*padding,
        // C_in] allocation and reads `T_in` rows from there. Requiring the
        // allocation shape (not just T_in) is what makes the row slice exact.
        cuda_require(it.dim() == 3 && it.size(0) == n && it.size(2) == c_in,
                     NT_CUDA_LAYOUT, "last convolution input shape differs from N,T_alloc,C_in");
        cuda_require(it.size(1) == int64_t(t_in) + 2 * int64_t(padding), NT_CUDA_LAYOUT,
                     "last convolution input rows are not T_in plus two paddings");
        cuda_require(it.stride(0) == in_n_stride && it.stride(1) == c_in && it.stride(2) == 1,
                     NT_CUDA_LAYOUT, "last convolution input strides differ from the view");
        d->handle = w->id;
        // ConvStack.cpp:232-236: `permute({2,1,0}).contiguous().flatten(0,1)`
        // then `transpose(0,1).contiguous()`, i.e. [C_out, W*C_in].
        cuda_require(wt.dim() == 2 && wt.size(0) == c_out &&
                             wt.size(1) == int64_t(window) * c_in,
                     NT_CUDA_LAYOUT, "last convolution weights shape differs from C_out,W*C_in");
        d->handle = b->id;
        cuda_require(bt.dim() == 1 && bt.size(0) == c_out, NT_CUDA_LAYOUT,
                     "last convolution bias shape differs from C_out");
        d->handle = out->id;
        // `wm.get_current_NTC_view()` for CUTLASS_TNC_I8 is
        // `current.narrow(0, start, T).transpose(1, 0)`: shape [N, T_out, C_out]
        // with the TNC parent's strides, so it is not contiguous.
        cuda_require(ot.dim() == 3 && ot.size(0) == n && ot.size(1) == t_out &&
                             ot.size(2) == c_out,
                     NT_CUDA_LAYOUT, "last convolution output shape differs from N,T_out,C_out");
        cuda_require(ot.stride(0) == out_n_stride && ot.stride(1) == out_t_stride &&
                             ot.stride(2) == 1,
                     NT_CUDA_LAYOUT, "last convolution output strides differ from the view");
        for (auto p : {in, w, b}) {
            storage_overlap(out, p, d, "last convolution output overlaps an input interval");
        }
        auto act = lastconv_activation(activation);
        auto koi_in = lastconv_type(in_type);
        auto koi_out = lastconv_type(out_type);
        auto koi_accum = lastconv_type(accum_type);
        c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream, s->device->ordinal));
        CUDAWork work{s, d, {in->tensor, w->tensor, b->tensor, out->tensor}, {}};
        work.run([&] {
            d->stage = NT_CUDA_ENQUEUE;
            // `in.slice(1, padding, None).data_ptr()` on a contiguous NTC tensor.
            void* in_window = static_cast<char*>(it.data_ptr()) +
                              size_t(padding) * size_t(c_in) * storage_dtype_size(in->dtype);
            *koi_result = host_convolution(s->stream, n, t_in, c_in, c_out, window, stride,
                                           padding, in_n_stride, out_n_stride, out_t_stride,
                                           in_window, ot.data_ptr(), nullptr, wt.data_ptr(),
                                           bt.data_ptr(), act, koi_in, koi_out, koi_accum);
        });
#else
        (void)stream_id; (void)in_id; (void)weight_id; (void)bias_id; (void)out_id;
        (void)n; (void)t_in; (void)c_in; (void)c_out; (void)window; (void)stride;
        (void)padding; (void)in_n_stride; (void)out_n_stride; (void)out_t_stride;
        (void)activation; (void)in_type; (void)out_type; (void)accum_type;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED, "ABI5 backend built without CUDA");
#endif
    });
}
int nt_cuda_linear_f16(uint64_t e, uint64_t stream_id, uint64_t in_id, uint64_t weight_id,
                       uint64_t bias_id, uint64_t out_id, int32_t size0, int32_t size1,
                       int32_t k, int32_t c_out, int32_t in_stride0, int32_t in_stride1,
                       int32_t out_stride0, int32_t out_stride1, uint32_t activation,
                       uint32_t in_type, uint32_t out_type, int32_t* koi_result,
                       nt_cuda_detail* d, char* err, size_t cap) {
    return cuda_boundary(d, err, cap, [&] {
        cuda_engine(e, d);
        cuda_require(koi_result, NT_CUDA_ARGUMENT, "linear needs a result out-parameter");
        *koi_result = -1;
#ifdef NT_ENABLE_CUDA
        auto s = cuda_stream(e, stream_id, d);
        auto in = cuda_storage(e, in_id, d);
        auto w = cuda_storage(e, weight_id, d);
        auto b = cuda_storage(e, bias_id, d);
        auto out = cuda_storage(e, out_id, d);
        for (auto p : {in, w, b, out}) {
            d->handle = p->id;
            same_device(s->device, p->device, d);
        }
        lastconv_type_matches(in, in_type, d, "linear input type differs from its storage");
        lastconv_type_matches(w, NT_KOI_F16, d, "linear weights must be F16");
        lastconv_type_matches(b, NT_KOI_F16, d, "linear bias must be F16");
        lastconv_type_matches(out, out_type, d, "linear output type differs from its storage");
        cuda_require(size0 > 0 && size1 > 0 && k > 0 && c_out > 0, NT_CUDA_ARGUMENT,
                     "linear geometry must be positive");
        const auto& it = in->tensor;
        const auto& wt = w->tensor;
        const auto& bt = b->tensor;
        const auto& ot = out->tensor;
        // `host_linear` reads windows straight out of the padded allocation: the
        // window is the K axis and `in_stride1` is `stride * C_in`
        // (ConvStack.cpp:295-300).
        storage_contiguous(in, d, "linear input must be the contiguous NTC allocation");
        storage_contiguous(w, d, "linear weights must be contiguous [C_out,K]");
        storage_contiguous(b, d, "linear bias must be contiguous [C_out]");
        d->handle = in->id;
        cuda_require(it.dim() == 3 && it.size(0) == size0, NT_CUDA_LAYOUT,
                     "linear input shape differs from size0,T_alloc,C_in");
        cuda_require(it.stride(0) == in_stride0 && it.stride(2) == 1, NT_CUDA_LAYOUT,
                     "linear input strides differ from the view");
        // The last window starts at (size1-1)*in_stride1 and spans k elements,
        // so that is what must fit inside the allocation. Using size1 instead of
        // size1-1 would reject the source's own last row.
        cuda_require(int64_t(size1 - 1) * int64_t(in_stride1) + int64_t(k) <=
                             int64_t(it.size(1)) * int64_t(it.size(2)),
                     NT_CUDA_LAYOUT, "linear window exceeds the input allocation");
        d->handle = w->id;
        cuda_require(wt.dim() == 2 && wt.size(0) == c_out && wt.size(1) == k, NT_CUDA_LAYOUT,
                     "linear weights shape differs from C_out,K");
        d->handle = b->id;
        cuda_require(bt.dim() == 1 && bt.size(0) == c_out, NT_CUDA_LAYOUT,
                     "linear bias shape differs from C_out");
        d->handle = out->id;
        cuda_require(ot.dim() == 3 && ot.size(0) == size0 && ot.size(1) == size1 &&
                             ot.size(2) == c_out,
                     NT_CUDA_LAYOUT, "linear output shape differs from size0,size1,C_out");
        cuda_require(ot.stride(0) == out_stride0 && ot.stride(1) == out_stride1 &&
                             ot.stride(2) == 1,
                     NT_CUDA_LAYOUT, "linear output strides differ from the view");
        for (auto p : {in, w, b}) {
            storage_overlap(out, p, d, "linear output overlaps an input interval");
        }
        auto act = lastconv_activation(activation);
        auto koi_in = lastconv_type(in_type);
        auto koi_out = lastconv_type(out_type);
        c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream, s->device->ordinal));
        CUDAWork work{s, d, {in->tensor, w->tensor, b->tensor, out->tensor}, {}};
        work.run([&] {
            d->stage = NT_CUDA_ENQUEUE;
            *koi_result = host_linear(s->stream, koi_in, act, koi_out, size0, size1, k, c_out,
                                      in_stride0, in_stride1, out_stride0, out_stride1,
                                      it.data_ptr(), wt.data_ptr(), ot.data_ptr(), nullptr,
                                      nullptr, bt.data_ptr());
        });
#else
        (void)stream_id; (void)in_id; (void)weight_id; (void)bias_id; (void)out_id;
        (void)size0; (void)size1; (void)k; (void)c_out; (void)in_stride0; (void)in_stride1;
        (void)out_stride0; (void)out_stride1; (void)activation; (void)in_type; (void)out_type;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED, "ABI5 backend built without CUDA");
#endif
    });
}
}
