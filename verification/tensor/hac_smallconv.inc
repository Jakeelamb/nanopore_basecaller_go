// HAC small convolution boundary. Included once from backend.cpp inside the
// single ABI translation unit, after the typed-storage operations.
//
// The pinned source path this mirrors is ConvStack.cpp:239-262:
//
//   auto out = wm.next_TC(T_out + 2 * output_T_padding, C_out, output_layout);
//   out = out.narrow(1, output_T_padding, T_out);
//   host_convolution_f16(stream, wm.N, C_in, C_out, T_in, winlen, stride,
//                        winlen / 2, int(out.stride(0)), in.data_ptr(),
//                        out.data_ptr(), w_device.data_ptr(), b_device.data_ptr(),
//                        get_koi_activation(params.activation));
//
// Everything above the single kernel call is Go's: weight packing, the working
// memory views, the crop and the batch stride. This file only validates the
// typed handles and makes the one named primitive call.
//
// `nt_cuda_weights_permute_cast_f16` is a verification-only oracle. It
// reproduces `weight.permute({2,1,0}).contiguous().flatten(0,1).to(kF16)` so the
// consumer can prove the pure-Go packer is bit-identical. The caller never uses
// it.
// koi.h carries no linkage specifier; the pinned Dorado source includes it inside
// `extern "C"` (ConvStack.cpp:12-14) because libkoi exports C symbols.
#ifdef NT_ENABLE_CUDA
extern "C" {
#include "koi.h"
}
#endif

namespace {
#ifdef NT_ENABLE_CUDA
// conv_activation accepts exactly the activations get_koi_activation can return
// for the pinned config loader (ConvStack.cpp:21-31). TANH_X5 is deliberately
// not accepted: no config activation maps to it.
KoiActivation conv_activation(uint32_t activation) {
    switch (activation) {
        case NT_KOI_IDENTITY:
            return KOI_IDENTITY;
        case NT_KOI_SWISH:
            return KOI_SWISH;
        case NT_KOI_SWISH_CLAMP:
            return KOI_SWISH_CLAMP;
        case NT_KOI_TANH:
            return KOI_TANH;
    }
    throw CUDAFailure(NT_CUDA_UNSUPPORTED, "unknown Koi activation for the small convolution");
}
#endif
}  // namespace

extern "C" {
int nt_cuda_conv_small_f16(uint64_t e, uint64_t stream_id, uint64_t in_id, uint64_t weight_id,
                           uint64_t bias_id, uint64_t out_id, int32_t n, int32_t c_in,
                           int32_t c_out, int32_t t_in, int32_t window, int32_t stride,
                           int32_t padding, int32_t out_n_stride, uint32_t activation,
                           nt_cuda_detail* d, char* err, size_t cap) {
    return cuda_boundary(d, err, cap, [&] {
        cuda_engine(e, d);
#ifdef NT_ENABLE_CUDA
        auto s = cuda_stream(e, stream_id, d);
        auto in = cuda_storage(e, in_id, d);
        auto w = cuda_storage(e, weight_id, d);
        auto b = cuda_storage(e, bias_id, d);
        auto out = cuda_storage(e, out_id, d);
        for (auto p : {in, w, b, out}) {
            d->handle = p->id;
            same_device(s->device, p->device, d);
            cuda_require(p->dtype == NT_DTYPE_F16, NT_CUDA_DTYPE,
                         "small convolution requires F16 storage");
        }
        cuda_require(n > 0 && c_in > 0 && c_out > 0 && t_in > 0 && window > 0 && stride > 0,
                     NT_CUDA_ARGUMENT, "small convolution geometry must be positive");
        // ConvStack.cpp:223 passes `params.winlen / 2`; an omitted input padding
        // is not a different convolution, it is a different call.
        cuda_require(padding == window / 2, NT_CUDA_ARGUMENT,
                     "small convolution padding must be window/2");
        const int64_t t_out = int64_t(t_in) / int64_t(stride);
        cuda_require(t_out > 0 && t_out * stride == int64_t(t_in),
                     NT_CUDA_ARGUMENT, "small convolution input length is not a stride multiple");

        const auto& it = in->tensor;
        const auto& wt = w->tensor;
        const auto& bt = b->tensor;
        const auto& ot = out->tensor;
        storage_contiguous(in, d, "small convolution input must be contiguous NTC");
        storage_contiguous(w, d, "small convolution weights must be contiguous [W*C_in,C_out]");
        storage_contiguous(b, d, "small convolution bias must be contiguous [C_out]");
        d->handle = in->id;
        cuda_require(it.dim() == 3 && it.size(0) == n && it.size(1) == t_in && it.size(2) == c_in,
                     NT_CUDA_LAYOUT, "small convolution input shape differs from N,T,C_in");
        d->handle = w->id;
        cuda_require(wt.dim() == 2 && wt.size(0) == int64_t(window) * c_in && wt.size(1) == c_out,
                     NT_CUDA_LAYOUT, "small convolution weights shape differs from W*C_in,C_out");
        d->handle = b->id;
        cuda_require(bt.dim() == 1 && bt.size(0) == c_out, NT_CUDA_LAYOUT,
                     "small convolution bias shape differs from C_out");
        d->handle = out->id;
        // The cropped output view keeps the parent's batch stride
        // (`out.stride(0)`), so it is not contiguous when output_T_padding is
        // nonzero. Requiring contiguity here would silently drop the nine rows.
        cuda_require(ot.dim() == 3 && ot.size(0) == n && ot.size(1) == t_out && ot.size(2) == c_out,
                     NT_CUDA_LAYOUT, "small convolution output shape differs from N,T_out,C_out");
        cuda_require(ot.stride(0) == out_n_stride && ot.stride(1) == c_out && ot.stride(2) == 1,
                     NT_CUDA_LAYOUT, "small convolution output batch stride differs from the view");
        for (auto p : {in, w, b}) {
            storage_overlap(out, p, d, "small convolution output overlaps an input interval");
        }
        auto act = conv_activation(activation);
        c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream, s->device->ordinal));
        CUDAWork work{s, d, {in->tensor, w->tensor, b->tensor, out->tensor}, {}};
        work.run([&] {
            d->stage = NT_CUDA_ENQUEUE;
            const int rc = host_convolution_f16(s->stream, n, c_in, c_out, t_in, window, stride,
                                               padding, out_n_stride, in->tensor.data_ptr(),
                                               out->tensor.data_ptr(), w->tensor.data_ptr(),
                                               b->tensor.data_ptr(), act);
            cuda_require(rc == 0, NT_CUDA_INTERNAL, "host_convolution_f16 failed");
        });
#else
        (void)stream_id; (void)in_id; (void)weight_id; (void)bias_id; (void)out_id;
        (void)n; (void)c_in; (void)c_out; (void)t_in; (void)window; (void)stride;
        (void)padding; (void)out_n_stride; (void)activation;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED, "ABI5 backend built without CUDA");
#endif
    });
}
int nt_cuda_weights_permute_cast_f16(uint64_t e, uint64_t stream_id, uint64_t weight_id,
                                     uint64_t bias_id, uint64_t packed_id,
                                     uint64_t packed_bias_id, nt_cuda_detail* d, char* err,
                                     size_t cap) {
    return cuda_boundary(d, err, cap, [&] {
        cuda_engine(e, d);
#ifdef NT_ENABLE_CUDA
        auto s = cuda_stream(e, stream_id, d);
        auto w = cuda_storage(e, weight_id, d);
        auto b = cuda_storage(e, bias_id, d);
        auto pw = cuda_storage(e, packed_id, d);
        auto pb = cuda_storage(e, packed_bias_id, d);
        for (auto p : {w, b}) {
            d->handle = p->id;
            same_device(s->device, p->device, d);
            cuda_require(p->dtype == NT_DTYPE_F32, NT_CUDA_DTYPE,
                         "weight reference requires F32 source storage");
        }
        for (auto p : {pw, pb}) {
            d->handle = p->id;
            same_device(s->device, p->device, d);
            cuda_require(p->dtype == NT_DTYPE_F16, NT_CUDA_DTYPE,
                         "weight reference requires F16 destination storage");
        }
        const auto& wt = w->tensor;
        const auto& bt = b->tensor;
        const auto& pt = pw->tensor;
        const auto& pbt = pb->tensor;
        d->handle = w->id;
        cuda_require(wt.dim() == 3, NT_CUDA_LAYOUT,
                     "weight reference source must be [C_out,C_in,W]");
        storage_contiguous(w, d, "weight reference source must be contiguous");
        storage_contiguous(b, d, "weight reference bias must be contiguous");
        storage_contiguous(pw, d, "weight reference destination must be contiguous");
        storage_contiguous(pb, d, "weight reference bias destination must be contiguous");
        const int64_t c_out = wt.size(0), c_in = wt.size(1), win = wt.size(2);
        d->handle = pw->id;
        cuda_require(pt.dim() == 2 && pt.size(0) == win * c_in && pt.size(1) == c_out,
                     NT_CUDA_LAYOUT, "weight reference destination must be [W*C_in,C_out]");
        d->handle = b->id;
        cuda_require(bt.dim() == 1 && bt.size(0) == c_out, NT_CUDA_LAYOUT,
                     "weight reference bias source must be [C_out]");
        d->handle = pb->id;
        cuda_require(pbt.dim() == 1 && pbt.size(0) == c_out, NT_CUDA_LAYOUT,
                     "weight reference bias destination must be [C_out]");
        c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream, s->device->ordinal));
        CUDAWork work{s, d, {w->tensor, b->tensor, pw->tensor, pb->tensor}, {}};
        work.run([&] {
            d->stage = NT_CUDA_ENQUEUE;
            auto opts = wt.options().dtype(at::kHalf);
            // Exactly ConvStack.cpp:230-233.
            pw->tensor.copy_(wt.permute({2, 1, 0}).contiguous().flatten(0, 1).to(opts));
            pb->tensor.copy_(bt.to(opts));
            cuda_runtime(cudaGetLastError(), d, "weight reference launch failed");
        });
#else
        (void)stream_id; (void)weight_id; (void)bias_id; (void)packed_id; (void)packed_bias_id;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED, "ABI5 backend built without CUDA");
#endif
    });
}
}
