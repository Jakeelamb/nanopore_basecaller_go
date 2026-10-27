// Typed CUDA storage. Included once from backend.cpp inside the single ABI
// translation unit. Every operation validates before enqueueing, keeps the
// reviewed synchronous-drain contract, and retains every owner until the drain
// completes. Dtypes mirror the pinned Koi type ids (koi.h:3).
#include "storage_byte_interval.h"
namespace {
#ifdef NT_ENABLE_CUDA
size_t storage_dtype_size(uint32_t dtype) {
    switch (dtype) {
        case NT_DTYPE_F16: return 2;
        case NT_DTYPE_I8: return 1;
        case NT_DTYPE_U8: return 1;
        case NT_DTYPE_I32: return 4;
        case NT_DTYPE_F32: return 4;
        case NT_DTYPE_E4M3: return 1;
    }
    throw CUDAFailure(NT_CUDA_DTYPE, "unknown storage dtype");
}
at::ScalarType storage_scalar(uint32_t dtype) {
    switch (dtype) {
        case NT_DTYPE_F16: return at::kHalf;
        case NT_DTYPE_I8: return at::kChar;
        case NT_DTYPE_U8: return at::kByte;
        case NT_DTYPE_I32: return at::kInt;
        case NT_DTYPE_F32: return at::kFloat;
        case NT_DTYPE_E4M3: return at::kFloat8_e4m3fn;
    }
    throw CUDAFailure(NT_CUDA_DTYPE, "unknown storage dtype");
}
#endif
bool storage_tag(int32_t tag) {
    switch (tag) {
        case NT_KOI_TAG_QKV:
        case NT_KOI_TAG_GATE_IFGO:
        case NT_KOI_TAG_W_U:
            return true;
    }
    // Single-character tags used literally by the pinned SUP/HAC source.
    switch (tag) {
        case 'N': case 'T': case 'C': case 't': case 'c':
        case 'M': case 'K': case 'm': case 'k':
        case 'H': case 'D': case 'd': case 'W': case 'n':
            return true;
    }
    return false;
}
void storage_tags(const int32_t* tags, size_t rank) {
    cuda_require(rank && rank <= NT_STORAGE_MAX_RANK, NT_CUDA_LAYOUT, "storage rank must be 1 through 8");
    cuda_require(tags, NT_CUDA_LAYOUT, "null storage tags");
    for (size_t i = 0; i < rank; ++i) cuda_require(storage_tag(tags[i]), NT_CUDA_LAYOUT, "unknown Koi dimension tag");
}
size_t storage_elements(const int64_t* shape, size_t rank) {
    cuda_require(shape && rank && rank <= NT_STORAGE_MAX_RANK, NT_CUDA_LAYOUT, "storage rank must be 1 through 8");
    size_t count = 1;
    for (size_t i = 0; i < rank; ++i) {
        cuda_require(shape[i] > 0, NT_CUDA_LAYOUT, "nonpositive storage dimension");
        auto dim = uint64_t(shape[i]);
        cuda_require(dim <= uint64_t(INT64_MAX) / 2 / count, NT_CUDA_LAYOUT, "storage element count overflow");
        count *= size_t(dim);
    }
    return count;
}
uint64_t storage_end(const int64_t* shape, const int64_t* strides, size_t rank, uint64_t offset) {
    storage_elements(shape, rank);
    cuda_require(strides, NT_CUDA_LAYOUT, "null strides");
    cuda_require(offset < uint64_t(INT64_MAX) / 2, NT_CUDA_LAYOUT, "storage offset overflow");
    uint64_t end = offset;
    for (size_t i = 0; i < rank; ++i) {
        cuda_require(strides[i] >= 0, NT_CUDA_LAYOUT, "negative storage stride");
        auto dim = uint64_t(shape[i] - 1), stride = uint64_t(strides[i]);
        cuda_require(!dim || stride <= (uint64_t(INT64_MAX) / 2 - end) / dim, NT_CUDA_LAYOUT, "storage span overflow");
        end += dim * stride;
    }
    cuda_require(end < uint64_t(INT64_MAX) / 2, NT_CUDA_LAYOUT, "storage span overflow");
    return end + 1;
}
#ifdef NT_ENABLE_CUDA
std::shared_ptr<CUDAStorageRecord> cuda_storage(uint64_t e, uint64_t id, nt_cuda_detail* d) {
    auto& r = cuda_engine(e, d);
    auto i = r.storages.find(id);
    if (i == r.storages.end()) cuda_unknown(e, id, d);
    auto p = i->second;
    d->handle = id;
    d->actual_device = p->device->id;
    d->actual_ordinal = p->device->ordinal;
    d->actual_dtype = p->dtype;
    d->expected_dtype = p->dtype;
    cuda_require(p->tensor.scalar_type() == storage_scalar(p->dtype), NT_CUDA_DTYPE, "storage record dtype mismatch");
    return p;
}
void storage_contiguous(const std::shared_ptr<CUDAStorageRecord>& p, nt_cuda_detail* d, const char* why) {
    d->handle = p->id;
    cuda_require(p->tensor.is_contiguous(), NT_CUDA_LAYOUT, why);
}
size_t storage_bytes(const std::shared_ptr<CUDAStorageRecord>& p) {
    return size_t(p->tensor.numel()) * storage_dtype_size(p->dtype);
}
void storage_overlap(const std::shared_ptr<CUDAStorageRecord>& a, const std::shared_ptr<CUDAStorageRecord>& b, nt_cuda_detail* d, const char* why) {
    if (a->allocation != b->allocation) return;
    d->handle = a->id;
    StorageByteInterval a_bytes, b_bytes;
    cuda_require(checked_storage_byte_interval(a->offset, a->end, storage_dtype_size(a->dtype), a_bytes) &&
                         checked_storage_byte_interval(b->offset, b->end, storage_dtype_size(b->dtype), b_bytes),
                 NT_CUDA_LAYOUT, "storage byte interval overflow or invalid endpoints");
    cuda_require(storage_byte_intervals_disjoint(a_bytes, b_bytes), NT_CUDA_LAYOUT, why);
}
#endif
}  // namespace
extern "C" {
int nt_cuda_storage_new(uint64_t e, uint64_t dev_id, uint64_t stream_id, uint32_t dtype,
                        const int64_t* shape, size_t rank, const int32_t* tags, const uint8_t* bytes,
                        size_t count, uint64_t* out, nt_cuda_detail* d, char* err, size_t cap) {
    return cuda_boundary(d, err, cap, [&] {
        cuda_engine(e, d);
        cuda_require(out, NT_CUDA_ARGUMENT, "null storage output");
        auto n = storage_elements(shape, rank);
        storage_tags(tags, rank);
#ifdef NT_ENABLE_CUDA
        auto size = storage_dtype_size(dtype);
        cuda_require(count == n * size, NT_CUDA_LAYOUT, "storage byte count differs from shape");
        cuda_require(!count || bytes, NT_CUDA_ARGUMENT, "null storage upload bytes");
        auto dev = cuda_device(e, dev_id, d);
        auto s = cuda_stream(e, stream_id, d);
        same_device(dev, s->device, d);
        c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream, dev->ordinal));
        auto p = std::make_shared<CUDAStorageRecord>();
        p->device = dev;
        p->id = fresh();
        p->allocation = p->id;
        p->dtype = dtype;
        p->storage = n;
        p->end = n;
        for (size_t i = 0; i < rank; ++i) p->tags[i] = tags[i];
        d->stage = NT_CUDA_ALLOCATE;
        p->tensor = at::empty(at::IntArrayRef(shape, rank),
                              at::TensorOptions().dtype(storage_scalar(dtype)).device(at::Device(at::kCUDA, dev->ordinal)));
        if (count) {
            auto host = std::make_shared<std::vector<uint8_t>>(bytes, bytes + count);
            CUDAWork work{s, d, {p->tensor}, {}};
            work.run([&] {
                d->stage = NT_CUDA_COPY;
                cuda_runtime(cudaMemcpyAsync(p->tensor.data_ptr(), host->data(), count, cudaMemcpyHostToDevice, s->stream), d, "storage upload failed");
            });
        }
        engines.at(e).storages.emplace(p->id, p);
        *out = p->id;
#else
        (void)dev_id; (void)stream_id; (void)dtype; (void)bytes; (void)count;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED, "backend built without CUDA");
#endif
    });
}
int nt_cuda_storage_view(uint64_t e, uint64_t parent_id, uint32_t dtype, uint64_t offset_elements,
                         const int64_t* shape, const int64_t* strides, const int32_t* tags, size_t rank,
                         uint64_t* out, nt_cuda_detail* d, char* err, size_t cap) {
    return cuda_boundary(d, err, cap, [&] {
        cuda_engine(e, d);
        cuda_require(out, NT_CUDA_ARGUMENT, "null storage view output");
        storage_elements(shape, rank);
        storage_tags(tags, rank);
#ifdef NT_ENABLE_CUDA
        auto parent = cuda_storage(e, parent_id, d);
        d->handle = parent_id;
        d->actual_dtype = parent->dtype;
        d->expected_dtype = dtype;
        cuda_require(dtype == parent->dtype, NT_CUDA_DTYPE, "storage view dtype differs from parent");
        cuda_require(offset_elements <= parent->end - parent->offset, NT_CUDA_LAYOUT, "storage view offset exceeds parent interval");
        auto start = parent->offset + offset_elements;
        auto end = storage_end(shape, strides, rank, start);
        cuda_require(end <= parent->end, NT_CUDA_LAYOUT, "storage view exceeds parent interval");
        auto p = std::make_shared<CUDAStorageRecord>(*parent);
        p->id = fresh();
        p->offset = start;
        p->end = end;
        for (size_t i = 0; i < rank; ++i) p->tags[i] = tags[i];
        p->tensor = parent->tensor.as_strided(at::IntArrayRef(shape, rank), at::IntArrayRef(strides, rank), int64_t(start));
        engines.at(e).storages.emplace(p->id, p);
        *out = p->id;
#else
        (void)parent_id; (void)dtype; (void)offset_elements; (void)strides;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED, "backend built without CUDA");
#endif
    });
}
int nt_cuda_storage_reinterpret(uint64_t e, uint64_t parent_id, uint32_t dtype, uint64_t offset_elements,
                                const int64_t* shape, const int64_t* strides, const int32_t* tags,
                                size_t rank, uint64_t* out, nt_cuda_detail* d, char* err, size_t cap) {
    return cuda_boundary(d, err, cap, [&] {
        cuda_engine(e, d);
        cuda_require(out, NT_CUDA_ARGUMENT, "null reinterpret output");
        storage_elements(shape, rank);
        storage_tags(tags, rank);
#ifdef NT_ENABLE_CUDA
        auto parent = cuda_storage(e, parent_id, d);
        auto psz = storage_dtype_size(parent->dtype);
        auto nsz = storage_dtype_size(dtype);
        d->handle = parent_id;
        d->actual_dtype = parent->dtype;
        d->expected_dtype = dtype;
        // `backing_tensor.view(dtype)` reinterprets a contiguous buffer. A
        // strided parent has no single byte interval to reinterpret.
        cuda_require(parent->tensor.is_contiguous(), NT_CUDA_LAYOUT,
                     "reinterpret parent must be contiguous");
        StorageByteInterval parent_bytes;
        uint64_t start = 0;
        cuda_require(checked_storage_byte_interval(parent->offset, parent->end, psz, parent_bytes) &&
                             checked_storage_reinterpret_offset(parent_bytes, nsz, offset_elements, start),
                     NT_CUDA_LAYOUT, "reinterpret offset is invalid or exceeds the parent byte interval");
        auto end = storage_end(shape, strides, rank, start);
        cuda_require(end <= parent_bytes.end / nsz, NT_CUDA_LAYOUT,
                     "reinterpret view exceeds the parent byte interval");
        auto p = std::make_shared<CUDAStorageRecord>(*parent);
        p->id = fresh();
        p->dtype = dtype;
        p->offset = start;
        p->end = end;
        p->storage = parent_bytes.end / nsz;
        for (size_t i = 0; i < rank; ++i) p->tags[i] = tags[i];
        auto flat = parent->tensor.view(storage_scalar(dtype));
        p->tensor = flat.as_strided(at::IntArrayRef(shape, rank), at::IntArrayRef(strides, rank),
                                    int64_t(start));
        engines.at(e).storages.emplace(p->id, p);
        *out = p->id;
#else
        (void)parent_id; (void)dtype; (void)offset_elements; (void)strides;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED, "ABI5 backend built without CUDA");
#endif
    });
}
int nt_cuda_storage_meta(uint64_t e, uint64_t id, nt_storage_meta* out, nt_cuda_detail* d, char* err, size_t cap) {
    return cuda_boundary(d, err, cap, [&] {
        cuda_engine(e, d);
        cuda_require(out, NT_CUDA_ARGUMENT, "null storage metadata output");
#ifdef NT_ENABLE_CUDA
        auto p = cuda_storage(e, id, d);
        nt_storage_meta m{};
        m.rank = p->tensor.dim();
        m.dtype = p->dtype;
        m.device = p->device->id;
        m.ordinal = p->device->ordinal;
        m.scale_dtype = p->scale_dtype;
        m.scale_present = p->scale_owner != 0;
        m.scale_tag = p->scale_tag;
        m.allocation = p->allocation;
        m.offset_elements = p->offset;
        m.storage_elements = p->storage;
        for (size_t i = 0; i < m.rank; ++i) {
            m.shape[i] = p->tensor.size(i);
            m.strides[i] = p->tensor.stride(i);
            m.tags[i] = p->tags[i];
        }
        *out = m;
#else
        (void)id;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED, "backend built without CUDA");
#endif
    });
}
int nt_cuda_storage_upload(uint64_t e, uint64_t stream_id, uint64_t id, const uint8_t* bytes, size_t count,
                           nt_cuda_detail* d, char* err, size_t cap) {
    return cuda_boundary(d, err, cap, [&] {
        cuda_engine(e, d);
#ifdef NT_ENABLE_CUDA
        auto p = cuda_storage(e, id, d);
        auto s = cuda_stream(e, stream_id, d);
        same_device(s->device, p->device, d);
        storage_contiguous(p, d, "storage upload requires a contiguous view");
        cuda_require(bytes && count == storage_bytes(p), NT_CUDA_LAYOUT, "storage upload count differs from view");
        c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream, p->device->ordinal));
        auto host = std::make_shared<std::vector<uint8_t>>(bytes, bytes + count);
        CUDAWork work{s, d, {p->tensor}, {}};
        work.run([&] {
            d->stage = NT_CUDA_COPY;
            cuda_runtime(cudaMemcpyAsync(p->tensor.data_ptr(), host->data(), count, cudaMemcpyHostToDevice, s->stream), d, "storage upload failed");
        });
#else
        (void)stream_id; (void)id; (void)bytes; (void)count;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED, "backend built without CUDA");
#endif
    });
}
int nt_cuda_storage_download(uint64_t e, uint64_t stream_id, uint64_t id, uint8_t* bytes, size_t count,
                             nt_cuda_detail* d, char* err, size_t cap) {
    return cuda_boundary(d, err, cap, [&] {
        cuda_engine(e, d);
#ifdef NT_ENABLE_CUDA
        auto p = cuda_storage(e, id, d);
        auto s = cuda_stream(e, stream_id, d);
        same_device(s->device, p->device, d);
        storage_contiguous(p, d, "storage download requires a contiguous view");
        cuda_require(bytes && count == storage_bytes(p), NT_CUDA_LAYOUT, "storage download count differs from view");
        c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream, p->device->ordinal));
        auto host = std::make_shared<std::vector<uint8_t>>(count);
        CUDAWork work{s, d, {p->tensor}, {}};
        work.run([&] {
            d->stage = NT_CUDA_COPY;
            cuda_runtime(cudaMemcpyAsync(host->data(), p->tensor.data_ptr(), count, cudaMemcpyDeviceToHost, s->stream), d, "storage download failed");
        });
        std::memcpy(bytes, host->data(), count);
#else
        (void)stream_id; (void)id; (void)bytes; (void)count;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED, "backend built without CUDA");
#endif
    });
}
int nt_cuda_storage_copy(uint64_t e, uint64_t stream_id, uint64_t dst_id, uint64_t src_id,
                         nt_cuda_detail* d, char* err, size_t cap) {
    return cuda_boundary(d, err, cap, [&] {
        cuda_engine(e, d);
#ifdef NT_ENABLE_CUDA
        auto s = cuda_stream(e, stream_id, d);
        auto dst = cuda_storage(e, dst_id, d);
        auto src = cuda_storage(e, src_id, d);
        for (auto p : {dst, src}) same_device(s->device, p->device, d);
        d->handle = dst_id;
        d->actual_dtype = src->dtype;
        d->expected_dtype = dst->dtype;
        cuda_require(dst->dtype == src->dtype, NT_CUDA_DTYPE, "storage copy dtype differs");
        storage_contiguous(dst, d, "storage copy destination requires a contiguous view");
        storage_contiguous(src, d, "storage copy source requires a contiguous view");
        cuda_require(dst->tensor.numel() == src->tensor.numel(), NT_CUDA_LAYOUT, "storage copy element counts differ");
        storage_overlap(dst, src, d, "storage copy source and destination intervals overlap");
        c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream, dst->device->ordinal));
        CUDAWork work{s, d, {dst->tensor, src->tensor}, {}};
        work.run([&] {
            d->stage = NT_CUDA_COPY;
            cuda_runtime(cudaMemcpyAsync(dst->tensor.data_ptr(), src->tensor.data_ptr(), storage_bytes(dst), cudaMemcpyDeviceToDevice, s->stream), d, "storage copy failed");
        });
#else
        (void)stream_id; (void)dst_id; (void)src_id;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED, "backend built without CUDA");
#endif
    });
}
int nt_cuda_storage_fill(uint64_t e, uint64_t stream_id, uint64_t id, const uint8_t* pattern, size_t count,
                         nt_cuda_detail* d, char* err, size_t cap) {
    return cuda_boundary(d, err, cap, [&] {
        cuda_engine(e, d);
#ifdef NT_ENABLE_CUDA
        auto p = cuda_storage(e, id, d);
        auto s = cuda_stream(e, stream_id, d);
        same_device(s->device, p->device, d);
        storage_contiguous(p, d, "storage fill requires a contiguous view");
        auto size = storage_dtype_size(p->dtype);
        cuda_require(pattern && count == size, NT_CUDA_LAYOUT, "storage fill pattern length differs from element size");
        auto elements = size_t(p->tensor.numel());
        auto bytes = elements * size;
        auto host = std::make_shared<std::vector<uint8_t>>(bytes);
        for (size_t i = 0; i < elements; ++i) std::memcpy(host->data() + i * size, pattern, size);
        c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream, p->device->ordinal));
        CUDAWork work{s, d, {p->tensor}, {}};
        work.run([&] {
            d->stage = NT_CUDA_COPY;
            cuda_runtime(cudaMemcpyAsync(p->tensor.data_ptr(), host->data(), bytes, cudaMemcpyHostToDevice, s->stream), d, "storage fill failed");
        });
#else
        (void)stream_id; (void)id; (void)pattern; (void)count;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED, "backend built without CUDA");
#endif
    });
}
int nt_cuda_storage_scale(uint64_t e, uint64_t id, uint64_t scale_id, int32_t scale_tag, nt_cuda_detail* d, char* err, size_t cap) {
    return cuda_boundary(d, err, cap, [&] {
        cuda_engine(e, d);
#ifdef NT_ENABLE_CUDA
        auto p = cuda_storage(e, id, d);
        if (scale_id == 0) {
            p->scale_owner = 0;
            p->scale_dtype = 0;
            p->scale_tag = 0;
            return;
        }
        cuda_require(storage_tag(scale_tag), NT_CUDA_LAYOUT, "unknown Koi scale tag");
        auto scale = cuda_storage(e, scale_id, d);
        same_device(p->device, scale->device, d);
        d->handle = scale_id;
        d->actual_dtype = scale->dtype;
        d->expected_dtype = NT_DTYPE_F32;
        cuda_require(scale_id != id, NT_CUDA_ARGUMENT, "storage cannot own itself as its Koi scale");
        d->handle = id;
        d->actual_dtype = p->dtype;
        d->expected_dtype = NT_DTYPE_I8;
        cuda_require(p->dtype == NT_DTYPE_I8, NT_CUDA_DTYPE, "Koi scale owner requires an I8 tensor");
        d->handle = scale_id;
        d->actual_dtype = scale->dtype;
        d->expected_dtype = NT_DTYPE_F32;
        cuda_require(scale->dtype == NT_DTYPE_F32, NT_CUDA_DTYPE, "Koi scale owner must be F32");
        cuda_require(scale->tensor.dim() == 1 && scale->tensor.numel() > 0, NT_CUDA_LAYOUT, "Koi scale owner must be a nonempty vector");
        p->scale_owner = scale_id;
        p->scale_dtype = scale->dtype;
        p->scale_tag = scale_tag;
#else
        (void)id; (void)scale_id;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED, "backend built without CUDA");
#endif
    });
}
int nt_cuda_storage_release(uint64_t e, uint64_t id, nt_cuda_detail* d, char* err, size_t cap) {
    return cuda_boundary(d, err, cap, [&] {
        cuda_engine(e, d);
#ifdef NT_ENABLE_CUDA
        auto p = cuda_storage(e, id, d);
        cuda_require(!p->device->poisoned, NT_CUDA_RUNTIME, "poisoned device retains storage until successful close drain");
        for (const auto& item : engines.at(e).storages) {
            if (item.second->scale_owner == id) {
                d->handle = item.second->id;
                throw CUDAFailure(NT_CUDA_OWNER, "storage is still the Koi scale owner of a live view");
            }
        }
        engines.at(e).storages.erase(id);
#else
        (void)id;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED, "backend built without CUDA");
#endif
    });
}
}
