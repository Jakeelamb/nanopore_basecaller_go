// One fixed C1024 K128 factorised layer. Go owns the seven-layer schedule.
extern "C" {
int nt_cuda_factorised_lstm(uint64_t e, uint64_t stream_id, const uint64_t* ids,
                            int32_t n, int32_t t, int32_t direction, int32_t parity,
                            nt_cuda_detail* d, char* err, size_t cap) {
    return cuda_boundary(d, err, cap, [&] {
        cuda_engine(e, d);
#ifdef NT_ENABLE_CUDA
        cuda_require(ids != nullptr, NT_CUDA_ARGUMENT, "missing FLSTM operands");
        auto s = cuda_stream(e, stream_id, d);
        cuda_require(n > 0 && n % 64 == 0 && t > 0 &&
                     int64_t(t)+1 <= (INT64_MAX - 2LL*n) / 516 / n &&
                     int64_t(t)+5 <= INT64_MAX / 1024 / n &&
                     (direction == -1 || direction == 1) && (parity == 0 || parity == 1),
                     NT_CUDA_ARGUMENT, "fixed FLSTM requires N64, positive T, direction +/-1, parity 0/1");
        cuda_require((direction == -1 && parity == 1) || (direction == 1 && parity == 0),
                     NT_CUDA_UNSUPPORTED, "fixed H05 requires source direction/parity pair");
        std::vector<std::shared_ptr<CUDAStorageRecord>> p;
        for (int i = 0; i < 11; ++i) {
            auto v = cuda_storage(e, ids[i], d);
            same_device(s->device, v->device, d);
            storage_contiguous(v, d, "FLSTM operands must be contiguous");
            cuda_require(reinterpret_cast<uintptr_t>(v->tensor.data_ptr()) % 16 == 0,
                         NT_CUDA_LAYOUT, "FLSTM operand pointer must be 16-byte aligned");
            p.push_back(v);
        }
        const std::vector<std::vector<int64_t>> shapes = {
            {int64_t(t)+5,n,1024}, {2LL*n}, {512LL*(int64_t(t)+1)*n},
            {4LL*(int64_t(t)+1)*n}, {128,1024}, {128,1024}, {128}, {128},
            {4096,256}, {4096}, {4096}};
        const uint32_t types[] = {NT_DTYPE_I8,NT_DTYPE_U8,NT_DTYPE_U8,NT_DTYPE_U8,
            NT_DTYPE_I8,NT_DTYPE_I8,NT_DTYPE_F16,NT_DTYPE_F16,NT_DTYPE_I8,NT_DTYPE_F16,NT_DTYPE_F16};
        for (int i=0;i<11;++i) {
            d->handle=p[i]->id;
            cuda_require(p[i]->dtype==types[i],NT_CUDA_DTYPE,"FLSTM operand dtype mismatch");
            cuda_require(p[i]->tensor.sizes().vec()==shapes[i],NT_CUDA_LAYOUT,"FLSTM operand shape mismatch");
            for (int j=i+1;j<11;++j) {
                if (i>=4) continue;
                storage_overlap(p[i],p[j],d,"FLSTM writable interval overlaps another operand");
            }
        }
        std::vector<at::Tensor> owners;
        for (auto v:p) owners.push_back(v->tensor);
        c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream,s->device->ordinal));
        CUDAWork work{s,d,std::move(owners),{}};
        work.run([&] {
            d->stage=NT_CUDA_ENQUEUE;
            int rc=host_factorised_lstm(s->stream,n,t,direction,parity,
                p[0]->tensor.data_ptr(),p[1]->tensor.data_ptr(),nullptr,
                p[4]->tensor.data_ptr(),p[6]->tensor.data_ptr(),p[5]->tensor.data_ptr(),p[7]->tensor.data_ptr(),
                p[2]->tensor.data_ptr(),p[3]->tensor.data_ptr(),p[8]->tensor.data_ptr(),
                p[9]->tensor.data_ptr(),p[10]->tensor.data_ptr());
            d->native_code=rc;
            cuda_require(rc==0,NT_CUDA_INTERNAL,"host_factorised_lstm failed");
            cuda_runtime(cudaGetLastError(),d,"host_factorised_lstm launch failed");
        });
#else
        (void)stream_id;(void)ids;(void)n;(void)t;(void)direction;(void)parity;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED,"ABI5 backend built without CUDA");
#endif
    });
}
}
