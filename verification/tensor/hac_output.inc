extern "C" {
int nt_cuda_factorised_linear(uint64_t e,uint64_t stream_id,uint64_t input_id,
    uint64_t down_weight_id,uint64_t down_scale_id,uint64_t down_id,
    uint64_t up_weight_id,uint64_t output_id,int64_t n,int64_t t,int32_t parity,
    float scale,nt_cuda_detail* d,char* err,size_t cap) {
    return cuda_boundary(d,err,cap,[&] {
        cuda_engine(e,d);
#ifdef NT_ENABLE_CUDA
        cuda_require(n>0&&t>0&&n<=INT_MAX&&t<=INT_MAX-5&&n%64==0&&t%32==0,
                     NT_CUDA_ARGUMENT,"factorised linear requires checked positive N64/T32 tiles");
        cuda_require(t+5<=INT_MAX/1024/n,NT_CUDA_ARGUMENT,
                     "factorised linear native element extent exceeds int");
        cuda_require(parity==1&&scale==5.f,NT_CUDA_ARGUMENT,
                     "factorised linear requires parity 1 and scale 5");
        auto s=cuda_stream(e,stream_id,d);
        const uint64_t ids[]={input_id,down_weight_id,down_scale_id,down_id,up_weight_id,output_id};
        const uint32_t types[]={NT_DTYPE_I8,NT_DTYPE_I8,NT_DTYPE_F16,NT_DTYPE_F16,NT_DTYPE_F16,NT_DTYPE_F16};
        const std::vector<std::vector<int64_t>> shapes={{t,n,1024},{128,1024},{128},{n*t*128},{1024,128},{n,t,1024}};
        const std::vector<std::vector<int32_t>> tags={{'T','N','C'},{'K','C'},{'K'},{'C'},{'C','K'},{'N','T','C'}};
        std::vector<std::shared_ptr<CUDAStorageRecord>> p;
        for(int i=0;i<6;++i) {
            auto v=cuda_storage(e,ids[i],d);same_device(s->device,v->device,d);
            cuda_require(v->dtype==types[i],NT_CUDA_DTYPE,"factorised linear operand dtype mismatch");
            cuda_require(v->tensor.sizes().vec()==shapes[i],NT_CUDA_LAYOUT,"factorised linear operand shape mismatch");
            storage_contiguous(v,d,"factorised linear operands must be contiguous");
            cuda_require(v->scale_owner==0,NT_CUDA_LAYOUT,"factorised linear requires explicit scales");
            for(size_t j=0;j<tags[i].size();++j)cuda_require(v->tags[j]==tags[i][j],NT_CUDA_LAYOUT,"factorised linear operand tag mismatch");
            cuda_require(reinterpret_cast<uintptr_t>(v->tensor.data_ptr())%16==0,
                         NT_CUDA_LAYOUT,"factorised linear pointer must be 16-byte aligned");
            p.push_back(v);
        }
        for(int i:{3,5})for(int j=0;j<6;++j)if(i!=j)
            storage_overlap(p[i],p[j],d,"factorised linear writable byte intervals overlap");
        std::vector<at::Tensor> owners;for(auto v:p)owners.push_back(v->tensor);
        c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream,s->device->ordinal));
        CUDAWork work{s,d,std::move(owners),{}};
        work.run([&] {
            d->stage=NT_CUDA_ENQUEUE;
            int rc=host_factorised_linear(s->stream,int(n),int(t),parity,
                p[0]->tensor.data_ptr(),p[1]->tensor.data_ptr(),p[2]->tensor.data_ptr(),
                p[3]->tensor.data_ptr(),p[4]->tensor.data_ptr(),nullptr,scale,nullptr,p[5]->tensor.data_ptr());
            d->native_code=rc;cuda_require(rc==0,NT_CUDA_INTERNAL,"host_factorised_linear failed");
            cuda_runtime(cudaGetLastError(),d,"host_factorised_linear launch failed");
        });
#else
        (void)stream_id;(void)input_id;(void)down_weight_id;(void)down_scale_id;(void)down_id;
        (void)up_weight_id;(void)output_id;(void)n;(void)t;(void)parity;(void)scale;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED,"ABI5 backend built without CUDA");
#endif
    });
}
}
