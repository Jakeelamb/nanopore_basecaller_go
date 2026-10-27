extern "C" {
int nt_cuda_storage_gemm_f16(uint64_t e,uint64_t stream_id,uint64_t a_id,uint64_t b_id,
                           uint64_t out_id,nt_cuda_detail* d,char* err,size_t cap) {
    return cuda_boundary(d,err,cap,[&] {
        cuda_engine(e,d);
#ifdef NT_ENABLE_CUDA
        auto s=cuda_stream(e,stream_id,d);
        auto a=cuda_storage(e,a_id,d),b=cuda_storage(e,b_id,d),out=cuda_storage(e,out_id,d);
        for(auto p:{a,b,out}) {
            same_device(s->device,p->device,d);
            cuda_require(p->dtype==NT_DTYPE_F16,NT_CUDA_DTYPE,"half storage GEMM requires half operands");
            auto& t=p->tensor;
            cuda_require(t.dim()==2&&t.stride(1)==1&&t.stride(0)>=t.size(1)&&
                         t.stride(0)<=INT_MAX&&t.size(0)<=INT_MAX&&t.size(1)<=INT_MAX,
                         NT_CUDA_LAYOUT,"half storage GEMM requires rank-two int32 leading dimensions");
            cuda_require(p->scale_owner==0,NT_CUDA_LAYOUT,"half GEMM does not accept scales");
        }
        cuda_require(a->tensor.size(1)==b->tensor.size(0)&&out->tensor.size(0)==a->tensor.size(0)&&
                     out->tensor.size(1)==b->tensor.size(1),NT_CUDA_LAYOUT,"half storage GEMM dimensions differ");
        storage_overlap(a,out,d,"half GEMM output overlaps A");
        storage_overlap(b,out,d,"half GEMM output overlaps B");
        source_half_gemm(s,a,b,out,d);
#else
        (void)stream_id;(void)a_id;(void)b_id;(void)out_id;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED,"backend built without CUDA");
#endif
    });
}

int nt_cuda_lstm_step_f16(uint64_t e,uint64_t stream_id,uint64_t bias_id,uint64_t gates_id,
                         uint64_t cell_id,uint64_t out_id,int32_t n,nt_cuda_detail* d,char* err,size_t cap) {
    return cuda_boundary(d,err,cap,[&] {
        cuda_engine(e,d);
        cuda_require(n>0&&n<=INT_MAX/4096,NT_CUDA_ARGUMENT,"half LSTM requires checked positive N");
#ifdef NT_ENABLE_CUDA
        auto s=cuda_stream(e,stream_id,d);
        auto bias=cuda_storage(e,bias_id,d),gates=cuda_storage(e,gates_id,d);
        auto cell=cuda_storage(e,cell_id,d),out=cuda_storage(e,out_id,d);
        std::vector<std::shared_ptr<CUDAStorageRecord>> values={bias,gates,cell,out};
        std::vector<std::vector<int64_t>> shapes={{4096},{n,4096},{n,1024},{n,1024}};
        for(size_t i=0;i<values.size();++i) {
            auto p=values[i];same_device(s->device,p->device,d);
            cuda_require(p->dtype==NT_DTYPE_F16,NT_CUDA_DTYPE,"half LSTM requires half operands");
            cuda_require(p->tensor.sizes().vec()==shapes[i],NT_CUDA_LAYOUT,"half LSTM operand shape mismatch");
            storage_contiguous(p,d,"half LSTM requires contiguous operands");
            cuda_require(p->scale_owner==0,NT_CUDA_LAYOUT,"half LSTM does not accept scales");
        }
        for(int writable:{2,3})for(int other=0;other<4;++other)if(writable!=other)
            storage_overlap(values[writable],values[other],d,"half LSTM writable operands overlap");
        c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream,s->device->ordinal));
        CUDAWork work{s,d,{bias->tensor,gates->tensor,cell->tensor,out->tensor},{}};
        work.run([&] {
            d->stage=NT_CUDA_ENQUEUE;
            host_lstm_step_f16(s->stream,n,1024,1024,true,bias->tensor.data_ptr(),
                               gates->tensor.data_ptr(),cell->tensor.data_ptr(),out->tensor.data_ptr());
            cuda_runtime(cudaGetLastError(),d,"half LSTM cell launch failed");
        });
#else
        (void)stream_id;(void)bias_id;(void)gates_id;(void)cell_id;(void)out_id;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED,"backend built without CUDA");
#endif
    });
}
}
