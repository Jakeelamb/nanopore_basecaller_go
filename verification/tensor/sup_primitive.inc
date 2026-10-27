namespace {
#ifdef NT_ENABLE_CUDA
struct SUPRole {
    uint32_t dtype;
    std::vector<int64_t> shape;
    std::vector<int32_t> tags;
    bool write=false;
    int32_t scale_tag=0;
};
SUPRole sup_role(uint32_t type,std::vector<int64_t> shape,std::string tags,bool write=false,int32_t scale=0) {
    return {type,std::move(shape),std::vector<int32_t>(tags.begin(),tags.end()),write,scale};
}
std::vector<SUPRole> sup_roles(uint32_t geometry,uint32_t op,int32_t selector) {
    cuda_require(geometry==NT_SUP_GEOMETRY_TINY||geometry==NT_SUP_GEOMETRY_FULL||geometry==NT_SUP_GEOMETRY_SHORT,NT_CUDA_ARGUMENT,"unknown fixed SUP geometry");
    const int64_t n=32,t=geometry==NT_SUP_GEOMETRY_TINY?64:geometry==NT_SUP_GEOMETRY_SHORT?512:1024,chunk=t*12,tiles=t/16,rows=n*t/16;
    auto h=[](std::vector<int64_t> s,std::string t,bool w=false){return sup_role(NT_DTYPE_F16,std::move(s),std::move(t),w);};
    auto f=[](std::vector<int64_t> s,std::string t,bool w=false){return sup_role(NT_DTYPE_E4M3,std::move(s),std::move(t),w);};
    auto i=[](std::vector<int64_t> s,std::string t,bool w=false){return sup_role(NT_DTYPE_I8,std::move(s),std::move(t),w,'C');};
    auto scale=[](std::vector<int64_t> s,bool w=false){auto n=s.size();return sup_role(NT_DTYPE_F32,std::move(s),std::string(n,'\0'),w);};
    auto ctr=[](){return sup_role(NT_DTYPE_I32,{1},"C",true);};
    auto half=[&](bool w=false){return h({n,tiles,64,16,8},"NTCtc",w);};
    cuda_require(op==NT_SUP_CONV_SILU || op==NT_SUP_LINEAR || selector==0,NT_CUDA_ARGUMENT,"unexpected SUP selector");
    switch(op) {
        case NT_SUP_CONV_SILU: {
            cuda_require(selector>=0 && selector<5,NT_CUDA_ARGUMENT,"SUP conv selector outside 0..4");
            const int ci[]={1,64,64,128,128},co[]={64,64,128,128,512},time[]={int(chunk),int(chunk),int(chunk),int(chunk/3),int(chunk/6)},st[]={1,1,3,2,2},w[]={5,5,9,9,5};
            auto j=selector;
            return {h({n,ci[j],time[j]},"NCT"),h({co[j],ci[j],w[j]},"NCW"),h({co[j]},"C"),h({n,co[j],time[j]/st[j]},"NCT",true)};
        }
        case NT_SUP_QUANTIZE_ROWS:return {h({n,t,512},"NTC"),sup_role(NT_DTYPE_I8,{n,t,512},"NTC",true),scale({n,t},true)};
        case NT_SUP_SINCOS:return {h({4096,32},"TD"),h({4096,32},"TD",true),h({4096,32},"TD",true)};
        case NT_SUP_LINEAR:
            if(selector==0)return {h({n,t,512},"NTC"),h({1024,512},"NC"),h({1024},"C"),h({n,t,1024},"NTC",true)};
            if(selector==1)return {h({n,2*t,512},"NTC"),h({4096,512},"NC"),h({n,2*t,4096},"NTC",true)};
            break;
        case NT_SUP_QKV_ROTARY:return {i({n,tiles,32,16,16},"NTCtc"),scale({n,t}),i({3,8,4,32,16,16},std::string("\0HDCdc",6)),scale({3,8,64}),h({tiles,8,16,8},"TDtd"),h({n,tiles,3,8,8,16,8},std::string("NT\0HDtd",7),true),ctr()};
        case NT_SUP_MASKED_ATTENTION:return {h({n,tiles,3,8,8,16,8},std::string("NT\0HDtd",7)),h({n,tiles,8,8,16,8},"NTHDtd",true)};
        case NT_SUP_PROJECTION:return {h({rows,64,16,8},"MKmk"),h({32,64,16,8},"NKnk"),h({512},"N"),h({rows,64,16,8},"MNmn",true),ctr()};
        case NT_SUP_NORM1:return {half(),half(true),h({512},"C"),f({n,tiles,32,16,16},"NTCtc",true)};
        case NT_SUP_SWIGLU:return {f({rows,32,16,16},"MKmk"),f({256,32,16,16},"NKnk"),f({rows,128,16,16},"MNmn",true),ctr()};
        case NT_SUP_FC2:return {f({rows,128,16,16},"MKmk"),f({32,128,16,16},"NKnk"),h({rows,64,16,8},"MNmn",true),ctr()};
        case NT_SUP_NORM2:return {half(),half(true),h({512},"C"),i({n,tiles,32,16,16},"NTCtc",true),scale({n,t},true)};
    }
    throw CUDAFailure(NT_CUDA_ARGUMENT,"unknown fixed SUP operation or selector");
}
KoiTensor sup_koi(const std::shared_ptr<CUDAStorageRecord>& p,const SUPRole& role,const std::shared_ptr<CUDAStorageRecord>& scale={}) {
    KoiTensor k{};k.data_ptr=p->tensor.data_ptr();k.ndims=int(role.shape.size());
    switch(p->dtype){case NT_DTYPE_F16:k.type_id=KOI_F16;break;case NT_DTYPE_E4M3:k.type_id=KOI_E4M3;break;case NT_DTYPE_I8:k.type_id=KOI_I8;break;default:throw CUDAFailure(NT_CUDA_DTYPE,"SUP Koi dtype unsupported");}
    for(int j=0;j<k.ndims;++j){k.dims[j].tag=role.tags[j];k.dims[j].size=p->tensor.size(j);k.dims[j].stride=p->tensor.stride(j);}
    if(scale){k.scale_tag=role.scale_tag;k.scale_type_id=KOI_F32;k.scale_data=scale->tensor.data_ptr();}
    return k;
}
#endif
}

extern "C" int nt_cuda_sup_primitive(uint64_t e,uint64_t stream_id,uint32_t geometry,uint32_t op,int32_t selector,
    const uint64_t* ids,size_t count,float alpha,nt_cuda_detail* d,char* err,size_t cap) {
    return cuda_boundary(d,err,cap,[&] {
        cuda_require(geometry==NT_SUP_GEOMETRY_TINY||geometry==NT_SUP_GEOMETRY_FULL||geometry==NT_SUP_GEOMETRY_SHORT,NT_CUDA_ARGUMENT,"unknown fixed SUP geometry");
        cuda_engine(e,d);
        cuda_require(ids&&count>0&&count<=8,NT_CUDA_ARGUMENT,"SUP operand list outside 1..8");
#ifdef NT_ENABLE_CUDA
        cuda_require((op==NT_SUP_NORM1||op==NT_SUP_NORM2)?alpha==2.4494897f:alpha==0.f,NT_CUDA_ARGUMENT,"SUP scalar differs from fixed source");
        auto s=cuda_stream(e,stream_id,d);
        std::vector<std::shared_ptr<CUDAStorageRecord>> p;
        for(size_t j=0;j<count;++j){auto v=cuda_storage(e,ids[j],d);same_device(s->device,v->device,d);p.push_back(v);}
        std::vector<SUPRole> roles;
        if(op==NT_SUP_COPY_STRIDED) {
            cuda_require(selector==0&&count==2,NT_CUDA_ARGUMENT,"SUP copy requires two operands");
            cuda_require(p[0]->dtype==p[1]->dtype&&p[0]->tensor.sizes()==p[1]->tensor.sizes(),NT_CUDA_LAYOUT,"SUP copy shape or dtype differs");
            for(size_t j=0;j<2;++j){SUPRole r{p[j]->dtype,p[j]->tensor.sizes().vec(),{},j==1,0};for(int64_t k=0;k<p[j]->tensor.dim();++k)r.tags.push_back(p[j]->tags[k]);roles.push_back(r);}
            std::vector<int> dims;for(int j=0;j<p[1]->tensor.dim();++j)dims.push_back(j);
            std::sort(dims.begin(),dims.end(),[&](int a,int b){return p[1]->tensor.stride(a)<p[1]->tensor.stride(b);});
            int64_t span=1;for(auto j:dims)if(p[1]->tensor.size(j)>1){auto stride=p[1]->tensor.stride(j),size=p[1]->tensor.size(j)-1;cuda_require(stride>=span&&stride<=(INT64_MAX-span)/size,NT_CUDA_LAYOUT,"SUP copy output overlaps itself or overflows");span+=stride*size;}
        } else roles=sup_roles(geometry,op,selector);
        cuda_require(count==roles.size(),NT_CUDA_ARGUMENT,"SUP operand count differs");
        for(size_t j=0;j<count;++j) {
            const auto& r=roles[j];auto v=p[j];d->handle=v->id;
            cuda_require(v->dtype==r.dtype,NT_CUDA_DTYPE,"SUP operand dtype differs");
            cuda_require(v->tensor.sizes().vec()==r.shape,NT_CUDA_LAYOUT,"SUP operand shape differs");
            if(op!=NT_SUP_COPY_STRIDED)storage_contiguous(v,d,"SUP operation requires contiguous operand");
            for(size_t k=0;k<r.tags.size();++k)cuda_require(v->tags[k]==r.tags[k],NT_CUDA_LAYOUT,"SUP operand tags differ");
            cuda_require(bool(v->scale_owner)==bool(r.scale_tag),NT_CUDA_LAYOUT,"SUP scale presence differs");
            if(r.scale_tag)cuda_require(v->scale_tag==r.scale_tag&&v->scale_dtype==NT_DTYPE_F32,NT_CUDA_LAYOUT,"SUP scale tag or type differs");
            if(r.write)for(size_t k=0;k<count;++k)if(j!=k)storage_overlap(v,p[k],d,"SUP writable byte intervals overlap");
        }
        if(op==NT_SUP_QKV_ROTARY)cuda_require(p[0]->scale_owner==p[1]->id&&p[2]->scale_owner==p[3]->id,NT_CUDA_OWNER,"SUP QKV scale owner differs");
        if(op==NT_SUP_NORM2)cuda_require(p[3]->scale_owner==p[4]->id,NT_CUDA_OWNER,"SUP norm2 scale owner differs");
        std::vector<at::Tensor> owners;for(auto v:p)owners.push_back(v->tensor);
        c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream,s->device->ordinal));
        CUDAWork work{s,d,std::move(owners),{}};
        work.run([&] {
            d->stage=NT_CUDA_ENQUEUE;
            auto t=[&](int j)->at::Tensor&{return p[j]->tensor;};
            if(op==NT_SUP_COPY_STRIDED)t(1).copy_(t(0));
            else if(op==NT_SUP_CONV_SILU){const int st[]={1,1,3,2,2},pad[]={2,2,4,4,2};auto z=at::conv1d(t(0),t(1),t(2),{st[selector]},{pad[selector]},{1},1);at::silu_(z);t(3).copy_(z);}
            else if(op==NT_SUP_QUANTIZE_ROWS){auto range=t(0).abs().amax(2);auto qscale=128/range;t(1).copy_((t(0)*qscale.unsqueeze(2)).round().clamp(-127,127).to(at::kChar));t(2).copy_(qscale.to(at::kFloat).reciprocal());}
            else if(op==NT_SUP_SINCOS){t(1).copy_(t(0).sin());t(2).copy_(t(0).cos());}
            else if(op==NT_SUP_LINEAR){if(selector==0){auto product=at::matmul(t(0),t(1).transpose(0,1));t(3).copy_(product+t(2));}else t(2).copy_(at::matmul(t(0),t(1).transpose(0,1)));}
            else {
                auto k=[&](int j,int scale=-1){return sup_koi(p[j],roles[j],scale<0?nullptr:p[scale]);};
                int rc=KOI_SUCCESS;
                if(op==NT_SUP_QKV_ROTARY){auto in=k(0,1),w=k(2,3),sc=k(4),out=k(5);rc=koi_qkv_rotary(s->stream,&in,&w,&sc,&out,nullptr,t(6).data_ptr<int>());}
                else if(op==NT_SUP_MASKED_ATTENTION){auto in=k(0),out=k(1);rc=koi_masked_attention(s->stream,127,128,&in,&out);}
                else if(op==NT_SUP_PROJECTION){auto in=k(0),w=k(1),b=k(2),out=k(3);rc=koi_linear(s->stream,&in,&w,&b,&out,t(4).data_ptr<int>());}
                else if(op==NT_SUP_NORM1){auto in=k(0),res=k(1),w=k(2),out=k(3);rc=koi_rmsnorm_residual(s->stream,&in,&res,alpha,&w,&res,&out,nullptr);}
                else if(op==NT_SUP_SWIGLU){auto in=k(0),w=k(1),out=k(2);rc=koi_mm_swiglu(s->stream,&in,&w,&out,t(3).data_ptr<int>(),0,false);}
                else if(op==NT_SUP_FC2){auto in=k(0),w=k(1),out=k(2);rc=koi_linear(s->stream,&in,&w,nullptr,&out,t(3).data_ptr<int>());}
                else if(op==NT_SUP_NORM2){auto in=k(0),res=k(1),w=k(2),out=k(3,4);rc=koi_rmsnorm_residual(s->stream,&in,&res,alpha,&w,&res,nullptr,&out);}
                d->native_code=rc;cuda_require(rc==KOI_SUCCESS,NT_CUDA_INTERNAL,"SUP Koi operation failed");
            }
            cuda_runtime(cudaGetLastError(),d,"SUP operation launch failed");
        });
#else
        (void)stream_id;(void)op;(void)selector;(void)alpha;
        throw CUDAFailure(NT_CUDA_UNSUPPORTED,"ABI5 backend built without CUDA");
#endif
    });
}
