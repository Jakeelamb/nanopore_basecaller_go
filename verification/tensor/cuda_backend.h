#pragma once
#include <climits>
#include <cstdio>
#include <cstdlib>
#include "drain_ownership.h"
namespace {
struct CUDAFailure : std::runtime_error {
 int status; CUDAFailure(int s,const char* why):std::runtime_error(why),status(s){}
};
void cuda_require(bool ok,int status,const char* why) {if(!ok)throw CUDAFailure(status,why);}
template<class F> int cuda_boundary(nt_cuda_detail* detail,char* error,size_t cap,F f) noexcept {
 if(!detail || !error || !cap)return NT_CUDA_ARGUMENT;
 *detail={};error[0]=0;
 int status=NT_CUDA_INTERNAL;
 try {std::lock_guard<std::mutex> lock(mutex);c10::InferenceMode guard;f();return NT_CUDA_OK;}
 catch(const CUDAFailure& ex){status=ex.status;auto n=std::min(cap-1,std::strlen(ex.what()));std::memcpy(error,ex.what(),n);error[n]=0;}
 catch(const std::exception& ex){auto n=std::min(cap-1,std::strlen(ex.what()));std::memcpy(error,ex.what(),n);error[n]=0;}
 catch(...){const char* text="unknown native CUDA exception";auto n=std::min(cap-1,std::strlen(text));std::memcpy(error,text,n);error[n]=0;}
 return status;
}
EngineRecord& cuda_engine(uint64_t e,nt_cuda_detail* d) {
 d->actual_engine=e;auto i=engines.find(e);cuda_require(i!=engines.end(),NT_CUDA_CLOSED,"closed engine");return i->second;
}
void cuda_unknown(uint64_t e,uint64_t id,nt_cuda_detail* d) {
 d->handle=id;
 for(auto& pair:engines) {
  auto& r=pair.second;
  auto t=r.tensors.find(id);
  if(t!=r.tensors.end()) {
   d->actual_engine=pair.first;d->actual_dtype=t->second.scalar_type()==at::kFloat?NT_SCALAR_F32:NT_SCALAR_I64;d->expected_dtype=NT_SCALAR_F16;
   cuda_require(pair.first==e,NT_CUDA_OWNER,"foreign engine handle");throw CUDAFailure(NT_CUDA_DTYPE,"CPU tensor is not a CUDA half handle");
  }
  if(r.devices.count(id)||r.streams.count(id)||r.halves.count(id)) {
   d->actual_engine=pair.first;cuda_require(pair.first==e,NT_CUDA_OWNER,"foreign engine handle");throw CUDAFailure(NT_CUDA_ARGUMENT,"wrong handle kind");
  }
 }
 throw CUDAFailure(NT_CUDA_CLOSED,"unknown or released handle");
}
size_t half_elements(const int64_t* shape,size_t rank) {
 cuda_require(shape && rank && rank<=NT_CUDA_MAX_RANK,NT_CUDA_LAYOUT,"half rank must be 1 through 8");size_t count=1;
 for(size_t i=0;i<rank;++i){cuda_require(shape[i]>0,NT_CUDA_LAYOUT,"nonpositive half dimension");auto n=uint64_t(shape[i]);cuda_require(n<=uint64_t(INT64_MAX)/2/count,NT_CUDA_LAYOUT,"half byte count overflow");count*=n;}return count;
}
uint64_t half_end(const int64_t* shape,const int64_t* strides,size_t rank,uint64_t offset) {
 half_elements(shape,rank);cuda_require(strides,NT_CUDA_LAYOUT,"null strides");cuda_require(offset<uint64_t(INT64_MAX)/2,NT_CUDA_LAYOUT,"half offset overflow");uint64_t end=offset;
 for(size_t i=0;i<rank;++i){cuda_require(strides[i]>=0,NT_CUDA_LAYOUT,"negative stride");auto dim=uint64_t(shape[i]-1),stride=uint64_t(strides[i]);cuda_require(!dim||stride<=(uint64_t(INT64_MAX)/2-end)/dim,NT_CUDA_LAYOUT,"half span overflow");end+=dim*stride;}
 cuda_require(end<uint64_t(INT64_MAX)/2,NT_CUDA_LAYOUT,"half span overflow");return end+1;
}
#ifdef NT_ENABLE_CUDA
std::shared_ptr<CUDADeviceRecord> cuda_device(uint64_t e,uint64_t id,nt_cuda_detail* d,bool healthy=true) {
 auto& r=cuda_engine(e,d);auto i=r.devices.find(id);if(i==r.devices.end())cuda_unknown(e,id,d);
 auto p=i->second;d->actual_device=p->id;d->actual_ordinal=p->ordinal;
 if(healthy)cuda_require(!p->poisoned,NT_CUDA_RUNTIME,"device owner poisoned by previous execution failure");return p;
}
std::shared_ptr<CUDAStreamRecord> cuda_stream(uint64_t e,uint64_t id,nt_cuda_detail* d,bool healthy=true) {
 auto& r=cuda_engine(e,d);auto i=r.streams.find(id);if(i==r.streams.end())cuda_unknown(e,id,d);
 auto p=i->second;d->handle=id;d->actual_device=p->device->id;d->actual_ordinal=p->device->ordinal;
 if(healthy)cuda_require(!p->device->poisoned,NT_CUDA_STREAM,"stream device is poisoned");return p;
}
std::shared_ptr<CUDAHalfRecord> cuda_half(uint64_t e,uint64_t id,nt_cuda_detail* d) {
 auto& r=cuda_engine(e,d);auto i=r.halves.find(id);if(i==r.halves.end())cuda_unknown(e,id,d);
 auto p=i->second;d->handle=id;d->actual_device=p->device->id;d->actual_ordinal=p->device->ordinal;
 d->actual_dtype=NT_SCALAR_F16;d->expected_dtype=NT_SCALAR_F16;
 cuda_require(p->tensor.scalar_type()==at::kHalf,NT_CUDA_DTYPE,"half record dtype mismatch");return p;
}
void same_device(const std::shared_ptr<CUDADeviceRecord>& expected,const std::shared_ptr<CUDADeviceRecord>& actual,nt_cuda_detail* d) {
 d->expected_device=expected->id;d->expected_ordinal=expected->ordinal;d->actual_device=actual->id;d->actual_ordinal=actual->ordinal;
 cuda_require(expected==actual,NT_CUDA_DEVICE,"different logical device owners");
}
#ifndef NT_STREAM_SYNCHRONIZE
#define NT_STREAM_SYNCHRONIZE cudaStreamSynchronize
#endif
void cuda_drain(const std::shared_ptr<CUDAStreamRecord>& s,nt_cuda_detail* d) {
 d->stage=NT_CUDA_DRAIN;auto status=NT_STREAM_SYNCHRONIZE(s->stream);d->native_code=status;
 if(status!=cudaSuccess){s->device->poisoned=true;throw CUDAFailure(NT_CUDA_RUNTIME,"stream drain failed; owners retained");}
}
void close_device_cuda(uint64_t e,const std::shared_ptr<CUDADeviceRecord>& dev,nt_cuda_detail* d) {
 c10::cuda::CUDAGuard guard(dev->ordinal);auto& r=engines.at(e);
 release_after_drain([&]{for(auto& item:r.streams)if(item.second->device==dev && item.second->stream)cuda_drain(item.second,d);},[&]{
 for(auto& item:r.streams)if(item.second->device==dev && item.second->stream){d->stage=NT_CUDA_DESTROY;auto status=cudaStreamDestroy(item.second->stream);d->native_code=status;cuda_require(status==cudaSuccess,NT_CUDA_RUNTIME,"stream destroy failed; owners retained");item.second->stream=nullptr;}
 for(auto i=r.halves.begin();i!=r.halves.end();)if(i->second->device==dev)i=r.halves.erase(i);else ++i;
 for(auto i=r.streams.begin();i!=r.streams.end();)if(i->second->device==dev)i=r.streams.erase(i);else ++i;
 dev->pending.reset();r.devices.erase(dev->id);});
}
struct CUDAWork {
 std::shared_ptr<CUDAStreamRecord> stream;nt_cuda_detail* detail;
 std::shared_ptr<CUDAWorkResources> owners;
 CUDAWork(std::shared_ptr<CUDAStreamRecord> s,nt_cuda_detail* d,std::vector<at::Tensor> tensors,std::vector<std::shared_ptr<std::vector<uint16_t>>> staging):stream(std::move(s)),detail(d),owners(std::make_shared<CUDAWorkResources>()) {
  cuda_require(!stream->device->pending,NT_CUDA_RUNTIME,"previous undrained operation retains device resources");
  owners->tensors=std::move(tensors);owners->tensors.reserve(owners->tensors.size()+4);owners->staging=std::move(staging);
  stream->device->pending=owners;
 }
 template<class F> void run(F enqueue) {
  nt_cuda_detail failure{};bool undrained=false;
  try {
   retain_until_drained(enqueue,[&]{cuda_drain(stream,detail);},[&]{stream->device->poisoned=true;failure=*detail;},[&]{undrained=true;});
  }catch(...){if(!undrained)stream->device->pending.reset();*detail=failure;throw;}
  stream->device->pending.reset();
 }
};
void cuda_runtime(cudaError_t status,nt_cuda_detail* d,const char* why) {d->native_code=status;cuda_require(status==cudaSuccess,NT_CUDA_RUNTIME,why);}
void close_engine_cuda(uint64_t e) {
 nt_cuda_detail detail{};auto& r=engines.at(e);
 for(auto& item:r.streams){c10::cuda::CUDAGuard guard(item.second->device->ordinal);if(item.second->stream)cuda_drain(item.second,&detail);}
 while(!r.devices.empty())close_device_cuda(e,r.devices.begin()->second,&detail);
}
#else
void close_engine_cuda(uint64_t) {}
#endif
}
extern "C" {
int nt_cuda_device_open(uint64_t e,int32_t ordinal,uint64_t* out,nt_cuda_detail* d,char* err,size_t cap) {
 return cuda_boundary(d,err,cap,[&]{auto& r=cuda_engine(e,d);cuda_require(out,NT_CUDA_ARGUMENT,"null output");d->actual_ordinal=ordinal;cuda_require(ordinal>=0,NT_CUDA_DEVICE,"negative CUDA ordinal");
#ifdef NT_ENABLE_CUDA
 int count=0;cuda_runtime(cudaGetDeviceCount(&count),d,"CUDA device enumeration failed");cuda_require(ordinal<count,NT_CUDA_DEVICE,"unavailable CUDA ordinal");
 if(std::getenv("NANOPORE_TENSOR_CUDA_DIAGNOSTICS")){
  int driver=0,runtime=0;cudaDeviceProp props{};cuda_runtime(cudaDriverGetVersion(&driver),d,"CUDA driver identity failed");cuda_runtime(cudaRuntimeGetVersion(&runtime),d,"CUDA runtime identity failed");cuda_runtime(cudaGetDeviceProperties(&props,ordinal),d,"CUDA device identity failed");
  char uuid[33];const char* digits="0123456789abcdef";for(size_t i=0;i<16;i++){auto b=static_cast<unsigned char>(props.uuid.bytes[i]);uuid[i*2]=digits[b>>4];uuid[i*2+1]=digits[b&15];}uuid[32]=0;
  std::fprintf(stderr,"nt_cuda_device driver=%d runtime=%d ordinal=%d uuid=%s\n",driver,runtime,ordinal,uuid);
 }
 auto dev=std::make_shared<CUDADeviceRecord>();dev->engine=e;dev->id=fresh();dev->ordinal=ordinal;r.devices.emplace(dev->id,dev);*out=dev->id;
#else
 (void)r;throw CUDAFailure(NT_CUDA_UNSUPPORTED,"ABI4 backend built without CUDA");
#endif
 });
}
int nt_cuda_device_close(uint64_t e,uint64_t id,nt_cuda_detail* d,char* err,size_t cap) {
 return cuda_boundary(d,err,cap,[&]{cuda_engine(e,d);
#ifdef NT_ENABLE_CUDA
 auto dev=cuda_device(e,id,d,false);close_device_cuda(e,dev,d);
#else
 cuda_unknown(e,id,d);
#endif
 });
}
int nt_cuda_stream_open(uint64_t e,uint64_t id,uint64_t* out,nt_cuda_detail* d,char* err,size_t cap) {
 return cuda_boundary(d,err,cap,[&]{cuda_engine(e,d);cuda_require(out,NT_CUDA_ARGUMENT,"null output");
#ifdef NT_ENABLE_CUDA
 auto dev=cuda_device(e,id,d);c10::cuda::CUDAGuard guard(dev->ordinal);auto stream=std::make_shared<CUDAStreamRecord>();stream->device=dev;stream->id=fresh();d->stage=NT_CUDA_ALLOCATE;cuda_runtime(cudaStreamCreateWithFlags(&stream->stream,cudaStreamNonBlocking),d,"stream creation failed");engines.at(e).streams.emplace(stream->id,stream);*out=stream->id;
#else
 cuda_unknown(e,id,d);
#endif
 });
}
int nt_cuda_stream_close(uint64_t e,uint64_t id,nt_cuda_detail* d,char* err,size_t cap) {
 return cuda_boundary(d,err,cap,[&]{cuda_engine(e,d);
#ifdef NT_ENABLE_CUDA
 auto s=cuda_stream(e,id,d,false);c10::cuda::CUDAGuard guard(s->device->ordinal);cuda_drain(s,d);d->stage=NT_CUDA_DESTROY;cuda_runtime(cudaStreamDestroy(s->stream),d,"stream destroy failed");s->stream=nullptr;engines.at(e).streams.erase(id);
#else
 cuda_unknown(e,id,d);
#endif
 });
}
int nt_cuda_half_new(uint64_t e,uint64_t dev_id,uint64_t stream_id,const int64_t* shape,size_t rank,const uint16_t* bits,size_t count,uint64_t* out,nt_cuda_detail* d,char* err,size_t cap) {
 return cuda_boundary(d,err,cap,[&]{cuda_engine(e,d);cuda_require(out&&bits,NT_CUDA_ARGUMENT,"null input or output");cuda_require(half_elements(shape,rank)==count,NT_CUDA_LAYOUT,"upload count differs from shape");
#ifdef NT_ENABLE_CUDA
 auto dev=cuda_device(e,dev_id,d);auto s=cuda_stream(e,stream_id,d);same_device(dev,s->device,d);c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream,dev->ordinal));
 auto p=std::make_shared<CUDAHalfRecord>();p->device=dev;p->id=fresh();p->allocation=p->id;p->storage=count;p->end=count;
 d->stage=NT_CUDA_ALLOCATE;p->tensor=at::empty(at::IntArrayRef(shape,rank),at::TensorOptions().dtype(at::kHalf).device(at::Device(at::kCUDA,dev->ordinal)));
 auto host=std::make_shared<std::vector<uint16_t>>(bits,bits+count);CUDAWork work{s,d,{p->tensor},{host}};
 work.run([&]{d->stage=NT_CUDA_COPY;cuda_runtime(cudaMemcpyAsync(p->tensor.data_ptr(),host->data(),count*2,cudaMemcpyHostToDevice,s->stream),d,"half upload failed");});engines.at(e).halves.emplace(p->id,p);*out=p->id;
#else
 (void)dev_id;(void)stream_id;throw CUDAFailure(NT_CUDA_UNSUPPORTED,"ABI4 backend built without CUDA");
#endif
 });
}
int nt_cuda_half_view(uint64_t e,uint64_t id,uint64_t offset,const int64_t* shape,const int64_t* strides,size_t rank,uint64_t* out,nt_cuda_detail* d,char* err,size_t cap) {
 return cuda_boundary(d,err,cap,[&]{cuda_engine(e,d);cuda_require(out,NT_CUDA_ARGUMENT,"null output");half_elements(shape,rank);
#ifdef NT_ENABLE_CUDA
 auto parent=cuda_half(e,id,d);cuda_require(offset<=parent->end-parent->offset,NT_CUDA_LAYOUT,"view offset exceeds parent interval");auto end=half_end(shape,strides,rank,parent->offset+offset);cuda_require(end<=parent->end,NT_CUDA_LAYOUT,"view exceeds parent interval");
 auto p=std::make_shared<CUDAHalfRecord>(*parent);p->id=fresh();p->offset=parent->offset+offset;p->end=end;p->tensor=parent->tensor.as_strided(at::IntArrayRef(shape,rank),at::IntArrayRef(strides,rank),int64_t(p->offset));engines.at(e).halves.emplace(p->id,p);*out=p->id;
#else
 (void)offset;(void)strides;cuda_unknown(e,id,d);
#endif
 });
}
int nt_cuda_half_meta(uint64_t e,uint64_t id,nt_half_meta* out,nt_cuda_detail* d,char* err,size_t cap) {
 return cuda_boundary(d,err,cap,[&]{cuda_engine(e,d);cuda_require(out,NT_CUDA_ARGUMENT,"null metadata output");
#ifdef NT_ENABLE_CUDA
 auto p=cuda_half(e,id,d);nt_half_meta m{};m.rank=p->tensor.dim();m.dtype=NT_SCALAR_F16;m.device=p->device->id;m.ordinal=p->device->ordinal;m.offset_elements=p->offset;m.storage_elements=p->storage;for(size_t i=0;i<m.rank;++i){m.shape[i]=p->tensor.size(i);m.strides[i]=p->tensor.stride(i);}*out=m;
#else
 cuda_unknown(e,id,d);
#endif
 });
}
int nt_cuda_half_bits(uint64_t e,uint64_t stream_id,uint64_t id,uint16_t* bits,size_t count,nt_cuda_detail* d,char* err,size_t cap) {
 return cuda_boundary(d,err,cap,[&]{cuda_engine(e,d);cuda_require(bits,NT_CUDA_ARGUMENT,"null download output");
#ifdef NT_ENABLE_CUDA
 auto p=cuda_half(e,id,d);auto s=cuda_stream(e,stream_id,d);same_device(s->device,p->device,d);cuda_require(uint64_t(p->tensor.numel())==count,NT_CUDA_LAYOUT,"download count differs from shape");c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream,p->device->ordinal));
 auto host=std::make_shared<std::vector<uint16_t>>(count);CUDAWork work{s,d,{p->tensor},{host}};
 work.run([&]{auto packed=p->tensor.contiguous();work.owners->tensors.push_back(packed);d->stage=NT_CUDA_COPY;cuda_runtime(cudaMemcpyAsync(host->data(),packed.data_ptr(),count*2,cudaMemcpyDeviceToHost,s->stream),d,"half download failed");});std::memcpy(bits,host->data(),count*2);
#else
 (void)stream_id;(void)count;cuda_unknown(e,id,d);
#endif
 });
}
int nt_cuda_half_release(uint64_t e,uint64_t id,nt_cuda_detail* d,char* err,size_t cap) {
 return cuda_boundary(d,err,cap,[&]{cuda_engine(e,d);
#ifdef NT_ENABLE_CUDA
 auto p=cuda_half(e,id,d);cuda_require(!p->device->poisoned,NT_CUDA_RUNTIME,"poisoned device retains half storage until successful close drain");engines.at(e).halves.erase(id);
#else
 cuda_unknown(e,id,d);
#endif
 });
}
int nt_cuda_gemm_f16(uint64_t e,uint64_t stream_id,uint64_t a_id,uint64_t b_id,uint64_t out_id,nt_cuda_detail* d,char* err,size_t cap) {
 return cuda_boundary(d,err,cap,[&]{cuda_engine(e,d);
#ifdef NT_ENABLE_CUDA
 auto s=cuda_stream(e,stream_id,d);auto a=cuda_half(e,a_id,d),b=cuda_half(e,b_id,d),out=cuda_half(e,out_id,d);for(auto p:{a,b,out}){d->handle=p->id;same_device(s->device,p->device,d);auto& t=p->tensor;cuda_require(t.dim()==2 && t.stride(1)==1 && t.stride(0)>=t.size(1) && t.stride(0)<=INT_MAX && t.size(0)<=INT_MAX && t.size(1)<=INT_MAX,NT_CUDA_LAYOUT,"GEMM requires rank two, unit final stride and int32 row-leading dimensions");}
 auto M=a->tensor.size(0),K=a->tensor.size(1),N=b->tensor.size(1);cuda_require(b->tensor.size(0)==K && out->tensor.size(0)==M && out->tensor.size(1)==N,NT_CUDA_LAYOUT,"GEMM dimensions differ");
 for(auto p:{a,b}){d->handle=p->id;cuda_require(p->allocation!=out->allocation || p->end<=out->offset || out->end<=p->offset,NT_CUDA_LAYOUT,"GEMM output overlaps an input interval");}
 c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream,s->device->ordinal));CUDAWork work{s,d,{a->tensor,b->tensor,out->tensor},{}};
 work.run([&]{d->stage=NT_CUDA_ENQUEUE;auto handle=at::cuda::getCurrentCUDABlasHandle();
 if(std::getenv("NANOPORE_TENSOR_CUDA_DIAGNOSTICS")){
  cudaStream_t actual=nullptr;cublasPointerMode_t pointer;cublasMath_t math;cublasAtomicsMode_t atomics;
  for(auto status:{cublasGetStream(handle,&actual),cublasGetPointerMode(handle,&pointer),cublasGetMathMode(handle,&math),cublasGetAtomicsMode(handle,&atomics)}){d->native_code=status;cuda_require(status==CUBLAS_STATUS_SUCCESS,NT_CUDA_CUBLAS,"cuBLAS diagnostics failed");}
  int driver=0,runtime=0,version=0;cudaDeviceProp props{};cuda_runtime(cudaDriverGetVersion(&driver),d,"CUDA driver identity failed");cuda_runtime(cudaRuntimeGetVersion(&runtime),d,"CUDA runtime identity failed");cuda_runtime(cudaGetDeviceProperties(&props,s->device->ordinal),d,"CUDA device identity failed");auto status=cublasGetVersion(handle,&version);d->native_code=status;cuda_require(status==CUBLAS_STATUS_SUCCESS,NT_CUDA_CUBLAS,"cuBLAS version identity failed");
  char uuid[33];const char* digits="0123456789abcdef";for(size_t i=0;i<16;i++){auto value=static_cast<unsigned char>(props.uuid.bytes[i]);uuid[i*2]=digits[value>>4];uuid[i*2+1]=digits[value&15];}uuid[32]=0;
  std::fprintf(stderr,"nt_cuda_gemm_f16 driver=%d runtime=%d cublas=%d ordinal=%d uuid=%s stream_matches=%d pointer_mode=%d math_mode=%d atomics_mode=%d\n",driver,runtime,version,s->device->ordinal,uuid,actual==s->stream,int(pointer),int(math),int(atomics));
 }
 const uint16_t zero=0,one=0x3c00;auto status=cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&one,b->tensor.data_ptr(),CUDA_R_16F,b->tensor.stride(0),a->tensor.data_ptr(),CUDA_R_16F,a->tensor.stride(0),&zero,out->tensor.data_ptr(),CUDA_R_16F,out->tensor.stride(0),CUDA_R_16F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);d->native_code=status;cuda_require(status==CUBLAS_STATUS_SUCCESS,NT_CUDA_CUBLAS,"source half GEMM enqueue failed");cuda_runtime(cudaGetLastError(),d,"source half GEMM launch failed");});
#else
 (void)stream_id;(void)b_id;(void)out_id;cuda_unknown(e,a_id,d);
#endif
 });
}
}
