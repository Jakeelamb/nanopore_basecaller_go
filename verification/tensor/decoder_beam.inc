#ifdef NT_ENABLE_CUDA
#include "decoder/beam_ptx.h"
namespace {
void prepare_decoder_beam(const std::shared_ptr<CUDADeviceRecord>& dev,nt_cuda_detail* d) {
 if(dev->beam){cuda_require(dev->beam->ready,NT_CUDA_RUNTIME,"failed beam program initialization retains resources until device close");return;}
 cuda_require(!dev->pending,NT_CUDA_RUNTIME,"beam module requires drained device");
 cudaDeviceProp properties{};cuda_runtime(cudaGetDeviceProperties(&properties,dev->ordinal),d,"beam device properties failed");
 cuda_require(properties.major==12 && properties.minor==0,NT_CUDA_UNSUPPORTED,"beam PTX requires SM120");
 cuda_runtime(cudaFree(nullptr),d,"beam CUDA context initialization failed");
 dev->beam=std::make_unique<DecoderBeamProgram>();
 CUjit_option option=CU_JIT_TARGET;
 void* value=reinterpret_cast<void*>(static_cast<uintptr_t>(CU_TARGET_COMPUTE_120));
 d->stage=NT_CUDA_ALLOCATE;
 decoder_driver(cuLinkCreate(1,&option,&value,&dev->beam->linker),d,"beam PTX linker creation failed");
 decoder_driver(cuLinkAddData(dev->beam->linker,CU_JIT_INPUT_PTX,const_cast<char*>(decoder_beam_ptx),sizeof(decoder_beam_ptx),"go-decoder-beam-numbers.ptx",0,nullptr,nullptr),d,"beam PTX compilation failed");
 void* cubin=nullptr;size_t size=0;
 decoder_driver(cuLinkComplete(dev->beam->linker,&cubin,&size),d,"beam link failed");
 cuda_require(cubin && size && size<=16*1024*1024,NT_CUDA_INTERNAL,"beam cubin extent invalid");
 auto first=static_cast<unsigned char*>(cubin);dev->beam->cubin.assign(first,first+size);
 decoder_driver(cuModuleLoadData(&dev->beam->module,dev->beam->cubin.data()),d,"beam module load failed");
 decoder_driver(cuModuleGetFunction(&dev->beam->numbers,dev->beam->module,"go_decoder_beam_numbers"),d,"beam numeric symbol absent");
 decoder_driver(cuLinkDestroy(dev->beam->linker),d,"beam linker close failed");dev->beam->linker=nullptr;dev->beam->ready=true;
}
}
#endif

extern "C" {
int nt_cuda_decoder_beam_program(uint64_t e,uint64_t device_id,uint8_t* bytes,size_t cap,size_t* written,
                               nt_cuda_detail* d,char* err,size_t error_cap) {
 return cuda_boundary(d,err,error_cap,[&] {
  cuda_engine(e,d);cuda_require(written,NT_CUDA_ARGUMENT,"beam program size output required");
#ifdef NT_ENABLE_CUDA
  cuda_require(bytes ||cap==0,NT_CUDA_ARGUMENT,"null beam program output requires size zero");
  auto dev=cuda_device(e,device_id,d);
  if(bytes) {
   cuda_require(dev->beam &&dev->beam->ready,NT_CUDA_ARGUMENT,"beam program size query must initialize the module first");
   *written=dev->beam->cubin.size();cuda_require(cap==*written,NT_CUDA_ARGUMENT,"exact beam cubin output extent required");
   std::memcpy(bytes,dev->beam->cubin.data(),cap);return;
  }
  c10::cuda::CUDAGuard guard(dev->ordinal);prepare_decoder_beam(dev,d);*written=dev->beam->cubin.size();
#else
  (void)device_id;(void)bytes;(void)cap;
  throw CUDAFailure(NT_CUDA_UNSUPPORTED,"backend built without CUDA");
#endif
 });
}

int nt_cuda_decoder_beam_numbers(uint64_t e,uint64_t stream_id,uint32_t operation,
                               uint64_t a_id,uint64_t b_id,uint64_t c_id,uint64_t out_id,
                               float clamp,float blank,nt_cuda_detail* d,char* err,size_t cap) {
 return cuda_boundary(d,err,cap,[&] {
  cuda_engine(e,d);
  cuda_require(operation>=1 &&operation<=6,NT_CUDA_ARGUMENT,"checked beam numeric operation required");
  cuda_require(std::isfinite(clamp)&&clamp>=0 &&std::isfinite(blank),NT_CUDA_ARGUMENT,"finite beam scalars required");
  if(operation!=2)cuda_require(clamp==0 &&!std::signbit(clamp) &&blank==0 &&!std::signbit(blank) &&b_id==0 &&c_id==0,
                              NT_CUDA_ARGUMENT,"unused beam operands and scalar bits must be zero");
#ifdef NT_ENABLE_CUDA
  auto s=cuda_stream(e,stream_id,d);
  auto a=cuda_storage(e,a_id,d),out=cuda_storage(e,out_id,d);
  std::shared_ptr<CUDAStorageRecord> b,c;
  std::vector<std::shared_ptr<CUDAStorageRecord>> inputs{a};
  if(operation==2){b=cuda_storage(e,b_id,d);c=cuda_storage(e,c_id,d);inputs.push_back(b);inputs.push_back(c);}
  cuda_require(a->tensor.dim()==2,NT_CUDA_LAYOUT,"beam input requires rank-two rows");
  const auto rows=a->tensor.size(0);
  cuda_require(rows>0 &&rows<=(1<<20),NT_CUDA_LAYOUT,"beam row count exceeds checked workspace");
  const int a_width=operation==1||operation==2?1:2;
  auto check=[&](const std::shared_ptr<CUDAStorageRecord>& p,uint32_t dtype,int width) {
   same_device(s->device,p->device,d);
   cuda_require(p->dtype==dtype,NT_CUDA_DTYPE,"beam operand dtype differs");
   cuda_require(p->tensor.dim()==2 &&p->tensor.size(0)==rows &&p->tensor.size(1)==width &&
                p->tensor.stride(0)==width &&p->tensor.stride(1)==1 &&p->scale_owner==0 &&p->tags[0]=='N' &&p->tags[1]=='C',
                NT_CUDA_LAYOUT,"beam operands require matching contiguous unscaled NC rows");
  };
  check(a,NT_DTYPE_F32,a_width);check(out,NT_DTYPE_F32,operation==2?6:1);
  if(operation==2){check(b,NT_DTYPE_F16,4);check(c,NT_DTYPE_F32,5);}
  for(const auto& p:inputs)storage_overlap(p,out,d,"beam output overlaps input bytes");
  cuda_require(s->device->beam &&s->device->beam->ready &&s->device->beam->numbers,NT_CUDA_ARGUMENT,"beam program must be initialized and authenticated first");
  c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream,s->device->ordinal));
  std::vector<at::Tensor> retained{a->tensor,out->tensor};
  if(b){retained.push_back(b->tensor);retained.push_back(c->tensor);}
  CUDAWork work{s,d,std::move(retained),{}};
  work.run([&] {
   auto ap=a->tensor.data_ptr(),op=out->tensor.data_ptr();
   void* bp=b?b->tensor.data_ptr():nullptr;void* cp=c?c->tensor.data_ptr():nullptr;
   uint32_t count=static_cast<uint32_t>(rows);
   void* args[]={&ap,&bp,&cp,&op,&count,&operation,&clamp,&blank};
   d->stage=NT_CUDA_ENQUEUE;
   decoder_driver(cuLaunchKernel(s->device->beam->numbers,(count+127)/128,1,1,128,1,1,0,reinterpret_cast<CUstream>(s->stream),args,nullptr),d,"beam numeric launch failed");
  });
#else
  (void)stream_id;(void)a_id;(void)out_id;
  throw CUDAFailure(NT_CUDA_UNSUPPORTED,"backend built without CUDA");
#endif
 });
}
}
