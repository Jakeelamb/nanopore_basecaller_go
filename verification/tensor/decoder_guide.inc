#ifdef NT_ENABLE_CUDA
#include "decoder/guide_ptx.h"
#include "decoder/forward_ptx.h"
#include "decoder/posterior_ptx.h"
#include "decoder/quality_ptx.h"
namespace {
void decoder_driver(CUresult rc,nt_cuda_detail* d,const char* why) {
 d->native_code=rc;cuda_require(rc==CUDA_SUCCESS,NT_CUDA_RUNTIME,why);
}
void prepare_decoder_guide(const std::shared_ptr<CUDADeviceRecord>& dev,nt_cuda_detail* d) {
 if(dev->guide){cuda_require(dev->guide->ready,NT_CUDA_RUNTIME,"failed decoder program initialization retains resources until device close");return;}
 cuda_require(!dev->pending,NT_CUDA_RUNTIME,"decoder module requires drained device");
 cudaDeviceProp properties{};cuda_runtime(cudaGetDeviceProperties(&properties,dev->ordinal),d,"decoder device properties failed");
 cuda_require(properties.major==12 && properties.minor==0,NT_CUDA_UNSUPPORTED,"decoder guide PTX is qualified only for SM120");
 cuda_runtime(cudaFree(nullptr),d,"decoder CUDA context initialization failed");
 dev->guide=std::make_unique<DecoderGuideProgram>();
 CUjit_option option=CU_JIT_TARGET;
 void* value=reinterpret_cast<void*>(static_cast<uintptr_t>(CU_TARGET_COMPUTE_120));
 d->stage=NT_CUDA_ALLOCATE;
 decoder_driver(cuLinkCreate(1,&option,&value,&dev->guide->linker),d,"decoder PTX linker creation failed");
 decoder_driver(cuLinkAddData(dev->guide->linker,CU_JIT_INPUT_PTX,const_cast<char*>(decoder_guide_ptx),sizeof(decoder_guide_ptx),"go-decoder-guide-frame.ptx",0,nullptr,nullptr),d,"decoder guide PTX compilation failed");
 decoder_driver(cuLinkAddData(dev->guide->linker,CU_JIT_INPUT_PTX,const_cast<char*>(decoder_forward_ptx),sizeof(decoder_forward_ptx),"go-decoder-forward-frame.ptx",0,nullptr,nullptr),d,"decoder forward PTX compilation failed");
 decoder_driver(cuLinkAddData(dev->guide->linker,CU_JIT_INPUT_PTX,const_cast<char*>(decoder_posterior_ptx),sizeof(decoder_posterior_ptx),"go-decoder-posterior-row.ptx",0,nullptr,nullptr),d,"decoder posterior PTX compilation failed");
 decoder_driver(cuLinkAddData(dev->guide->linker,CU_JIT_INPUT_PTX,const_cast<char*>(decoder_quality_ptx),sizeof(decoder_quality_ptx),"go-decoder-quality-numbers.ptx",0,nullptr,nullptr),d,"decoder quality PTX compilation failed");
 void* cubin=nullptr;size_t size=0;
 decoder_driver(cuLinkComplete(dev->guide->linker,&cubin,&size),d,"decoder guide link failed");
 cuda_require(cubin && size && size<=16*1024*1024,NT_CUDA_INTERNAL,"decoder guide cubin extent invalid");
 auto first=static_cast<unsigned char*>(cubin);dev->guide->cubin.assign(first,first+size);
 decoder_driver(cuModuleLoadData(&dev->guide->module,dev->guide->cubin.data()),d,"decoder guide module load failed");
 decoder_driver(cuModuleGetFunction(&dev->guide->frame,dev->guide->module,"go_decoder_guide_frame"),d,"decoder guide frame symbol absent");
 decoder_driver(cuModuleGetFunction(&dev->guide->forward,dev->guide->module,"go_decoder_forward_frame"),d,"decoder forward frame symbol absent");
 decoder_driver(cuModuleGetFunction(&dev->guide->posterior,dev->guide->module,"go_decoder_posterior_row"),d,"decoder posterior row symbol absent");
 decoder_driver(cuModuleGetFunction(&dev->guide->quality,dev->guide->module,"go_decoder_quality_numbers"),d,"decoder quality numeric symbol absent");
 decoder_driver(cuLinkDestroy(dev->guide->linker),d,"decoder guide linker close failed");dev->guide->linker=nullptr;dev->guide->ready=true;
}
}
#endif

extern "C" {
int nt_cuda_decoder_guide_program(uint64_t e,uint64_t device_id,uint8_t* bytes,size_t cap,size_t* written,
                                nt_cuda_detail* d,char* err,size_t error_cap) {
 return cuda_boundary(d,err,error_cap,[&] {
  cuda_engine(e,d);cuda_require(written,NT_CUDA_ARGUMENT,"decoder program size output required");
#ifdef NT_ENABLE_CUDA
  cuda_require(bytes ||cap==0,NT_CUDA_ARGUMENT,"null decoder program output requires size zero");
  auto dev=cuda_device(e,device_id,d);
  if(bytes) {
   cuda_require(dev->guide &&dev->guide->ready,NT_CUDA_ARGUMENT,"decoder program size query must initialize the module first");
   *written=dev->guide->cubin.size();
   cuda_require(cap==*written,NT_CUDA_ARGUMENT,"exact decoder cubin output extent required");
   std::memcpy(bytes,dev->guide->cubin.data(),cap);return;
  }
  c10::cuda::CUDAGuard guard(dev->ordinal);prepare_decoder_guide(dev,d);*written=dev->guide->cubin.size();
#else
  (void)device_id;(void)bytes;(void)cap;
  throw CUDAFailure(NT_CUDA_UNSUPPORTED,"backend built without CUDA");
#endif
 });
}
int nt_cuda_decoder_guide_frame(uint64_t e,uint64_t stream_id,uint64_t score_id,uint64_t next_id,uint64_t out_id,
                              float clamp,float blank,nt_cuda_detail* d,char* err,size_t cap) {
 return cuda_boundary(d,err,cap,[&] {
  cuda_engine(e,d);
  cuda_require(std::isfinite(clamp)&&clamp>=0 &&std::isfinite(blank),NT_CUDA_ARGUMENT,"finite decoder clamp and blank required");
#ifdef NT_ENABLE_CUDA
  auto s=cuda_stream(e,stream_id,d);
  auto scores=cuda_storage(e,score_id,d),next=cuda_storage(e,next_id,d),out=cuda_storage(e,out_id,d);
  for(auto p:{scores,next,out}) {
   same_device(s->device,p->device,d);
   cuda_require(p->tensor.dim()==2 &&p->tensor.stride(1)==1 &&p->tensor.stride(0)>=p->tensor.size(1) &&p->scale_owner==0,
                NT_CUDA_LAYOUT,"decoder guide requires rank-two row-major unscaled operands");
  }
  cuda_require(scores->dtype==NT_DTYPE_F16 &&next->dtype==NT_DTYPE_F32 &&out->dtype==NT_DTYPE_F32,
               NT_CUDA_DTYPE,"decoder guide requires half scores and float guides");
  for(auto p:{scores,next,out})cuda_require(p->tags[0]=='N' &&p->tags[1]=='C',NT_CUDA_LAYOUT,"decoder guide operand NC tags required");
  const auto n=scores->tensor.size(0),states=next->tensor.size(1);
  cuda_require((states==256 &&n>0 &&n%64==0)||(states==1024 &&n>0 &&n%32==0),NT_CUDA_LAYOUT,"decoder guide requires fixed HAC/SUP batch tiles");
  cuda_require(n<=INT_MAX &&scores->tensor.size(1)==4*states &&next->tensor.size(0)==n &&out->tensor.sizes()==next->tensor.sizes(),
               NT_CUDA_LAYOUT,"decoder guide dimensions differ");
  storage_contiguous(next,d,"decoder next guide rows must be contiguous");
  storage_contiguous(out,d,"decoder output guide rows must be contiguous");
  storage_overlap(scores,out,d,"decoder guide output overlaps scores");
  storage_overlap(next,out,d,"decoder guide rows overlap");
  cuda_require(s->device->guide &&s->device->guide->ready &&s->device->guide->frame,NT_CUDA_ARGUMENT,"decoder guide program must be initialized and retained first");
  c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream,s->device->ordinal));
  CUDAWork work{s,d,{scores->tensor,next->tensor,out->tensor},{}};
  work.run([&] {
   auto score_ptr=scores->tensor.data_ptr(),next_ptr=next->tensor.data_ptr(),out_ptr=out->tensor.data_ptr();
   uint64_t stride=scores->tensor.stride(0);uint32_t count=states;
   void* args[]={&score_ptr,&next_ptr,&out_ptr,&stride,&count,&clamp,&blank};
   d->stage=NT_CUDA_ENQUEUE;
   decoder_driver(cuLaunchKernel(s->device->guide->frame,unsigned(n),1,1,unsigned(states),1,1,0,reinterpret_cast<CUstream>(s->stream),args,nullptr),d,"decoder guide frame launch failed");
  });
#else
  (void)stream_id;(void)score_id;(void)next_id;(void)out_id;(void)clamp;(void)blank;
  throw CUDAFailure(NT_CUDA_UNSUPPORTED,"backend built without CUDA");
#endif
 });
}
}
