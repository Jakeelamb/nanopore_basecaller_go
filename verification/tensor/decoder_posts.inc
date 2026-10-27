// Go selects all scan frames, numeric rows, groups, output slots and counters.
extern "C" int nt_cuda_decoder_post_numbers(uint64_t e,uint64_t stream_id,uint32_t op,
    uint64_t a_id,uint64_t b_id,uint64_t out_id,uint64_t aux_id,float first,float second,
    nt_cuda_detail* d,char* err,size_t cap) {
 return cuda_boundary(d,err,cap,[&] {
  cuda_engine(e,d);
  cuda_require(op>=1&&op<=5,NT_CUDA_ARGUMENT,"unknown decoder numeric operation");
  cuda_require(std::isfinite(first)&&std::isfinite(second),NT_CUDA_ARGUMENT,"finite decoder numeric scalars required");
  if(op==1)cuda_require(first>=0,NT_CUDA_ARGUMENT,"nonnegative forward clamp required");
  else if(op!=4)cuda_require(first==0&&!std::signbit(first)&&second==0&&!std::signbit(second),NT_CUDA_ARGUMENT,"unused decoder scalar bits must be positive zero");
  cuda_require(op>=4||aux_id==0,NT_CUDA_ARGUMENT,"unused numeric auxiliary output must be zero");
#ifdef NT_ENABLE_CUDA
  auto s=cuda_stream(e,stream_id,d);
  auto a=cuda_storage(e,a_id,d),b=cuda_storage(e,b_id,d),out=cuda_storage(e,out_id,d);
  std::shared_ptr<CUDAStorageRecord> aux;
  std::vector<std::shared_ptr<CUDAStorageRecord>> operands{a,b,out};
  if(op>=4){aux=cuda_storage(e,aux_id,d);operands.push_back(aux);}
  for(const auto& p:operands) {
   same_device(s->device,p->device,d);
   cuda_require(p->tensor.dim()==2&&p->tensor.stride(1)==1&&p->tensor.stride(0)>=p->tensor.size(1)&&p->scale_owner==0,
                NT_CUDA_LAYOUT,"decoder numeric operands require unscaled row-major matrices");
   cuda_require(p->tags[0]=='N'&&p->tags[1]=='C',NT_CUDA_LAYOUT,"decoder numeric NC tags required");
  }
  const auto n=a->tensor.size(0);
  cuda_require(n>0&&n<=INT_MAX&&b->tensor.size(0)==n&&out->tensor.size(0)==n,NT_CUDA_LAYOUT,"decoder numeric row counts differ");
  CUfunction fn=nullptr;
  cuda_require(s->device->guide&&s->device->guide->ready,NT_CUDA_ARGUMENT,"complete decoder program must be initialized and retained first");
  if(op<=2) {
   const auto states=b->tensor.size(1);
   cuda_require((states==256&&n%64==0)||(states==1024&&n%32==0),NT_CUDA_LAYOUT,"posterior requires fixed HAC/SUP batch tiles");
   cuda_require(b->dtype==NT_DTYPE_F32&&out->tensor.sizes()==b->tensor.sizes(),NT_CUDA_DTYPE,"posterior float rows or output shape differ");
   storage_contiguous(b,d,"posterior float rows must be contiguous");
   if(op==1) {
    cuda_require(a->dtype==NT_DTYPE_F16&&out->dtype==NT_DTYPE_F32,NT_CUDA_DTYPE,"forward alpha requires half scores and float output");
    cuda_require(a->tensor.size(1)==4*states,NT_CUDA_LAYOUT,"forward alpha score width differs");
    storage_contiguous(out,d,"forward alpha output must be contiguous");fn=s->device->guide->forward;
   } else {
    cuda_require(a->dtype==NT_DTYPE_F32&&out->dtype==NT_DTYPE_F16,NT_CUDA_DTYPE,"posterior normalization requires float input and half output");
    cuda_require(a->tensor.sizes()==b->tensor.sizes(),NT_CUDA_LAYOUT,"posterior alpha/beta dimensions differ");
    storage_contiguous(a,d,"posterior alpha must be contiguous");fn=s->device->guide->posterior;
   }
  } else {
   const auto width=a->tensor.size(1);
   cuda_require(n<=(1<<20)&&width>0&&width<=(1<<20)&&n<=(1<<27)/width,NT_CUDA_LAYOUT,"quality numeric rows exceed workspace");
   for(const auto& p:operands)storage_contiguous(p,d,"quality numeric operands must be contiguous");
   cuda_require(a->dtype==(op==3?NT_DTYPE_F16:NT_DTYPE_F32)&&b->dtype==NT_DTYPE_I32&&out->dtype==NT_DTYPE_F32,NT_CUDA_DTYPE,"quality numeric operand dtype differs");
   cuda_require(b->tensor.size(1)==(op==5?2:1)&&out->tensor.size(1)==(op==5?3:1)&&(op!=3||width==9),NT_CUDA_LAYOUT,"quality numeric widths differ");
   if(aux)cuda_require(aux->dtype==NT_DTYPE_U8&&aux->tensor.size(0)==n&&aux->tensor.size(1)==1,NT_CUDA_LAYOUT,"quality auxiliary output must be U8 NC rows");
   fn=s->device->guide->quality;
  }
  storage_overlap(a,out,d,"decoder numeric output overlaps first input");
  storage_overlap(b,out,d,"decoder numeric output overlaps second input");
  if(aux)for(const auto& p:{a,b,out})storage_overlap(p,aux,d,"quality auxiliary output overlaps another operand");
  cuda_require(fn,NT_CUDA_ARGUMENT,"decoder numeric function absent");
  c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(s->stream,s->device->ordinal));
  std::vector<at::Tensor> retained{a->tensor,b->tensor,out->tensor};if(aux)retained.push_back(aux->tensor);
  CUDAWork work{s,d,std::move(retained),{}};
  work.run([&] {
   auto ap=a->tensor.data_ptr(),bp=b->tensor.data_ptr(),outp=out->tensor.data_ptr();void* auxiliary=aux?aux->tensor.data_ptr():nullptr;
   uint64_t stride=op==1?a->tensor.stride(0):out->tensor.stride(0);uint32_t states=b->tensor.size(1),rows=n,width=a->tensor.size(1);
   void* forward_args[]={&ap,&bp,&outp,&stride,&states,&first,&second};
   void* posterior_args[]={&ap,&bp,&outp,&stride,&states};
   void* quality_args[]={&ap,&bp,&outp,&auxiliary,&rows,&width,&op,&first,&second};
   auto args=op==1?forward_args:op==2?posterior_args:quality_args;
   d->stage=NT_CUDA_ENQUEUE;
   decoder_driver(cuLaunchKernel(fn,op<=2?unsigned(n):(rows+127)/128,1,1,op<=2?states:128,1,1,0,reinterpret_cast<CUstream>(s->stream),args,nullptr),d,"decoder numeric launch failed");
  });
#else
  (void)stream_id;(void)a_id;(void)b_id;(void)out_id;
  throw CUDAFailure(NT_CUDA_UNSUPPORTED,"backend built without CUDA");
#endif
 });
}
