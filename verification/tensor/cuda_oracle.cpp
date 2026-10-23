#include "runtime.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace spdlog { void error(const char* text,int code) {std::cerr<<text<<" "<<code<<"\n";} }
namespace details {
#include "cuda_source_block.h"
}
namespace {
void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
std::vector<uint16_t> load(const char* path){
 std::ifstream file(path,std::ios::binary|std::ios::ate);if(!file)throw std::runtime_error("missing raw half input");auto n=file.tellg();if(n<=0||n%2)throw std::runtime_error("bad raw half byte count");std::vector<uint16_t> bits(size_t(n)/2);file.seekg(0);file.read(reinterpret_cast<char*>(bits.data()),n);if(!file)throw std::runtime_error("short input read");return bits;
}
void save(const std::string& path,const std::vector<uint16_t>& bits){std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(bits.data()),bits.size()*2);if(!file)throw std::runtime_error("raw output write failed");}
int64_t integer(const char* raw){size_t end=0;auto n=std::stoll(raw,&end);if(raw[end]||n<0)throw std::runtime_error("invalid unsigned decimal argument");return n;}
at::Tensor upload(const std::vector<uint16_t>& bits,cudaStream_t stream,int ordinal){auto t=at::empty({int64_t(bits.size())},at::TensorOptions().dtype(at::kHalf).device(at::Device(at::kCUDA,ordinal)));check(cudaMemcpyAsync(t.data_ptr(),bits.data(),bits.size()*2,cudaMemcpyHostToDevice,stream));check(cudaStreamSynchronize(stream));return t;}
std::vector<uint16_t> download(const at::Tensor& t,cudaStream_t stream){std::vector<uint16_t> bits(t.numel());check(cudaMemcpyAsync(bits.data(),t.data_ptr(),bits.size()*2,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));return bits;}
at::Tensor view(const at::Tensor& root,int64_t rows,int64_t cols,int64_t stride,int64_t offset){if(rows<=0||cols<=0||stride<cols||rows>INT_MAX||cols>INT_MAX||stride>INT_MAX||offset>=root.numel()||(rows-1)>(root.numel()-1-offset)/stride||(rows-1)*stride+cols>root.numel()-offset)throw std::runtime_error("invalid matrix span");return root.as_strided({rows,cols},{stride,1},offset);}
void diagnostics(cudaStream_t stream,int ordinal){
 int driver=0,runtime=0;check(cudaDriverGetVersion(&driver));check(cudaRuntimeGetVersion(&runtime));cudaDeviceProp props{};check(cudaGetDeviceProperties(&props,ordinal));auto handle=at::cuda::getCurrentCUDABlasHandle();int version=0;cublasPointerMode_t pointer;cublasMath_t math;cublasAtomicsMode_t atomics;cudaStream_t actual;
 if(cublasGetVersion(handle,&version)!=CUBLAS_STATUS_SUCCESS||cublasGetStream(handle,&actual)!=CUBLAS_STATUS_SUCCESS||cublasGetPointerMode(handle,&pointer)!=CUBLAS_STATUS_SUCCESS||cublasGetMathMode(handle,&math)!=CUBLAS_STATUS_SUCCESS||cublasGetAtomicsMode(handle,&atomics)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS diagnostics failed");
 std::cout<<"driver="<<driver<<" runtime="<<runtime<<" cublas="<<version<<" ordinal="<<ordinal<<" device="<<props.name<<" stream_matches="<<(actual==stream)<<" pointer_mode="<<int(pointer)<<" math_mode="<<int(math)<<" atomics_mode="<<int(atomics)<<" uuid=";
 const auto* uuid=reinterpret_cast<const unsigned char*>(props.uuid.bytes);const char* digits="0123456789abcdef";for(size_t i=0;i<16;i++)std::cout<<digits[uuid[i]>>4]<<digits[uuid[i]&15];std::cout<<"\n";
 for(const char* name:{"CUDA_VISIBLE_DEVICES","CUBLAS_WORKSPACE_CONFIG","DORADO_MATMUL_FP16_MODE"}){auto value=std::getenv(name);std::cout<<name<<"="<<(value?value:"")<<"\n";}
 std::ifstream maps("/proc/self/maps");std::cout<<"loaded_maps_begin\n"<<maps.rdbuf()<<"loaded_maps_end\n";
}
}
void run_case(const std::vector<std::string>& args,const std::string& endpoint,cudaStream_t raw,int ordinal){
 if(args.size()!=15)throw std::runtime_error("batch row requires 15 fields");
 std::string mode=args[0],prefix=args[1];if(mode!="copy"&&mode!="gemm")throw std::runtime_error("unknown mode");
 auto ah=load(args[2].c_str()),bh=load(args[3].c_str()),ch=load(args[4].c_str());bool shared=integer(args[5].c_str())==1;
 auto number=[&](size_t i){return integer(args[i].c_str());};auto m=number(6),n=number(7),k=number(8),as=number(9),bs=number(10),cs=number(11),ao=number(12),bo=number(13),co=number(14);
 auto ar=upload(ah,raw,ordinal),br=upload(bh,raw,ordinal),cr=shared?ar:upload(ch,raw,ordinal);
 if(mode=="gemm"){
 auto a=view(ar,m,k,as,ao),b=view(br,k,n,bs,bo),c=view(cr,m,n,cs,co);
 if(endpoint=="source-f16")details::matmul_f16_cublas(a,b,c);
 else{const float one=1,zero=0;auto status=cublasGemmEx(at::cuda::getCurrentCUDABlasHandle(),CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&one,b.data_ptr(),CUDA_R_16F,bs,a.data_ptr(),CUDA_R_16F,as,&zero,c.data_ptr(),CUDA_R_16F,cs,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("float32 mutant failed");}
 check(cudaGetLastError());check(cudaStreamSynchronize(raw));
 }
 save(prefix+"-a.u16le",download(ar,raw));save(prefix+"-b.u16le",download(br,raw));save(prefix+"-c.u16le",download(cr,raw));
 std::cout<<"completed "<<prefix<<"\n";
}
int main(int argc,char** argv){
 try{
 if(argc!=5||std::string(argv[1])!="--batch")throw std::runtime_error("expected --batch plan-path source-f16|mutant-f32 ordinal");
 std::string endpoint=argv[3];if(endpoint!="source-f16"&&endpoint!="mutant-f32")throw std::runtime_error("only extracted cuBLAS endpoint or labeled float32 mutant is supported");
 auto ordinal=integer(argv[4]);if(ordinal>INT_MAX)throw std::runtime_error("ordinal overflow");
 std::ifstream plan(argv[2]);if(!plan)throw std::runtime_error("missing batch plan");std::vector<std::vector<std::string>> cases;std::string line;
 while(std::getline(plan,line)){std::istringstream input(line);std::vector<std::string> fields;std::string value;while(input>>std::quoted(value))fields.push_back(value);if(!input.eof()||fields.size()!=15)throw std::runtime_error("invalid batch row");cases.push_back(std::move(fields));}
 if(cases.empty()||cases.size()>37)throw std::runtime_error("batch size outside bounded corpus");
 initialize_tensor_runtime();c10::InferenceMode inference;c10::cuda::CUDAGuard device{int(ordinal)};cudaStream_t raw;check(cudaStreamCreateWithFlags(&raw,cudaStreamNonBlocking));
 {c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(raw,int(ordinal)));for(const auto& args:cases)run_case(args,endpoint,raw,ordinal);diagnostics(raw,ordinal);}
 check(cudaStreamDestroy(raw));return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
