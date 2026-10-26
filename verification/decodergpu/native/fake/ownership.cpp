// Host-only fault fixture. No CUDA/Koi implementation or library is linked.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include "cuda_runtime_api.h"

namespace {
const char* mode="";
const void* borrowed[64]{};
unsigned borrowed_count=0;
bool faulted=false;
const char* output_path=nullptr;
bool fault(const char* name){return std::strcmp(mode,name)==0;}
void release(void* p) noexcept {
    if(p)for(unsigned i=0;i<borrowed_count;++i)if(borrowed[i]==p){
        std::fputs("fake-owner-released-before-drain\n",stderr);std::fflush(stderr);std::_Exit(91);
    }
    std::free(p);
}
}
void* operator new(std::size_t n){if(void*p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void*p) noexcept {release(p);}
void operator delete[](void*p) noexcept {release(p);}
void operator delete(void*p,std::size_t) noexcept {release(p);}
void operator delete[](void*p,std::size_t) noexcept {release(p);}

#define main d01_oracle_main
#define D01_LIFETIME_ONLY
namespace d01_gate {void phase(const std::string&){};}
namespace d01_image {struct Observer {void start(){}void stage(unsigned){}template<class...T>void checkpoint(T&&...){}template<class...T>void retain(T&&...){}void require_records(){}void stop(){}};}
#include "../oracle.cpp"
#undef main

const char* cudaGetErrorString(cudaError_t){return "injected fake runtime failure";}
cudaError_t cudaGetDevice(int*p){*p=0;return 0;}
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*p,int){std::strcpy(p->name,"D01 host fake, no device");p->major=12;p->minor=0;return 0;}
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*p,unsigned){*p=reinterpret_cast<void*>(1);return 0;}
cudaError_t cudaMalloc(void**p,std::size_t n){*p=std::malloc(n);if(!*p)return 1;std::memset(*p,0xa5,n);return 0;}
cudaError_t cudaMemcpyAsync(void*dst,const void*src,std::size_t n,cudaMemcpyKind kind,cudaStream_t){
    if(borrowed_count==64)std::_Exit(93);
    borrowed[borrowed_count++]=kind==cudaMemcpyHostToDevice?src:dst;
    std::memcpy(dst,src,n);
    if(!faulted&&kind==cudaMemcpyDeviceToHost&&fault("raw-capture-close")){
        std::filesystem::create_symlink("/dev/full",std::filesystem::path(output_path)/"back-guide-chunks.bin");faulted=true;
    }
    if(!faulted&&((kind==cudaMemcpyHostToDevice&&fault("h2d-enqueue"))||(kind==cudaMemcpyDeviceToHost&&fault("d2h-enqueue")))){faulted=true;return 1;}
    return 0;
}
cudaError_t cudaMemsetAsync(void*p,int v,std::size_t n,cudaStream_t){std::memset(p,v,n);return 0;}
cudaError_t cudaStreamSynchronize(cudaStream_t){
    if(!faulted&&borrowed_count&&((fault("h2d-completion")&&borrowed_count==2)||((fault("d2h-completion")||fault("final-drain"))&&borrowed_count==1))){
        faulted=true;std::fputs("fake-completion-failed\n",stderr);return 1;
    }
    if(faulted&&fault("final-drain")){
        std::fprintf(stderr,"fake-drain-failed borrowed=%u\n",borrowed_count);return 1;
    }
    if(faulted)std::fprintf(stderr,"fake-drain-ok borrowed=%u\n",borrowed_count);
    borrowed_count=0;return 0;
}
cudaError_t cudaGetLastError(){return 0;}
cudaError_t cudaFree(void*p){if(borrowed_count)std::_Exit(92);std::fputs("fake-device-free\n",stderr);std::free(p);return 0;}
cudaError_t cudaStreamDestroy(cudaStream_t){return 0;}
extern "C" {
int host_back_guide_step(D01_FAKE_ARGS){return 0;}
int host_beam_search_step(D01_FAKE_ARGS){return 0;}
int host_compute_posts_step(D01_FAKE_ARGS){return 0;}
int host_run_decode(D01_FAKE_ARGS,int){return 0;}
}

int main(int argc,char**argv){
    if(argc==2&&std::strcmp(argv[1],"owner-control")==0){
        auto raw=std::vector<uint8_t>(16);borrowed[borrowed_count++]=raw.data();
        return 0; // Its destructor must trip the detector before main returns.
    }
    if(argc==2&&std::strcmp(argv[1],"raw-close")==0){
        // /dev/full accepts open; a one-byte buffered write fails on close.
        try{const char byte='x';write("/dev/full",&byte,1);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 0;}
        std::fputs("raw-close-failure-was-ignored\n",stderr);return 94;
    }
    if(argc!=7)return 95;
    mode=argv[1];output_path=argv[6];
    // Test-only trust root for zero-score fixture. Production uses its fixed H01
    // digest; all real validation/control flow below is the included producer.
    const auto zeros=std::vector<uint8_t>(scores_bytes);
    const auto zero_sha=hex_sha(zeros.data(),zeros.size());scores_sha=zero_sha.c_str();
    return d01_oracle_main(argc-1,argv+1);
}
