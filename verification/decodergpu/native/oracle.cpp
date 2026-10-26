// Verification only. These four opaque host calls are not a decoder port.
#include <cuda_runtime_api.h>
extern "C" {
#include "koi.h"
}
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "sha256.inc"

namespace {
constexpr size_t N=64,T=32,C=1024,W=32;
constexpr size_t scores_bytes=N*T*C*2,aux_bytes=N*(T+1)*(C+4*W);
const char* scores_sha="56b26176b4272906612eb75690087182a72527f94a9171d9937e5dd80b0d12db";
std::vector<uint8_t> read(const std::filesystem::path& p,size_t size) {
    if(std::filesystem::file_size(p)!=size) throw std::runtime_error("fixed input byte count differs");
    std::vector<uint8_t>b(size);std::ifstream f(p,std::ios::binary);
    if(!f.read(reinterpret_cast<char*>(b.data()),size))throw std::runtime_error("input read failed");return b;
}
void write(const std::filesystem::path& p,const void* data,size_t size) {
    std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(data),size);
    f.close();
    if(!f)throw std::runtime_error("raw observation write failed");
}
uint32_t u32(const uint8_t*p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}
void checked(cudaError_t rc,const char* op){if(rc!=cudaSuccess)throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(rc));}
std::string now(){auto x=std::chrono::system_clock::now();auto t=std::chrono::system_clock::to_time_t(x);std::ostringstream s;s<<std::put_time(std::gmtime(&t),"%FT%TZ");return s.str();}
struct Buffer {const char*name;void*data;size_t bytes;};
}

#ifndef D01_LIFETIME_ONLY
#include "runtime_gate.inc"
#include "image_observer.inc"
#ifdef D01_NUMERIC_IDENTITY
#include "numeric_identity.inc"
#endif
#endif

int main(int argc,char**argv){
    cudaStream_t stream=nullptr;std::vector<Buffer> buffers;bool failed=false;std::ofstream events,manifest;
    // Async copies borrow these allocations. Keep every host owner outside the
    // throwing scope, including snapshots whose completion reports an error.
    std::vector<uint8_t> params,chunks,scores;
    std::vector<std::vector<uint8_t>> snapshots;
    d01_image::Observer image_observer;bool image_started=false;std::filesystem::path observation;
    try{
        if(argc!=6 || std::string(argv[1])!="--fixed-hac-staged-oracle")throw std::runtime_error("requires explicit fixed oracle mode, params, descriptors, scores, fresh output");
        if(std::endian::native!=std::endian::little)throw std::runtime_error("little endian host required");
        params=read(argv[2],52);chunks=read(argv[3],N*16);scores=read(argv[4],scores_bytes);
        const uint32_t expected[]={1,N,T,C,W,0,0x3f828f5c,0xbe99999a,0x42c80000,0x40000000,0};
        if(std::memcmp(params.data(),"D01FIX01",8))throw std::runtime_error("parameter schema differs");
        for(size_t i=0;i<11;++i)if(u32(params.data()+8+4*i)!=expected[i])throw std::runtime_error("fixed scalar bits/shape differ");
        for(size_t i=0;i<N;++i){const uint32_t d[]={uint32_t(i*T),T,uint32_t(i*T),0};for(size_t j=0;j<4;++j)if(u32(chunks.data()+16*i+4*j)!=d[j])throw std::runtime_error("fixed descriptor differs");}
        if(hex_sha(scores.data(),scores.size())!=scores_sha)throw std::runtime_error("immutable H01 score digest differs");
        for(size_t i=0;i<scores.size();i+=2)if(((uint16_t(scores[i])|(uint16_t(scores[i+1])<<8))&0x7c00)==0x7c00)throw std::runtime_error("nonfinite input score rejected");
        std::filesystem::path out(argv[5]);if(!std::filesystem::create_directory(out))throw std::runtime_error("fresh native observation directory required");observation=out;
        write(out/"scores-original.f16",scores.data(),scores.size());write(out/"params.bin",params.data(),params.size());write(out/"chunks-original.bin",chunks.data(),chunks.size());
        events.open(out/"events.jsonl");manifest.open(out/"buffers.tsv");
        buffers.reserve(8);snapshots.reserve(32);
        auto event=[&](const std::string&name,const std::string&state){events<<"{\"stage\":\""<<name<<"\",\"state\":\""<<state<<"\",\"utc\":\""<<now()<<"\"}\n";events.flush();if(!events)throw std::runtime_error("event write failed");};
        d01_gate::phase("pre-model");
        image_started=true;image_observer.start();
        int device=0;checked(cudaGetDevice(&device),"get device");cudaDeviceProp props{};checked(cudaGetDeviceProperties(&props,device),"device properties");
        if(props.major!=12 || props.minor!=0)throw std::runtime_error("first oracle requires SM120 device domain");
        write(out/"device-name.txt",props.name,std::strlen(props.name));
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream create");
        auto alloc=[&](const char*name,size_t bytes){void*p=nullptr;checked(cudaMalloc(&p,bytes),"allocate");buffers.push_back({name,p,bytes});return p;};
        void*dchunks=alloc("chunks",N*16),*results=alloc("results",N*32),*working=alloc("scores",scores_bytes),*aux=alloc("aux",aux_bytes),*path=alloc("path",N*(T+1)*4),*moves=alloc("moves",N*T),*sequence=alloc("sequence",N*T),*quality=alloc("qstring",N*T);
        checked(cudaMemcpyAsync(dchunks,chunks.data(),chunks.size(),cudaMemcpyHostToDevice,stream),"upload chunks");
        checked(cudaMemcpyAsync(working,scores.data(),scores.size(),cudaMemcpyHostToDevice,stream),"upload immutable scores copy");
        for(const auto&b:buffers)if(std::string(b.name)=="path" || std::string(b.name)=="moves" || std::string(b.name)=="sequence" || std::string(b.name)=="qstring")checked(cudaMemsetAsync(b.data,0,b.bytes,stream),"zero source-initialized output");
        checked(cudaStreamSynchronize(stream),"input completion");
        image_observer.checkpoint(out,"inputs");
#if defined(D01_NUMERIC_IDENTITY) && !defined(D01_LIFETIME_ONLY)
        d01_numeric::retain(out,"inputs");
#endif
        d01_gate::phase("pre-forward");
        float clamp=std::bit_cast<float>(expected[5]),scale=std::bit_cast<float>(expected[6]),shift=std::bit_cast<float>(expected[7]),cut=std::bit_cast<float>(expected[8]),blank=std::bit_cast<float>(expected[9]);
        auto capture=[&](const std::string&stage){
            for(const auto&b:buffers){snapshots.emplace_back(b.bytes);auto&raw=snapshots.back();checked(cudaMemcpyAsync(raw.data(),b.data,b.bytes,cudaMemcpyDeviceToHost,stream),"snapshot");checked(cudaStreamSynchronize(stream),"snapshot completion");std::string name=stage+"-"+b.name+".bin";write(out/name,raw.data(),raw.size());manifest<<stage<<'\t'<<b.name<<'\t'<<raw.size()<<'\t'<<hex_sha(raw.data(),raw.size())<<'\t'<<name<<'\n';manifest.flush();if(!manifest)throw std::runtime_error("manifest write failed");}
        };
        using First=int(*)(void*,void*,void*,int,void*,float,int,void*,void*,void*,void*,void*,void*,float,float,int,float,float);
        const First first[]={host_back_guide_step,host_beam_search_step,host_compute_posts_step};
        const char*names[]={"back-guide","beam-search","posterior","decode"};
        const char*phases[]={"post-back-guide","post-beam-search","post-posterior","post-staged"};
        for(int stage=0;stage<4;++stage){
            image_observer.stage(stage+1);
            event(names[stage],"before-call");int rc;
            if(stage<3)rc=first[stage](stream,dchunks,results,N,working,clamp,C,aux,path,moves,nullptr,sequence,quality,scale,shift,W,cut,blank);
            else rc=host_run_decode(stream,dchunks,results,N,working,clamp,C,aux,path,moves,nullptr,sequence,quality,scale,shift,W,cut,blank,0);
            if(rc!=0)throw std::runtime_error(std::string(names[stage])+" host status="+std::to_string(rc));
            checked(cudaGetLastError(),"stage launch");checked(cudaStreamSynchronize(stream),"stage completion");image_observer.checkpoint(out,names[stage]);
#if defined(D01_NUMERIC_IDENTITY) && !defined(D01_LIFETIME_ONLY)
            d01_numeric::retain(out,names[stage]);
#endif
            capture(names[stage]);event(names[stage],"captured");d01_gate::phase(phases[stage]);
        }
        image_observer.require_records();image_observer.stop();image_started=false;
        if(hex_sha(scores.data(),scores.size())!=scores_sha)throw std::runtime_error("immutable host scores changed");
        event("all","completed");
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';failed=true;if(image_started && !observation.empty()){try{image_observer.retain(observation,"failure");}catch(const std::exception&retention){std::cerr<<"image failure retention: "<<retention.what()<<'\n';}}}
    if(stream){const auto rc=cudaStreamSynchronize(stream);if(rc!=cudaSuccess){std::cerr<<"final drain failed; host and device owners retained until process teardown: "<<cudaGetErrorString(rc)<<'\n';events.flush();manifest.flush();std::cerr.flush();std::_Exit(1);}}
    for(const auto&b:buffers)if(cudaFree(b.data)!=cudaSuccess)failed=true;
    if(stream && cudaStreamDestroy(stream)!=cudaSuccess)failed=true;
    return failed?1:0;
}
