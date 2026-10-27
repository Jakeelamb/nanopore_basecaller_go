#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "../sha256.inc"
using CUcontext=void*;
using cudaError_t=int;
struct cudaDeviceProp {struct {char bytes[16];}uuid;int major,minor;};
std::string mode;
void checked(int status,const char*){if(status)throw std::runtime_error("fake CUDA query failure");}
int cudaGetDevice(int*p){*p=0;return 0;}
int cudaGetDeviceProperties(cudaDeviceProp*p,int){std::memset(p->uuid.bytes,0x11,16);p->major=12;p->minor=0;return 0;}
int cudaDriverGetVersion(int*p){*p=13040;return mode=="driver-query"?1:0;}
int cudaRuntimeGetVersion(int*p){*p=13040;return 0;}
int cuptiGetContextId(CUcontext,uint32_t*p){*p=mode=="context-uid"?0:9;return 0;}
int cuptiGetDeviceId(CUcontext,uint32_t*p){*p=mode=="context-device"?1:0;return 0;}
void write(const std::filesystem::path&p,const void*data,size_t bytes){
    if(mode=="close-failure")throw std::runtime_error("fake raw observation close failure");
    std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(data),bytes);f.close();if(!f)throw std::runtime_error("fake raw close failure");
}
namespace d01_image {
constexpr size_t max_modules=16,max_launches=512;
struct State{size_t modules=0,launches=0;struct Row{uintptr_t context=0;};std::array<Row,max_modules>module;std::array<Row,max_launches>launch;}state;
struct Locked{};
void checked_cupti(int status,const char*){if(status)throw std::runtime_error("fake CUPTI query failure");}
}
#include "../numeric_identity.inc"
int main(int argc,char**argv){
    if(argc!=3)return 2;mode=argv[1];std::filesystem::create_directory(argv[2]);
    d01_image::state.modules=1;d01_image::state.module[0].context=1;d01_image::state.launches=2;d01_image::state.launch[0].context=1;d01_image::state.launch[1].context=1;
    try{d01_numeric::retain(argv[2],"decode");return mode=="positive"?0:3;}
    catch(const std::exception&e){std::cerr<<e.what()<<'\n';return mode=="positive"?4:1;}
}
