#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/InferenceMode.h>
#include <cuda_runtime_api.h>
#include <torch/nn.h>
#include <torch/serialize.h>
#include "nn/LinearLayer.h"
#include "nn/WorkingMemory.h"
#include "torch_utils/module_utils.h"
#define private public
#include "nn/FactorisedLinearLayer.h"
#undef private

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int64_t arena_bytes = 7782400;
constexpr int64_t full_offset = 5357568;
constexpr int64_t full_bytes = 2424832;
constexpr int64_t down_bytes = 524288;
constexpr int64_t scores_bytes = 4194304;

void require(bool ok,const char* why) {
    if(!ok)throw std::runtime_error(why);
}
std::vector<char> read(const std::string& path) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    require(bool(in),"cannot open input");
    auto size=in.tellg();require(size>=0,"cannot measure input");in.seekg(0);
    std::vector<char> result(static_cast<size_t>(size));
    if(!result.empty())require(bool(in.read(result.data(),size)),"cannot read input");
    return result;
}
void write(const std::string& path,const void* data,size_t bytes) {
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    require(bool(out),"cannot create observation");
    out.write(static_cast<const char*>(data),std::streamsize(bytes));out.close();
    require(bool(out),"cannot finish observation");
}
void maps(const std::string& path) {
    std::ifstream in("/proc/self/maps");require(bool(in),"cannot read runtime maps");
    std::ostringstream text;text<<in.rdbuf();const auto raw=text.str();
    require(!raw.empty(),"runtime maps are empty");write(path,raw.data(),raw.size());
}
at::Tensor weight(const std::string& path,const std::vector<int64_t>& shape) {
    std::vector<at::Tensor> tensors;torch::load(tensors,path,at::Device(at::kCPU));
    require(tensors.size()==1,"archive must contain one tensor");
    require(tensors[0].scalar_type()==at::kFloat&&tensors[0].sizes().vec()==shape,
            "archive role, dtype, or shape mismatch");
    return tensors[0];
}
void emit(std::ofstream& manifest,const std::string& root,const std::string& role,
          const at::Tensor& tensor,at::ScalarType dtype,const std::vector<int64_t>& shape) {
    require(tensor.defined()&&tensor.is_cuda()&&tensor.scalar_type()==dtype&&
            tensor.sizes().vec()==shape&&tensor.is_contiguous(),"live source cache descriptor mismatch");
    auto cpu=tensor.detach().to(at::kCPU).contiguous();
    const auto bytes=size_t(cpu.numel())*cpu.element_size();
    const auto file=role+".bin";write(root+"/"+file,cpu.data_ptr(),bytes);
    manifest<<role<<'\t'<<c10::toString(dtype)<<'\t';
    for(size_t i=0;i<shape.size();++i)manifest<<(i?",":"")<<shape[i];
    manifest<<'\t'<<bytes<<'\t'<<file<<'\n';
}
}

int main(int argc,char** argv) {
    try {
        std::map<std::string,std::string> args;
        for(int i=1;i<argc;i+=2) {
            require(i+1<argc,"option requires a value");
            std::string name=argv[i];
            require(name=="--down-weight"||name=="--up-weight"||name=="--input-parent"||
                    name=="--expected-parent"||name=="--out"||name=="--ordinal","unknown option");
            require(args.emplace(name,argv[i+1]).second,"duplicate option");
        }
        require(args.size()==6,"six explicit options required");
        const auto out=args.at("--out");
        require(std::filesystem::create_directory(out),"fresh output directory required");
        maps(out+"/runtime-before.maps");
        const auto source=read(args.at("--input-parent"));
        const auto expected=read(args.at("--expected-parent"));
        require(source.size()==arena_bytes&&expected.size()==arena_bytes,"fixed H01 parent extent required");
        size_t parsed=0;int ordinal=std::stoi(args.at("--ordinal"),&parsed);
        require(parsed==args.at("--ordinal").size()&&ordinal>=0,"invalid ordinal");
        const auto device=at::Device(at::kCUDA,ordinal);
        c10::cuda::CUDAGuard guard(device);c10::InferenceMode inference;
        cudaDeviceProp properties{};
        require(cudaGetDeviceProperties(&properties,ordinal)==cudaSuccess,"cannot observe selected CUDA device");
        std::ostringstream identity;identity<<"ordinal="<<ordinal<<"\nuuid=GPU-";
        for(int i=0;i<16;++i) {
            if(i==4||i==6||i==8||i==10)identity<<'-';
            identity<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(uint8_t(properties.uuid.bytes[i]));
        }
        identity<<'\n';const auto device_text=identity.str();write(out+"/selected-device.txt",device_text.data(),device_text.size());

        dorado::nn::FactorisedLinearLayer projection(1024,128,1024);
        dorado::utils::load_state_dict(*projection,{
            weight(args.at("--down-weight"),{128,1024}),
            weight(args.at("--up-weight"),{1024,128})});
        projection->to(device,at::kHalf);
        require(projection->dn->weight.sizes().vec()==std::vector<int64_t>({128,1024})&&
                projection->up->weight.sizes().vec()==std::vector<int64_t>({1024,128}),"stock semantic weight roles changed");

        dorado::nn::WorkingMemory wm(64);
        wm.reservation_bytes=arena_bytes;wm.current_bytes=full_bytes;
        wm.backing_tensor=at::from_blob(const_cast<char*>(source.data()),{arena_bytes/2},
            at::TensorOptions().dtype(at::kHalf)).to(device,false,true);
        wm.current=wm.backing_tensor.view(at::kChar).narrow(0,full_offset,full_bytes).view({37,64,1024});
        wm.layout=dorado::nn::TensorLayout::CUTLASS_TNC_I8;
        wm.N=64;wm.T=32;wm.C=1024;wm.is_input_to_rev_lstm=false;
        maps(out+"/runtime-ready.maps");

        projection->run_koi(wm,nullptr);

        auto parent=wm.backing_tensor.to(at::kCPU).contiguous();
        write(out+"/candidate-backing.bin",parent.data_ptr(),arena_bytes);
        maps(out+"/runtime-after.maps");
        require(wm.layout==dorado::nn::TensorLayout::NTC&&wm.N==64&&wm.T==32&&wm.C==1024&&
                wm.current.scalar_type()==at::kHalf&&wm.current.sizes().vec()==std::vector<int64_t>({64,32,1024})&&
                wm.current.is_contiguous()&&wm.current.storage_offset()*2==down_bytes,
                "source projection output metadata changed");
        const auto* bytes=static_cast<const char*>(parent.data_ptr());
        require(std::memcmp(bytes,expected.data(),down_bytes)==0,"source projection down scratch differs from H01");
        require(std::memcmp(bytes+down_bytes,expected.data()+down_bytes,scores_bytes)==0,
                "source projection scores differ from H01");
        require(std::memcmp(bytes+down_bytes+scores_bytes,source.data()+down_bytes+scores_bytes,
                            arena_bytes-down_bytes-scores_bytes)==0,"source projection changed input or untouched arena");

        std::ofstream manifest(out+"/caches.tsv");require(bool(manifest),"cannot create cache descriptor");
        manifest<<"nanopore-hac-output-caches/v1\n";
        emit(manifest,out,"down",projection->device_dn_weight_,at::kChar,{128,1024});
        emit(manifest,out,"down-scale",projection->device_dn_weight_scale_,at::kHalf,{128});
        emit(manifest,out,"up",projection->device_up_weight_,at::kHalf,{1024,128});
        manifest.close();require(bool(manifest),"cannot finish cache descriptor");
        std::cout<<"source projection only: down="<<down_bytes<<" scores="<<scores_bytes
                 <<" cache_bytes=393472; exact H01 parity\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<e.what()<<'\n';return 1;
    }
}
