#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/types.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>
#include <c10/core/InferenceMode.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
// SCAN_SOURCE
}
namespace dorado::basecall::decode::inner {
// SCORE_SOURCE
}
namespace dorado::utils {
// INITIALIZATION_SOURCE
}
namespace inner = dorado::basecall::decode::inner;
uint32_t read_u32(std::istream& in) {
    unsigned char b[4];in.read(reinterpret_cast<char*>(b),4);if(!in)throw std::runtime_error("truncated input");
    return uint32_t(b[0]) | uint32_t(b[1])<<8 | uint32_t(b[2])<<16 | uint32_t(b[3])<<24;
}
void write_tensor(const std::filesystem::path& path,const at::Tensor& tensor) {
    auto data=tensor.contiguous();std::ofstream out(path,std::ios::binary);
    out.write(reinterpret_cast<const char*>(data.data_ptr<float>()),data.numel()*sizeof(float));
    if(!out)throw std::runtime_error("cannot write native result");
}
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    try {
        const uint16_t endian=1;if(*reinterpret_cast<const unsigned char*>(&endian)!=1)throw std::runtime_error("little endian host required");
        dorado::utils::initialise_torch();dorado::utils::make_torch_deterministic();
        at::InferenceMode inference_mode_guard;
        std::ifstream input(argv[1],std::ios::binary);if(!input)throw std::runtime_error("cannot read input");
        if(read_u32(input)!=0x43524631)throw std::runtime_error("bad CRF protocol version");
        const auto cases=read_u32(input);
        for(uint32_t c=0;c<cases;++c) {
            const auto blocks=read_u32(input),states=read_u32(input),blank_bits=read_u32(input);
            if(blocks<1 || blocks>4096 || states<4 || states>16384)throw std::runtime_error("invalid fixture dimensions");
            struct {float blank_score;} options;
            std::memcpy(&options.blank_score,&blank_bits,sizeof(float));
            auto t_scores=at::empty({int64_t(blocks),1,int64_t(4*states)},at::TensorOptions().dtype(at::kFloat));
            input.read(reinterpret_cast<char*>(t_scores.data_ptr<float>()),t_scores.numel()*sizeof(float));
            if(!input)throw std::runtime_error("truncated score payload");
// POSTERIOR_SOURCE
            auto directory=std::filesystem::path(argv[2])/std::to_string(c);std::filesystem::create_directories(directory);
            write_tensor(directory/"scores.bin",t_scores);write_tensor(directory/"forward.bin",fwd);
            write_tensor(directory/"backward.bin",bwd);write_tensor(directory/"posts.bin",posts);
        }
        char extra;if(input.get(extra))throw std::runtime_error("trailing fixture bytes");
        std::cout<<"completed "<<cases<<" original CPU CRF scan fixtures\n";
        return 0;
    }catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}
}
