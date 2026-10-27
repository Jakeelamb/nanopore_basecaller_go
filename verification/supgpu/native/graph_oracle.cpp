// Verification only. The reference executable links only original Dorado TUs.
// The observer executable links generated source copies ahead of the archives.
// The observed forward records caches and operations in their actual source
// order. No cache is reconstructed after remove_bits.
#include "supgpu_observer.h"
#include "basecall/crf_utils.h"
#include "config/BasecallModelConfig.h"
#include "model/TxModel.h"
#include "torch_utils/gpu_profiling.h"
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace {
int phase = 0;
std::string output;
std::ofstream events;
std::ofstream coarse;
std::map<dorado::nn::TxEncoderImpl *, int> layers;
unsigned int gate_sequence = 0;

std::string quote(const std::string &s) {
    std::ostringstream o; o << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') o << '\\' << char(c);
        else if (c < 32) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else o << char(c);
    }
    return o.str() + '"';
}

std::string sha256(const void *data, size_t size) {
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    auto r=[](uint32_t x,int n){return (x>>n)|(x<<(32-n));};
    auto p=static_cast<const unsigned char *>(data);
    size_t blocks=(size+9+63)/64;
    for(size_t block=0;block<blocks;block++) {
        unsigned char b[64]={};
        for(size_t i=0;i<64;i++){size_t pos=block*64+i;if(pos<size)b[i]=p[pos];else if(pos==size)b[i]=0x80;}
        if(block+1==blocks){uint64_t bits=uint64_t(size)*8;for(int i=0;i<8;i++)b[63-i]=static_cast<unsigned char>(bits>>(i*8));}
        uint32_t w[64];for(int i=0;i<16;i++)w[i]=(uint32_t(b[4*i])<<24)|(uint32_t(b[4*i+1])<<16)|(uint32_t(b[4*i+2])<<8)|b[4*i+3];
        for(int i=16;i<64;i++)w[i]=w[i-16]+(r(w[i-15],7)^r(w[i-15],18)^(w[i-15]>>3))+w[i-7]+(r(w[i-2],17)^r(w[i-2],19)^(w[i-2]>>10));
        uint32_t a=h[0],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7],bb=h[1];
        for(int i=0;i<64;i++){uint32_t s=z+(r(e,6)^r(e,11)^r(e,25))+((e&f)^((~e)&g))+k[i]+w[i];uint32_t v=(r(a,2)^r(a,13)^r(a,22))+((a&bb)^(a&c)^(bb&c));z=g;g=f;f=e;e=d+s;d=c;c=bb;bb=a;a=s+v;}
        uint32_t q[8]={a,bb,c,d,e,f,g,z};for(int i=0;i<8;i++)h[i]+=q[i];
    }
    std::ostringstream o;for(uint32_t v:h)o<<std::hex<<std::setw(8)<<std::setfill('0')<<v;return o.str();
}

std::string payload(const void *data,size_t bytes) {
    auto sha=sha256(data,bytes);auto relative="payloads/"+sha+".bin";auto path=output+"/"+relative;
    if(!std::filesystem::exists(path)) {std::ofstream f(path,std::ios::binary);f.write(static_cast<const char *>(data),static_cast<std::streamsize>(bytes));f.close();if(!f)throw std::runtime_error("payload write failed");}
    else if(std::filesystem::file_size(path)!=bytes)throw std::runtime_error("payload digest collision/size mismatch");
    return "{\"path\":"+quote(relative)+",\"sha256\":"+quote(sha)+",\"bytes\":"+std::to_string(bytes)+"}";
}

std::string tensor_pin(const at::Tensor &t) {auto cpu=t.detach().to(at::kCPU).contiguous();return payload(cpu.const_data_ptr(),size_t(cpu.nbytes()));}
template<class T>std::string array(const T &values){std::ostringstream o;o<<'[';bool first=true;for(auto v:values){if(!first)o<<',';first=false;o<<v;}return o.str()+']';}

std::string tensor_json(const supgpu::TensorArg &arg) {
    const auto &t=arg.tensor;if(!t.defined() || t.numel()==0)throw std::runtime_error("required tensor is absent: "+arg.name);
    auto bytes=t.storage().nbytes();
    auto all=at::from_blob(t.storage().data_ptr().get(),{int64_t(bytes)},t.options().dtype(at::kByte)).to(at::kCPU).contiguous();
    auto offset=t.storage_offset()*int64_t(t.element_size());
    std::ostringstream o;
    o<<"{\"name\":"<<quote(arg.name)<<",\"dtype\":"<<quote(c10::toString(t.scalar_type()))
      <<",\"shape\":"<<array(t.sizes())<<",\"strides\":"<<array(t.strides())<<",\"tags\":"<<array(arg.tags)
      <<",\"scale_tag\":"<<arg.scale_tag<<",\"offset_bytes\":"<<offset<<",\"data\":"<<tensor_pin(t)
      <<",\"backing\":"<<payload(all.const_data_ptr(),size_t(bytes))<<",\"regions\":[{\"begin\":0,\"end\":"<<bytes
      <<",\"state\":\"initialized\",\"rule\":\"fixed-source-full-allocation/v1\"}]}";
    return o.str();
}

std::string dtype(KoiTypeId t){switch(t){case KOI_NONE:return "None";case KOI_I8:return "Char";case KOI_I32:return "Int";case KOI_E4M3:return "Float8_e4m3fn";case KOI_F16:return "Half";case KOI_F32:return "Float";default:throw std::runtime_error("unsupported Koi type");}}
std::string descriptor_json(const supgpu::DescriptorArg &arg){
    const auto &d=*arg.value;if(d.ndims<=0 || d.ndims>KOI_TENSOR_MAX_DIMS || !d.data_ptr)throw std::runtime_error("invalid Koi descriptor");
    std::vector<int64_t> shape,strides;std::vector<int> tags;for(int i=0;i<d.ndims;i++){shape.push_back(d.dims[i].size);strides.push_back(d.dims[i].stride);tags.push_back(d.dims[i].tag);}
    return "{\"name\":"+quote(arg.name)+",\"dtype\":"+quote(dtype(d.type_id))+",\"shape\":"+array(shape)+",\"strides\":"+array(strides)+",\"tags\":"+array(tags)+",\"scale_tag\":"+std::to_string(d.scale_tag)+",\"scale_dtype\":"+quote(dtype(d.scale_type_id))+"}";
}

void event(const std::string &name,int layer,std::initializer_list<supgpu::TensorArg> tensors,std::initializer_list<supgpu::DescriptorArg> descriptors={}) {
    events<<"{\"name\":"<<quote(name)<<",\"layer\":"<<layer<<",\"tensors\":[";
    bool first=true;for(const auto &t:tensors){if(!first)events<<',';first=false;events<<tensor_json(t);}events<<"],\"descriptors\":[";first=true;for(const auto &d:descriptors){if(!first)events<<',';first=false;events<<descriptor_json(d);}events<<"]}\n";events.flush();if(!events)throw std::runtime_error("event write failed");
}

void maps(const std::string &name){std::ifstream in("/proc/self/maps");std::ofstream out(output+"/"+name);out<<in.rdbuf();out.close();if(!out)throw std::runtime_error("runtime maps write failed");}
}

namespace supgpu {
void reference_boundary(const std::string &name,const at::Tensor &t){coarse<<"{\"name\":"<<quote(name)<<",\"data\":"<<tensor_pin(t)<<"}\n";coarse.flush();if(!coarse)throw std::runtime_error("reference boundary write failed");}
void check(int result){if(result!=0)throw std::runtime_error("Koi operation failed: "+std::to_string(result));}
void emit(const std::string &name,int layer,std::initializer_list<TensorArg> tensors){if(phase==2)event(name,layer,tensors);}
void encoder(dorado::nn::TxEncoderImpl *layer,const std::string &stage,std::initializer_list<TensorArg> tensors,std::initializer_list<DescriptorArg> descriptors){if(phase!=2)return;auto index=layers.at(layer);std::ostringstream name;name<<"encoder."<<std::setw(2)<<std::setfill('0')<<index<<'.'<<stage;event(name.str(),index,tensors,descriptors);}
void cache(dorado::nn::TxEncoderImpl *p,bool after){
    if(phase!=2)return;auto index=layers.at(p);std::ostringstream name;name<<"encoder."<<std::setw(2)<<std::setfill('0')<<index<<".cache."<<(after?"after":"before")<<"_remove_bits";
    event(name.str(),index,{{"qkv_half",p->wqkv_weights_f16.t},{"qkv_int8",p->wqkv_weights_i8.t},{"qkv_scale",p->wqkv_weights_i8.scale},
          {"sincos",p->sincos_bfr},{"proj_weight",p->proj_weight},{"proj_bias",p->proj_bias},{"norm1_weight",p->t_res_weights},
          {"norm2_weight",p->t_res2_weights},{"fc1_half",p->t_fc1_wts_f16.t},{"fc1_fp8",p->t_fc1_wts_f8.t},{"fc2_fp8",p->t_fc2_wts}});
}

}

#include "reference_sequence.inc"

// Separate from audit FDs 3/4 so a background dlopen rendezvous cannot consume
// a main-thread phase approval. These calls add no model/tensor arithmetic.
void runtime_gate(const std::string &name) {
    maps(name+".maps");
    auto seq=++gate_sequence;
    auto line="{\"schema\":\"nanopore-supgpu-loader-gate/v1\",\"kind\":\"phase\",\"sequence\":"+std::to_string(seq)+",\"cookie\":0,\"path\":"+quote(name)+",\"namespace\":0,\"pid\":"+std::to_string(getpid())+"}\n";
    size_t sent=0;while(sent<line.size()){auto n=write(5,line.data()+sent,line.size()-sent);if(n<=0)throw std::runtime_error("phase request pipe failed");sent+=size_t(n);}
    std::string reply;char c;while(reply.size()<128){if(read(6,&c,1)!=1)throw std::runtime_error("phase approval pipe failed");reply+=c;if(c=='\n')break;}
    if(reply!="allow phase "+std::to_string(seq)+"\n")throw std::runtime_error("runtime phase rejected: "+name);
}

int main(int argc,char **argv) {
    try {
        if(argc!=8)throw std::runtime_error("usage: oracle reference|observer MODEL INPUT OUT BATCH CHUNK capture|attest-only (private audit/phase FDs required)");
        std::string mode=argv[1];if(mode!="reference" && mode!="observer")throw std::runtime_error("unknown mode");
#if SUPGPU_OBSERVER
        if(mode!="observer")throw std::runtime_error("observed TU binary cannot be the untouched reference");
#else
        if(mode!="reference")throw std::runtime_error("untouched binary cannot emit observed stages");
#endif
        int n=std::stoi(argv[5]),chunk=std::stoi(argv[6]);if(n!=32 || chunk!=768)throw std::runtime_error("only the frozen N32/chunk768 case is admitted by this adapter");
        output=argv[4];if(!std::filesystem::create_directory(output))throw std::runtime_error("output already exists");std::filesystem::create_directory(output+"/payloads");
        std::string action=argv[7];if(action!="capture" && action!="attest-only")throw std::runtime_error("unknown action");
        runtime_gate("pre-model");
        if(action=="attest-only"){maps("runtime.maps");return 0;}
        std::ifstream input_file(argv[3],std::ios::binary);std::vector<char> input((std::istreambuf_iterator<char>(input_file)),{});
        if(input.size()!=size_t(n*chunk*2))throw std::runtime_error("input byte count mismatch");
        torch::NoGradGuard no_grad;at::set_num_threads(2);at::set_num_interop_threads(1);
        c10::cuda::CUDAGuard guard(0);auto device=c10::Device(c10::kCUDA,0);
        cudaDeviceProp properties{};if(cudaGetDeviceProperties(&properties,0)!=cudaSuccess)throw std::runtime_error("cannot observe capture device");
        int count=0,driver=0,runtime=0;if(cudaGetDeviceCount(&count)!=cudaSuccess || cudaDriverGetVersion(&driver)!=cudaSuccess || cudaRuntimeGetVersion(&runtime)!=cudaSuccess || count!=1 || driver!=13040 || runtime!=13040 || properties.major!=12 || properties.minor!=0 || properties.multiProcessorCount!=36)throw std::runtime_error("capture device/runtime differs from retained P01");
        std::ostringstream uuid;for(unsigned char byte:properties.uuid.bytes)uuid<<std::hex<<std::setw(2)<<std::setfill('0')<<int(byte);
        if(uuid.str()!="807d7ae76a99c79512c223fc801d38e7")throw std::runtime_error("capture device differs from pinned P01");
        std::cout<<"device_uuid="<<uuid.str()<<" driver="<<driver<<" runtime="<<runtime<<" compute=12.0 sms=36"<<std::endl;
        auto config=dorado::config::load_model_config(argv[2]);
        if(!config.tx || config.tx->tx.depth!=18 || config.stride_inner()!=12 || config.chunk_size_granularity()!=768)throw std::runtime_error("SUP config mismatch");
        auto holder=dorado::basecall::load_crf_model(config,torch::TensorOptions().device(device).dtype(torch::kF16));
        auto &model=holder->get<dorado::basecall::model::TxModelImpl>();
        if(!model.tx_encoder->use_koi_tiled || !model.tx_encoder->use_i8 || model.tx_encoder->layer_vec.size()!=18)throw std::runtime_error("source selected an unsupported encoder path");
        for(size_t i=0;i<model.tx_encoder->layer_vec.size();i++)layers[model.tx_encoder->layer_vec[i].get()]=int(i);
        auto x=at::from_blob(input.data(),{n,1,chunk},at::TensorOptions().dtype(at::kHalf)).to(device).contiguous();
        events.open(output+"/events.jsonl");if(!events)throw std::runtime_error("cannot open event stream");
        runtime_gate("pre-forward");
        if(mode=="reference"){
            auto scores=model.forward(x,nullptr);std::ofstream f(output+"/scores.json");f<<tensor_pin(scores)<<'\n';f.close();if(!f)throw std::runtime_error("score receipt write failed");
            runtime_gate("post-full");
            coarse.open(output+"/coarse.jsonl");auto sequence_scores=reference_sequence(model,x);
            if(tensor_pin(sequence_scores)!=tensor_pin(scores))throw std::runtime_error("stock module sequence does not reproduce untouched full forward");
            coarse.close();if(!coarse)throw std::runtime_error("reference boundary close failed");
            runtime_gate("post-staged");
        }else{
            phase=2;auto staged=model.forward(x,nullptr);std::ofstream g(output+"/staged.json");g<<tensor_pin(staged)<<'\n';g.close();if(!g)throw std::runtime_error("staged score write failed");
            runtime_gate("post-observed");
        }
        maps("runtime.maps");events.close();if(!events)throw std::runtime_error("event stream close failed");return 0;
    }catch(const std::exception &e){if(!output.empty()){try{maps("failure.maps");}catch(...){}}std::cerr<<"supgpu oracle: "<<e.what()<<'\n';return 1;}
}
