#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <torch/nn.h>
#include <torch/serialize.h>
#include "nn/FLSTMStack.h"
#include "nn/WorkingMemory.h"
#include "torch_utils/module_utils.h"
#include "h06_observer.h"

#include <bit>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef H06_OBSERVED
#define H06_OBSERVED 0
#endif

namespace {
#include "../../../nativeproof/sha256.inc"
constexpr int N=64,T=32,C=1024,K=128;
constexpr size_t input_bytes=size_t(T+3)*N*C*2;
constexpr size_t temp_bytes=size_t(N)*(2*K+4*C)*2;
constexpr size_t arena_bytes=input_bytes+temp_bytes;

void require(bool ok,const std::string& message){if(!ok)throw std::runtime_error(message);}
std::vector<char> read_file(const std::string& path){
    std::ifstream in(path,std::ios::binary|std::ios::ate);require(bool(in),"cannot open "+path);
    auto size=in.tellg();require(size>=0,"cannot measure "+path);in.seekg(0);
    std::vector<char> data(static_cast<size_t>(size));
    if(!data.empty())require(bool(in.read(data.data(),std::streamsize(data.size()))),"cannot read "+path);
    return data;
}
std::vector<char> read_pin(const std::string& path,const std::string& pin){
    auto data=read_file(path);require(pin.size()==64&&hex_sha(data.data(),data.size())==pin,"input digest differs: "+path);return data;
}
void write_file(const std::string& path,const void* data,size_t size){
    std::ofstream out(path,std::ios::binary|std::ios::trunc);require(bool(out),"cannot create "+path);
    out.write(static_cast<const char*>(data),std::streamsize(size));out.close();require(bool(out),"cannot finish "+path);
}
std::string dims(const at::Tensor& t){std::ostringstream s;for(int64_t i=0;i<t.dim();++i)s<<(i?",":"")<<t.size(i);return s.str();}

struct Pending {at::Tensor device,host;};
struct Run {
    std::vector<char> input,expected;
    std::vector<at::Tensor> weights;
    dorado::nn::FLSTMStack stack{nullptr};
    dorado::nn::WorkingMemory wm{N};
    std::vector<Pending> pending;
    std::map<std::string,std::string> runtime;
    std::string out;
    std::ofstream tensors,trace;
    cudaStream_t stream{};
    bool stream_ready=false;
    int cache_layers=0,steps=0,layers=0,first_state=0;

    void record_runtime(const std::string& tag){
        std::ifstream maps("/proc/self/maps");require(bool(maps),"cannot read runtime maps");
        std::ostringstream raw;raw<<maps.rdbuf();const auto text=raw.str();write_file(out+"/runtime-"+tag+".maps",text.data(),text.size());
        std::istringstream lines(text);std::string line;
        while(std::getline(lines,line)){
            std::istringstream fields(line);std::string address,perms,offset,device,inode,path;
            if(!(fields>>address>>perms>>offset>>device>>inode))continue;
            std::getline(fields,path);path.erase(0,path.find_first_not_of(' '));
            if(path.empty()||path[0]!='/')continue;
            if(path.find(".so")==std::string::npos&&perms.find('x')==std::string::npos)continue;
            require(!path.ends_with(" (deleted)"),"deleted runtime code");
            const auto actual=std::filesystem::canonical(path).string();
            require(runtime.contains(actual),"runtime object outside prospective policy: "+actual);
        }
    }
    at::Tensor emit(const std::string& stage,int layer,int tick,int ti,int th,int to,
                    const std::string& role,const at::Tensor& value){
        require(value.defined()&&value.is_cuda()&&value.scalar_type()==at::kHalf,"observation requires device half tensor");
        pending.emplace_back(Pending{value.detach(),{}});auto& p=pending.back();
        p.device=p.device.contiguous();p.host=at::empty(p.device.sizes(),at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
        const size_t size=size_t(p.device.numel())*2;
        require(cudaMemcpyAsync(p.host.data_ptr(),p.device.data_ptr(),size,cudaMemcpyDeviceToHost,stream)==cudaSuccess,"observation enqueue failed");
        require(cudaStreamSynchronize(stream)==cudaSuccess,"observation drain failed");
        const std::string file=stage+"-l"+std::to_string(layer)+"-t"+std::to_string(tick)+"-"+role+".bin";
        write_file(out+"/"+file,p.host.data_ptr(),size);
        tensors<<stage<<'\t'<<layer<<'\t'<<tick<<'\t'<<ti<<'\t'<<th<<'\t'<<to<<'\t'<<role<<"\tHalf\t"<<dims(p.host)<<'\t'<<size<<'\t'<<hex_sha(p.host.data_ptr(),size)<<'\t'<<file<<'\n';
        tensors.flush();require(bool(tensors),"cannot finish observation manifest row");
        auto result=p.host;pending.clear();return result;
    }
};
Run* active=nullptr;

std::vector<at::Tensor> load_weights(const std::string& manifest,const std::string& pin){
    const auto raw=read_pin(manifest,pin);std::istringstream rows(std::string(raw.begin(),raw.end()));std::string line;
    require(bool(std::getline(rows,line))&&line=="nanopore-hac-half-weights/v1","weight manifest schema differs");
    const std::vector<std::string> suffix={"dn_weight_ih","dn_weight_hh","up_weight_ih","up_weight_hh","up_bias_ih","up_bias_hh"};
    const std::vector<std::vector<int64_t>> shapes={{K,C},{K,C},{4*C,K},{4*C,K},{4*C},{4*C}};
    std::vector<at::Tensor> weights;
    for(int layer=0;layer<7;++layer)for(int part=0;part<6;++part){
        require(bool(std::getline(rows,line)),"weight manifest truncated");std::istringstream row(line);std::string name,sha,path;
        require(bool(std::getline(row,name,'\t'))&&bool(std::getline(row,sha,'\t'))&&bool(std::getline(row,path))&&path.find('\t')==std::string::npos,"weight row differs");
        const auto expected=std::to_string(4+layer)+".rnn."+suffix[part]+".tensor";
        require(name==expected&&std::filesystem::path(path).filename()==expected,"weight role differs");
        (void)read_pin(path,sha);std::vector<at::Tensor> data;torch::load(data,path,at::Device(at::kCPU));
        require(data.size()==1&&data[0].scalar_type()==at::kFloat&&data[0].sizes().vec()==shapes[part],"weight archive shape or dtype differs");
        weights.push_back(data[0]);
    }
    require(!std::getline(rows,line),"unexpected weight manifest tail");return weights;
}
}

namespace h06_observer {
void caches(int layer,const at::Tensor& ih,const at::Tensor& hh,const at::Tensor& up,const at::Tensor& bias){
    require(H06_OBSERVED&&active&&layer==active->cache_layers,"cache observer sequence differs");
    active->emit("cache",layer,-1,-1,-1,-1,"down-ih",ih);active->emit("cache",layer,-1,-1,-1,-1,"down-hh",hh);
    active->emit("cache",layer,-1,-1,-1,-1,"up",up);active->emit("cache",layer,-1,-1,-1,-1,"bias",bias);++active->cache_layers;
}
void before_step(int layer,int tick,int ti,int th,int to,const at::Tensor& gates,const at::Tensor& cell){
    if(layer!=0||tick!=0)return;
    require(H06_OBSERVED&&active&&active->first_state==0,"initial cell observer sequence differs");
    active->emit("before",layer,tick,ti,th,to,"gates",gates);active->emit("before",layer,tick,ti,th,to,"cell",cell);++active->first_state;
}
void after_step(int layer,int tick,int ti,int th,int to,const at::Tensor& down,const at::Tensor& gates,const at::Tensor& cell,const at::Tensor& value){
    require(H06_OBSERVED&&active&&active->steps==layer*T+tick,"step observer sequence differs");
    const bool reverse=layer%2==0;const int input=reverse?T-tick:2+tick;
    require(ti==input&&th==input+(reverse?2:-2)&&to==input+(reverse?1:-1),"source timestep rows differ");
    active->emit("step",layer,tick,ti,th,to,"down",down);active->emit("step",layer,tick,ti,th,to,"gates",gates);
    active->emit("step",layer,tick,ti,th,to,"cell",cell);active->emit("step",layer,tick,ti,th,to,"output",value);
    active->trace<<layer<<'\t'<<tick<<'\t'<<ti<<'\t'<<th<<'\t'<<to<<'\n';active->trace.flush();require(bool(active->trace),"cannot finish timestep trace");++active->steps;
}
void layer_output(int layer,const at::Tensor& inout){
    require(H06_OBSERVED&&active&&active->layers==layer&&active->steps==(layer+1)*T,"layer observer sequence differs");
    active->emit("layer",layer,-1,-1,-1,-1,"inout",inout);++active->layers;
}
}

int main(int argc,char**argv){
    auto run=std::make_unique<Run>();active=run.get();int exit_code=1;
    try{
        require(std::endian::native==std::endian::little,"little-endian host required");
        std::map<std::string,std::string> args;
        for(int i=1;i<argc;i+=2){require(i+1<argc,"option value absent");require(args.emplace(argv[i],argv[i+1]).second,"duplicate option");}
        const std::vector<std::string> keys={"--weights","--weights-sha","--input","--input-sha","--out","--ordinal","--uuid","--expected","--expected-sha","--runtime-policy","--runtime-sha"};
        require(args.size()==keys.size(),"explicit closed options required");for(const auto& key:keys)require(args.contains(key),"missing option "+key);
        require(args.at("--ordinal")=="0","fixed visible ordinal zero required");
        const char* mode=std::getenv("DORADO_MATMUL_FP16_MODE");require(mode&&std::string(mode)=="CUBLAS","explicit CUBLAS half mode required");
        run->input=read_pin(args.at("--input"),args.at("--input-sha"));require(run->input.size()==input_bytes,"fixed half input extent required");
        for(size_t i=0;i<input_bytes;i+=2){uint16_t bits;std::memcpy(&bits,run->input.data()+i,2);require((bits&0x7c00)!=0x7c00,"finite input fixture required");}
        if(H06_OBSERVED){run->expected=read_pin(args.at("--expected"),args.at("--expected-sha"));require(run->expected.size()==arena_bytes,"reference arena extent differs");}
        const auto policy=read_pin(args.at("--runtime-policy"),args.at("--runtime-sha"));std::istringstream lines(std::string(policy.begin(),policy.end()));std::string line;
        while(std::getline(lines,line)){auto tab=line.find('\t');require(tab!=std::string::npos&&line.size()-tab-1==64,"runtime policy row differs");const auto path=line.substr(0,tab);require(std::filesystem::canonical(path).string()==path&&run->runtime.emplace(path,line.substr(tab+1)).second,"runtime policy path differs");}
        require(run->runtime.size()>=10,"complete prospective runtime required");
        run->out=args.at("--out");require(std::filesystem::create_directory(run->out),"fresh output required");
        run->tensors.open(run->out+"/tensors.tsv");run->trace.open(run->out+"/trace.tsv");require(bool(run->tensors)&&bool(run->trace),"cannot create manifests");
        run->tensors<<"nanopore-hac-half-tensors/v1\n";run->trace<<"layer\ttick\tinput\thidden\toutput\n";
        run->record_runtime("initial");run->weights=load_weights(args.at("--weights"),args.at("--weights-sha"));
        c10::cuda::CUDAGuard guard(0);c10::InferenceMode inference;run->stream=c10::cuda::getCurrentCUDAStream().stream();run->stream_ready=true;
        cudaDeviceProp props{};require(cudaGetDeviceProperties(&props,0)==cudaSuccess,"cannot observe device");
        std::ostringstream uuid;uuid<<"GPU-";for(int i=0;i<16;++i){if(i==4||i==6||i==8||i==10)uuid<<'-';uuid<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(uint8_t(props.uuid.bytes[i]));}
        require(uuid.str()==args.at("--uuid")&&props.major==12&&props.minor==0,"pinned GPU identity differs");
        auto options=at::TensorOptions().dtype(at::kHalf).device(at::Device(at::kCUDA,0));
        run->stack=dorado::nn::FLSTMStack(7,C,K,true,options);dorado::utils::load_state_dict(*run->stack,run->weights);
        run->stack->to(at::kHalf);run->stack->to(at::Device(at::kCUDA,0));run->stack->eval();
        run->wm.next_TC(T,C,dorado::nn::TensorLayout::CUBLAS_TNC);run->stack->reserve_working_memory(run->wm);
        require(run->wm.reservation_bytes==int64_t(arena_bytes),"source reservation differs");run->wm.allocate_backing_tensor(at::Device(at::kCUDA,0));
        auto input=run->wm.next_TC(T,C,dorado::nn::TensorLayout::CUBLAS_TNC);
        require(input.sizes().vec()==std::vector<int64_t>({T+3,N,C})&&input.data_ptr()==run->wm.backing_tensor.data_ptr(),"source input placement differs");
        require(cudaMemcpyAsync(input.data_ptr(),run->input.data(),input_bytes,cudaMemcpyHostToDevice,run->stream)==cudaSuccess,"input upload failed");
        require(cudaStreamSynchronize(run->stream)==cudaSuccess,"input upload drain failed");run->record_runtime("ready");
        run->stack->run_koi(run->wm,nullptr);
        require(cudaGetLastError()==cudaSuccess&&cudaStreamSynchronize(run->stream)==cudaSuccess,"source stack drain failed");
        require(!run->wm.is_input_to_rev_lstm&&run->wm.layout==dorado::nn::TensorLayout::CUBLAS_TNC,"final source orientation/layout differs");
        auto arena=run->emit("final",-1,-1,-1,-1,-1,"arena",run->wm.backing_tensor);
        run->emit("final",-1,-1,-1,-1,-1,"inout",run->wm.current);run->emit("final",-1,-1,-1,-1,-1,"hidden-ntc",run->wm.get_current_NTC_view());
        if(H06_OBSERVED){require(std::memcmp(arena.data_ptr(),run->expected.data(),arena_bytes)==0,"observed final arena differs from untouched reference");require(run->cache_layers==7&&run->steps==7*T&&run->layers==7&&run->first_state==1,"observation event count differs");}
        run->record_runtime("after");require(hex_sha(run->input.data(),run->input.size())==args.at("--input-sha"),"host input changed");
        run->tensors.close();run->trace.close();require(bool(run->tensors)&&bool(run->trace),"cannot finish manifests");
        std::cout<<(H06_OBSERVED?"observed":"untouched")<<" half FLSTM completed: arena="<<arena_bytes<<" caches="<<run->cache_layers<<" steps="<<run->steps<<'\n';exit_code=0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';}
    if(run->stream_ready&&cudaStreamSynchronize(run->stream)!=cudaSuccess){std::cerr<<"final drain failed; owners retained until process teardown\n";std::cerr.flush();std::cout.flush();std::_Exit(1);}
    active=nullptr;return exit_code;
}
