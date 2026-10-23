// Verification adapter only. The build inserts hash-checked, verbatim Dorado
// function definitions at SOURCE_FUNCTIONS. These minimal module declarations
// expose storage for the adapter; they do not claim the original class ABI.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/nn.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#include "config/common.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

namespace dorado::utils {
struct ScopedProfileRange { ScopedProfileRange(const char*,int) {} };
// INITIALIZATION_FUNCTIONS
}
namespace dorado::nn {
struct AuxiliaryData;
struct FLSTMLayerImpl : torch::nn::Module {
    FLSTMLayerImpl(int,int,at::TensorOptions);
    at::Tensor forward(at::Tensor);
    int C_;
    at::Tensor dn_weight_ih_,dn_weight_hh_,up_weight_ih_,up_weight_hh_,up_bias_ih_,up_bias_hh_;
};
TORCH_MODULE(FLSTMLayer);
struct FLSTMStackImpl : torch::nn::Module {
    FLSTMStackImpl(int,int,int,bool,at::TensorOptions);
    at::Tensor forward(at::Tensor);
    int C_,K_; bool first_reverse_; std::vector<FLSTMLayer> layers_;
};
TORCH_MODULE(FLSTMStack);
struct ConvStackImpl : torch::nn::Module {
    struct ConvLayer {
        explicit ConvLayer(const config::ConvParams&);
        config::ConvParams params; torch::nn::Conv1d conv{nullptr};
    };
    explicit ConvStackImpl(const std::vector<config::ConvParams>&);
    at::Tensor forward(at::Tensor);
    std::vector<ConvLayer> layers;
};
TORCH_MODULE(ConvStack);
struct LinearCRFImpl : torch::nn::Module {
    LinearCRFImpl(int,int,bool,bool);
    at::Tensor forward(at::Tensor);
    bool bias; static constexpr int scale=5;
    torch::nn::Linear linear{nullptr}; torch::nn::Tanh activation{nullptr};
};
TORCH_MODULE(LinearCRF);
struct ClampImpl : torch::nn::Module {
    ClampImpl(float,float,bool);at::Tensor forward(at::Tensor);
    bool active;float min,max;
};
TORCH_MODULE(Clamp);
// SOURCE_FUNCTIONS
}
namespace dorado::basecall::model {
struct CRFModelImpl : torch::nn::Module {
    torch::nn::Sequential encoder{nullptr};
    at::Tensor forward(const at::Tensor&,nn::AuxiliaryData*);
};
TORCH_MODULE(CRFModel);
// MODEL_FORWARD
}

namespace {
using namespace dorado::nn;
using dorado::basecall::model::CRFModel;
namespace fs=std::filesystem;
struct Ref { std::string name,path; std::vector<int64_t> shape; };
struct Fixture {
    std::string name,kind; int C,K,layers; bool reverse;
    Ref input; std::vector<Ref> weights,stages;
};
fs::path root;
at::Tensor values(at::IntArrayRef shape,int seed,float scale) {
    auto t=at::empty(shape,at::kFloat); auto*p=t.data_ptr<float>();
    for(int64_t i=0;i<t.numel();++i) p[i]=float(((i*7+seed*11)%31)-15)*scale;
    return t;
}
Ref store(const std::string& prefix,const std::string& name,const at::Tensor&t,bool archive) {
    auto x=t.contiguous();std::string path=prefix+"/"+name;
    fs::create_directories((root/path).parent_path());
    std::ofstream out(root/(path+".bin"),std::ios::binary);
    out.write(reinterpret_cast<const char*>(x.data_ptr<float>()),x.numel()*sizeof(float));
    if(!out)throw std::runtime_error("write fixture failed");
    if(archive)torch::save(std::vector<at::Tensor>{x},(root/(path+".tensor")).string());
    return {name,path, x.sizes().vec()};
}
void assign(torch::nn::Module& module,int seed,float scale) {
    int i=0;for(auto p:module.parameters())p.copy_(values(p.sizes(),seed+i++,scale));
}
void refJSON(std::ostream&out,const Ref&r) {
    out<<"{\"name\":\""<<r.name<<"\",\"path\":\""<<r.path<<"\",\"shape\":[";
    for(size_t i=0;i<r.shape.size();++i){if(i)out<<',';out<<r.shape[i];}out<<"]}";
}
void listJSON(std::ostream&out,const std::vector<Ref>&rs) {
    out<<'[';for(size_t i=0;i<rs.size();++i){if(i)out<<',';refJSON(out,rs[i]);}out<<']';
}
void writeManifest(const std::vector<Fixture>&fixtures) {
    std::ofstream out(root/"fixtures.json");out<<'[';
    for(size_t i=0;i<fixtures.size();++i){if(i)out<<',';auto&f=fixtures[i];
        out<<"{\"name\":\""<<f.name<<"\",\"kind\":\""<<f.kind<<"\",\"channels\":"<<f.C<<",\"inner\":"<<f.K<<",\"layers\":"<<f.layers<<",\"reverse\":"<<(f.reverse?"true":"false")<<",\"input\":";
        refJSON(out,f.input);out<<",\"weights\":";listJSON(out,f.weights);out<<",\"stages\":";listJSON(out,f.stages);out<<'}';
    }out<<"]\n";if(!out)throw std::runtime_error("manifest write failed");
}
void appendRNNWeights(Fixture&f,FLSTMStack&rnn) {
    int i=0;for(auto p:rnn->parameters())f.weights.push_back(store(f.name,"w"+std::to_string(i++),p,true));
}
Fixture rnnFixture(int N,int T,int layers,bool reverse) {
    Fixture f;f.name="rnn_n"+std::to_string(N)+"_t"+std::to_string(T)+"_l"+std::to_string(layers)+"_r"+std::to_string(reverse);f.kind="rnn";f.C=4;f.K=2;f.layers=layers;f.reverse=reverse;
    FLSTMStack rnn(layers,4,2,reverse,at::TensorOptions().dtype(at::kFloat));assign(*rnn,3,.125f);
    auto input=values({N,T,4},5,.25f);f.input=store(f.name,"input",input,true);appendRNNWeights(f,rnn);
    auto x=input;bool direction=!reverse;
    for(int i=0;i<layers;++i){if(i>0||reverse){x=x.flip(1);direction^=1;}x=rnn->layers_[i]->forward(x);f.stages.push_back(store(f.name,"rnn"+std::to_string(i),x,false));}
    f.stages.push_back(store(f.name,"rnns",rnn->forward(input),false));return f;
}
Fixture graphFixture(int N,int L,const std::string&model) {
    const bool real=!model.empty();const int C=real?1024:4,K=real?128:2,D=real?128:2,O=real?1024:8;
    Fixture f;f.name=real?"stock_hac_n1_l96":"graph_n"+std::to_string(N)+"_l"+std::to_string(L);f.kind="graph";f.C=C;f.K=K;f.layers=7;f.reverse=true;
    std::vector<dorado::config::ConvParams> cv{{1,real?16:4,5,1,dorado::config::Activation::SWISH,0},{real?16:4,real?16:4,5,1,dorado::config::Activation::SWISH,0},{real?16:4,C,19,6,dorado::config::Activation::TANH,0}};
    ConvStack conv(cv);FLSTMStack rnn(7,C,K,true,at::TensorOptions().dtype(at::kFloat));LinearCRF lin1(C,D,false,false),lin2(D,O,false,true);
    std::vector<at::Tensor> weights;
    for(auto p:conv->parameters())weights.push_back(p);for(auto p:rnn->parameters())weights.push_back(p);for(auto p:lin1->parameters())weights.push_back(p);for(auto p:lin2->parameters())weights.push_back(p);
    if(real){
        std::vector<std::string> names;for(int i=0;i<3;++i){names.push_back(std::to_string(i)+".conv.weight.tensor");names.push_back(std::to_string(i)+".conv.bias.tensor");}
        for(int i=4;i<11;++i)for(const auto&s:{"dn_weight_ih","dn_weight_hh","up_weight_ih","up_weight_hh","up_bias_ih","up_bias_hh"})names.push_back(std::to_string(i)+".rnn."+s+".tensor");
        names.push_back("11.linear.weight.tensor");names.push_back("12.linear.weight.tensor");
        for(size_t i=0;i<weights.size();++i){std::vector<at::Tensor> loaded;torch::load(loaded,(fs::path(model)/names[i]).string(),at::Device(at::kCPU));if(loaded.size()!=1||loaded[0].sizes()!=weights[i].sizes())throw std::runtime_error("stock weight shape mismatch");weights[i].copy_(loaded[0]);}
    }else{
        assign(*conv,11,.03125f);assign(*rnn,3,.125f);assign(*lin1,7,.0625f);assign(*lin2,9,.125f);
        for(size_t i=0;i<weights.size();++i)f.weights.push_back(store(f.name,"w"+std::to_string(i),weights[i],true));
    }
    auto input=values({N,1,L},13,.0625f);f.input=store(f.name,"input",input,true);
    auto x=input;
    for(size_t i=0;i<cv.size();++i){ConvStack one(std::vector<dorado::config::ConvParams>{cv[i]});one->parameters()[0].copy_(weights[2*i]);one->parameters()[1].copy_(weights[2*i+1]);x=one->forward(x).transpose(1,2);f.stages.push_back(store(f.name,"conv"+std::to_string(i),x,false));}
    x=conv->forward(input);f.stages.push_back(store(f.name,"conv_ntc",x,false));
    auto y=x;for(int i=0;i<7;++i){y=y.flip(1);y=rnn->layers_[i]->forward(y);f.stages.push_back(store(f.name,"rnn"+std::to_string(i),y,false));}
    x=rnn->forward(x);f.stages.push_back(store(f.name,"rnns",x,false));
    x=lin1->forward(x);f.stages.push_back(store(f.name,"linear1",x,false));
    auto raw=lin2->linear(x);f.stages.push_back(store(f.name,"linear2",raw,false));
    CRFModel full;auto&encoder=full->encoder;
    auto&convs=conv;auto&rnns=rnn;auto&linear1=lin1;auto&linear2=lin2;
    auto clamp1=Clamp(-5.0,5.0,false);
    using torch::nn::Sequential;
    // MODEL_SEQUENCE
    x=full->forward(input,nullptr);f.stages.push_back(store(f.name,"scores",x,false));return f;
}
}
int main(int argc,char**argv){
    try{
        if(argc!=2&&argc!=3)return 2;root=argv[1];fs::create_directories(root);
        dorado::utils::initialise_torch();dorado::utils::make_torch_deterministic();at::InferenceMode guard;
        std::vector<Fixture> fixtures;
        if(argc==3)fixtures.push_back(graphFixture(1,96,argv[2]));else{
            for(int N:{1,2})for(int T:{1,2,7})for(int layers:{1,2,3,7})for(bool reverse:{false,true})fixtures.push_back(rnnFixture(N,T,layers,reverse));
            for(int N:{1,2})for(int L:{6,12,42})fixtures.push_back(graphFixture(N,L,""));
        }
        writeManifest(fixtures);std::cout<<"wrote "<<fixtures.size()<<" source-linked CPU fixtures\n";return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
