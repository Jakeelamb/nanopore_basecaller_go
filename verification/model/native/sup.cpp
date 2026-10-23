// Selected original CPU functions are inserted after whole-file hash checks.
// Module declarations adapt storage and defaults, without claiming Dorado's ABI.
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <torch/nn.h>
#include <torch/serialize.h>
#include <torch/torch.h>
#include "config/common.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <vector>
#include <cmath>

namespace spdlog {template<class... T> void error(const char*,T&&...) {}}
namespace dorado::config {
// CONFIG_DEFINITIONS
}
namespace dorado::utils {
struct ScopedProfileRange { ScopedProfileRange(const char*,int) {} };
template<class T> T get_dev_opt(const char*,T value){return value;}
// MATH_FUNCTIONS
// INITIALIZATION_FUNCTIONS
}
namespace dorado::nn {
using namespace config;
using namespace torch::nn;
using Slice = torch::indexing::Slice;
struct AuxiliaryData;
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
struct GatedMLPImpl : torch::nn::Module {
    GatedMLPImpl(int,int);at::Tensor forward(const at::Tensor&);
    int in_features,hidden_features;torch::nn::Linear fc1{nullptr},fc2{nullptr};
};
TORCH_MODULE(GatedMLP);
struct RotaryEmbeddingImpl : torch::nn::Module {
    RotaryEmbeddingImpl(int,float,int,const at::TensorOptions&);
    at::Tensor forward(at::Tensor&);at::Tensor get_inv_freqs()const;
    void assert_forward_dims(const at::Tensor&)const;
    const int64_t dim,max_seq_len;const float theta;const at::TensorOptions options;
};
TORCH_MODULE(RotaryEmbedding);
using MaskKey=std::pair<int64_t,torch::Device>;
struct MaskKeyHash {
    size_t operator()(const MaskKey&key)const{return std::hash<int64_t>{}(key.first)^(std::hash<torch::Device>{}(key.second)<<1);}
};
struct MultiHeadAttentionImpl : torch::nn::Module {
    MultiHeadAttentionImpl(int,int,bool,bool,const std::pair<int,int>&,float,int,const at::TensorOptions&);
    at::Tensor forward(at::Tensor);at::Tensor get_attn_window_mask(int64_t);
    at::Tensor build_attn_window_mask(int64_t)const;
    at::Tensor capture_attention(const at::Tensor&);
    const int d_model,nhead,head_dim,num_splits;const std::pair<int,int>attn_window;
    const float theta;const int max_seq_len;const at::TensorOptions options;
    std::unordered_map<MaskKey,at::Tensor,MaskKeyHash>mask_cache;
    torch::nn::Linear wqkv{nullptr},out_proj{nullptr};RotaryEmbedding rotary_emb{nullptr};
};
TORCH_MODULE(MultiHeadAttention);
struct RMSNormImpl : torch::nn::Module {
    explicit RMSNormImpl(int);at::Tensor forward(at::Tensor);
    at::Tensor weight;const int hidden_size;const float eps{1e-5f};
};
TORCH_MODULE(RMSNorm);
struct TxEncoderImpl : torch::nn::Module {
    TxEncoderImpl(const config::TxEncoderParams&,const at::TensorOptions&);
    at::Tensor forward(at::Tensor);
    config::TxEncoderParams params;
    MultiHeadAttention self_attn{nullptr};GatedMLP ff{nullptr};RMSNorm norm1{nullptr},norm2{nullptr};
};
TORCH_MODULE(TxEncoder);
struct TxEncoderStackImpl : torch::nn::Module {
    TxEncoderStackImpl(const config::TxEncoderParams&,const at::TensorOptions&);
    at::Tensor forward(const at::Tensor&,AuxiliaryData*);
    bool use_i8=false;torch::nn::Sequential stack{nullptr};std::vector<TxEncoder>layer_vec;
};
TORCH_MODULE(TxEncoderStack);
struct LinearUpsampleImpl : torch::nn::Module {
    LinearUpsampleImpl(int,int);explicit LinearUpsampleImpl(const config::LinearUpsampleParams&);
    at::Tensor forward(const at::Tensor&);const int scale_factor;torch::nn::Linear linear{nullptr};
};
TORCH_MODULE(LinearUpsample);
struct LinearScaledCRFImpl : torch::nn::Module {
    explicit LinearScaledCRFImpl(const config::CRFEncoderParams&);at::Tensor forward(const at::Tensor&);
    bool scale_applied=false;torch::nn::Linear linear{nullptr};config::CRFEncoderParams m_params;
};
TORCH_MODULE(LinearScaledCRF);
// SOURCE_FUNCTIONS
at::Tensor MultiHeadAttentionImpl::capture_attention(const at::Tensor&qkv) {
    const int64_t N=qkv.size(1),T=qkv.size(3),C=d_model;
    auto x=at::empty({N,T,C},qkv.options());at::Tensor attn_output_ntc;
    // ATTENTION_CAPTURE
    return attn_output_ntc;
}
}
namespace dorado::basecall::model {
struct TxModelImpl : torch::nn::Module {
    nn::ConvStack convs{nullptr};nn::TxEncoderStack tx_encoder{nullptr};
    nn::LinearUpsample tx_decoder{nullptr};nn::LinearScaledCRF crf{nullptr};
    at::Tensor forward(const at::Tensor&,nn::AuxiliaryData*);
};
TORCH_MODULE(TxModel);
// MODEL_FORWARD
}
namespace {
using namespace dorado::nn;
namespace fs=std::filesystem;
struct Ref {std::string name,path;std::vector<int64_t>shape;};
struct Fixture {
    std::string name,kind;int channels=8,heads=2,hidden=12,layers=1,upper=127,lower=128,max_sequence=4096;
    float alpha=2.4494897f;
    Ref input;std::vector<Ref>weights,stages;
};
fs::path root;
auto options=at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
at::Tensor values(at::IntArrayRef shape,int seed,float scale) {
    auto t=at::empty(shape,options);auto*p=t.data_ptr<float>();
    for(int64_t i=0;i<t.numel();++i)p[i]=float(((i*7+seed*11)%31)-15)*scale;
    return t;
}
Ref store(const std::string&prefix,const std::string&name,const at::Tensor&t,bool archive) {
    auto x=t.contiguous();std::string path=prefix+"/"+name;fs::create_directories((root/path).parent_path());
    std::ofstream out(root/(path+".bin"),std::ios::binary);
    out.write(reinterpret_cast<const char*>(x.data_ptr<float>()),x.numel()*sizeof(float));
    if(!out)throw std::runtime_error("write fixture failed");
    if(archive)torch::save(std::vector<at::Tensor>{x},(root/(path+".tensor")).string());
    return{name,path,x.sizes().vec()};
}
void stage(Fixture&f,const std::string&name,const at::Tensor&x){f.stages.push_back(store(f.name,name,x,false));}
void assign(torch::nn::Module&m,int seed,float scale) {
    int i=0;for(auto p:m.parameters())p.copy_(values(p.sizes(),seed+i++,scale));
}
void weights(Fixture&f,torch::nn::Module&m) {
    for(auto p:m.parameters())f.weights.push_back(store(f.name,"w"+std::to_string(f.weights.size()),p,true));
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
    for(size_t i=0;i<fixtures.size();++i){if(i)out<<',';const auto&f=fixtures[i];
        out<<"{\"name\":\""<<f.name<<"\",\"kind\":\""<<f.kind<<"\",\"channels\":"<<f.channels<<",\"heads\":"<<f.heads<<",\"hidden\":"<<f.hidden<<",\"layers\":"<<f.layers<<",\"upper\":"<<f.upper<<",\"lower\":"<<f.lower<<",\"max_sequence\":"<<f.max_sequence<<",\"alpha\":"<<std::setprecision(9)<<f.alpha<<",\"input\":";
        refJSON(out,f.input);out<<",\"weights\":";listJSON(out,f.weights);out<<",\"stages\":";listJSON(out,f.stages);out<<'}';
    }out<<"]\n";if(!out)throw std::runtime_error("manifest write failed");
}
dorado::config::TxEncoderParams params(const Fixture&f) {
    dorado::config::TxEncoderParams p;p.d_model=f.channels;p.nhead=f.heads;p.dim_feedforward=f.hidden;
    p.depth=f.layers;p.attn_window={f.upper,f.lower};p.deepnorm_alpha=f.alpha;p.max_seq_len=f.max_sequence;return p;
}
Fixture ropeFixture(int N,int T,int H,int D) {
    Fixture f;f.name="rope_n"+std::to_string(N)+"_t"+std::to_string(T)+"_h"+std::to_string(H)+"_d"+std::to_string(D);f.kind="rope";f.channels=D;f.heads=H;
    RotaryEmbedding rope(D,10000.f,f.max_sequence,options);
    auto x=values({N,T,3,H,D},5,.03125f);f.input=store(f.name,"input",x,true);
    stage(f,"inv_freq",rope->get_inv_freqs());stage(f,"cos",rope->named_buffers()["cos_freqs"]);stage(f,"sin",rope->named_buffers()["sin_freqs"]);
    stage(f,"rope",rope->forward(x));return f;
}
Fixture rmsFixture() {
    Fixture f;f.name="rms_n2_t7";f.kind="rms";RMSNorm norm(8);assign(*norm,3,.125f);
    auto x=values({2,7,8},5,.0009765625f);f.input=store(f.name,"input",x,true);weights(f,*norm);
    stage(f,"rms",norm->forward(x.clone()));return f;
}
Fixture mlpFixture() {
    Fixture f;f.name="mlp_n2_t7";f.kind="mlp";GatedMLP mlp(8,12);assign(*mlp,3,.0625f);
    auto x=values({2,7,8},5,.25f);f.input=store(f.name,"input",x,true);weights(f,*mlp);
    stage(f,"mlp",mlp->forward(x));return f;
}
Fixture upsampleFixture() {
    Fixture f;f.name="upsample_n2_t3";f.kind="upsample";LinearUpsample up(8,2);assign(*up,3,.125f);
    auto x=values({2,3,8},5,.25f);f.input=store(f.name,"input",x,true);weights(f,*up);
    stage(f,"upsample",up->forward(x));return f;
}
Fixture scaledFixture() {
    Fixture f;f.name="scaled_crf_twice";f.kind="scaled_crf";
    LinearScaledCRF crf(dorado::config::CRFEncoderParams{8,2,2,5.f,2.f,true,{}});assign(*crf,3,.125f);
    auto x=values({2,3,8},5,.25f);f.input=store(f.name,"input",x,true);weights(f,*crf);
    stage(f,"scores",crf->forward(x));stage(f,"scaled_weight",crf->linear->weight);stage(f,"scores_repeat",crf->forward(x));return f;
}
at::Tensor encoderCapture(Fixture&f,TxEncoder&encoder,const at::Tensor&input,const std::string&prefix) {
    auto&attn=encoder->self_attn;auto&ff=encoder->ff;
    auto qkv=attn->wqkv(input).view({input.size(0),input.size(1),3,f.heads,f.channels/f.heads});stage(f,prefix+"qkv",qkv);
    auto rotated=attn->rotary_emb->forward(qkv);stage(f,prefix+"rope",rotated);
    stage(f,prefix+"attention",attn->capture_attention(rotated));
    auto projection=attn->forward(input);stage(f,prefix+"projection",projection);
    auto alpha=encoder->named_buffers()["deepnorm_alpha"];
    auto norm1=encoder->norm1(projection+input*alpha);stage(f,prefix+"norm1",norm1);
    stage(f,prefix+"fc1",ff->fc1(norm1));
    auto chunks=ff->fc1(norm1).chunk(2,-1);stage(f,prefix+"gate",torch::nn::functional::silu(chunks[1]).mul_(chunks[0]));
    auto feedforward=ff->forward(norm1);stage(f,prefix+"ff",feedforward);
    auto out=encoder->forward(input);stage(f,prefix+"norm2",out);return out;
}
Fixture encoderFixture(int N,int T,int depth) {
    Fixture f;f.name="encoder_n"+std::to_string(N)+"_t"+std::to_string(T)+"_l"+std::to_string(depth);f.kind="encoder";f.layers=depth;
    TxEncoderStack encoder(params(f),options);assign(*encoder,3,.03125f);
    auto x=values({N,T,f.channels},5,.25f);f.input=store(f.name,"input",x,true);weights(f,*encoder);
    auto out=x;for(int i=0;i<depth;++i)out=encoderCapture(f,encoder->layer_vec[i],out,"encoder"+std::to_string(i)+".");
    stage(f,"stack",encoder->forward(x,nullptr));
    if(T>256){
        auto first=encoder->layer_vec[0]->self_attn;
        auto qkv=first->wqkv(x).view({N,T,3,f.heads,f.channels/f.heads});auto rot=first->rotary_emb(qkv);
        auto full=at::scaled_dot_product_attention(rot[0],rot[1],rot[2],first->get_attn_window_mask(T));
        auto split=first->capture_attention(rot).view({N,T,f.heads,f.channels/f.heads}).transpose(1,2);
        if(at::equal(full,split))throw std::runtime_error("long attention fixture fails to discriminate whole-mask replacement");
    }
    return f;
}
std::vector<std::string>stockNames() {
    std::vector<std::string>names;
    for(int i=0;i<5;++i)for(const auto*s:{"weight","bias"})names.push_back("conv."+std::to_string(i)+".conv."+s+".tensor");
    for(int i=0;i<18;++i)for(const auto*s:{"self_attn.Wqkv.weight","self_attn.out_proj.weight","self_attn.out_proj.bias","ff.fc1.weight","ff.fc2.weight","norm1.weight","norm2.weight"})names.push_back("transformer_encoder."+std::to_string(i)+"."+s+".tensor");
    for(const auto*s:{"upsample.linear.weight.tensor","upsample.linear.bias.tensor","crf.linear.weight.tensor"})names.push_back(s);
    return names;
}
Fixture graphFixture(int N,int L,const std::string&model) {
    const bool real=!model.empty();Fixture f;f.name=real?"stock_sup_n1_l768":"sup_graph_n"+std::to_string(N)+"_l"+std::to_string(L);f.kind="sup_graph";
    f.channels=real?512:8;f.heads=real?8:2;f.hidden=real?2048:12;f.layers=real?18:2;
    const int A=real?64:4,B=real?128:4;
    std::vector<dorado::config::ConvParams>cv{{1,A,5,1,dorado::config::Activation::SWISH,0},{A,A,5,1,dorado::config::Activation::SWISH,0},{A,B,9,3,dorado::config::Activation::SWISH,0},{B,B,9,2,dorado::config::Activation::SWISH,0},{B,f.channels,5,2,dorado::config::Activation::SWISH,0}};
    ConvStack conv(cv);TxEncoderStack enc(params(f),options);LinearUpsample up(f.channels,2);
    LinearScaledCRF crf(dorado::config::CRFEncoderParams{f.channels,real?4:2,real?5:2,5.f,2.f,true,{}});
    std::vector<at::Tensor>ws;
    for(auto*m:std::vector<torch::nn::Module*>{conv.get(),enc.get(),up.get(),crf.get()})for(auto p:m->parameters())ws.push_back(p);
    if(real){auto names=stockNames();if(names.size()!=ws.size())throw std::runtime_error("stock weight count mismatch");
        for(size_t i=0;i<ws.size();++i){std::vector<at::Tensor>loaded;torch::load(loaded,(fs::path(model)/names[i]).string(),at::Device(at::kCPU));if(loaded.size()!=1||loaded[0].sizes()!=ws[i].sizes())throw std::runtime_error("stock weight shape mismatch");ws[i].copy_(loaded[0]);}
    }else{assign(*conv,7,.03125f);assign(*enc,3,.03125f);assign(*up,9,.0625f);assign(*crf,11,.125f);for(auto w:ws)f.weights.push_back(store(f.name,"w"+std::to_string(f.weights.size()),w,true));}
    auto input=values({N,1,L},13,.0625f);f.input=store(f.name,"input",input,true);
    auto x=input;
    for(size_t i=0;i<cv.size();++i){ConvStack one(std::vector<dorado::config::ConvParams>{cv[i]});one->parameters()[0].copy_(ws[2*i]);one->parameters()[1].copy_(ws[2*i+1]);x=one->forward(x).transpose(1,2);stage(f,"conv"+std::to_string(i),x);}
    x=conv->forward(input);stage(f,"conv_ntc",x);
    for(int i=0;i<f.layers;++i)x=encoderCapture(f,enc->layer_vec[i],x,"encoder"+std::to_string(i)+".");
    x=up->forward(x);stage(f,"upsample",x);
    dorado::basecall::model::TxModel full;full->convs=conv;full->tx_encoder=enc;full->tx_decoder=up;full->crf=crf;
    stage(f,"scores",full->forward(input,nullptr));stage(f,"scores_repeat",full->forward(input,nullptr));return f;
}
}
int main(int argc,char**argv) {
    try {
        if(argc!=2&&argc!=3)return 2;root=argv[1];fs::create_directories(root);
        dorado::utils::initialise_torch();dorado::utils::make_torch_deterministic();at::InferenceMode guard;
        std::vector<Fixture>fixtures;
        if(argc==3)fixtures.push_back(graphFixture(1,768,argv[2]));
        else {
            fixtures.push_back(ropeFixture(2,7,2,64));fixtures.push_back(ropeFixture(1,260,2,4));
            fixtures.push_back(rmsFixture());fixtures.push_back(mlpFixture());fixtures.push_back(upsampleFixture());fixtures.push_back(scaledFixture());
            for(auto shape:std::vector<std::pair<int,int>>{{1,1},{2,7},{1,260}})fixtures.push_back(encoderFixture(shape.first,shape.second,shape.second==260?1:2));
            fixtures.push_back(graphFixture(1,24,""));fixtures.push_back(graphFixture(2,84,""));
        }
        writeManifest(fixtures);std::cout<<"wrote "<<fixtures.size()<<" source-linked SUP CPU fixtures\n";return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
