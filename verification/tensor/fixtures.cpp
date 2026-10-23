#include <ATen/ATen.h>
#include "runtime.h"
#include <torch/serialize.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
void write_tensor(std::ostream& out,const char* name,const at::Tensor& value) {
    auto t=value.contiguous();out<<'"'<<name<<"\":{\"shape\":[";
    for(int64_t i=0;i<t.dim();++i){if(i)out<<',';out<<t.size(i);}
    out<<"],\"data_bits\":[";
    for(int64_t i=0;i<t.numel();++i){uint32_t bits;std::memcpy(&bits,t.data_ptr<float>()+i,sizeof(bits));if(i)out<<',';out<<bits;}
    out<<"]}";
}
int main(int argc,char**argv) {
    if(argc!=2 && argc!=3) return 2;
    try {
        initialize_tensor_runtime();
        c10::InferenceMode guard;
        std::string root=argv[1];
        auto f=at::arange(6,at::TensorOptions().dtype(at::kFloat)).reshape({2,3});
        torch::save(std::vector<at::Tensor>{f},root+"/float32.tensor");
        torch::save(std::vector<at::Tensor>{f.to(at::kLong)},root+"/int64.tensor");
        torch::save(std::vector<at::Tensor>{f,f},root+"/multiple.tensor");
        torch::save(std::vector<at::Tensor>{},root+"/empty.tensor");
        {
            auto input=at::tensor({1000.f,1001.f,999.f,-1000.f,-1000.f,-1000.f}).reshape({2,3});
            auto indices=at::tensor({2,0,2},at::TensorOptions().dtype(at::kLong));
            std::ofstream out(root+"/primitive-reference.json");out<<'{';
            write_tensor(out,"logsumexp",at::logsumexp(input,{1},false));out<<',';
            write_tensor(out,"logsumexp_keepdim",at::logsumexp(input,{1},true));out<<',';
            write_tensor(out,"softmax",at::softmax(input,1));out<<',';
            write_tensor(out,"index_select",at::index_select(f,1,indices));out<<"}\n";
            if(!out)return 4;
        }
        {
            auto tied=at::tensor({2,1,1,0,2,0},at::TensorOptions().dtype(at::kLong));
            auto order=at::argsort(tied,0,false).contiguous();
            std::ofstream out(root+"/argsort-reference.json");out<<"[";
            for(int64_t i=0;i<order.numel();++i){if(i)out<<',';out<<order.data_ptr<int64_t>()[i];}
            out<<"]\n";if(!out)return 4;
        }
        {
            auto x=at::tensor({0.125f,0.5f,1.25f,4.f,9.f,16.f}).reshape({2,3});
            auto qkv=(at::arange(3*2*2*9*4,at::TensorOptions().dtype(at::kFloat))*0.013f-0.5f).reshape({3,2,2,9,4});
            auto q=qkv[0].narrow(-2,2,5),k=qkv[1].narrow(-2,0,7),v=qkv[2].narrow(-2,0,7);
            auto full_mask=at::ones({9,9},at::TensorOptions().dtype(at::kBool)).triu_(-2).tril_(3);
            auto mask=full_mask.narrow(0,2,5).narrow(1,0,7);
            std::ofstream out(root+"/attention-reference.json");out<<'{';
            write_tensor(out,"input",x);out<<',';
            write_tensor(out,"qkv",qkv);out<<',';
            write_tensor(out,"mean",x.mean(-1,false));out<<',';
            write_tensor(out,"mean_keepdim",x.mean(-1,true));out<<',';
            write_tensor(out,"mean_transpose",x.transpose(0,1).mean(1,true));out<<',';
            write_tensor(out,"rsqrt",at::rsqrt(x));out<<',';
            write_tensor(out,"reciprocal",x.reciprocal());out<<',';
            write_tensor(out,"sin",at::sin(x));out<<',';
            write_tensor(out,"cos",at::cos(x));out<<',';
            write_tensor(out,"sub",x-x.narrow(0,0,1));out<<',';
            write_tensor(out,"square",x.square());out<<',';
            write_tensor(out,"masked",at::scaled_dot_product_attention(q,k,v,mask));out<<',';
            write_tensor(out,"masked_contiguous",at::scaled_dot_product_attention(q,k,v,mask.contiguous()));out<<',';
            write_tensor(out,"broadcast",at::scaled_dot_product_attention(q,k,v,mask.reshape({1,1,5,7})));out<<',';
            write_tensor(out,"unmasked",at::scaled_dot_product_attention(q,k,v));out<<',';
            write_tensor(out,"blocked",at::scaled_dot_product_attention(q,k,v,at::zeros({5,7},at::TensorOptions().dtype(at::kBool))));
            out<<"}\n";if(!out)return 4;
            // Dorado slices a full T by T mask with row stride T before calling SDPA.
            for(const int64_t size : {173,271}) {
                auto a=(at::arange(3*2*size*64,at::TensorOptions().dtype(at::kFloat))*0.00013f-0.5f).reshape({3,1,2,size,64});
                int64_t qb=size==173?80:136,qe=qb+24,kvb=std::max<int64_t>(0,qb-128),kve=std::min<int64_t>(size,qe+127);
                auto m=at::ones({size,size},at::TensorOptions().dtype(at::kBool)).triu_(-127).tril_(128).narrow(0,qb,qe-qb).narrow(1,kvb,kve-kvb);
                auto query=a[0].narrow(-2,qb,qe-qb),key=a[1].narrow(-2,kvb,kve-kvb),value=a[2].narrow(-2,kvb,kve-kvb);
                auto source=at::scaled_dot_product_attention(query,key,value,m).contiguous();
                auto copied=at::scaled_dot_product_attention(query,key,value,m.contiguous()).contiguous();
                if(std::memcmp(source.data_ptr<float>(),copied.data_ptr<float>(),source.numel()*sizeof(float))!=0) throw std::runtime_error("source mask stride changes CPU SDPA bits");
                std::cout<<"source mask row stride "<<size<<" matches copied mask bitwise, heads=2 head_dim=64 query="<<qe-qb<<" keys="<<kve-kvb<<'\n';
            }
        }
        if(argc==3) {
            std::vector<at::Tensor> weights;
            torch::load(weights,argv[2],at::Device(at::kCPU));
            if(weights.size()!=1 || weights[0].scalar_type()!=at::kFloat) return 3;
            auto w=weights[0].contiguous();
            std::ofstream out(root+"/real-weight.json");
            out<<"{\"shape\":[";
            for(int64_t i=0;i<w.dim();++i) {if(i) out<<',';out<<w.size(i);}
            out<<"],\"data_bits\":[";
            for(int64_t i=0;i<w.numel();++i){uint32_t bits;std::memcpy(&bits,w.data_ptr<float>()+i,sizeof(bits));if(i) out<<',';out<<bits;}
            out<<"]}\n";
            if(!out) return 4;
        }
        std::cout<<"created independent Torch vector archives\n";
        return 0;
    } catch(const std::exception& ex) {std::cerr<<ex.what()<<'\n';return 1;}
}
