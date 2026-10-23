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
