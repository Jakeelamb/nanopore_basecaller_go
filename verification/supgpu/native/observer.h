#pragma once
#include <ATen/ATen.h>
#include <initializer_list>
#include <string>
#include <vector>
extern "C" {
#include "koi.h"
}

namespace dorado::nn { struct TxEncoderImpl; }
namespace supgpu {
struct TensorArg {
    std::string name;
    at::Tensor tensor;
    std::vector<int> tags;
    int scale_tag;
    TensorArg(std::string name_, at::Tensor tensor_, std::vector<int> tags_ = {}, int scale_tag_ = 0)
        : name(std::move(name_)), tensor(std::move(tensor_)), tags(std::move(tags_)), scale_tag(scale_tag_) {}
};
struct DescriptorArg { std::string name; const KoiTensor *value; };
void emit(const std::string &, int, std::initializer_list<TensorArg>);
void encoder(dorado::nn::TxEncoderImpl *, const std::string &, std::initializer_list<TensorArg>, std::initializer_list<DescriptorArg> = {});
void cache(dorado::nn::TxEncoderImpl *, bool);
void check(int);
void reference_boundary(const std::string &,const at::Tensor &);
}
