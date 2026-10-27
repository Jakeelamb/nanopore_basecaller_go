#pragma once
#include <ATen/core/TensorBody.h>

namespace h06_observer {
void caches(int layer,const at::Tensor& down_ih,const at::Tensor& down_hh,
            const at::Tensor& up,const at::Tensor& bias);
void before_step(int layer,int tick,int input,int hidden,int output,
                 const at::Tensor& gates,const at::Tensor& cell);
void after_step(int layer,int tick,int input,int hidden,int output,
                const at::Tensor& down,const at::Tensor& gates,
                const at::Tensor& cell,const at::Tensor& value);
void layer_output(int layer,const at::Tensor& inout);
}
