#pragma once
#include <cstddef>
using cudaError_t = int;
using cudaStream_t = void*;
constexpr cudaError_t cudaSuccess = 0;
constexpr unsigned cudaStreamNonBlocking = 1;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
struct cudaDeviceProp { char name[256]; int major; int minor; };
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaGetDevice(int*);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*,int);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*,unsigned);
cudaError_t cudaMalloc(void**,std::size_t);
cudaError_t cudaMemcpyAsync(void*,const void*,std::size_t,cudaMemcpyKind,cudaStream_t);
cudaError_t cudaMemsetAsync(void*,int,std::size_t,cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaGetLastError();
cudaError_t cudaFree(void*);
cudaError_t cudaStreamDestroy(cudaStream_t);
