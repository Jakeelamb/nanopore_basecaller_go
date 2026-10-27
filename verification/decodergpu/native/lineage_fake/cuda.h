#pragma once
#include <cstdint>
#define CUDAAPI
using CUresult=int;using CUdevice=int;using CUcontext=void*;using CUmodule=void*;using CUfunction=void*;using CUjit_option=int;
struct CUuuid {char bytes[16];};
constexpr int CUDA_SUCCESS=0,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR=75,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR=76;
CUresult cuInit(unsigned);CUresult cuDriverGetVersion(int*);CUresult cuDeviceGet(CUdevice*,int);CUresult cuDeviceGetUuid(CUuuid*,CUdevice);CUresult cuDeviceGetAttribute(int*,int,CUdevice);CUresult cuDevicePrimaryCtxRetain(CUcontext*,CUdevice);CUresult cuCtxSetCurrent(CUcontext);CUresult cuCtxGetCurrent(CUcontext*);CUresult cuModuleLoadDataEx(CUmodule*,const void*,unsigned,CUjit_option*,void**);CUresult cuModuleGetFunction(CUfunction*,CUmodule,const char*);CUresult cuModuleUnload(CUmodule);CUresult cuDevicePrimaryCtxRelease(CUdevice);
