#include "cuda.h"
struct cuModuleLoadDataEx_params {CUmodule*module;const void*image;unsigned numOptions;CUjit_option*options;void**optionValues;};
struct cuModuleGetFunction_params {CUfunction*hfunc;CUmodule hmod;const char*name;};
struct cuModuleUnload_params {CUmodule hmod;};
