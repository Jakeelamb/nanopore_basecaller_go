#include "cuda.h"
#include "cupti.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {
CUpti_CallbackFunc callback=nullptr;bool driver_on=false,resource_on=false;CUcontext context=nullptr;uint32_t correlation=0;unsigned lookup=0;std::vector<char> module;
std::string mode(){const char*v=std::getenv("D01_LINEAGE_FAKE_CASE");return v?v:"positive";}
void api(unsigned id,int site,const char*name,void*params,int result,uint32_t corr){if(!driver_on || !callback)return;CUpti_CallbackData data{};data.callbackSite=site;data.functionName=name;data.functionParams=params;data.functionReturnValue=&result;data.context=context;data.contextUid=context?4:0;data.correlationId=corr;callback(nullptr,CUPTI_CB_DOMAIN_DRIVER_API,id,&data);}
void resource(){if(!resource_on || !callback)return;CUpti_ModuleResourceData moduleData{23,module.size(),module.data()};CUpti_ResourceData data{};data.context=context;data.resourceDescriptor=&moduleData;callback(nullptr,CUPTI_CB_DOMAIN_RESOURCE,CUPTI_CBID_RESOURCE_MODULE_LOADED,&data);}
struct Destructor {~Destructor(){if(const char*p=std::getenv("D01_LINEAGE_FAKE_DESTRUCTOR")){std::ofstream f(p);f<<"normal CPU owner teardown\n";}}} destructor;
}
CUptiResult cuptiSubscribe(CUpti_SubscriberHandle*out,CUpti_CallbackFunc cb,void*){callback=cb;*out=reinterpret_cast<void*>(1);return 0;}
CUptiResult cuptiEnableDomain(uint32_t on,CUpti_SubscriberHandle,CUpti_CallbackDomain){driver_on=on;return 0;}
CUptiResult cuptiEnableCallback(uint32_t on,CUpti_SubscriberHandle,CUpti_CallbackDomain,CUpti_CallbackId){resource_on=on;return 0;}
CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle){callback=nullptr;return 0;}
CUptiResult cuptiGetContextId(CUcontext,uint32_t*out){*out=4;return 0;}
CUresult cuInit(unsigned){const auto pid=std::to_string(getpid());::setenv("SUPGPU_TARGET_PID",pid.c_str(),1);if(mode()=="api-overflow")for(unsigned i=0;i<257;++i)api(1,0,"cuInit",nullptr,0,++correlation);const char*p=std::getenv("D01_LINEAGE_FAKE_MODULE");if(!p)return 1;std::ifstream f(p,std::ios::binary);module.assign(std::istreambuf_iterator<char>(f),{});if(mode()=="module-overflow")module.resize((1<<20)+1);return module.empty()?1:0;}
CUresult cuDriverGetVersion(int*out){*out=13040;return 0;}
CUresult cuDeviceGet(CUdevice*out,int){*out=0;return 0;}
CUresult cuDeviceGetUuid(CUuuid*out,CUdevice){std::memset(out->bytes,0x11,16);return 0;}
CUresult cuDeviceGetAttribute(int*out,int attribute,CUdevice){*out=attribute==75?12:0;return 0;}
CUresult cuDevicePrimaryCtxRetain(CUcontext*out,CUdevice){*out=reinterpret_cast<void*>(0x1234);return 0;}
CUresult cuCtxSetCurrent(CUcontext ctx){context=ctx;return 0;}
CUresult cuCtxGetCurrent(CUcontext*out){*out=context;return 0;}
CUresult cuModuleLoadDataEx(CUmodule*out,const void*input,unsigned count,CUjit_option*options,void**values){
 const uint32_t corr=++correlation;cuModuleLoadDataEx_params params{out,input,count,options,values};api(20,0,"cuModuleLoadDataEx",&params,0,corr);*out=reinterpret_cast<void*>(0x99);
 if(mode()=="launch")api(307,0,"cuLaunchKernel",nullptr,0,++correlation);
 if(mode()=="async-resource"){std::thread t(resource);t.join();}
 else if(mode()!="lookup-resource" && mode()!="missing-resource")resource();
 if(mode()=="copy-lifetime"){std::fill(module.begin(),module.end(),0);static_cast<uint8_t*>(const_cast<void*>(input))[0]^=1;}
 api(20,1,"cuModuleLoadDataEx",&params,0,corr);return 0;
}
CUresult cuModuleGetFunction(CUfunction*out,CUmodule hmod,const char*name){const auto corr=++correlation;cuModuleGetFunction_params params{out,hmod,name};api(23,0,"cuModuleGetFunction",&params,0,corr);*out=reinterpret_cast<void*>(0x200+(++lookup));if(mode()=="lookup-resource" && lookup==1)resource();api(23,1,"cuModuleGetFunction",&params,0,corr);return 0;}
CUresult cuModuleUnload(CUmodule hmod){const auto corr=++correlation;cuModuleUnload_params params{hmod};api(22,0,"cuModuleUnload",&params,0,corr);const int result=mode()=="cleanup-failure"?1:0;api(22,1,"cuModuleUnload",&params,result,corr);if(mode()=="write-close-failure"){const char*p=std::getenv("D01_LINEAGE_FAKE_OUT");if(p)std::filesystem::create_symlink("/dev/full",std::filesystem::path(p)/"observation.json");}return result;}
CUresult cuDevicePrimaryCtxRelease(CUdevice){const auto corr=++correlation;api(544,0,"cuDevicePrimaryCtxRelease_v2",nullptr,0,corr);context=nullptr;api(544,1,"cuDevicePrimaryCtxRelease_v2",nullptr,0,corr);return 0;}
