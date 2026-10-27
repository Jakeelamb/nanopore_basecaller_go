#include <cuda.h>
#include <cupti.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include "sha256.inc"
#include "runtime_gate.inc"
void cupti_check(CUptiResult rc,const char*name){if(rc!=CUPTI_SUCCESS)throw std::runtime_error(std::string(name)+" CUPTI="+std::to_string(rc));}
#include "lineage_observer.inc"
extern char**environ;
namespace {
void driver(CUresult rc,const char*name){if(rc!=CUDA_SUCCESS)throw std::runtime_error(std::string(name)+" CUDA="+std::to_string(rc));}
std::string quote(const std::string&s){std::string to="\"";for(unsigned char c:s){if(c=='"' || c=='\\'){to+='\\';to+=c;}else if(c<32){const char*hex="0123456789abcdef";to+="\\u00";to+=hex[c>>4];to+=hex[c&15];}else to+=c;}return to+'"';}
void raw(const std::filesystem::path&p,const void*data,size_t bytes){std::ofstream f(p,std::ios::binary);if(!f)throw std::runtime_error("raw open failed");f.write(static_cast<const char*>(data),bytes);f.close();if(!f)throw std::runtime_error("raw close failed");}
void finish(std::ofstream&f){f.close();if(!f)throw std::runtime_error("observation close failed");}
struct Result {CUcontext primary=nullptr,current=nullptr;uint32_t uid=0;CUuuid uuid{};int major=0,minor=0,version=0;unsigned loads=0,lookups=0,unloads=0,releases=0;bool cleanup=false;std::string error;};
void retain(const std::filesystem::path&out,const std::string&input,Result&r){
    using namespace d01_lineage;Locked held;
    raw(out/"load-input.bin",state.input.data(),state.inputs?input_bytes:0);
    std::ofstream f(out/"observation.json");if(!f)throw std::runtime_error("observation open failed");
    f<<"{\n\"Schema\":\"nanopore-decoder-load-observation/v1\",\n\"PID\":"<<getpid()<<",\n\"Error\":"<<quote(r.error)<<",\n\"Context\":{\"Primary\":"<<reinterpret_cast<uintptr_t>(r.primary)<<",\"Current\":"<<reinterpret_cast<uintptr_t>(r.current)<<",\"UID\":"<<r.uid<<",\"UUID\":\"";
    const char*hex="0123456789abcdef";for(unsigned char c:r.uuid.bytes)f<<hex[c>>4]<<hex[c&15];
    f<<"\",\"Major\":"<<r.major<<",\"Minor\":"<<r.minor<<",\"DriverVersion\":"<<r.version<<",\"Environment\":[";bool comma=false;for(char**e=environ;*e;++e){if(comma)f<<',';comma=true;f<<quote(*e);}f<<"]},\n\"Input\":{\"Path\":"<<quote(input)<<",\"SHA256\":"<<quote(hex_sha(state.input.data(),state.inputs?input_bytes:0))<<"},\"InputCopy\":{\"Path\":"<<quote((out/"load-input.bin").string())<<",\"SHA256\":"<<quote(hex_sha(state.input.data(),state.inputs?input_bytes:0))<<"},\"InputBytes\":"<<(state.inputs?input_bytes:0)<<",\"InputAddress\":"<<reinterpret_cast<uintptr_t>(state.expected)<<",\"InputOperation\":"<<state.input_operation<<",\n\"API\":[";
    for(size_t i=0;i<state.apis;++i){const auto&a=state.api[i];if(i)f<<',';f<<"{\"Sequence\":"<<a.sequence<<",\"Thread\":"<<a.thread<<",\"Context\":"<<a.context<<",\"ContextUID\":"<<a.uid<<",\"Correlation\":"<<a.correlation<<",\"Operation\":"<<a.operation<<",\"Module\":"<<a.module<<",\"Function\":"<<a.function<<",\"Image\":"<<a.image<<",\"Options\":"<<a.options<<",\"OptionValues\":"<<a.option_values<<",\"OptionCount\":"<<a.option_count<<",\"Callback\":"<<a.callback<<",\"Site\":"<<a.site<<",\"Result\":"<<a.result<<",\"Name\":"<<quote(a.name)<<",\"Symbol\":"<<quote(a.symbol)<<'}';}
    f<<"],\n\"Modules\":[";for(size_t i=0;i<state.modules;++i){const auto&m=state.module[i];const auto name="module-"+std::to_string(i)+".cubin";raw(out/name,m.bytes.data(),m.retained);if(i)f<<',';f<<"{\"Sequence\":"<<m.sequence<<",\"Thread\":"<<m.thread<<",\"Context\":"<<m.context<<",\"ModuleID\":"<<m.id<<",\"Operation\":"<<m.operation<<",\"Bytes\":"<<m.observed<<",\"Payload\":{\"Path\":"<<quote((out/name).string())<<",\"SHA256\":"<<quote(hex_sha(m.bytes.data(),m.retained))<<"}}";}
    f<<"],\n\"CallbackError\":"<<state.error.load()<<",\"LaunchCallbacks\":"<<state.launches<<",\"LoadCalls\":"<<r.loads<<",\"LookupCalls\":"<<r.lookups<<",\"UnloadCalls\":"<<r.unloads<<",\"ReleaseCalls\":"<<r.releases<<",\"CleanupComplete\":"<<(r.cleanup?"true":"false")<<"}\n";finish(f);
}
}
int main(int argc,char**argv){
    Result r;CUmodule module=nullptr;bool retained=false,own_out=false,tried_retain=false;d01_lineage::Observer observer;std::vector<uint8_t> input;std::filesystem::path out;std::string input_path;int code=1;
    try {
        if(argc!=6 || std::string(argv[1])!="--decoder-load-lineage")throw std::runtime_error("fixed lineage argv required");
        const std::string architecture=argv[2],mode=argv[3];input_path=argv[4];out=std::filesystem::absolute(argv[5]);
        if((architecture!="sm_120" && architecture!="sm_120a") || (mode!="none" && mode!="four-original-functions"))throw std::runtime_error("unsupported lineage case");
        std::ifstream f(input_path,std::ios::binary);if(!f)throw std::runtime_error("original input open failed");input.assign(std::istreambuf_iterator<char>(f),{});if(f.bad() || input.size()!=d01_lineage::input_bytes)throw std::runtime_error("original input extent differs");f.close();if(!f)throw std::runtime_error("original input close failed");
        const std::string expected=architecture=="sm_120"?"292621a0efb7da3a9345797b469d73fb04ee04a88ad1aa0e52c94944214da938":"8c5d3abd2702a48d99a4c6f69e47de6235f5e280c468c38d74a910fda5acea40";
        if(hex_sha(input.data(),input.size())!=expected)throw std::runtime_error("original input digest differs");
        if(!std::filesystem::create_directory(out))throw std::runtime_error("fresh lineage output required");
        own_out=true;
        d01_gate::phase("pre-model");observer.start(input.data());driver(cuInit(0),"cuInit");driver(cuDriverGetVersion(&r.version),"driver version");CUdevice device;driver(cuDeviceGet(&device,0),"device0");driver(cuDeviceGetUuid(&r.uuid,device),"UUID");driver(cuDeviceGetAttribute(&r.major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device),"major");driver(cuDeviceGetAttribute(&r.minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device),"minor");if(r.major!=12 || r.minor!=0)throw std::runtime_error("fixed SM120 family required");
        driver(cuDevicePrimaryCtxRetain(&r.primary,device),"primary retain");retained=true;driver(cuCtxSetCurrent(r.primary),"primary current");driver(cuCtxGetCurrent(&r.current),"current context");cupti_check(cuptiGetContextId(r.current,&r.uid),"context UID");
        d01_gate::phase("pre-load");++r.loads;driver(cuModuleLoadDataEx(&module,input.data(),0,nullptr,nullptr),"original cubin load");d01_gate::phase("post-load");
        if(mode=="four-original-functions")for(const char*name:{"back_guide_step","beam_search_step","compute_posts_step","decode_step"}){CUfunction function=nullptr;++r.lookups;driver(cuModuleGetFunction(&function,module,name),"function lookup");if(!function)throw std::runtime_error("empty function handle");}
        d01_gate::phase("post-lookup");++r.unloads;driver(cuModuleUnload(module),"module unload");module=nullptr;++r.releases;driver(cuDevicePrimaryCtxRelease(device),"primary release");retained=false;r.cleanup=true;observer.stop();d01_gate::phase("post-cleanup");if(d01_lineage::state.error.load())throw std::runtime_error("retained callback failure");tried_retain=true;retain(out,input_path,r);code=0;
    } catch(const std::exception&e) {r.error=e.what();std::cerr<<r.error<<'\n';}
    if(code){bool failed=false;if(module){++r.unloads;if(cuModuleUnload(module)!=CUDA_SUCCESS)failed=true;else module=nullptr;}if(retained){++r.releases;if(cuDevicePrimaryCtxRelease(0)!=CUDA_SUCCESS)failed=true;else retained=false;}r.cleanup=!failed;if(observer.subscriber){try{observer.stop();}catch(...){failed=true;}}if(own_out && !tried_retain){tried_retain=true;try{retain(out,input_path,r);}catch(const std::exception&e){std::cerr<<e.what()<<'\n';failed=true;}}std::cerr.flush();if(failed)::_Exit(2);}
    return code;
}
