#include <unistd.h>
#include <iterator>
extern char** environ;

namespace d01_numeric {
std::string quote(const std::string&s){
    std::string out="\"";
    const char*hex="0123456789abcdef";
    for(unsigned char c:s){if(c=='"'||c=='\\'){out+='\\';out+=c;}else if(c<32){out+="\\u00";out+=hex[c>>4];out+=hex[c&15];}else out+=c;}
    return out+'"';
}
void retain(const std::filesystem::path&out,const std::string&stage){
    int device=0,driver=0,runtime=0;
    checked(cudaGetDevice(&device),"identity device");
    cudaDeviceProp props{};checked(cudaGetDeviceProperties(&props,device),"identity device properties");
    checked(cudaDriverGetVersion(&driver),"identity driver API");
    checked(cudaRuntimeGetVersion(&runtime),"identity runtime API");
    std::array<uintptr_t,d01_image::max_modules+d01_image::max_launches> contexts{};size_t count=0;
    {
        d01_image::Locked held;
        auto add=[&](uintptr_t value){if(!value)throw std::runtime_error("identity context absent");for(size_t i=0;i<count;++i)if(contexts[i]==value)return;if(count==contexts.size())throw std::runtime_error("identity context capacity");contexts[count++]=value;};
        for(size_t i=0;i<d01_image::state.modules;++i)add(d01_image::state.module[i].context);
        for(size_t i=0;i<d01_image::state.launches;++i)add(d01_image::state.launch[i].context);
    }
    std::sort(contexts.begin(),contexts.begin()+count);
    std::ifstream input("/proc/driver/nvidia/version",std::ios::binary);
    if(!input)throw std::runtime_error("identity kernel version open");
    std::string kernel((std::istreambuf_iterator<char>(input)),{});
    if(input.bad()||kernel.empty()||kernel.size()>16384)throw std::runtime_error("identity kernel version extent/read");
    input.close();if(!input)throw std::runtime_error("identity kernel version close");
    const char*hex="0123456789abcdef";std::string uuid;
    for(unsigned char c:props.uuid.bytes){uuid+=hex[c>>4];uuid+=hex[c&15];}
    std::ostringstream body;
    body<<"{\"Stage\":"<<quote(stage)<<",\"PID\":"<<getpid()<<",\"DeviceOrdinal\":"<<device<<",\"UUID\":"<<quote(uuid)<<",\"Major\":"<<props.major<<",\"Minor\":"<<props.minor<<",\"DriverAPI\":"<<driver<<",\"RuntimeAPI\":"<<runtime<<",\"KernelVersion\":"<<quote(kernel)<<",\"Environment\":[";
    size_t entries=0;
    for(char**e=environ;*e;++e){if(entries==64||std::strlen(*e)>4096)throw std::runtime_error("identity environment capacity");if(entries++)body<<',';body<<quote(*e);}
    body<<"],\"Contexts\":[";
    for(size_t i=0;i<count;++i){uint32_t uid=0,ordinal=0;auto context=reinterpret_cast<CUcontext>(contexts[i]);d01_image::checked_cupti(cuptiGetContextId(context,&uid),"identity context UID");d01_image::checked_cupti(cuptiGetDeviceId(context,&ordinal),"identity context device");if(!uid||ordinal!=static_cast<uint32_t>(device))throw std::runtime_error("identity context device/UID differs");if(i)body<<',';body<<"{\"Pointer\":"<<contexts[i]<<",\"UID\":"<<uid<<",\"DeviceOrdinal\":"<<ordinal<<'}';}
    body<<"]}\n";const auto bytes=body.str();write(out/("identity-"+stage+".json"),bytes.data(),bytes.size());
}
}
