#include <cupti.h>
#include <cupti_driver_cbid.h>
#include <array>
#include <algorithm>
#include <atomic>

namespace d01_image {
constexpr size_t max_modules=16,max_cubin=1024*1024,max_functions=512,max_launches=512;
constexpr size_t activity_buffers=8,activity_bytes=64*1024,name_bytes=256;
struct Module {uintptr_t context=0;uint32_t id=0,stage=0;size_t observed=0,retained=0;std::array<uint8_t,max_cubin> bytes{};};
struct Function {uint32_t id=0,context=0,module=0,index=0,stage=0;char name[name_bytes]{};};
struct Launch {uintptr_t context=0;uint32_t uid=0,correlation=0,cbid=0,site=0,stage=0;int32_t result=0;char name[name_bytes]{};};
struct ActivityBuffer {alignas(8) std::array<uint8_t,activity_bytes> bytes{};bool borrowed=false;};
struct State {
    std::atomic_flag lock=ATOMIC_FLAG_INIT;
    std::atomic<unsigned> failure{0},stage{0};
    size_t modules=0,functions=0,launches=0;
    std::array<Module,max_modules> module;
    std::array<Function,max_functions> function;
    std::array<Launch,max_launches> launch;
    std::array<ActivityBuffer,activity_buffers> activity;
};
State state;
struct Locked {Locked(){while(state.lock.test_and_set(std::memory_order_acquire)){}}~Locked(){state.lock.clear(std::memory_order_release);}};
void fail(unsigned code) noexcept {unsigned zero=0;state.failure.compare_exchange_strong(zero,code);}
bool name(char*to,const char*from) noexcept {
    if(!from){fail(1);return false;}
    for(size_t i=0;i<name_bytes;++i){to[i]=from[i];if(!from[i])return true;}
    to[name_bytes-1]=0;fail(2);return false;
}
void CUPTIAPI callback(void*,CUpti_CallbackDomain domain,CUpti_CallbackId id,const void*data) noexcept {
    try {
        if(!data){fail(3);return;}
        Locked held;
        if(domain==CUPTI_CB_DOMAIN_RESOURCE && id==CUPTI_CBID_RESOURCE_MODULE_LOADED){
            const auto&r=*static_cast<const CUpti_ResourceData*>(data);
            const auto*m=static_cast<const CUpti_ModuleResourceData*>(r.resourceDescriptor);
            if(!m || !m->pCubin || !m->cubinSize){fail(4);return;}
            if(state.modules==max_modules){fail(5);return;}
            auto&to=state.module[state.modules++];to.context=reinterpret_cast<uintptr_t>(r.context);to.id=m->moduleId;to.stage=state.stage.load();to.observed=m->cubinSize;to.retained=std::min(m->cubinSize,max_cubin);
            std::memcpy(to.bytes.data(),m->pCubin,to.retained);
            if(to.observed!=to.retained)fail(6);
        } else if(domain==CUPTI_CB_DOMAIN_DRIVER_API){
            const auto&r=*static_cast<const CUpti_CallbackData*>(data);
            if(state.launches==max_launches){fail(7);return;}
            auto&to=state.launch[state.launches++];to.context=reinterpret_cast<uintptr_t>(r.context);to.uid=r.contextUid;to.correlation=r.correlationId;to.cbid=id;to.site=r.callbackSite;to.stage=state.stage.load();name(to.name,r.symbolName);
            if(r.callbackSite==CUPTI_API_EXIT){if(!r.functionReturnValue)fail(8);else std::memcpy(&to.result,r.functionReturnValue,sizeof(to.result));}
            else if(r.callbackSite!=CUPTI_API_ENTER)fail(9);
        } else fail(10);
    } catch(...) {fail(11);}
}
void CUPTIAPI requested(uint8_t**buffer,size_t*size,size_t*records) noexcept {
    try {
        *buffer=nullptr;*size=0;*records=0;
        Locked held;
        for(auto&b:state.activity)if(!b.borrowed){b.borrowed=true;*buffer=b.bytes.data();*size=b.bytes.size();return;}
        fail(12);
    }catch(...){fail(13);}
}
void CUPTIAPI completed(CUcontext,uint32_t,uint8_t*buffer,size_t size,size_t valid) noexcept {
    try {
        ActivityBuffer*owner=nullptr;
        {Locked held;for(auto&b:state.activity)if(b.bytes.data()==buffer && b.borrowed)owner=&b;}
        if(!owner || size!=activity_bytes || valid>size){fail(14);return;}
        CUpti_Activity*record=nullptr;
        for(;;){
            const auto rc=cuptiActivityGetNextRecord(buffer,valid,&record);
            if(rc==CUPTI_ERROR_MAX_LIMIT_REACHED)break;
            if(rc!=CUPTI_SUCCESS || !record){fail(15);break;}
            if(record->kind!=CUPTI_ACTIVITY_KIND_FUNCTION){fail(16);continue;}
            const auto&r=*reinterpret_cast<const CUpti_ActivityFunction*>(record);
            Locked held;
            if(state.functions==max_functions){fail(17);break;}
            auto&to=state.function[state.functions++];to.id=r.id;to.context=r.contextId;to.module=r.moduleId;to.index=r.functionIndex;to.stage=state.stage.load();name(to.name,r.name);
        }
        {Locked held;owner->borrowed=false;}
    } catch(...) {fail(18);}
}
constexpr CUpti_CallbackId launch_ids[]={CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz,CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx,CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz};
void checked_cupti(CUptiResult rc,const char*op){if(rc!=CUPTI_SUCCESS)throw std::runtime_error(std::string(op)+" status="+std::to_string(static_cast<int>(rc)));}
struct Observer {
    CUpti_SubscriberHandle subscriber=nullptr;
    void start(){
        checked_cupti(cuptiSubscribe(&subscriber,callback,nullptr),"CUPTI subscribe");
        checked_cupti(cuptiEnableCallback(1,subscriber,CUPTI_CB_DOMAIN_RESOURCE,CUPTI_CBID_RESOURCE_MODULE_LOADED),"CUPTI module callback");
        for(auto id:launch_ids)checked_cupti(cuptiEnableCallback(1,subscriber,CUPTI_CB_DOMAIN_DRIVER_API,id),"CUPTI launch callback");
        checked_cupti(cuptiActivityRegisterCallbacks(requested,completed),"CUPTI activity buffers");
        checked_cupti(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_FUNCTION),"CUPTI function records only");
    }
    void stage(unsigned n){state.stage.store(n);}
    void retain(const std::filesystem::path&out,const std::string&label){
        const auto dir=out/("images-"+label);if(!std::filesystem::create_directory(dir))throw std::runtime_error("fresh image observation directory required");
        Locked held;
        std::ofstream modules(dir/"modules.tsv"),functions(dir/"functions.tsv"),launches(dir/"launches.tsv"),status(dir/"status.txt");
        for(size_t i=0;i<state.modules;++i){const auto&m=state.module[i];const auto file="module-"+std::to_string(i)+".cubin";write(dir/file,m.bytes.data(),m.retained);modules<<i<<'\t'<<m.context<<'\t'<<m.id<<'\t'<<m.stage<<'\t'<<m.observed<<'\t'<<m.retained<<'\t'<<hex_sha(m.bytes.data(),m.retained)<<'\t'<<file<<'\n';}
        for(size_t i=0;i<state.functions;++i){const auto&f=state.function[i];functions<<f.id<<'\t'<<f.context<<'\t'<<f.module<<'\t'<<f.index<<'\t'<<f.stage<<'\t'<<f.name<<'\n';}
        for(size_t i=0;i<state.launches;++i){const auto&l=state.launch[i];launches<<l.context<<'\t'<<l.uid<<'\t'<<l.correlation<<'\t'<<l.cbid<<'\t'<<l.site<<'\t'<<l.stage<<'\t'<<l.result<<'\t'<<l.name<<'\n';}
        status<<"callback_error="<<state.failure.load()<<"\nmodules="<<state.modules<<"\nfunctions="<<state.functions<<"\nlaunches="<<state.launches<<'\n';
        modules.close();functions.close();launches.close();status.close();
        if(!modules || !functions || !launches || !status)throw std::runtime_error("image observation close failed");
    }
    void checkpoint(const std::filesystem::path&out,const std::string&label){
        checked_cupti(cuptiActivityFlushAll(0),"CUPTI function flush after stage sync");
        size_t dropped=0;checked_cupti(cuptiActivityGetNumDroppedRecords(nullptr,0,&dropped),"CUPTI dropped function records");
        if(dropped)fail(19);
        retain(out,label);
        if(state.failure.load())throw std::runtime_error("CUPTI observer retained failure="+std::to_string(state.failure.load()));
    }
    void stop(){
        checked_cupti(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_FUNCTION),"CUPTI function disable");
        checked_cupti(cuptiUnsubscribe(subscriber),"CUPTI unsubscribe");subscriber=nullptr;
    }
    void require_records(){if(state.modules==0 || state.functions==0 || state.launches==0)throw std::runtime_error("CUPTI selected-code records unavailable; module-only evidence cannot qualify functions");}
};
}
