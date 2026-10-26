#include <cupti.h>
#include <cupti_driver_cbid.h>
#include <atomic>
#include <array>
#include <sys/syscall.h>

namespace d01_lineage {
constexpr size_t input_bytes=261192, max_image=1<<20, max_modules=4, max_api=256, text_bytes=256;
struct API {uint64_t sequence=0,thread=0,context=0,uid=0,correlation=0,operation=0,module=0,function=0,image=0,options=0,option_values=0;uint32_t callback=0,site=0,option_count=0;int32_t result=0;char name[text_bytes]{},symbol[text_bytes]{};};
struct Module {uint64_t sequence=0,thread=0,context=0,id=0,operation=0,observed=0,retained=0;std::array<uint8_t,max_image> bytes{};};
struct State {std::atomic_flag lock=ATOMIC_FLAG_INIT;std::atomic<unsigned> error{0};uint64_t sequence=0,launches=0,input_operation=0;size_t apis=0,modules=0,inputs=0;const void*expected=nullptr;std::array<uint8_t,input_bytes> input{};std::array<API,max_api> api{};std::array<Module,max_modules> module{};};
State state;
thread_local uint64_t active=0;
struct Locked {Locked(){while(state.lock.test_and_set(std::memory_order_acquire)){}}~Locked(){state.lock.clear(std::memory_order_release);}};
void fail(unsigned code) noexcept {unsigned zero=0;state.error.compare_exchange_strong(zero,code);}
bool text(char*to,const char*from) noexcept {if(!from){fail(1);return false;}for(size_t i=0;i<text_bytes;++i){to[i]=from[i];if(!from[i])return true;}to[text_bytes-1]=0;fail(2);return false;}
uint64_t thread() noexcept {return static_cast<uint64_t>(::syscall(SYS_gettid));}
void CUPTIAPI callback(void*,CUpti_CallbackDomain domain,CUpti_CallbackId id,const void*data) noexcept {
    try {
        if(!data){fail(3);return;}
        Locked held;
        if(domain==CUPTI_CB_DOMAIN_RESOURCE && id==CUPTI_CBID_RESOURCE_MODULE_LOADED){
            const auto&r=*static_cast<const CUpti_ResourceData*>(data);const auto*m=static_cast<const CUpti_ModuleResourceData*>(r.resourceDescriptor);
            if(!m || !m->pCubin || !m->cubinSize){fail(4);return;}
            if(state.modules==max_modules){fail(5);return;}
            auto&to=state.module[state.modules++];to.sequence=++state.sequence;to.thread=thread();to.context=reinterpret_cast<uintptr_t>(r.context);to.id=m->moduleId;to.operation=active;to.observed=m->cubinSize;to.retained=std::min(m->cubinSize,max_image);std::memcpy(to.bytes.data(),m->pCubin,to.retained);
            if(!active || !to.context || to.observed!=to.retained)fail(6);
            return;
        }
        if(domain!=CUPTI_CB_DOMAIN_DRIVER_API){fail(7);return;}
        if(state.apis==max_api){fail(8);return;}
        const auto&r=*static_cast<const CUpti_CallbackData*>(data);auto&to=state.api[state.apis++];to.sequence=++state.sequence;to.thread=thread();to.context=reinterpret_cast<uintptr_t>(r.context);to.uid=r.contextUid;to.correlation=r.correlationId;to.callback=id;to.site=r.callbackSite;text(to.name,r.functionName);
        if(std::strstr(to.name,"Launch")){++state.launches;fail(9);}
        const bool load=id==CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx,lookup=id==CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction;
        if(r.callbackSite==CUPTI_API_ENTER){
            if(load || lookup){if(active){fail(10);return;}active=to.sequence;}
            to.operation=active;
            if(load){
                const auto*p=static_cast<const cuModuleLoadDataEx_params*>(r.functionParams);
                if(!p || p->image!=state.expected || !p->module || p->numOptions || p->options || p->optionValues || state.inputs){fail(11);return;}
                std::memcpy(state.input.data(),p->image,input_bytes);++state.inputs;state.input_operation=to.operation;
            }
        } else if(r.callbackSite==CUPTI_API_EXIT){
            to.operation=active;
            if(!r.functionReturnValue){fail(12);return;}std::memcpy(&to.result,r.functionReturnValue,sizeof(to.result));
            if(load){const auto*p=static_cast<const cuModuleLoadDataEx_params*>(r.functionParams);if(!p || !p->module){fail(13);return;}if(to.result==0)to.module=reinterpret_cast<uintptr_t>(*p->module);}
        } else {fail(14);return;}
        if(load){const auto*p=static_cast<const cuModuleLoadDataEx_params*>(r.functionParams);if(!p){fail(19);return;}to.image=reinterpret_cast<uintptr_t>(p->image);to.option_count=p->numOptions;to.options=reinterpret_cast<uintptr_t>(p->options);to.option_values=reinterpret_cast<uintptr_t>(p->optionValues);}
        if(lookup){
            const auto*p=static_cast<const cuModuleGetFunction_params*>(r.functionParams);
            if(!p || !p->hfunc){fail(15);return;}to.module=reinterpret_cast<uintptr_t>(p->hmod);text(to.symbol,p->name);if(r.callbackSite==CUPTI_API_EXIT && to.result==0)to.function=reinterpret_cast<uintptr_t>(*p->hfunc);
        }
        if(id==CUPTI_DRIVER_TRACE_CBID_cuModuleUnload){const auto*p=static_cast<const cuModuleUnload_params*>(r.functionParams);if(!p){fail(18);return;}to.module=reinterpret_cast<uintptr_t>(p->hmod);}
        if((load || lookup) && r.callbackSite==CUPTI_API_EXIT){if(!active)fail(16);active=0;}
    } catch(...) {fail(17);}
}
struct Observer {
    CUpti_SubscriberHandle subscriber=nullptr;
    void start(const void*input){state.expected=input;cupti_check(cuptiSubscribe(&subscriber,callback,nullptr),"subscribe");cupti_check(cuptiEnableDomain(1,subscriber,CUPTI_CB_DOMAIN_DRIVER_API),"all driver callbacks");cupti_check(cuptiEnableCallback(1,subscriber,CUPTI_CB_DOMAIN_RESOURCE,CUPTI_CBID_RESOURCE_MODULE_LOADED),"module callbacks");}
    void stop(){cupti_check(cuptiEnableDomain(0,subscriber,CUPTI_CB_DOMAIN_DRIVER_API),"disable driver callbacks");cupti_check(cuptiEnableCallback(0,subscriber,CUPTI_CB_DOMAIN_RESOURCE,CUPTI_CBID_RESOURCE_MODULE_LOADED),"disable module callbacks");cupti_check(cuptiUnsubscribe(subscriber),"unsubscribe");subscriber=nullptr;}
};
}
