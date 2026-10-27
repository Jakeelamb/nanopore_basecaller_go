#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "cupti.h"
#include "../sha256.inc"
void write(const std::filesystem::path&p,const void*data,size_t n){std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(data),n);f.close();if(!f)throw std::runtime_error("fake raw close failed");}
#include "../image_observer.inc"

CUpti_CallbackFunc resource_callback;
CUpti_BuffersCallbackRequestFunc buffer_request;
CUpti_BuffersCallbackCompleteFunc buffer_complete;
CUpti_ActivityFunction function_record;
std::string mode;
CUptiResult cuptiSubscribe(CUpti_SubscriberHandle*p,CUpti_CallbackFunc cb,void*){*p=reinterpret_cast<void*>(1);resource_callback=cb;return 0;}
CUptiResult cuptiEnableCallback(uint32_t enabled,CUpti_SubscriberHandle,CUpti_CallbackDomain domain,CUpti_CallbackId id){if(enabled!=1 || (domain!=3 && domain!=1) || (domain==3 && id!=6))return 99;return 0;}
CUptiResult cuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc a,CUpti_BuffersCallbackCompleteFunc b){buffer_request=a;buffer_complete=b;return 0;}
CUptiResult cuptiActivityEnable(int kind){return kind==26?0:98;}
CUptiResult cuptiActivityDisable(int kind){return kind==26?0:98;}
CUptiResult cuptiActivityGetNextRecord(uint8_t*,size_t,CUpti_Activity**r){if(mode=="callback-exception")throw std::runtime_error("fake callback parser exception");if(*r)return 1;*r=reinterpret_cast<CUpti_Activity*>(&function_record);return 0;}
CUptiResult cuptiActivityGetNumDroppedRecords(CUcontext,uint32_t,size_t*n){*n=mode=="dropped"?1:0;return 0;}
CUptiResult cuptiActivityFlushAll(uint32_t flags){return flags==0?0:97;}
CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle){return 0;}

int main(int argc,char**argv){
    if(argc!=3)return 96;mode=argv[1];std::filesystem::path out(argv[2]);std::filesystem::create_directory(out);
    d01_image::Observer observer;observer.start();
    std::vector<char> bytes(mode=="oversize"?d01_image::max_cubin+1:32,'c');
    CUpti_ModuleResourceData m{7,bytes.size(),bytes.data()};CUpti_ResourceData resource{reinterpret_cast<void*>(1),{},&m};
    resource_callback(nullptr,3,6,&resource);bytes[0]='X';
    const char*names[]={"back_guide_step","beam_search_step","compute_posts_step","decode_step"};
    for(unsigned stage=1;stage<=4;++stage){
        observer.stage(stage);int result=0;char name[256]{};std::strcpy(name,names[stage-1]);
        CUpti_CallbackData launch{};launch.callbackSite=0;launch.symbolName=name;launch.context=reinterpret_cast<void*>(1);launch.contextUid=9;launch.correlationId=stage;
        resource_callback(nullptr,1,307,&launch);launch.callbackSite=1;launch.functionReturnValue=&result;resource_callback(nullptr,1,307,&launch);
        function_record={26,stage,9,7,stage,0,name};
        uint8_t*buffer=nullptr;size_t size=0,records=0;buffer_request(&buffer,&size,&records);buffer_complete(nullptr,0,buffer,size,sizeof(function_record));
        name[0]='X';
    }
    if(mode=="malformed")resource_callback(nullptr,3,6,nullptr);
    if(mode=="capacity")for(unsigned i=0;i<17;++i)resource_callback(nullptr,3,6,&resource);
    if(mode=="buffer-capacity")for(unsigned i=0;i<9;++i){uint8_t*b=nullptr;size_t n=0,r=0;buffer_request(&b,&n,&r);}
    bool failed=false;
    try{observer.checkpoint(out,"final");observer.require_records();observer.stop();}catch(const std::exception&e){std::cout<<e.what()<<'\n';failed=true;}
    const bool want=mode!="positive";
    if(failed!=want)return 95;
    std::cout<<"callback-failure-retained="<<d01_image::state.failure.load()<<"\n";
    return 0;
}
