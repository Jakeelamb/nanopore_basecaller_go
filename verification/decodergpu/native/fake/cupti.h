#pragma once
#include <cstddef>
#include <cstdint>
#define CUPTIAPI
using CUcontext=void*;
using CUptiResult=int;
using CUpti_CallbackDomain=int;
using CUpti_CallbackId=uint32_t;
using CUpti_SubscriberHandle=void*;
constexpr int CUPTI_SUCCESS=0,CUPTI_ERROR_MAX_LIMIT_REACHED=1;
constexpr int CUPTI_CB_DOMAIN_RESOURCE=3,CUPTI_CB_DOMAIN_DRIVER_API=1,CUPTI_CBID_RESOURCE_MODULE_LOADED=6;
constexpr int CUPTI_API_ENTER=0,CUPTI_API_EXIT=1,CUPTI_ACTIVITY_KIND_FUNCTION=26;
struct CUpti_ResourceData {CUcontext context;union {void*stream;}resourceHandle;void*resourceDescriptor;};
struct CUpti_ModuleResourceData {uint32_t moduleId;size_t cubinSize;const char*pCubin;};
struct CUpti_CallbackData {int callbackSite;const char*functionName;const void*functionParams;void*functionReturnValue;const char*symbolName;CUcontext context;uint32_t contextUid;uint64_t*correlationData;uint32_t correlationId;};
struct CUpti_Activity {int kind;};
struct CUpti_ActivityFunction {int kind;uint32_t id,contextId,moduleId,functionIndex,pad;const char*name;};
using CUpti_CallbackFunc=void(*)(void*,CUpti_CallbackDomain,CUpti_CallbackId,const void*);
using CUpti_BuffersCallbackRequestFunc=void(*)(uint8_t**,size_t*,size_t*);
using CUpti_BuffersCallbackCompleteFunc=void(*)(CUcontext,uint32_t,uint8_t*,size_t,size_t);
CUptiResult cuptiSubscribe(CUpti_SubscriberHandle*,CUpti_CallbackFunc,void*);
CUptiResult cuptiEnableCallback(uint32_t,CUpti_SubscriberHandle,CUpti_CallbackDomain,CUpti_CallbackId);
CUptiResult cuptiActivityRegisterCallbacks(CUpti_BuffersCallbackRequestFunc,CUpti_BuffersCallbackCompleteFunc);
CUptiResult cuptiActivityEnable(int);
CUptiResult cuptiActivityDisable(int);
CUptiResult cuptiActivityGetNextRecord(uint8_t*,size_t,CUpti_Activity**);
CUptiResult cuptiActivityGetNumDroppedRecords(CUcontext,uint32_t,size_t*);
CUptiResult cuptiActivityFlushAll(uint32_t);
CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle);
