#pragma once
#include "cuda.h"
#include "generated_cuda_meta.h"
#include <cstddef>
#define CUPTIAPI
using CUptiResult=int;using CUpti_CallbackDomain=int;using CUpti_CallbackId=uint32_t;using CUpti_SubscriberHandle=void*;
constexpr int CUPTI_SUCCESS=0,CUPTI_CB_DOMAIN_RESOURCE=3,CUPTI_CB_DOMAIN_DRIVER_API=1,CUPTI_CBID_RESOURCE_MODULE_LOADED=6,CUPTI_API_ENTER=0,CUPTI_API_EXIT=1;
struct CUpti_ResourceData {CUcontext context;union{void*stream;}resourceHandle;void*resourceDescriptor;};
struct CUpti_ModuleResourceData {uint32_t moduleId;size_t cubinSize;const char*pCubin;};
struct CUpti_CallbackData {int callbackSite;const char*functionName;const void*functionParams;void*functionReturnValue;const char*symbolName;CUcontext context;uint32_t contextUid;uint64_t*correlationData;uint32_t correlationId;};
using CUpti_CallbackFunc=void(*)(void*,CUpti_CallbackDomain,CUpti_CallbackId,const void*);
CUptiResult cuptiSubscribe(CUpti_SubscriberHandle*,CUpti_CallbackFunc,void*);CUptiResult cuptiEnableDomain(uint32_t,CUpti_SubscriberHandle,CUpti_CallbackDomain);CUptiResult cuptiEnableCallback(uint32_t,CUpti_SubscriberHandle,CUpti_CallbackDomain,CUpti_CallbackId);CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle);CUptiResult cuptiGetContextId(CUcontext,uint32_t*);
