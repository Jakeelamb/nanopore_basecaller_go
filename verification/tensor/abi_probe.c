#include "abi.h"
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

static int failed(const char* message) {
    fprintf(stderr,"%s\n",message);
    return 1;
}

int main(int argc,char** argv) {
    if(argc!=2) return failed("expected one ABI 5 library path");
    void* library=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!library) return failed(dlerror());
    uint32_t (*version)(void);
    int (*open_engine)(uint64_t*,char*,size_t);
    int (*close_engine)(uint64_t,char*,size_t);
    int (*new_tensor)(uint64_t,const int64_t*,size_t,const float*,size_t,uint64_t*,char*,size_t);
    int (*sdpa)(uint64_t,uint64_t,uint64_t,uint64_t,const int64_t*,size_t,const uint8_t*,size_t,uint64_t*,char*,size_t);
    #define LOAD(field,name) do {*(void**)(&field)=dlsym(library,name);if(!field)return failed("missing " name);} while(0)
    LOAD(version,"nt_version");LOAD(open_engine,"nt_open");LOAD(close_engine,"nt_close");LOAD(new_tensor,"nt_new");LOAD(sdpa,"nt_sdpa");
    #undef LOAD
    if(version()!=NT_ABI_VERSION) return failed("unexpected ABI version");
    char error[NT_ERROR_CAP];
    uint64_t engine=0,tensor=0;
    const int64_t shape[]={2};
    const float data[]={1,2};
    if(open_engine(&engine,error,sizeof(error))) return failed(error);
    if(new_tensor(engine,shape,1,data,2,&tensor,error,sizeof(error))) return failed(error);
    for(size_t cap=0;cap<=31;cap=cap==0?1:cap==1?31:32) {
        unsigned char storage[64];memset(storage,0x5a,sizeof(storage));
        uint64_t out=UINT64_MAX;
        int status=sdpa(engine,tensor,tensor,tensor,NULL,0,NULL,0,&out,(char*)storage+1,cap);
        if(status==0 || out!=UINT64_MAX) return failed("native exception published a handle");
        if(storage[0]!=0x5a) return failed("error buffer underflow");
        for(size_t i=cap+1;i<sizeof(storage);++i) if(storage[i]!=0x5a) return failed("error buffer overflow");
        if(cap && !memchr(storage+1,0,cap)) return failed("error buffer lacks bounded terminator");
        if(cap==0 && storage[1]!=0x5a) return failed("zero-capacity error buffer changed");
        printf("native SDPA exception bounded at error capacity %zu\n",cap);
    }
    int (*device_open)(uint64_t,int32_t,uint64_t*,nt_cuda_detail*,char*,size_t);
    int (*new_indices)(uint64_t,const int64_t*,size_t,const int64_t*,size_t,uint64_t*,char*,size_t);
    *(void**)(&new_indices)=dlsym(library,"nt_new_int64");
    if(!new_indices)return failed("missing nt_new_int64");
    int (*half_meta)(uint64_t,uint64_t,nt_half_meta*,nt_cuda_detail*,char*,size_t);
    *(void**)(&device_open)=dlsym(library,"nt_cuda_device_open");
    *(void**)(&half_meta)=dlsym(library,"nt_cuda_half_meta");
    if(!device_open||!half_meta)return failed("missing CUDA symbols");
    nt_cuda_detail detail;nt_half_meta meta;memset(&meta,0x5a,sizeof(meta));
    for(size_t cap=0;cap<=31;cap=cap==0?1:cap==1?31:32){
        unsigned char storage[64];memset(storage,0x5a,sizeof(storage));
        uint64_t candidate=UINT64_MAX;
        int status=device_open(engine,-1,&candidate,&detail,(char*)storage+1,cap);
        if(status==0||candidate!=UINT64_MAX)return failed("CUDA validation published handle");
        if(storage[0]!=0x5a)return failed("CUDA error underflow");
        for(size_t i=cap+1;i<sizeof(storage);++i)if(storage[i]!=0x5a)return failed("CUDA error overflow");
        if(cap&&!memchr(storage+1,0,cap))return failed("CUDA error lacks terminator");
    }
    if(half_meta(engine,tensor,&meta,&detail,error,sizeof(error))!=NT_CUDA_DTYPE||detail.actual_dtype!=NT_SCALAR_F32||detail.expected_dtype!=NT_SCALAR_F16)return failed("CPU float32 entered half metadata");
    uint64_t indices=0;const int64_t idx[]={0,1};
    if(new_indices(engine,shape,1,idx,2,&indices,error,sizeof(error)))return failed(error);
    if(half_meta(engine,indices,&meta,&detail,error,sizeof(error))!=NT_CUDA_DTYPE||detail.actual_dtype!=NT_SCALAR_I64)return failed("CPU int64 entered half metadata");
    for(size_t i=0;i<sizeof(meta);++i)if(((unsigned char*)&meta)[i]!=0x5a)return failed("rejected metadata output changed");
    if(half_meta(engine,UINT64_MAX,&meta,&detail,error,sizeof(error))!=NT_CUDA_CLOSED)return failed("unknown half handle accepted");
    uint64_t foreign=0;
    if(open_engine(&foreign,error,sizeof(error)))return failed(error);
    if(half_meta(foreign,tensor,&meta,&detail,error,sizeof(error))!=NT_CUDA_OWNER)return failed("foreign engine accepted half input");
    if(close_engine(foreign,error,sizeof(error)))return failed(error);
    uint64_t out=UINT64_MAX;
    if(device_open(engine,-1,&out,NULL,error,sizeof(error))!=NT_CUDA_ARGUMENT||out!=UINT64_MAX)return failed("null CUDA detail accepted");
    if(device_open(engine,-1,&out,&detail,NULL,31)!=NT_CUDA_ARGUMENT||out!=UINT64_MAX)return failed("null CUDA error accepted");
    if(sdpa(engine,tensor,tensor,tensor,NULL,0,NULL,0,&out,NULL,31)==0 || out!=UINT64_MAX) return failed("null error buffer accepted");
    if(close_engine(engine,error,sizeof(error))) return failed(error);
    dlclose(library);
    puts("ABI 5 direct C exception checks passed");
    return 0;
}
