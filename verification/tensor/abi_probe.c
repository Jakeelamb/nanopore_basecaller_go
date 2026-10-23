#include "abi.h"
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

static int failed(const char* message) {
    fprintf(stderr,"%s\n",message);
    return 1;
}

int main(int argc,char** argv) {
    if(argc!=2) return failed("expected one ABI 3 library path");
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
    uint64_t out=UINT64_MAX;
    if(sdpa(engine,tensor,tensor,tensor,NULL,0,NULL,0,&out,NULL,31)==0 || out!=UINT64_MAX) return failed("null error buffer accepted");
    if(close_engine(engine,error,sizeof(error))) return failed(error);
    dlclose(library);
    puts("ABI 3 direct C exception checks passed");
    return 0;
}
