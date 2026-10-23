//go:build cgo && linux

#include "loader.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
struct nt_library {
 void* dl;
 uint32_t (*version)(void);
 int (*open)(uint64_t*,char*,size_t);
 int (*close)(uint64_t,char*,size_t);
 int (*new_tensor)(uint64_t,const int64_t*,size_t,const float*,size_t,uint64_t*,char*,size_t);
 int (*new_int64)(uint64_t,const int64_t*,size_t,const int64_t*,size_t,uint64_t*,char*,size_t);
 int (*load)(uint64_t,const char*,uint64_t*,char*,size_t);
 int (*release)(uint64_t,uint64_t,char*,size_t);
 int (*shape)(uint64_t,uint64_t,int64_t*,size_t,size_t*,char*,size_t);
 int (*data)(uint64_t,uint64_t,float*,size_t,char*,size_t);
 int (*data_int64)(uint64_t,uint64_t,int64_t*,size_t,char*,size_t);
 int (*run)(uint64_t,int,const uint64_t*,size_t,const int64_t*,size_t,const double*,size_t,uint64_t*,char*,size_t);
};
nt_library* nt_library_open(const char* path,char* err,size_t cap) {
 nt_library* l=calloc(1,sizeof(*l));
 if(!l) {snprintf(err,cap,"loader allocation failed"); return NULL;}
 l->dl=dlopen(path,RTLD_NOW|RTLD_LOCAL);
 if(!l->dl) {snprintf(err,cap,"dlopen: %s",dlerror()); free(l); return NULL;}
 #define LOAD(field,symbol) do { *(void**)(&l->field)=dlsym(l->dl,symbol); if(!l->field) {snprintf(err,cap,"missing ABI symbol %s",symbol); nt_library_close(l); return NULL;} } while(0)
 LOAD(version,"nt_version");
 if(l->version()!=NT_ABI_VERSION) {snprintf(err,cap,"unsupported tensor ABI version %u",l->version()); nt_library_close(l); return NULL;}
 LOAD(open,"nt_open"); LOAD(close,"nt_close");
 LOAD(new_tensor,"nt_new"); LOAD(load,"nt_load"); LOAD(release,"nt_release");
 LOAD(shape,"nt_shape"); LOAD(data,"nt_data"); LOAD(run,"nt_run");
 LOAD(new_int64,"nt_new_int64");LOAD(data_int64,"nt_data_int64");
 #undef LOAD
 return l;
}
void nt_library_close(nt_library* l) {if(l) {if(l->dl) dlclose(l->dl); free(l);}}
int nt_call_open(nt_library*l,uint64_t*out,char*e,size_t c){return l->open(out,e,c);}
int nt_call_close(nt_library*l,uint64_t id,char*e,size_t c){return l->close(id,e,c);}
int nt_call_new(nt_library*l,uint64_t id,const int64_t*s,size_t n,const float*d,size_t z,uint64_t*out,char*e,size_t c){return l->new_tensor(id,s,n,d,z,out,e,c);}
int nt_call_new_int64(nt_library*l,uint64_t id,const int64_t*s,size_t n,const int64_t*d,size_t z,uint64_t*out,char*e,size_t c){return l->new_int64(id,s,n,d,z,out,e,c);}
int nt_call_load(nt_library*l,uint64_t id,const char*p,uint64_t*out,char*e,size_t c){return l->load(id,p,out,e,c);}
int nt_call_release(nt_library*l,uint64_t id,uint64_t t,char*e,size_t c){return l->release(id,t,e,c);}
int nt_call_shape(nt_library*l,uint64_t id,uint64_t t,int64_t*s,size_t n,size_t*r,char*e,size_t c){return l->shape(id,t,s,n,r,e,c);}
int nt_call_data(nt_library*l,uint64_t id,uint64_t t,float*d,size_t n,char*e,size_t c){return l->data(id,t,d,n,e,c);}
int nt_call_data_int64(nt_library*l,uint64_t id,uint64_t t,int64_t*d,size_t n,char*e,size_t c){return l->data_int64(id,t,d,n,e,c);}
int nt_call_run(nt_library*l,uint64_t id,int op,const uint64_t*t,size_t n,const int64_t*p,size_t pn,const double*f,size_t fn,uint64_t*out,char*e,size_t c){return l->run(id,op,t,n,p,pn,f,fn,out,e,c);}
