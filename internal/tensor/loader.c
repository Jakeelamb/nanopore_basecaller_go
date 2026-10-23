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
 int (*sdpa)(uint64_t,uint64_t,uint64_t,uint64_t,const int64_t*,size_t,const uint8_t*,size_t,uint64_t*,char*,size_t);
 int (*cuda_device_open)(uint64_t engine, int32_t ordinal, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap);
 int (*cuda_device_close)(uint64_t engine, uint64_t device, nt_cuda_detail *detail, char *error, size_t error_cap);
 int (*cuda_stream_open)(uint64_t engine, uint64_t device, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap);
 int (*cuda_stream_close)(uint64_t engine, uint64_t stream, nt_cuda_detail *detail, char *error, size_t error_cap);
 int (*cuda_half_new)(uint64_t engine, uint64_t device, uint64_t stream, const int64_t *shape, size_t rank, const uint16_t *bits, size_t count, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap);
 int (*cuda_half_view)(uint64_t engine, uint64_t parent, uint64_t offset_elements, const int64_t *shape, const int64_t *strides, size_t rank, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap);
 int (*cuda_half_meta)(uint64_t engine, uint64_t half, nt_half_meta *out, nt_cuda_detail *detail, char *error, size_t error_cap);
 int (*cuda_half_bits)(uint64_t engine, uint64_t stream, uint64_t half, uint16_t *bits, size_t count, nt_cuda_detail *detail, char *error, size_t error_cap);
 int (*cuda_half_release)(uint64_t engine, uint64_t half, nt_cuda_detail *detail, char *error, size_t error_cap);
 int (*cuda_gemm_f16)(uint64_t engine, uint64_t stream, uint64_t a, uint64_t b, uint64_t out, nt_cuda_detail *detail, char *error, size_t error_cap);
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
 LOAD(sdpa,"nt_sdpa");
 LOAD(cuda_device_open,"nt_cuda_device_open");
 LOAD(cuda_device_close,"nt_cuda_device_close");
 LOAD(cuda_stream_open,"nt_cuda_stream_open");
 LOAD(cuda_stream_close,"nt_cuda_stream_close");
 LOAD(cuda_half_new,"nt_cuda_half_new");
 LOAD(cuda_half_view,"nt_cuda_half_view");
 LOAD(cuda_half_meta,"nt_cuda_half_meta");
 LOAD(cuda_half_bits,"nt_cuda_half_bits");
 LOAD(cuda_half_release,"nt_cuda_half_release");
 LOAD(cuda_gemm_f16,"nt_cuda_gemm_f16");
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
int nt_call_sdpa(nt_library*l,uint64_t id,uint64_t q,uint64_t k,uint64_t v,const int64_t*s,size_t rank,const uint8_t*m,size_t n,uint64_t*out,char*e,size_t c){return l->sdpa(id,q,k,v,s,rank,m,n,out,e,c);}

int nt_call_cuda_device_open(nt_library *library, uint64_t engine, int32_t ordinal, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap) { return library->cuda_device_open(engine,ordinal,out,detail,error,error_cap); }

int nt_call_cuda_device_close(nt_library *library, uint64_t engine, uint64_t device, nt_cuda_detail *detail, char *error, size_t error_cap) { return library->cuda_device_close(engine,device,detail,error,error_cap); }

int nt_call_cuda_stream_open(nt_library *library, uint64_t engine, uint64_t device, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap) { return library->cuda_stream_open(engine,device,out,detail,error,error_cap); }

int nt_call_cuda_stream_close(nt_library *library, uint64_t engine, uint64_t stream, nt_cuda_detail *detail, char *error, size_t error_cap) { return library->cuda_stream_close(engine,stream,detail,error,error_cap); }

int nt_call_cuda_half_new(nt_library *library, uint64_t engine, uint64_t device, uint64_t stream, const int64_t *shape, size_t rank, const uint16_t *bits, size_t count, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap) { return library->cuda_half_new(engine,device,stream,shape,rank,bits,count,out,detail,error,error_cap); }

int nt_call_cuda_half_view(nt_library *library, uint64_t engine, uint64_t parent, uint64_t offset_elements, const int64_t *shape, const int64_t *strides, size_t rank, uint64_t *out, nt_cuda_detail *detail, char *error, size_t error_cap) { return library->cuda_half_view(engine,parent,offset_elements,shape,strides,rank,out,detail,error,error_cap); }

int nt_call_cuda_half_meta(nt_library *library, uint64_t engine, uint64_t half, nt_half_meta *out, nt_cuda_detail *detail, char *error, size_t error_cap) { return library->cuda_half_meta(engine,half,out,detail,error,error_cap); }

int nt_call_cuda_half_bits(nt_library *library, uint64_t engine, uint64_t stream, uint64_t half, uint16_t *bits, size_t count, nt_cuda_detail *detail, char *error, size_t error_cap) { return library->cuda_half_bits(engine,stream,half,bits,count,detail,error,error_cap); }

int nt_call_cuda_half_release(nt_library *library, uint64_t engine, uint64_t half, nt_cuda_detail *detail, char *error, size_t error_cap) { return library->cuda_half_release(engine,half,detail,error,error_cap); }

int nt_call_cuda_gemm_f16(nt_library *library, uint64_t engine, uint64_t stream, uint64_t a, uint64_t b, uint64_t out, nt_cuda_detail *detail, char *error, size_t error_cap) { return library->cuda_gemm_f16(engine,stream,a,b,out,detail,error,error_cap); }
