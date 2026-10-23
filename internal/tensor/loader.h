#include "abi.h"
typedef struct nt_library nt_library;
nt_library* nt_library_open(const char*, char*, size_t);
void nt_library_close(nt_library*);
int nt_call_open(nt_library*,uint64_t*,char*,size_t);
int nt_call_close(nt_library*,uint64_t,char*,size_t);
int nt_call_new(nt_library*,uint64_t,const int64_t*,size_t,const float*,size_t,uint64_t*,char*,size_t);
int nt_call_new_int64(nt_library*,uint64_t,const int64_t*,size_t,const int64_t*,size_t,uint64_t*,char*,size_t);
int nt_call_load(nt_library*,uint64_t,const char*,uint64_t*,char*,size_t);
int nt_call_release(nt_library*,uint64_t,uint64_t,char*,size_t);
int nt_call_shape(nt_library*,uint64_t,uint64_t,int64_t*,size_t,size_t*,char*,size_t);
int nt_call_data(nt_library*,uint64_t,uint64_t,float*,size_t,char*,size_t);
int nt_call_data_int64(nt_library*,uint64_t,uint64_t,int64_t*,size_t,char*,size_t);
int nt_call_run(nt_library*,uint64_t,int,const uint64_t*,size_t,const int64_t*,size_t,const double*,size_t,uint64_t*,char*,size_t);
