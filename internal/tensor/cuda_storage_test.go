//go:build cgo && linux

package tensor

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// fakeStorageTypes mirrors the ABI structs without including abi.h, so the fake
// backend can declare its untouched stubs as plain C functions.
const fakeStorageTypes = `
#include <stdint.h>
#include <stddef.h>
typedef struct nt_cuda_detail {
 uint32_t stage; int32_t native_code; uint64_t handle; uint64_t actual_engine;
 uint64_t expected_device; uint64_t actual_device;
 int32_t expected_ordinal; int32_t actual_ordinal;
 uint32_t expected_dtype; uint32_t actual_dtype;
} nt_cuda_detail;
typedef struct nt_storage_meta {
 uint32_t rank; uint32_t dtype; uint64_t device; int32_t ordinal;
 uint32_t scale_dtype; uint32_t scale_present; int32_t scale_tag;
 uint64_t allocation; uint64_t offset_elements; uint64_t storage_elements;
 int64_t shape[8]; int64_t strides[8]; int32_t tags[8];
} nt_storage_meta;
`

// fakeStorageBackend is a host-memory implementation of the typed storage ABI.
// It exercises the real loader, cgo marshalling and Go validation without a
// GPU. The real device path is captured separately by the maintained CUDA unit.
const fakeStorageBackend = `
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
typedef struct { uint64_t id; uint32_t dtype, scale_dtype; int32_t scale_tag; uint64_t offset, storage, end, allocation, scale, engine, device; int32_t rank, tags[8]; int64_t shape[8], strides[8]; uint8_t *data; } rec;
static rec table[256]; static uint64_t next_id=1;
static size_t dsize(uint32_t t){switch(t){case 1:return 2;case 2:case 3:case 6:return 1;case 4:case 5:return 4;}return 0;}
static rec* find(uint64_t e,uint64_t id){for(int i=0;i<256;i++)if(table[i].id==id){if(table[i].engine!=e)return NULL;return &table[i];}return NULL;}
static uint64_t fresh(void){return next_id++;}
uint32_t nt_version(void){return 4;}
int nt_open(uint64_t*out,char*e,size_t c){(void)e;(void)c;*out=1;return 0;}
int nt_close(uint64_t id,char*e,size_t c){(void)id;(void)e;(void)c;return 0;}
int nt_cuda_device_open(uint64_t e,int32_t o,uint64_t*out,nt_cuda_detail*d,char*er,size_t c){(void)e;(void)d;(void)er;(void)c;if(o<0)return 3;*out=100;return 0;}
int nt_cuda_device_close(uint64_t e,uint64_t id,nt_cuda_detail*d,char*er,size_t c){(void)e;(void)id;(void)d;(void)er;(void)c;return 0;}
int nt_cuda_stream_open(uint64_t e,uint64_t dev,uint64_t*out,nt_cuda_detail*d,char*er,size_t c){(void)e;(void)dev;(void)d;(void)er;(void)c;*out=200;return 0;}
int nt_cuda_stream_close(uint64_t e,uint64_t id,nt_cuda_detail*d,char*er,size_t c){(void)e;(void)id;(void)d;(void)er;(void)c;return 0;}
int nt_cuda_storage_new(uint64_t e,uint64_t dev,uint64_t stream,uint32_t dtype,const int64_t*shape,size_t rank,const int32_t*tags,const uint8_t*bytes,size_t count,uint64_t*out,nt_cuda_detail*d,char*er,size_t c){
 (void)dev;(void)stream;(void)d;(void)er;(void)c;size_t n=1;for(size_t i=0;i<rank;i++){if(shape[i]<=0)return 7;n*=(size_t)(shape[i]);}
 if(!dsize(dtype))return 4;if(count!=n*dsize(dtype))return 7;
 for(int i=0;i<256;i++)if(!table[i].id){table[i].id=fresh();table[i].engine=e;table[i].device=100;table[i].dtype=dtype;table[i].rank=(int32_t)rank;table[i].storage=n;table[i].end=n;table[i].allocation=table[i].id;table[i].data=malloc(n*dsize(dtype));if(count)memcpy(table[i].data,bytes,count);for(size_t j=rank;j-->0;){table[i].strides[j]=j+1<rank?table[i].shape[j+1]*table[i].strides[j+1]:1;table[i].shape[j]=shape[j];table[i].tags[j]=tags[j];}*out=table[i].id;return 0;}
 return 1;}
int nt_cuda_storage_view(uint64_t e,uint64_t parent,uint32_t dtype,uint64_t offset,const int64_t*shape,const int64_t*strides,const int32_t*tags,size_t rank,uint64_t*out,nt_cuda_detail*d,char*er,size_t c){
 (void)d;(void)er;(void)c;rec*p=find(e,parent);if(!p)return 2;if(dtype!=p->dtype)return 4;
 uint64_t end=p->offset+offset;for(size_t i=0;i<rank;i++){if(strides[i]<0)return 7;end+=(shape[i]-1)*strides[i];}
 if(end>=p->end)return 7;
 for(int i=0;i<256;i++)if(!table[i].id){table[i]=*p;table[i].id=fresh();table[i].offset=p->offset+offset;table[i].end=end+1;table[i].rank=(int32_t)rank;for(size_t j=0;j<rank;j++){table[i].shape[j]=shape[j];table[i].strides[j]=strides[j];table[i].tags[j]=tags[j];}*out=table[i].id;return 0;}
 return 1;}
static int contig(rec*p){if(p->rank<=0)return 1;if(p->strides[p->rank-1]!=1)return 0;
 for(int i=p->rank-2;i>=0;i--)if(p->strides[i]!=p->shape[i+1]*p->strides[i+1])return 0;return 1;}
int nt_cuda_storage_reinterpret(uint64_t e,uint64_t parent,uint32_t dtype,uint64_t off,
 const int64_t*shape,const int64_t*strides,const int32_t*tags,size_t rank,uint64_t*out,
 nt_cuda_detail*d,char*er,size_t c){
 (void)d;(void)er;(void)c;rec*p=find(e,parent);if(!p)return 2;
 if(!contig(p))return 7;
 size_t nsz=dsize(dtype),psz=dsize(p->dtype);if(!nsz||!psz)return 4;
 uint64_t start=(uint64_t)p->offset*psz+off*nsz;if(start%nsz)return 7;
 uint64_t st=start/nsz,end=st;
 for(size_t i=0;i<rank;i++){if(strides[i]<0)return 7;end+=(uint64_t)(shape[i]-1)*(uint64_t)strides[i];}
 uint64_t limit=(uint64_t)p->end*psz/nsz;if(end>=limit)return 7;
 for(int i=0;i<256;i++)if(!table[i].id){table[i]=*p;table[i].id=fresh();table[i].dtype=dtype;table[i].offset=st;table[i].end=end+1;table[i].storage=limit;table[i].rank=(int32_t)rank;for(size_t j=0;j<rank;j++){table[i].shape[j]=shape[j];table[i].strides[j]=strides[j];table[i].tags[j]=tags[j];}*out=table[i].id;return 0;}
 return 1;}
int nt_cuda_storage_meta(uint64_t e,uint64_t id,nt_storage_meta*m,nt_cuda_detail*d,char*er,size_t c){
 (void)d;(void)er;(void)c;rec*p=find(e,id);if(!p)return 2;memset(m,0,sizeof(*m));m->rank=p->rank;m->dtype=p->dtype;m->device=p->device;m->allocation=p->allocation;m->offset_elements=p->offset;m->storage_elements=p->storage;m->scale_dtype=p->scale_dtype;m->scale_present=p->scale!=0;m->scale_tag=p->scale_tag;for(int i=0;i<p->rank;i++){m->shape[i]=p->shape[i];m->strides[i]=p->strides[i];m->tags[i]=p->tags[i];}return 0;}
int nt_cuda_storage_upload(uint64_t e,uint64_t stream,uint64_t id,const uint8_t*bytes,size_t count,nt_cuda_detail*d,char*er,size_t c){
 (void)stream;(void)d;(void)er;(void)c;rec*p=find(e,id);if(!p)return 2;size_t n=1;for(int i=0;i<p->rank;i++)n*=(size_t)(p->shape[i]);if(count!=n*dsize(p->dtype))return 7;memcpy(p->data+p->offset*dsize(p->dtype),bytes,count);return 0;}
int nt_cuda_storage_download(uint64_t e,uint64_t stream,uint64_t id,uint8_t*bytes,size_t count,nt_cuda_detail*d,char*er,size_t c){
 (void)stream;(void)d;(void)er;(void)c;rec*p=find(e,id);if(!p)return 2;size_t n=1;for(int i=0;i<p->rank;i++)n*=(size_t)(p->shape[i]);if(count!=n*dsize(p->dtype))return 7;memcpy(bytes,p->data+p->offset*dsize(p->dtype),count);return 0;}
int nt_cuda_storage_copy(uint64_t e,uint64_t stream,uint64_t dst,uint64_t src,nt_cuda_detail*d,char*er,size_t c){
 (void)stream;(void)d;(void)er;(void)c;rec*a=find(e,dst),*b=find(e,src);if(!a||!b)return 2;if(a->dtype!=b->dtype)return 4;if(a->allocation==b->allocation&&!(a->end<=b->offset||b->end<=a->offset))return 7;size_t n=1;for(int i=0;i<b->rank;i++)n*=(size_t)(b->shape[i]);memcpy(a->data+a->offset*dsize(a->dtype),b->data+b->offset*dsize(b->dtype),n*dsize(a->dtype));return 0;}
int nt_cuda_storage_fill(uint64_t e,uint64_t stream,uint64_t id,const uint8_t*pattern,size_t count,nt_cuda_detail*d,char*er,size_t c){
 (void)stream;(void)d;(void)er;(void)c;rec*p=find(e,id);if(!p)return 2;size_t sz=dsize(p->dtype);if(count!=sz)return 7;size_t n=1;for(int i=0;i<p->rank;i++)n*=(size_t)(p->shape[i]);for(size_t i=0;i<n;i++)memcpy(p->data+(p->offset+i)*sz,pattern,sz);return 0;}
int nt_cuda_storage_scale(uint64_t e,uint64_t id,uint64_t scale,int32_t tag,nt_cuda_detail*d,char*er,size_t c){
 (void)d;(void)er;(void)c;rec*p=find(e,id);if(!p)return 2;if(!scale){p->scale=0;p->scale_dtype=0;p->scale_tag=0;return 0;}rec*s=find(e,scale);if(!s)return 2;if(p->dtype!=2)return 4;if(s->dtype!=5)return 4;p->scale=scale;p->scale_dtype=s->dtype;p->scale_tag=tag;return 0;}
int nt_cuda_storage_release(uint64_t e,uint64_t id,nt_cuda_detail*d,char*er,size_t c){
 (void)d;(void)er;(void)c;rec*p=find(e,id);if(!p)return 2;for(int i=0;i<256;i++)if(table[i].scale==id)return 3;if(p->allocation==p->id&&p->offset==0)free(p->data);memset(p,0,sizeof(*p));return 0;}
static uint16_t h16(float f){_Float16 h=(_Float16)f;uint16_t b;memcpy(&b,&h,2);return b;}
/* Host-memory stand-in for the small-convolution boundary. It recomputes the
   same argument contract the native guard enforces, so a Go request that drops
   or swaps a geometry field is rejected here instead of silently accepted. */
int nt_cuda_conv_small_f16(uint64_t e,uint64_t stream,uint64_t in,uint64_t w,uint64_t b,uint64_t out,
 int32_t n,int32_t cin,int32_t cout,int32_t tin,int32_t win,int32_t stride,int32_t pad,int32_t ons,
 uint32_t act,nt_cuda_detail*d,char*er,size_t c){
 (void)stream;(void)d;(void)er;(void)c;
 rec*ri=find(e,in),*rw=find(e,w),*rb=find(e,b),*ro=find(e,out);
 if(!ri||!rw||!rb||!ro)return 2;
 for(int k=0;k<4;k++){rec*r=k==0?ri:k==1?rw:k==2?rb:ro;if(r->dtype!=1)return 4;}
 if(n<=0||cin<=0||cout<=0||tin<=0||win<=0||stride<=0)return 3;
 if(pad!=win/2)return 7;
 if(tin%stride)return 7;
 if(ri->rank!=3||ri->shape[0]!=n||ri->shape[1]!=tin||ri->shape[2]!=cin)return 7;
 if(rw->rank!=2||rw->shape[0]!=(int64_t)win*cin||rw->shape[1]!=cout)return 7;
 if(rb->rank!=1||rb->shape[0]!=cout)return 7;
 if(ro->rank!=3||ro->shape[0]!=n||ro->shape[1]!=tin/stride||ro->shape[2]!=cout)return 7;
 if(ro->strides[0]!=ons||ro->strides[1]!=cout||ro->strides[2]!=1)return 7;
 if(act>3)return 8;
 return 0;}
/* Host-memory reproduction of weight.permute({2,1,0}).contiguous().flatten(0,1).to(kF16). */
int nt_cuda_weights_permute_cast_f16(uint64_t e,uint64_t stream,uint64_t w,uint64_t b,uint64_t pw,uint64_t pb,nt_cuda_detail*d,char*er,size_t c){
 (void)stream;(void)d;(void)er;(void)c;
 rec*rw=find(e,w),*rb=find(e,b),*rp=find(e,pw),*rpb=find(e,pb);
 if(!rw||!rb||!rp||!rpb)return 2;
 if(rw->dtype!=5||rb->dtype!=5||rp->dtype!=1||rpb->dtype!=1)return 4;
 if(rw->rank!=3)return 7;
 int64_t co=rw->shape[0],ci=rw->shape[1],win=rw->shape[2];
 if(rp->rank!=2||rp->shape[0]!=win*ci||rp->shape[1]!=co)return 7;
 if(rb->rank!=1||rb->shape[0]!=co||rpb->rank!=1||rpb->shape[0]!=co)return 7;
 const float*src=(const float*)rw->data;uint16_t*dst=(uint16_t*)rp->data;
 for(int64_t o=0;o<co;o++)for(int64_t i=0;i<ci;i++)for(int64_t k=0;k<win;k++)
  dst[(k*ci+i)*co+o]=h16(src[(o*ci+i)*win+k]);
 for(int64_t o=0;o<co;o++)((uint16_t*)rpb->data)[o]=h16(((const float*)rb->data)[o]);
 return 0;}
/* Host-memory stand-in for the last-convolution boundary. It recomputes the
   same argument contract the native guard enforces, so a Go request that drops
   or swaps a geometry field is rejected here instead of silently accepted.
   Koi type ids are not nt_dtype ids; map one to the other explicitly. */
static uint32_t koi_dtype(uint32_t t){switch(t){case 2:return 2;case 3:return 4;case 4:return 6;case 5:return 1;case 6:return 5;}return 0;}
int nt_cuda_conv_koi_f16(uint64_t e,uint64_t stream,uint64_t in,uint64_t w,uint64_t b,uint64_t out,
 int32_t n,int32_t tin,int32_t cin,int32_t cout,int32_t win,int32_t stride,int32_t pad,
 int32_t ins,int32_t ons,int32_t ots,uint32_t act,uint32_t it,uint32_t ot,uint32_t at,
 int32_t*res,nt_cuda_detail*d,char*er,size_t c){
 (void)stream;(void)d;(void)er;(void)c;
 if(!res)return 3;*res=-1;
 rec*ri=find(e,in),*rw=find(e,w),*rb=find(e,b),*ro=find(e,out);
 if(!ri||!rw||!rb||!ro)return 2;
 if(ri->dtype!=koi_dtype(it)||ro->dtype!=koi_dtype(ot)||rw->dtype!=1||rb->dtype!=1)return 4;
 if(at!=5&&at!=6)return 4;
 if(n<=0||tin<=0||cin<=0||cout<=0||win<=0||stride<=0)return 3;
 if(pad!=win/2)return 7;
 if(tin%stride)return 7;
 int32_t tout=tin/stride;
 if(ri->rank!=3||ri->shape[0]!=n||ri->shape[1]!=(int64_t)tin+2*(int64_t)pad||ri->shape[2]!=cin)return 7;
 if(ri->strides[0]!=ins||ri->strides[1]!=cin||ri->strides[2]!=1)return 7;
 if(rw->rank!=2||rw->shape[0]!=cout||rw->shape[1]!=(int64_t)win*cin)return 7;
 if(rb->rank!=1||rb->shape[0]!=cout)return 7;
 if(ro->rank!=3||ro->shape[0]!=n||ro->shape[1]!=tout||ro->shape[2]!=cout)return 7;
 if(ro->strides[0]!=ons||ro->strides[1]!=ots||ro->strides[2]!=1)return 7;
 if(act<1||act>3)return 8;
 *res=0;return 0;}
int nt_cuda_linear_f16(uint64_t e,uint64_t stream,uint64_t in,uint64_t w,uint64_t b,uint64_t out,
 int32_t size0,int32_t size1,int32_t k,int32_t cout,int32_t is0,int32_t is1,
 int32_t os0,int32_t os1,uint32_t act,uint32_t it,uint32_t ot,
 int32_t*res,nt_cuda_detail*d,char*er,size_t c){
 (void)stream;(void)d;(void)er;(void)c;
 if(!res)return 3;*res=-1;
 rec*ri=find(e,in),*rw=find(e,w),*rb=find(e,b),*ro=find(e,out);
 if(!ri||!rw||!rb||!ro)return 2;
 if(ri->dtype!=koi_dtype(it)||ro->dtype!=koi_dtype(ot)||rw->dtype!=1||rb->dtype!=1)return 4;
 if(size0<=0||size1<=0||k<=0||cout<=0)return 3;
 if(ri->rank!=3||ri->shape[0]!=size0)return 7;
 if(ri->strides[0]!=is0||ri->strides[2]!=1)return 7;
 if((int64_t)(size1-1)*(int64_t)is1+(int64_t)k>(int64_t)ri->shape[1]*(int64_t)ri->shape[2])return 7;
 if(rw->rank!=2||rw->shape[0]!=cout||rw->shape[1]!=k)return 7;
 if(rb->rank!=1||rb->shape[0]!=cout)return 7;
 if(ro->rank!=3||ro->shape[0]!=size0||ro->shape[1]!=size1||ro->shape[2]!=cout)return 7;
 if(ro->strides[0]!=os0||ro->strides[1]!=os1||ro->strides[2]!=1)return 7;
 if(act<1||act>3)return 8;
 *res=0;return 0;}
`

// The loader resolves every ABI symbol at open, so the host-memory backend also
// declares the untouched CPU tensor and half entrypoints.
var fakeStorageStubs = []string{
	"nt_new", "nt_load", "nt_release", "nt_shape", "nt_data", "nt_run", "nt_new_int64", "nt_data_int64", "nt_sdpa",
	"nt_cuda_half_new", "nt_cuda_half_view", "nt_cuda_half_meta", "nt_cuda_half_bits", "nt_cuda_half_release", "nt_cuda_gemm_f16",
}

func fakeStorageEngine(t *testing.T) *Engine {
	t.Helper()
	dir := t.TempDir()
	source := filepath.Join(dir, "fake-storage.c")
	body := fakeStorageTypes + fakeStorageBackend
	for _, name := range fakeStorageStubs {
		body += "void " + name + "(void){}\n"
	}
	if err := os.WriteFile(source, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	lib := filepath.Join(dir, "fake-storage.so")
	cmd := exec.Command("cc", "-shared", "-fPIC", "-I", "../../internal/tensor", source, "-o", lib)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("fake storage backend: %s: %v", out, err)
	}
	e, err := Open(lib)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := e.Close(); err != nil {
			t.Error(err)
		}
	})
	return e
}

func TestCUDAStorageAndKoiDescriptors(t *testing.T) {
	e := fakeStorageEngine(t)
	d, err := e.NewCUDADevice(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := d.Close(); err != nil {
			t.Error(err)
		}
	})
	s, err := d.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})

	t.Run("roundtrip", func(t *testing.T) {
		bits := []byte{1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0}
		b, err := d.NewStorage(s, DTypeF16, []int64{2, 3}, []KoiTag{TagN, TagT}, bits)
		if err != nil {
			t.Fatal(err)
		}
		got, err := b.Download(s)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, bits) {
			t.Fatalf("roundtrip %v", got)
		}
		v, err := b.View(1, []int64{1, 2}, []int64{3, 1}, []KoiTag{TagN, TagC})
		if err != nil {
			t.Fatal(err)
		}
		if err := v.Upload(s, []byte{9, 0, 8, 0}); err != nil {
			t.Fatal(err)
		}
		got, err = b.Download(s)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, []byte{1, 0, 9, 0, 8, 0, 4, 0, 5, 0, 6, 0}) {
			t.Fatalf("view upload %v", got)
		}
		if err := b.Fill(s, []byte{7, 0}); err != nil {
			t.Fatal(err)
		}
		got, _ = b.Download(s)
		for i := 0; i < len(got); i += 2 {
			if got[i] != 7 || got[i+1] != 0 {
				t.Fatalf("fill %v", got)
			}
		}
		meta, err := b.Meta()
		if err != nil {
			t.Fatal(err)
		}
		if meta.DType != DTypeF16 || len(meta.Shape) != 2 || meta.StorageElements != 6 {
			t.Fatalf("meta %+v", meta)
		}
		for _, h := range []*Storage{v, b} {
			if err := h.Close(); err != nil {
				t.Fatal(err)
			}
		}
	})

	t.Run("copy", func(t *testing.T) {
		src, err := d.NewStorage(s, DTypeF32, []int64{3}, []KoiTag{TagC}, []byte{0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64})
		if err != nil {
			t.Fatal(err)
		}
		dst, err := d.NewStorage(s, DTypeF32, []int64{3}, []KoiTag{TagC}, make([]byte, 12))
		if err != nil {
			t.Fatal(err)
		}
		if err := src.CopyTo(s, dst); err != nil {
			t.Fatal(err)
		}
		got, err := dst.Download(s)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, []byte{0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64}) {
			t.Fatalf("copy %v", got)
		}
		// A destination view that overlaps its own source interval must reject.
		view, err := dst.View(1, []int64{2}, []int64{1}, []KoiTag{TagC})
		if err != nil {
			t.Fatal(err)
		}
		if err := dst.CopyTo(s, view); err == nil {
			t.Fatal("overlapping copy accepted")
		}
		if err := src.Close(); err != nil {
			t.Fatal(err)
		}
		if err := dst.Close(); err != nil {
			t.Fatal(err)
		}
		if err := view.Close(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("koi-descriptor", func(t *testing.T) {
		scale, err := d.NewStorage(s, DTypeF32, []int64{2}, []KoiTag{TagC}, make([]byte, 8))
		if err != nil {
			t.Fatal(err)
		}
		i8, err := d.NewStorage(s, DTypeI8, []int64{2, 2}, []KoiTag{TagN, TagK}, []byte{1, 2, 3, 4})
		if err != nil {
			t.Fatal(err)
		}
		if err := i8.BindScale(scale, TagK); err != nil {
			t.Fatal(err)
		}
		k, err := i8.KoiTensor()
		if err != nil {
			t.Fatal(err)
		}
		if k.DType != KoiI8 || k.Scale == nil || k.Scale.DType != KoiF32 || k.Scale.Tag != TagK || len(k.Dims) != 2 {
			t.Fatalf("koi %+v", k)
		}
		u8, err := d.NewStorage(s, DTypeU8, []int64{1}, []KoiTag{TagC}, []byte{0})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := u8.KoiTensor(); err == nil {
			t.Fatal("U8 storage accepted as a Koi tensor")
		}
		if err := u8.Close(); err != nil {
			t.Fatal(err)
		}
		if err := scale.BindScale(i8, TagK); err == nil {
			t.Fatal("scale binding on F32 accepted")
		}
		if err := i8.BindScale(i8, TagK); err == nil {
			t.Fatal("self scale accepted")
		}
		if err := i8.BindScale(scale, KoiTag(999)); err == nil {
			t.Fatal("unknown scale tag accepted")
		}
		// A live scale owner blocks release of its storage.
		if err := scale.Close(); err == nil {
			t.Fatal("scale owner released while bound")
		}
		if err := i8.BindScale(nil, 0); err != nil {
			t.Fatal(err)
		}
		if err := scale.Close(); err != nil {
			t.Fatal(err)
		}
		if err := i8.Close(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("validation", func(t *testing.T) {
		if _, err := d.NewStorage(s, DTypeNone, []int64{1}, []KoiTag{TagN}, nil); err == nil {
			t.Fatal("unknown dtype accepted")
		}
		if _, err := d.NewStorage(s, DTypeF16, []int64{0}, []KoiTag{TagN}, nil); err == nil {
			t.Fatal("zero dimension accepted")
		}
		if _, err := d.NewStorage(s, DTypeF16, []int64{2}, []KoiTag{TagN}, make([]byte, 2)); err == nil {
			t.Fatal("short upload accepted")
		}
		if _, err := d.NewStorage(s, DTypeF16, []int64{1}, []KoiTag{KoiTag(7)}, make([]byte, 2)); err == nil {
			t.Fatal("unknown tag accepted")
		}
		b, err := d.NewStorage(s, DTypeF16, []int64{2}, []KoiTag{TagN}, make([]byte, 4))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := b.View(^uint64(0), []int64{1}, []int64{1}, []KoiTag{TagN}); err == nil {
			t.Fatal("offset overflow accepted")
		}
		if _, err := b.View(0, []int64{3}, []int64{1}, []KoiTag{TagN}); err == nil {
			t.Fatal("out-of-parent view accepted")
		}
		if _, err := b.View(0, []int64{2}, []int64{-1}, []KoiTag{TagN}); err == nil {
			t.Fatal("negative stride accepted")
		}
		if _, err := b.View(0, []int64{2}, []int64{1}, []KoiTag{TagN, TagT}); err == nil {
			t.Fatal("tag rank mismatch accepted")
		}
		other := fakeStorageEngine(t)
		if _, err := other.backend.cuda(cudaRequest{op: "storage_meta", storage: b.state.id}); err == nil {
			t.Fatal("foreign engine storage accepted")
		}
		if err := b.Close(); err != nil {
			t.Fatal(err)
		}
		if err := b.Close(); err != nil {
			t.Fatal("idempotent close", err)
		}
		if _, err := b.Download(s); !errors.Is(err, ErrClosed) {
			t.Fatalf("closed storage download: %v", err)
		}
	})
}
