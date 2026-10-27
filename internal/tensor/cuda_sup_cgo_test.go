//go:build cgo && linux

package tensor

import (
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

const fakeSUPBackend = `
int nt_cuda_sup_primitive(uint64_t e,uint64_t stream,uint32_t geometry,uint32_t op,int32_t selector,const uint64_t*ids,size_t count,float alpha,nt_cuda_detail*d,char*err,size_t cap){
 (void)stream;(void)d;(void)err;(void)cap;if(!ids||count<1||count>8)return 1;
 rec*p[8];for(size_t i=0;i<count;i++){p[i]=find(e,ids[i]);if(!p[i])return 2;}
 if(op==1){if(count!=2||p[0]->dtype!=p[1]->dtype||p[0]->rank!=p[1]->rank)return 7;size_t n=1;for(int j=0;j<p[0]->rank;j++){if(p[0]->shape[j]!=p[1]->shape[j])return 7;n*=p[0]->shape[j];}
 for(size_t i=0;i<n;i++){size_t rem=i;uint64_t src=p[0]->offset,dst=p[1]->offset;for(int j=p[0]->rank;j-->0;){size_t c=rem%p[0]->shape[j];rem/=p[0]->shape[j];src+=c*p[0]->strides[j];dst+=c*p[1]->strides[j];}memcpy(p[1]->data+dst*dsize(p[1]->dtype),p[0]->data+src*dsize(p[0]->dtype),dsize(p[0]->dtype));}return 0;}
 int out=-1;switch(op){case 2:out=3;break;case 3:case 4:out=1;break;case 5:out=selector?2:3;break;case 6:out=5;break;case 7:case 9:case 12:out=1;break;case 8:out=3;break;case 10:case 11:out=2;break;default:return 1;}
 if((size_t)out>=count)return 1;uint8_t*target=p[out]->data+p[out]->offset*dsize(p[out]->dtype);memcpy(target,&op,4);memcpy(target+4,&selector,4);memcpy(target+8,ids,count*8);memcpy(target+8+count*8,&alpha,4);memcpy(target+12+count*8,&geometry,4);return 0;
}
`

func TestSUPPrimitiveMarshalAndOwnershipThroughActualLoader(t *testing.T) {
	e := fakeStorageEngine(t)
	d, err := e.NewCUDADevice(0)
	if err != nil {
		t.Fatal(err)
	}
	s, err := d.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	defer s.Close()
	for op := SUPConvSiLU; op <= SUPNorm2; op++ {
		selectors := []int{0}
		if op == SUPConvSiLU {
			selectors = []int{0, 1, 2, 3, 4}
		}
		if op == SUPLinear {
			selectors = []int{0, 1}
		}
		for _, selector := range selectors {
			specs, err := SUPPrimitiveSpecs(SUPGeometryTiny, op, selector)
			if err != nil {
				t.Fatal(err)
			}
			handles := make([]*Storage, len(specs))
			for i, sp := range specs {
				n := 1
				for _, x := range sp.Shape {
					n *= int(x)
				}
				handles[i], err = d.NewStorage(s, sp.DType, sp.Shape, sp.Tags, make([]byte, n*sp.DType.Size()))
				if err != nil {
					t.Fatal(err)
				}
			}
			if op == SUPQKVRotary {
				if err = handles[0].BindScale(handles[1], TagC); err != nil {
					t.Fatal(err)
				}
				if err = handles[2].BindScale(handles[3], TagC); err != nil {
					t.Fatal(err)
				}
			}
			if op == SUPNorm2 {
				if err = handles[3].BindScale(handles[4], TagC); err != nil {
					t.Fatal(err)
				}
			}
			if err = d.SUPMath(s, SUPGeometryTiny, op, selector, handles...); err != nil {
				t.Fatalf("op%d/%d: %v", op, selector, err)
			}
			output := []int{0, 0, 3, 1, 1, 3, 5, 1, 3, 1, 2, 2, 1}[op]
			if op == SUPLinear && selector == 1 {
				output = 2
			}
			got, err := handles[output].Download(s)
			if err != nil {
				t.Fatal(err)
			}
			if binary.LittleEndian.Uint32(got) != uint32(op) || int32(binary.LittleEndian.Uint32(got[4:])) != int32(selector) {
				t.Fatal("operation/selector marshalling differs")
			}
			for i, h := range handles {
				if binary.LittleEndian.Uint64(got[8+i*8:]) != h.state.id {
					t.Fatalf("op%d operand%d marshalling differs", op, i)
				}
			}
			alpha := float32(0)
			if op == SUPNorm1 || op == SUPNorm2 {
				alpha = SUPDeepNormAlpha
			}
			if binary.LittleEndian.Uint32(got[8+len(handles)*8:]) != math.Float32bits(alpha) {
				t.Fatal("Go scalar marshalling differs")
			}
			if binary.LittleEndian.Uint32(got[12+len(handles)*8:]) != uint32(SUPGeometryTiny) {
				t.Fatal("geometry marshalling differs")
			}
			if err = d.SUPMath(s, SUPGeometry(99), op, selector, handles...); err == nil {
				t.Fatal("unknown geometry reached permissive fake backend")
			}
			if op != SUPSinCos {
				if err = d.SUPMath(s, SUPGeometryFull, op, selector, handles...); err == nil {
					t.Fatal("foreign whole operand profile reached permissive fake backend")
				}
			}
			if err = d.SUPMath(s, SUPGeometryTiny, op, selector, handles[:len(handles)-1]...); err == nil {
				t.Fatal("missing operand accepted")
			}
			for _, h := range handles {
				if h.DType() == DTypeI8 {
					if err = h.BindScale(nil, 0); err != nil {
						t.Fatal(err)
					}
				}
			}
			for i := len(handles) - 1; i >= 0; i-- {
				if err = handles[i].Close(); err != nil {
					t.Fatal(err)
				}
			}
			if err = d.SUPMath(s, SUPGeometryTiny, op, selector, handles...); err == nil {
				t.Fatal("closed operands accepted")
			}
		}
	}
	other, err := e.NewCUDADevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer other.Close()
	h, err := other.NewStorage(s, DTypeF16, []int64{2}, []KoiTag{TagC}, []byte{0, 0, 0, 0})
	if err == nil || h != nil {
		t.Fatal("foreign stream accepted at allocation")
	}
}

func TestSUPStridedCopyThroughActualLoader(t *testing.T) {
	e := fakeStorageEngine(t)
	d, err := e.NewCUDADevice(0)
	if err != nil {
		t.Fatal(err)
	}
	s, err := d.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	defer s.Close()
	in, err := d.NewStorage(s, DTypeI8, []int64{2, 3}, []KoiTag{TagM, TagN}, []byte{0, 1, 2, 3, 4, 5})
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	parent, err := d.NewStorage(s, DTypeI8, []int64{3, 2}, []KoiTag{TagN, TagM}, make([]byte, 6))
	if err != nil {
		t.Fatal(err)
	}
	defer parent.Close()
	out, err := parent.View(0, []int64{2, 3}, []int64{1, 2}, []KoiTag{TagM, TagN})
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	if err = d.SUPMath(s, SUPGeometryTiny, SUPCopyStrided, 0, in, out); err != nil {
		t.Fatal(err)
	}
	got, err := parent.Download(s)
	if err != nil || string(got) != string([]byte{0, 3, 1, 4, 2, 5}) {
		t.Fatalf("strided copy=%v err=%v", got, err)
	}
	source, err := in.Download(s)
	if err != nil || string(source) != string([]byte{0, 1, 2, 3, 4, 5}) {
		t.Fatal("copy mutated input")
	}
	if err = d.SUPMath(s, SUPGeometryTiny, SUPCopyStrided, 0, in, in); err == nil {
		t.Fatal("overlapping copy accepted")
	}
}

func TestSUPPrimitiveMandatorySymbolRejectedByActualLoader(t *testing.T) {
	raw, err := os.ReadFile("loader.c")
	if err != nil {
		t.Fatal(err)
	}
	matches := regexp.MustCompile(`LOAD\([^,]+,"([^"]+)"\)`).FindAllStringSubmatch(string(raw), -1)
	body := "unsigned int nt_version(void){return 5;}\n"
	for _, m := range matches {
		if m[1] != "nt_cuda_sup_primitive" && m[1] != "nt_version" {
			body += "void " + m[1] + "(void){}\n"
		}
	}
	dir := t.TempDir()
	src := filepath.Join(dir, "missing-sup.c")
	lib := filepath.Join(dir, "missing-sup.so")
	if err = os.WriteFile(src, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command("cc", "-shared", "-fPIC", src, "-o", lib).CombinedOutput(); err != nil {
		t.Fatalf("fake C compile: %s %v", out, err)
	}
	if _, err = Open(lib); err == nil || !strings.Contains(err.Error(), "missing ABI symbol nt_cuda_sup_primitive") {
		t.Fatal("missing mandatory SUP symbol was not rejected", err)
	}
}

func TestSUPPrimitiveGeometryThroughActualLoader(t *testing.T) {
	e := fakeStorageEngine(t)
	d, err := e.NewCUDADevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	s, err := d.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	specs, err := SUPPrimitiveSpecs(SUPGeometryFull, SUPSinCos, 0)
	if err != nil {
		t.Fatal(err)
	}
	handles := make([]*Storage, len(specs))
	for i, spec := range specs {
		handles[i], err = d.NewStorage(s, spec.DType, spec.Shape, spec.Tags, make([]byte, 4096*32*2))
		if err != nil {
			t.Fatal(err)
		}
		defer handles[i].Close()
	}
	for _, geometry := range []SUPGeometry{SUPGeometryTiny, SUPGeometryFull, SUPGeometryShort} {
		if err = d.SUPMath(s, geometry, SUPSinCos, 0, handles...); err != nil {
			t.Fatal(err)
		}
		got, err := handles[1].Download(s)
		if err != nil {
			t.Fatal(err)
		}
		if binary.LittleEndian.Uint32(got) != uint32(SUPSinCos) || binary.LittleEndian.Uint32(got[12+len(handles)*8:]) != uint32(geometry) {
			t.Fatal("geometry did not cross actual loader independently of constant RoPE shape", geometry)
		}
	}
}
