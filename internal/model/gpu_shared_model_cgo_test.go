//go:build cgo && linux

package model

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

type sharedModelFake struct {
	engine        *tensor.Engine
	device        *tensor.Device
	stream        *tensor.Stream
	fault, events string
}

func sharedModelFixtureSource(t *testing.T, file, name string) ast.Expr {
	t.Helper()
	f, err := parser.ParseFile(token.NewFileSet(), file, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, decl := range f.Decls {
		g, ok := decl.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, spec := range g.Specs {
			v, ok := spec.(*ast.ValueSpec)
			if ok && len(v.Names) == 1 && v.Names[0].Name == name && len(v.Values) == 1 {
				return v.Values[0]
			}
		}
	}
	t.Fatalf("missing maintained fake source %s", name)
	return nil
}

func sharedModelFixtureString(t *testing.T, file, name string) string {
	t.Helper()
	literal, ok := sharedModelFixtureSource(t, file, name).(*ast.BasicLit)
	if !ok || literal.Kind != token.STRING {
		t.Fatal("fake source is not a string literal")
	}
	body, err := strconv.Unquote(literal.Value)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func newSharedModelFake(t *testing.T) *sharedModelFake {
	t.Helper()
	dir := t.TempDir()
	if root := os.Getenv("MODEL_FAKE_CC_RECEIPTS"); root != "" {
		if err := os.MkdirAll(root, 0700); err != nil {
			t.Fatal(err)
		}
		var err error
		dir, err = os.MkdirTemp(root, "shared-model-")
		if err != nil {
			t.Fatal(err)
		}
	}
	f := &sharedModelFake{fault: filepath.Join(dir, "fault"), events: filepath.Join(dir, "events")}
	f.setFault(t, "")
	t.Setenv("MODEL_FAKE_FAULT", f.fault)
	t.Setenv("MODEL_FAKE_EVENTS", f.events)
	storage := filepath.Join("..", "tensor", "cuda_storage_test.go")
	body := sharedModelFixtureString(t, storage, "fakeStorageTypes") + sharedModelFixtureString(t, storage, "fakeStorageBackend") + sharedModelFixtureString(t, filepath.Join("..", "tensor", "cuda_sup_cgo_test.go"), "fakeSUPBackend")
	body = strings.ReplaceAll(body, "table[256]", "table[4096]")
	body = strings.ReplaceAll(body, "i<256", "i<4096")
	body = strings.Replace(body, "*out=200;return 0;", "static uint64_t next_stream=200;*out=next_stream++;return 0;", 1)
	oldFree := "if(p->allocation==p->id&&p->offset==0)free(p->data);memset(p,0,sizeof(*p));return 0;"
	newFree := "uint8_t*data=p->data;uint64_t allocation=p->allocation;memset(p,0,sizeof(*p));for(int i=0;i<4096;i++)if(table[i].id&&table[i].allocation==allocation)return 0;free(data);return 0;"
	if strings.Count(body, oldFree) != 1 {
		t.Fatal("maintained fake release shape changed")
	}
	body = strings.Replace(body, oldFree, newFree, 1)
	for _, name := range []string{"nt_close", "nt_cuda_device_close", "nt_cuda_stream_close", "nt_cuda_storage_new", "nt_cuda_storage_view", "nt_cuda_storage_scale", "nt_cuda_storage_release"} {
		body = strings.ReplaceAll(body, name+"(", "base_"+name+"(")
	}
	stubs, ok := sharedModelFixtureSource(t, storage, "fakeStorageStubs").(*ast.CompositeLit)
	if !ok {
		t.Fatal("maintained fake stub list changed")
	}
	for _, entry := range stubs.Elts {
		literal, ok := entry.(*ast.BasicLit)
		if !ok {
			t.Fatal("fake stub is not literal")
		}
		name, err := strconv.Unquote(literal.Value)
		if err != nil {
			t.Fatal(err)
		}
		body += "void " + name + "(void){}\n"
	}
	body += sharedModelFaultC
	source, library := filepath.Join(dir, "fake-shared-model.c"), filepath.Join(dir, "fake-shared-model.so")
	if err := os.WriteFile(source, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("cc", "-shared", "-fPIC", "-I", "../../internal/tensor", source, "-o", library)
	start := time.Now().UTC()
	output, failure := cmd.CombinedOutput()
	exit := 0
	if failure != nil {
		exit = -1
		if e, ok := failure.(*exec.ExitError); ok {
			exit = e.ExitCode()
		}
	}
	receipt := map[string]any{"test": t.Name(), "argv": cmd.Args, "started_utc": start, "finished_utc": time.Now().UTC(), "exit": exit, "source_sha256": fmt.Sprintf("%x", sha256.Sum256([]byte(body))), "combined_sha256": fmt.Sprintf("%x", sha256.Sum256(output)), "synthetic_host_loader_only": true}
	encoded, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "compiler.json"), append(encoded, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(dir, "compiler.log"), output, 0600); err != nil {
		t.Fatal(err)
	}
	if failure != nil {
		t.Fatalf("synthetic loader compiler: %v: %s", failure, output)
	}
	t.Logf("synthetic model ownership loader %s", dir)
	f.engine, err = tensor.Open(library)
	if err != nil {
		t.Fatal(err)
	}
	f.device, err = f.engine.NewCUDADevice(0)
	if err != nil {
		t.Fatal(err)
	}
	f.stream, err = f.device.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		f.setFault(t, "")
		if err := f.stream.Close(); err != nil {
			t.Error(err)
		}
		if err := f.device.Close(); err != nil {
			t.Error(err)
		}
		if err := f.engine.Close(); err != nil {
			t.Error(err)
		}
	})
	return f
}

func (f *sharedModelFake) setFault(t *testing.T, text string) {
	t.Helper()
	if err := os.WriteFile(f.fault, []byte(text), 0600); err != nil {
		t.Fatal(err)
	}
}
func (f *sharedModelFake) log(t *testing.T) []string {
	t.Helper()
	data, err := os.ReadFile(f.events)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	return strings.FieldsFunc(strings.TrimSpace(string(data)), func(c rune) bool { return c == '\n' })
}
func (f *sharedModelFake) storage(t *testing.T, dtype tensor.DType, shape []int64, tags []tensor.KoiTag) *tensor.Storage {
	t.Helper()
	n := 1
	for _, x := range shape {
		n *= int(x)
	}
	h, err := f.device.NewStorage(f.stream, dtype, shape, tags, make([]byte, n*dtype.Size()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := h.Close(); err != nil {
			t.Error(err)
		}
	})
	return h
}

const sharedModelFaultC = `
static unsigned long long new_calls=0;
static int model_fault(const char*op,uint64_t id){
 const char*path=getenv("MODEL_FAKE_FAULT");if(!path)return 0;FILE*f=fopen(path,"r");if(!f)return 0;
 char names[8][32];unsigned long long ids[8];int left[8],n=0,hit=0;
 while(n<8&&fscanf(f,"%31s %llu %d",names[n],&ids[n],&left[n])==3)n++;fclose(f);
 for(int i=0;i<n;i++)if(!hit&&left[i]>0&&!strcmp(names[i],op)&&(ids[i]==0||ids[i]==id)){left[i]--;hit=1;}
 if(hit){f=fopen(path,"w");if(!f)abort();for(int i=0;i<n;i++)fprintf(f,"%s %llu %d\n",names[i],ids[i],left[i]);fclose(f);}return hit;
}
static void model_log(const char*op,uint64_t id,uint64_t extra){const char*p=getenv("MODEL_FAKE_EVENTS");if(!p)return;FILE*f=fopen(p,"a");if(!f)abort();fprintf(f,"%s %llu %llu\n",op,(unsigned long long)id,(unsigned long long)extra);fclose(f);}
static void model_clear(void){for(int i=0;i<4096;i++)if(table[i].id){uint8_t*data=table[i].data;for(int j=i;j<4096;j++)if(table[j].id&&table[j].data==data)memset(&table[j],0,sizeof(rec));free(data);}}
int nt_close(uint64_t id,char*e,size_t c){model_clear();return base_nt_close(id,e,c);}
int nt_cuda_device_close(uint64_t e,uint64_t id,nt_cuda_detail*d,char*er,size_t c){if(model_fault("device",id))return 3;model_clear();return base_nt_cuda_device_close(e,id,d,er,c);}
int nt_cuda_stream_close(uint64_t e,uint64_t id,nt_cuda_detail*d,char*er,size_t c){model_log("stream",id,0);if(model_fault("stream",id))return 3;return base_nt_cuda_stream_close(e,id,d,er,c);}
int nt_cuda_storage_new(uint64_t e,uint64_t dev,uint64_t stream,uint32_t dtype,const int64_t*shape,size_t rank,const int32_t*tags,const uint8_t*bytes,size_t count,uint64_t*out,nt_cuda_detail*d,char*er,size_t c){
 ++new_calls;if(model_fault("new",new_calls)){model_log("new-failed",new_calls,0);return 3;}int r=base_nt_cuda_storage_new(e,dev,stream,dtype,shape,rank,tags,bytes,count,out,d,er,c);if(!r)model_log("new",*out,count);return r;}
int nt_cuda_storage_view(uint64_t e,uint64_t parent,uint32_t dtype,uint64_t offset,const int64_t*shape,const int64_t*strides,const int32_t*tags,size_t rank,uint64_t*out,nt_cuda_detail*d,char*er,size_t c){
 int r=base_nt_cuda_storage_view(e,parent,dtype,offset,shape,strides,tags,rank,out,d,er,c);if(!r){const char*p=getenv("MODEL_FAKE_EVENTS");if(p){FILE*f=fopen(p,"a");if(!f)abort();fprintf(f,"view %llu %llu %zu %llu",(unsigned long long)*out,(unsigned long long)parent,rank,(unsigned long long)offset);for(size_t i=0;i<rank;i++)fprintf(f," %lld",(long long)shape[i]);for(size_t i=0;i<rank;i++)fprintf(f," %lld",(long long)strides[i]);for(size_t i=0;i<rank;i++)fprintf(f," %d",tags[i]);fputc('\n',f);fclose(f);}}return r;}
int nt_cuda_storage_scale(uint64_t e,uint64_t id,uint64_t scale,int32_t tag,nt_cuda_detail*d,char*er,size_t c){model_log(scale?"bind":"unbind",id,scale);if(!scale&&model_fault("unbind",id))return 3;return base_nt_cuda_storage_scale(e,id,scale,tag,d,er,c);}
int nt_cuda_storage_release(uint64_t e,uint64_t id,nt_cuda_detail*d,char*er,size_t c){model_log("release",id,0);if(model_fault("release",id))return 3;return base_nt_cuda_storage_release(e,id,d,er,c);}
`

func syntheticHACGPUWeights(t *testing.T) HACGPUWeights {
	t.Helper()
	spec, err := Pinned(HAC)
	if err != nil {
		t.Fatal(err)
	}
	receipts, err := WeightManifest(HAC)
	if err != nil {
		t.Fatal(err)
	}
	w := HACGPUWeights{spec: spec, receipts: receipts}
	for i, p := range hacGPUConvParams() {
		w.convs[i] = hacGPUConvWeights{make([]float32, p.COut*p.CIn*p.WinLen), make([]float32, p.COut)}
	}
	w.fused = make([]FLSTMCache, 7)
	for i := range w.fused {
		w.fused[i] = FLSTMCache{Params: FLSTMParams{C: 1024, K: 128}, DownIH: make([]int8, 128*1024), DownHH: make([]int8, 128*1024), DownScaleIH: make([]uint16, 128), DownScaleHH: make([]uint16, 128), Up: make([]int8, 4096*256), UpScale: make([]uint16, 4096), UpBias: make([]uint16, 4096)}
	}
	w.output = HACOutputCache{down: make([]int8, 128*1024), downScale: make([]uint16, 128), up: make([]uint16, 1024*128)}
	return w
}

func syntheticSUPGPUWeights(t *testing.T) SUPGPUWeights {
	t.Helper()
	spec, err := Pinned(SUP)
	if err != nil {
		t.Fatal(err)
	}
	receipts, err := WeightManifest(SUP)
	if err != nil {
		t.Fatal(err)
	}
	w := SUPGPUWeights{spec: spec, receipts: receipts}
	half := supHalfBuffer([]uint16{0x3800}, []int64{1}, "C")
	for i := range w.convs {
		w.convs[i] = [2]supGPUBuffer{half, half}
	}
	for i := range w.encoders {
		values := map[string]supGPUBuffer{}
		for _, name := range []string{"qkv_half", "proj_weight", "proj_bias", "norm1_weight", "norm2_weight", "fc1_half", "fc1_fp8", "fc2_fp8"} {
			values[name] = half
		}
		values["qkv_int8"] = supByteBuffer([]byte{1}, tensor.DTypeI8, []int64{1}, "C")
		values["qkv_scale"] = supF32Buffer([]float32{1}, []int64{1})
		w.encoders[i] = SUPEncoderGPUCache{before: values, after: values}
	}
	w.upsample = [2]supGPUBuffer{half, half}
	w.crf = half
	return w
}

func TestHACSharedModelKeepsCachesAcrossExplicitProfiles(t *testing.T) {
	f := newSharedModelFake(t)
	w := syntheticHACGPUWeights(t)
	m, err := NewHACGPU(f.device, f.stream, w)
	if m != nil {
		t.Cleanup(func() {
			if err := m.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	cacheIDs := func(m *HACGPU) []uint64 {
		var handles []*tensor.Storage
		for _, cache := range m.small {
			handles = append(handles, cache.Weights, cache.Bias)
		}
		handles = append(handles, m.last.Weights, m.last.Bias)
		for _, cache := range m.fused.caches {
			handles = append(handles, cache.DownIH, cache.DownHH, cache.DownScaleIH, cache.DownScaleHH, cache.Up, cache.UpScale, cache.UpBias)
		}
		handles = append(handles, m.output.down, m.output.downScale, m.output.up)
		ids := make([]uint64, len(handles))
		for i, handle := range handles {
			meta, err := handle.Meta()
			if err != nil {
				t.Fatal(err)
			}
			ids[i] = meta.Allocation
		}
		return ids
	}
	ids := cacheIDs(m)
	cacheBytes, err := m.CacheTensors(f.stream)
	if err != nil {
		t.Fatal(err)
	}
	stop := errors.New("stop after real input transport")
	workspaces := map[uint64]bool{}
	for _, profile := range []HACGPUProfile{SourceDefaultHACGPUProfile(), SourceShortHACGPUProfile(), SourceDefaultHACGPUProfile(), SourceShortHACGPUProfile(), SourceDefaultHACGPUProfile(), SourceShortHACGPUProfile()} {
		p, err := PlanHACGPU(profile)
		if err != nil {
			t.Fatal(err)
		}
		s, err := f.device.NewStream()
		if err != nil {
			t.Fatal(err)
		}
		input := f.storage(t, tensor.DTypeF16, []int64{64, 1, profile.Chunk}, []tensor.KoiTag{tensor.TagN, tensor.TagC, tensor.TagT})
		before := len(f.log(t))
		result, err := m.ForwardObserved(context.Background(), s, input, p, HACGPUBatch{64}, 0, func(g HACGPUObservation) error {
			if g.Name != "input" || g.View.Geometry.T != profile.Chunk {
				t.Fatal("forward changed selected plan")
			}
			meta, err := g.Workspace.Meta()
			if err != nil || workspaces[meta.Allocation] {
				t.Fatal("forward reused an invocation workspace", err)
			}
			workspaces[meta.Allocation] = true
			return stop
		})
		if result != nil || !errors.Is(err, stop) || len(m.pending) != 0 {
			t.Fatalf("input transport/cleanup: %v", err)
		}
		allocations := 0
		for _, event := range f.log(t)[before:] {
			if strings.HasPrefix(event, "new ") {
				allocations++
			}
		}
		if allocations != 1 {
			t.Fatalf("input-stage forward allocated %d owners; want one workspace and no cache uploads", allocations)
		}
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(ids, cacheIDs(m)) {
			t.Fatal("profile transport replaced or released persistent cache")
		}
		current, err := m.CacheTensors(f.stream)
		if err != nil || !reflect.DeepEqual(current, cacheBytes) {
			t.Fatal("profile transport changed persistent cache bytes", err)
		}
		if err := input.Close(); err != nil {
			t.Fatal(err)
		}
	}
	other, err := NewHACGPU(f.device, f.stream, w)
	if other != nil {
		t.Cleanup(func() {
			if err := other.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	if slices.Equal(ids, cacheIDs(other)) {
		t.Fatal("separate model shared a global cache owner")
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := other.CacheTensors(f.stream); err != nil {
		t.Fatal("closing one model released another model", err)
	}
}

func TestSUPSharedModelKeepsRoPEAndScaleBindingsAcrossProfiles(t *testing.T) {
	f := newSharedModelFake(t)
	m, err := NewSUPGPU(context.Background(), f.device, f.stream, syntheticSUPGPUWeights(t))
	if m != nil {
		t.Cleanup(func() {
			if err := m.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	fullMeta, err := m.sincos.Meta()
	if err != nil {
		t.Fatal(err)
	}
	fullBytes, err := m.sincos.Download(f.stream)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(fullMeta.Shape, []int64{256, 8, 16, 8}) {
		t.Fatal("persistent RoPE is not full source table")
	}
	persistent := make([]tensor.StorageMeta, 18)
	for i, cache := range m.encoders {
		persistent[i], err = cache["qkv_int8"].Meta()
		if err != nil || !persistent[i].ScalePresent {
			t.Fatal("persistent QKV scale missing", err)
		}
	}
	stop := errors.New("stop after real input transport")
	for _, profile := range []SUPGPUProfile{FullFixedSUPGPUProfile(), ShortFixedSUPGPUProfile(), FullFixedSUPGPUProfile(), ShortFixedSUPGPUProfile(), FullFixedSUPGPUProfile(), ShortFixedSUPGPUProfile()} {
		p, err := PlanSUPGPU(profile)
		if err != nil {
			t.Fatal(err)
		}
		s, err := f.device.NewStream()
		if err != nil {
			t.Fatal(err)
		}
		input := f.storage(t, tensor.DTypeF16, []int64{32, 1, profile.Chunk}, tagValues("NCT"))
		before := len(f.log(t))
		result, err := m.ForwardObserved(context.Background(), s, input, p, func(g SUPGPUObservation) error {
			if g.Name != "input" || !slices.Equal(g.Tensors[0].Shape, []int64{32, 1, profile.Chunk}) {
				t.Fatal("forward changed selected plan")
			}
			return stop
		})
		if result != nil || !errors.Is(err, stop) || len(m.pending) != 0 {
			t.Fatalf("SUP input transport/cleanup: %v", err)
		}
		var prefix []string
		allocations := 0
		for _, event := range f.log(t)[before:] {
			fields := strings.Fields(event)
			if fields[0] == "new" {
				allocations++
			}
			if len(fields) > 3 && fields[0] == "view" && fields[2] == strconv.FormatUint(fullMeta.Allocation, 10) {
				prefix = fields
			}
		}
		if allocations != 1 {
			t.Fatalf("input-stage forward allocated %d owners; want one activation and no cache uploads", allocations)
		}
		want := []string{"4", "0", strconv.FormatInt(profile.EncoderT/16, 10), "8", "16", "8", "1024", "128", "8", "1", "84", "68", "116", "100"}
		if len(prefix) < 3 || !reflect.DeepEqual(prefix[3:], want) {
			t.Fatalf("wrong invocation RoPE prefix: %v", prefix)
		}
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
		for i, cache := range m.encoders {
			meta, err := cache["qkv_int8"].Meta()
			if err != nil || !reflect.DeepEqual(meta, persistent[i]) {
				t.Fatal("profile cleanup changed persistent scale owner", err)
			}
		}
		meta, err := m.sincos.Meta()
		if err != nil || !reflect.DeepEqual(meta, fullMeta) {
			t.Fatal("profile changed full RoPE metadata", err)
		}
		bits, err := m.sincos.Download(f.stream)
		if err != nil || !slices.Equal(bits, fullBytes) {
			t.Fatal("profile changed full RoPE bytes", err)
		}
		if err := input.Close(); err != nil {
			t.Fatal(err)
		}
	}
	short, _ := PlanSUPGPU(ShortFixedSUPGPUProfile())
	wrong := f.storage(t, tensor.DTypeF16, []int64{32, 1, 12288}, tagValues("NCT"))
	if _, err := m.Forward(context.Background(), f.stream, wrong, short); !errors.Is(err, ErrGPUPlanLayout) {
		t.Fatalf("mixed input/plan admitted: %v", err)
	}
	other, err := NewSUPGPU(context.Background(), f.device, f.stream, syntheticSUPGPUWeights(t))
	if other != nil {
		t.Cleanup(func() {
			if err := other.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	otherMeta, err := other.sincos.Meta()
	if err != nil || otherMeta.Allocation == fullMeta.Allocation {
		t.Fatal("separate SUP model shared its RoPE owner", err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	for i, cache := range other.encoders {
		meta, err := cache["qkv_int8"].Meta()
		if err != nil || !meta.ScalePresent || meta.Allocation == persistent[i].Allocation {
			t.Fatal("closing one SUP model affected the other's scale/cache owner", err)
		}
	}
	if _, err := other.sincos.Download(f.stream); err != nil {
		t.Fatal("closing one SUP model affected the other's RoPE bytes", err)
	}
}

func TestSUPActivationCleanupDetachesViewsWithoutChangingPersistentBinding(t *testing.T) {
	f := newSharedModelFake(t)
	persistent := f.storage(t, tensor.DTypeI8, []int64{1}, tagValues("C"))
	persistentScale := f.storage(t, tensor.DTypeF32, []int64{1}, tagValues("C"))
	if err := persistent.BindScale(persistentScale, tensor.TagC); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := persistent.BindScale(nil, 0); err != nil {
			t.Error(err)
		}
	})
	activation := f.storage(t, tensor.DTypeI8, []int64{1}, tagValues("C"))
	scale := f.storage(t, tensor.DTypeF32, []int64{1}, tagValues("C"))
	if err := activation.BindScale(scale, tensor.TagC); err != nil {
		t.Fatal(err)
	}
	view, err := activation.View(0, []int64{1}, []int64{1}, tagValues("C"))
	if err != nil {
		t.Fatal(err)
	}
	if err := scale.Close(); err == nil {
		t.Fatal("released scale while activation/view still borrow it")
	}
	failed, err := closeSUPStorages([]*tensor.Storage{activation, scale, view})
	if err != nil || len(failed) != 0 {
		t.Fatal("activation cleanup retained bindings or owners", err)
	}
	meta, err := persistent.Meta()
	if err != nil || !meta.ScalePresent {
		t.Fatal("activation cleanup detached the persistent scale", err)
	}
	if _, err := persistentScale.Meta(); err != nil {
		t.Fatal("activation cleanup released the persistent scale", err)
	}
}

func TestGPUModelUploadFailureRetainsTopLevelOwner(t *testing.T) {
	for _, lane := range []string{"HAC", "SUP"} {
		t.Run(lane, func(t *testing.T) {
			f := newSharedModelFake(t)
			var owner interface{ Close() error }
			var err error
			if lane == "HAC" {
				f.setFault(t, "new 2 1\nrelease 1 2\n")
				var m *HACGPU
				m, err = NewHACGPU(f.device, f.stream, syntheticHACGPUWeights(t))
				if m == nil || m.small[0] == nil || m.small[0].Weights == nil {
					t.Fatal("lost HAC partial owner", err)
				}
				owner = m
			} else {
				f.setFault(t, "new 2 1\nrelease 1 1\n")
				var m *SUPGPU
				m, err = NewSUPGPU(context.Background(), f.device, f.stream, syntheticSUPGPUWeights(t))
				if m == nil || len(m.owned) != 1 {
					t.Fatal("lost SUP partial owner", err)
				}
				owner = m
			}
			if err == nil {
				t.Fatal("upload failure was suppressed")
			}
			if err := owner.Close(); err != nil {
				t.Fatal("top-level cleanup retry failed", err)
			}
			before := len(f.log(t))
			if err := owner.Close(); err != nil || len(f.log(t)) != before {
				t.Fatal("successful top-level close repeated releases", err)
			}
		})
	}
}

func TestGPUCacheUploadFailureRetainsPartialOwnerForCloseRetry(t *testing.T) {
	f := newSharedModelFake(t)
	f.setFault(t, "new 2 1\nrelease 1 1\n")
	p := hacGPUConvParams()[0]
	cache, err := NewHACSmallConv(f.device, f.stream, p, make([]float32, p.COut*p.CIn*p.WinLen), make([]float32, p.COut))
	if err == nil || cache == nil || cache.Weights == nil {
		t.Fatal("upload/cleanup failure lost partial cache", err)
	}
	if _, err := cache.Weights.Meta(); err != nil {
		t.Fatal("failed owner no longer live", err)
	}
	if err := cache.Close(); err != nil || cache.Weights != nil || cache.Bias != nil {
		t.Fatal("close retry did not release/clear cache", err)
	}
	before := len(f.log(t))
	if err := cache.Close(); err != nil || len(f.log(t)) != before {
		t.Fatal("successful close repeated native releases", err)
	}
}

func TestSUPFailedScaleDetachRetainsOwnersAndRecoversAfterDeviceClose(t *testing.T) {
	f := newSharedModelFake(t)
	m, err := NewSUPGPU(context.Background(), f.device, f.stream, syntheticSUPGPUWeights(t))
	if err != nil {
		t.Fatal(err)
	}
	scale := m.encoders[0]["qkv_scale"]
	i8 := m.encoders[0]["qkv_int8"]
	id, err := i8.Meta()
	if err != nil {
		t.Fatal(err)
	}
	f.setFault(t, fmt.Sprintf("unbind %d 1\n", id.Allocation))
	count := len(m.owned)
	if err := m.Close(); err == nil || len(m.owned) != count {
		t.Fatal("failed detach discarded dependent owners", err)
	}
	if _, err := scale.Meta(); err != nil {
		t.Fatal("failed detach released needed scale", err)
	}
	p, _ := PlanSUPGPU(FixedSUPGPUProfile())
	if _, err := m.Forward(context.Background(), f.stream, nil, p); !errors.Is(err, ErrClosed) {
		t.Fatal("closing model resumed", err)
	}
	if err := m.Close(); err != nil || len(m.owned) != 0 {
		t.Fatal("scale detach retry failed", err)
	}
	m, err = NewSUPGPU(context.Background(), f.device, f.stream, syntheticSUPGPUWeights(t))
	if err != nil {
		t.Fatal(err)
	}
	if err := f.device.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil || len(m.owned) != 0 {
		t.Fatal("successful device recovery left Go owners unrecoverable", err)
	}
}

func TestHACForwardCleanupFailureRetainsWorkspaceAndStopsReuse(t *testing.T) {
	f := newSharedModelFake(t)
	m, err := NewHACGPU(f.device, f.stream, syntheticHACGPUWeights(t))
	if m != nil {
		t.Cleanup(func() {
			if err := m.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	p, err := PlanHACGPU(FixedHACGPUProfile())
	if err != nil {
		t.Fatal(err)
	}
	input := f.storage(t, tensor.DTypeF16, []int64{64, 1, 192}, tagValues("NCT"))
	stop := errors.New("stop after workspace failure armed")
	var allocation uint64
	result, err := m.ForwardObserved(context.Background(), f.stream, input, p, HACGPUBatch{64}, 0, func(g HACGPUObservation) error {
		meta, err := g.Workspace.Meta()
		if err != nil {
			t.Fatal(err)
		}
		allocation = meta.Allocation
		f.setFault(t, fmt.Sprintf("release %d 1\n", allocation))
		return stop
	})
	if result != nil || !errors.Is(err, stop) || len(m.pending) != 1 {
		t.Fatal("failed workspace owner lost", err, len(m.pending))
	}
	meta, err := m.pending[0].Meta()
	if err != nil || meta.Allocation != allocation {
		t.Fatal("wrong retained workspace", err)
	}
	before := len(f.log(t))
	if _, err := m.Forward(context.Background(), f.stream, input, p, HACGPUBatch{64}); err == nil || !strings.Contains(err.Error(), "cleanup pending") || len(f.log(t)) != before {
		t.Fatal("model ran through unresolved cleanup", err)
	}
	if err := m.Close(); err != nil || len(m.pending) != 0 {
		t.Fatal("workspace close retry failed", err)
	}
	if _, err := m.Forward(context.Background(), f.stream, input, p, HACGPUBatch{64}); !errors.Is(err, ErrClosed) {
		t.Fatal("closed model resumed", err)
	}
}

func TestSUPSharedCacheColdWarmObservationUsesActualUploadedBytes(t *testing.T) {
	f := newSharedModelFake(t)
	w := syntheticSUPGPUWeights(t)
	before := map[string]supGPUBuffer{}
	for name, value := range w.encoders[0].before {
		before[name] = value
	}
	before["proj_weight"] = supHalfBuffer([]uint16{0x3801}, []int64{1}, "C")
	w.encoders[0].before = before
	m, err := NewSUPGPU(context.Background(), f.device, f.stream, w)
	if m != nil {
		t.Cleanup(func() {
			if err := m.Close(); err != nil {
				t.Error(err)
			}
		})
	}
	if err != nil {
		t.Fatal(err)
	}
	var observations []SUPGPUObservation
	v := &supGPUInvocation{model: m, ctx: context.Background(), stream: f.stream, observe: func(g SUPGPUObservation) error { observations = append(observations, g); return nil }}
	if err := v.observeEncoderCache(0); err != nil {
		t.Fatal(err)
	}
	if len(observations) != 2 || observations[0].Name != "encoder.00.cache.before_remove_bits" || observations[0].Origin != SUPGPUPreparedCache || observations[1].Name != "encoder.00.cache.after_remove_bits" || observations[1].Origin != SUPGPUUploaded {
		t.Fatal("cold cache provenance/order lost", observations)
	}
	find := func(g SUPGPUObservation) []byte {
		for _, x := range g.Tensors {
			if x.Name == "proj_weight" {
				return x.Data
			}
		}
		t.Fatal("missing projection cache")
		return nil
	}
	if !slices.Equal(find(observations[0]), []byte{1, 0x38}) || !slices.Equal(find(observations[1]), []byte{0, 0x38}) {
		t.Fatal("cold prepared/actual distinction lost")
	}
	if err := m.encoders[0]["proj_weight"].Upload(f.stream, []byte{2, 0x38}); err != nil {
		t.Fatal(err)
	}
	observations = nil
	if err := v.observeEncoderCache(0); err != nil {
		t.Fatal(err)
	}
	if len(observations) != 1 || observations[0].Name != "encoder.00.cache.warm" || observations[0].Origin != SUPGPUUploaded || !slices.Equal(find(observations[0]), []byte{2, 0x38}) {
		t.Fatal("warm cache hid actual uploaded mutation", observations)
	}
	v.observe = nil
	if err := v.observeEncoderCache(1); err != nil {
		t.Fatal(err)
	}
	observations = nil
	v.observe = func(g SUPGPUObservation) error { observations = append(observations, g); return nil }
	if err := v.observeEncoderCache(1); err != nil || len(observations) != 1 || observations[0].Name != "encoder.01.cache.warm" {
		t.Fatal("late observer replayed cold preparation", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	v.ctx = ctx
	if err := v.observeEncoderCache(2); !errors.Is(err, context.Canceled) || m.cacheEntered[2] {
		t.Fatal("cancelled entry consumed cold event", err)
	}
	v.ctx = context.Background()
	stop := errors.New("observer stopped")
	v.observe = func(g SUPGPUObservation) error { return stop }
	if err := v.observeEncoderCache(2); !errors.Is(err, stop) || !m.cacheEntered[2] {
		t.Fatal("observer failure lost existing-cache state", err)
	}
	observations = nil
	v.observe = func(g SUPGPUObservation) error { observations = append(observations, g); return nil }
	if err := v.observeEncoderCache(2); err != nil || len(observations) != 1 || observations[0].Name != "encoder.02.cache.warm" {
		t.Fatal("observer retry repeated cold preparation", err)
	}
}
