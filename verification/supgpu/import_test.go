package supgpu

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"syscall"
	"testing"
)

func tinyTensor(t *testing.T) (string, Tensor, tensorSpec) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "data.bin")
	body := []byte{1, 2, 3, 4}
	if err := os.WriteFile(path, body, 0600); err != nil {
		t.Fatal(err)
	}
	p, err := pin(path)
	if err != nil {
		t.Fatal(err)
	}
	p.Path = "data.bin"
	s := spec("input", "Half", []int64{2}, "C")
	v := Tensor{Name: "input", DType: "Half", Shape: []int64{2}, Strides: []int64{1}, Tags: []int{int('C')}, ScaleTag: 0, OffsetBytes: 0, Data: p, Backing: p, Regions: []Range{{0, 4, "initialized", "fixed-source-full-allocation/v1"}}}
	return dir, v, s
}

func TestClosedTensorImporterAndEffectiveControls(t *testing.T) {
	dir, v, s := tinyTensor(t)
	if err := verifyTensor(dir, v, s); err != nil {
		t.Fatal(err)
	}
	mutations := map[string]func(*Tensor){
		"dtype": func(v *Tensor) { v.DType = "Float" }, "shape": func(v *Tensor) { v.Shape = []int64{1, 2} }, "stride": func(v *Tensor) { v.Strides = []int64{2} }, "tag": func(v *Tensor) { v.Tags = []int{int('N')} }, "scale_tag": func(v *Tensor) { v.ScaleTag = int('K') }, "offset": func(v *Tensor) { v.OffsetBytes = 2 }, "backing_extent": func(v *Tensor) { v.Backing.Bytes = 2 }, "payload_pin": func(v *Tensor) { v.Data.SHA256 = strings.Repeat("0", 64) }, "path_escape": func(v *Tensor) { v.Data.Path = "../data.bin" }, "mask": func(v *Tensor) { v.Regions = []Range{{0, 4, "excluded", "observed-mismatch"}} }, "missing_live_bytes": func(v *Tensor) { v.Regions = []Range{{0, 2, "initialized", "fixed-source-full-allocation/v1"}} },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			raw, _ := json.Marshal(v)
			var m Tensor
			if err := json.Unmarshal(raw, &m); err != nil {
				t.Fatal(err)
			}
			mutate(&m)
			if err := verifyTensor(dir, m, s); err == nil {
				t.Fatal("corrupt tensor accepted")
			}
		})
	}
	if err := os.WriteFile(filepath.Join(dir, "data.bin"), []byte{1, 2, 3, 5}, 0600); err != nil {
		t.Fatal(err)
	}
	if err := verifyTensor(dir, v, s); err == nil {
		t.Fatal("corrupt payload accepted")
	}
}

func TestClosedJSONRejectsAmbiguousFields(t *testing.T) {
	_, v, _ := tinyTensor(t)
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	var good Tensor
	if err = decodeClosed(raw, &good); err != nil {
		t.Fatal(err)
	}
	for name, bad := range map[string][]byte{
		"duplicate": bytes.Replace(raw, []byte(`"name":"input"`), []byte(`"name":"input","name":"input"`), 1),
		"case":      bytes.Replace(raw, []byte(`"name"`), []byte(`"Name"`), 1),
		"missing":   bytes.Replace(raw, []byte(`"name":"input",`), nil, 1),
		"null":      bytes.Replace(raw, []byte(`"tags":[67]`), []byte(`"tags":null`), 1),
		"unknown":   bytes.Replace(raw, []byte(`"name":"input"`), []byte(`"name":"input","extra":1`), 1),
		"trailing":  append(append([]byte{}, raw...), []byte(` {}`)...),
	} {
		t.Run(name, func(t *testing.T) {
			var x Tensor
			if err := decodeClosed(bad, &x); err == nil {
				t.Fatal("ambiguous JSON accepted")
			}
		})
	}
}

func TestPayloadSymlinkAndFIFORejectWithoutBlocking(t *testing.T) {
	dir, v, _ := tinyTensor(t)
	if err := os.Symlink(filepath.Join(dir, "data.bin"), filepath.Join(dir, "link.bin")); err != nil {
		t.Fatal(err)
	}
	v.Data.Path = "link.bin"
	if err := verifyPin(dir, v.Data); err == nil {
		t.Fatal("symlink accepted")
	}
	fifo := filepath.Join(dir, "fifo")
	if err := syscall.Mkfifo(fifo, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := pin(fifo); err == nil {
		t.Fatal("FIFO accepted")
	}
}

func TestFiniteRegistryFromSourceGeometry(t *testing.T) {
	g := Geometry{32, 768, 64, 12, 768, 600}
	events := eventRegistry(g)
	if len(events) != 191 || events[0].Name != "input" || events[7].Name != "encoder.00.cache.before_remove_bits" || events[len(events)-1].Name != "scores" {
		t.Fatalf("wrong finite registry: %d events", len(events))
	}
	seen := map[string]bool{}
	layers := map[int]int{}
	for _, e := range events {
		if seen[e.Name] {
			t.Fatal("duplicate event")
		}
		seen[e.Name] = true
		if e.Layer >= 0 {
			layers[e.Layer]++
		}
	}
	if len(layers) != 18 {
		t.Fatal("missing encoder")
	}
	for layer, n := range layers {
		if n != 10 {
			t.Fatalf("layer %d has %d observations", layer, n)
		}
	}
	names := WeightNames()
	if len(names) != 139 || names[10] != "transformer_encoder.0.self_attn.Wqkv.weight.tensor" || names[138] != "crf.linear.weight.tensor" {
		t.Fatal("weight source order changed")
	}
}

func TestNativePhasesRequireSeparateAdmission(t *testing.T) {
	if _, err := RunBuild("does-not-exist", "", false); err == nil || !strings.Contains(err.Error(), "separate coordinator admission") {
		t.Fatal(err)
	}
	if _, err := RunCapture("", "", "", "", "", "", "does-not-exist", false); err == nil || !strings.Contains(err.Error(), "separate coordinator admission") {
		t.Fatal(err)
	}
}

func TestDepfileCompleteHeaderList(t *testing.T) {
	path := filepath.Join(t.TempDir(), "unit.d")
	if err := os.WriteFile(path, []byte("unit.o: /source/unit.cpp \\\n /usr/include/a.h /path/with\\ space/b.h\n"), 0600); err != nil {
		t.Fatal(err)
	}
	got, err := DepfileInputs(path)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"/source/unit.cpp", "/usr/include/a.h", "/path/with space/b.h"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("%q != %q", got, want)
	}
	if err := os.WriteFile(path, []byte("unit.o:"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := DepfileInputs(path); err == nil {
		t.Fatal("empty header closure accepted")
	}
}

func TestCacheRoundingRelationAndUnroundedCacheControl(t *testing.T) {
	dir, v, _ := tinyTensor(t)
	v.Name = "qkv_half"
	pre := Event{Name: "encoder.00.cache.before_remove_bits", Tensors: []Tensor{v}}
	rounded := []byte{0, 2, 0, 4}
	path := filepath.Join(dir, "rounded.bin")
	if err := os.WriteFile(path, rounded, 0600); err != nil {
		t.Fatal(err)
	}
	p, err := pin(path)
	if err != nil {
		t.Fatal(err)
	}
	p.Path = "rounded.bin"
	v.Data = p
	post := Event{Name: "encoder.00.cache.after_remove_bits", Tensors: []Tensor{v}}
	if err = verifyCacheTransition(dir, pre, post); err != nil {
		t.Fatal(err)
	}
	pre.Tensors[0].Name = "qkv_int8"
	post.Tensors[0].Name = "qkv_int8"
	if err = verifyCacheTransition(dir, pre, post); err == nil {
		t.Fatal("changed unrounded cache accepted")
	}
}

func TestNativeDescriptorContractPreservesTagsAndReciprocalScale(t *testing.T) {
	g := Geometry{32, 768, 64, 12, 768, 600}
	qkv := descriptorRegistry("encoder.00.qkv", g)
	if len(qkv) != 4 || qkv[0].ScaleTag != int('C') || qkv[0].ScaleDType != "Float" || qkv[1].Tags[0] != 0 || qkv[3].Tags[2] != 0 {
		t.Fatal("QKV descriptor source tags/scales changed")
	}
	norm1 := descriptorRegistry("encoder.00.norm1", g)
	norm2 := descriptorRegistry("encoder.00.norm2", g)
	if norm1[4].DType != "Float8_e4m3fn" || norm2[4].DType != "Char" || norm2[4].ScaleTag != int('C') {
		t.Fatal("conditional FP8/residual descriptor changed")
	}
	if !reflect.DeepEqual(norm1[1].Shape, norm1[3].Shape) || !reflect.DeepEqual(norm2[1].Shape, norm2[3].Shape) {
		t.Fatal("half residual output shape changed")
	}
}
