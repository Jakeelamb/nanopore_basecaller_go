//go:build cgo && linux

package tensor

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/cudafixture"
)

func TestNativeCUDAUnsupported(t *testing.T) {
	if os.Getenv("NANOPORE_TENSOR_CUDA") == "1" {
		t.Skip("CUDA-enabled tier exercises real devices separately")
	}
	e := nativeEngine(t)
	_, err := e.NewCUDADevice(0)
	var unavailable *CUDAUnsupportedError
	if !errors.As(err, &unavailable) || !errors.Is(err, ErrCUDAUnavailable) {
		t.Fatalf("CPU-only ABI4 CUDA error: %v", err)
	}
	if _, err := e.NewCUDADevice(-1); err == nil {
		t.Fatal("negative ordinal accepted")
	}
	a := newTensor(t, e, []int64{1}, []float32{1})
	idx := newIndices(t, e, []int64{1}, []int64{0})
	for _, id := range []uint64{a.id, idx.handle.id} {
		_, err := e.backend.cuda(cudaRequest{op: "half_meta", half: id})
		var dtype *DTypeError
		if !errors.As(err, &dtype) {
			t.Fatalf("CPU handle passed as half: %v", err)
		}
	}
	other := nativeEngine(t)
	if _, err := other.backend.cuda(cudaRequest{op: "half_meta", half: a.id}); err == nil {
		t.Fatal("foreign engine accepted")
	}
}
func cudaEngine(t *testing.T) (*Engine, *Device, *Stream) {
	t.Helper()
	if os.Getenv("NANOPORE_TENSOR_CUDA") != "1" {
		t.Skip("CUDA runtime gate not configured; ordinary tests do not initialize CUDA")
	}
	e := nativeEngine(t)
	ordinal := 0
	if raw := os.Getenv("NANOPORE_TENSOR_CUDA_ORDINAL"); raw != "" {
		var err error
		ordinal, err = strconv.Atoi(raw)
		if err != nil {
			t.Fatal(err)
		}
	}
	checkLoadedRuntime(t)
	d, err := e.NewCUDADevice(ordinal)
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
	return e, d, s
}
func TestNativeCUDAAllHalfBits(t *testing.T) {
	_, _, s := cudaEngine(t)
	defer checkLoadedRuntime(t)
	bits := make([]uint16, 65536)
	for i := range bits {
		bits[i] = uint16(i)
	}
	h, err := s.NewHalf([]int64{65536}, bits)
	if err != nil {
		t.Fatal(err)
	}
	got, err := s.HalfBits(h)
	if err != nil {
		t.Fatal(err)
	}
	for i, b := range bits {
		if got[i] != b {
			t.Fatalf("half payload %04x became %04x", b, got[i])
		}
	}
}
func TestNativeCUDAOwnershipAndLayouts(t *testing.T) {
	e, d, s := cudaEngine(t)
	defer checkLoadedRuntime(t)
	bits := make([]uint16, 1024)
	for i := range bits {
		bits[i] = 0x3c00
	}
	root, err := s.NewHalf([]int64{1024}, bits)
	if err != nil {
		t.Fatal(err)
	}
	view, err := root.View([]int64{16, 16}, []int64{16, 1}, 0)
	if err != nil {
		t.Fatal(err)
	}
	out, err := root.View([]int64{16, 16}, []int64{16, 1}, 512)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.GEMMHalf(view, view, view); err == nil {
		t.Fatal("overlap accepted")
	}
	if err := s.GEMMHalf(view, view, out); err != nil {
		t.Fatal(err)
	}
	meta, err := view.Layout()
	if err != nil {
		t.Fatal(err)
	}
	meta.Shape[0] = 999
	again, err := view.Layout()
	if err != nil || again.Shape[0] != 16 {
		t.Fatal("aliased layout slices", err)
	}
	for _, v := range []struct {
		shape, stride []int64
		offset        uint64
	}{{[]int64{2}, []int64{1}, 1024}, {[]int64{2}, []int64{-1}, 0}, {[]int64{16, 16}, []int64{15, 1}, 0}} {
		h, err := root.View(v.shape, v.stride, v.offset)
		if err == nil {
			if err := s.GEMMHalf(h, view, out); err == nil {
				t.Fatal("invalid GEMM layout accepted")
			}
			h.Close()
		}
	}
	if err := root.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.HalfBits(view); err != nil {
		t.Fatal("view lost parent storage", err)
	}
	other, err := e.NewCUDADevice(int(d.state.ordinal))
	if err != nil {
		t.Fatal(err)
	}
	defer other.Close()
	foreign, err := other.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := foreign.HalfBits(view); err == nil {
		t.Fatal("same-ordinal foreign device accepted")
	}
	if err := view.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	next, err := d.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	got, err := next.HalfBits(out)
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range got {
		if b != 0x4c00 {
			t.Fatalf("completed output changed after operand/stream close: %04x", b)
		}
	}
	copyOut := *out
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				_, err := next.HalfBits(out)
				if err != nil && !errors.Is(err, ErrClosed) {
					t.Error(err)
				}
			}
		}()
	}
	if err := copyOut.Close(); err != nil {
		t.Fatal(err)
	}
	wg.Wait()
}
func TestNativeCUDASourceFixtures(t *testing.T) {
	if os.Getenv("NANOPORE_TENSOR_CUDA") != "1" {
		t.Skip("CUDA runtime gate not configured")
	}
	var capture struct {
		Cases []struct {
			Case   string `json:"case"`
			Prefix string `json:"source_prefix"`
		} `json:"cases"`
		Inputs map[string]string `json:"input_sha256"`
	}
	path := os.Getenv("NANOPORE_TENSOR_CUDA_REPLAY")
	if path == "" {
		t.Fatal("saved source replay manifest required")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &capture); err != nil {
		t.Fatal(err)
	}
	if len(capture.Cases) != 37 {
		t.Fatal("saved corpus requires 37 cases")
	}
	type prepared struct {
		c        cudafixture.Case
		data     map[string][]uint16
		expected map[string][]uint16
	}
	cases := make([]prepared, 0, 37)
	seen := map[string]bool{}
	for _, saved := range capture.Cases {
		c, data, err := cudafixture.Load(saved.Case)
		if err != nil {
			t.Fatal(err)
		}
		if seen[c.Name] || filepath.Base(filepath.Dir(saved.Case)) != c.Name {
			t.Fatal("duplicate or renamed source case")
		}
		seen[c.Name] = true
		item := prepared{c, data, map[string][]uint16{}}
		for _, role := range []string{"a", "b", "c"} {
			p := saved.Prefix + "-" + role + ".u16le"
			raw, err := os.ReadFile(p)
			if err != nil {
				t.Fatal(err)
			}
			sum := sha256.Sum256(raw)
			if capture.Inputs[p] != hex.EncodeToString(sum[:]) {
				t.Fatal("source output byte pin mismatch")
			}
			root := data[c.Views[role].Allocation]
			if len(raw) != len(root)*2 {
				t.Fatal("source allocation length differs")
			}
			bits := make([]uint16, len(root))
			for i := range bits {
				bits[i] = uint16(raw[2*i]) | uint16(raw[2*i+1])<<8
			}
			item.expected[role] = bits
		}
		cases = append(cases, item)
	}
	_, _, stream := cudaEngine(t)
	defer checkLoadedRuntime(t)
	for _, item := range cases {
		t.Run(item.c.Name, func(t *testing.T) {
			roots := map[string]*HalfBuffer{}
			for name, bits := range item.data {
				h, err := stream.NewHalf([]int64{int64(len(bits))}, bits)
				if err != nil {
					t.Fatal(err)
				}
				roots[name] = h
				t.Cleanup(func() {
					if err := h.Close(); err != nil {
						t.Error(err)
					}
				})
			}
			views := map[string]*HalfBuffer{}
			for role, v := range item.c.Views {
				h, err := roots[v.Allocation].View(v.Shape, v.Strides, v.Offset)
				if err != nil {
					t.Fatal(err)
				}
				views[role] = h
				t.Cleanup(func() {
					if err := h.Close(); err != nil {
						t.Error(err)
					}
				})
			}
			if item.c.Mode == "gemm" {
				if err := stream.GEMMHalf(views["a"], views["b"], views["c"]); err != nil {
					t.Fatal(err)
				}
			}
			for _, role := range []string{"a", "b", "c"} {
				got, err := stream.HalfBits(roots[item.c.Views[role].Allocation])
				if err != nil {
					t.Fatal(err)
				}
				for i, b := range got {
					want := item.expected[role][i]
					if b != want {
						t.Fatalf("%s parent bit %d: got %04x want %04x", role, i, b, want)
					}
				}
			}
		})
	}
}

func TestNativeCUDAABIBoundary(t *testing.T) {
	e, d, s := cudaEngine(t)
	defer checkLoadedRuntime(t)
	h, err := s.NewHalf([]int64{4, 4}, make([]uint16, 16))
	if err != nil {
		t.Fatal(err)
	}
	other, err := e.NewCUDADevice(int(d.state.ordinal))
	if err != nil {
		t.Fatal(err)
	}
	defer other.Close()
	foreign, err := other.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	cpu := newTensor(t, e, []int64{1}, []float32{1})
	idx := newIndices(t, e, []int64{1}, []int64{0})
	for _, id := range []uint64{cpu.id, idx.handle.id} {
		_, err := e.backend.cuda(cudaRequest{op: "half_meta", half: id})
		var dtype *DTypeError
		if !errors.As(err, &dtype) {
			t.Fatal("native dtype", err)
		}
	}
	for _, q := range []cudaRequest{
		{op: "half_new", device: d.state.id, stream: s.state.id, shape: []int64{-1}, bits: []uint16{0}},
		{op: "half_new", device: d.state.id, stream: s.state.id, shape: []int64{1 << 62, 8}, bits: []uint16{0}},
		{op: "half_new", device: d.state.id, stream: s.state.id, shape: []int64{2}, bits: []uint16{0}},
		{op: "half_new", device: d.state.id, stream: foreign.state.id, shape: []int64{1}, bits: []uint16{0}},
		{op: "half_bits", stream: foreign.state.id, half: h.state.id, bits: make([]uint16, 16)},
		{op: "half_bits", stream: s.state.id, half: h.state.id, bits: make([]uint16, 15)},
		{op: "half_bits", stream: h.state.id, half: h.state.id, bits: make([]uint16, 16)},
		{op: "half_meta", half: ^uint64(0)},
		{op: "half_view", half: h.state.id, shape: []int64{2}, strides: []int64{-1}},
		{op: "half_view", half: h.state.id, shape: []int64{2}, strides: []int64{1}, offset: ^uint64(0)},
		{op: "gemm_f16", stream: s.state.id, a: h.state.id, b: h.state.id, out: h.state.id},
	} {
		if _, err := e.backend.cuda(q); err == nil {
			t.Fatalf("native accepted %+v", q)
		}
	}
	if _, err := e.backend.data(h.state.id, 16); err == nil {
		t.Fatal("half entered CPU float copy")
	}
	if _, err := e.backend.dataInt64(h.state.id, 16); err == nil {
		t.Fatal("half entered CPU index copy")
	}
	otherEngine := nativeEngine(t)
	if _, err := otherEngine.backend.cuda(cudaRequest{op: "half_meta", half: h.state.id}); err == nil {
		t.Fatal("foreign engine half accepted")
	}
	before, err := s.HalfBits(h)
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range before {
		if b != 0 {
			t.Fatal("validation changed destination")
		}
	}
	if err := h.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := e.backend.cuda(cudaRequest{op: "half_meta", half: h.state.id}); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
}

func TestNativeCloseFailureRetainsLibrary(t *testing.T) {
	if os.Getenv("NANOPORE_TENSOR_BACKEND") == "" {
		t.Skip("native gate not configured")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "close-failure.so")
	source := filepath.Join(dir, "close-failure.c")
	body := `#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
uint32_t nt_version(void){return 4;}
int nt_open(uint64_t* out,char* e,size_t cap){(void)e;(void)cap;*out=1;return 0;}
int nt_close(uint64_t id,char* e,size_t cap){(void)id;static int attempts=0;if(++attempts==1){snprintf(e,cap,"injected drain failure");return 1;}return 0;}
`
	for _, name := range []string{"nt_new", "nt_load", "nt_release", "nt_shape", "nt_data", "nt_run", "nt_new_int64", "nt_data_int64", "nt_sdpa", "nt_cuda_device_open", "nt_cuda_device_close", "nt_cuda_stream_open", "nt_cuda_stream_close", "nt_cuda_half_new", "nt_cuda_half_view", "nt_cuda_half_meta", "nt_cuda_half_bits", "nt_cuda_half_release", "nt_cuda_gemm_f16",
		"nt_cuda_storage_new", "nt_cuda_storage_view", "nt_cuda_storage_meta", "nt_cuda_storage_upload",
		"nt_cuda_storage_download", "nt_cuda_storage_copy", "nt_cuda_storage_fill", "nt_cuda_storage_scale",
		"nt_cuda_storage_release"} {
		body += "void " + name + "(void){}\n"
	}
	if err := os.WriteFile(source, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	if raw, err := exec.Command("cc", "-shared", "-fPIC", source, "-o", path).CombinedOutput(); err != nil {
		t.Fatalf("fake close library: %s: %v", raw, err)
	}
	e, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	copyE := *e
	if err := e.Close(); err == nil || !strings.Contains(err.Error(), "injected drain failure") {
		t.Fatal(err)
	}
	if e.closed || e.backend.library == nil {
		t.Fatal("failed close released engine or loader")
	}
	maps, err := os.ReadFile("/proc/self/maps")
	if err != nil || !strings.Contains(string(maps), path) {
		t.Fatal("failed close unloaded library", err)
	}
	if err := copyE.Close(); err != nil {
		t.Fatal(err)
	}
	if !e.closed || e.backend.library != nil {
		t.Fatal("successful retry did not close shared engine")
	}
}
