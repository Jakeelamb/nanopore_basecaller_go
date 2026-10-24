package tensor

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
)

func TestShapeValidation(t *testing.T) {
	for _, shape := range [][]int64{{-1}, {math.MaxInt64, 2}, make([]int64, 33)} {
		if _, err := elementCount(shape); err == nil {
			t.Fatalf("accepted invalid shape %v", shape)
		}
	}
	for _, shape := range [][]int64{nil, {}, {2, 3}, {0, 4}} {
		if _, err := elementCount(shape); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := Open(""); err == nil {
		t.Fatal("empty library path accepted")
	}
	if _, err := Open("foo\x00bar"); err == nil {
		t.Fatal("NUL library path accepted")
	}
	var e Engine
	if !errors.Is(e.Close(), ErrClosed) {
		t.Fatal("zero engine close must reject")
	}
}
func nativeEngine(t *testing.T) *Engine {
	t.Helper()
	path := os.Getenv("NANOPORE_TENSOR_BACKEND")
	if path == "" {
		t.Skip("native tensor gate skipped; set NANOPORE_TENSOR_BACKEND to the qualified CPU backend")
	}
	e, err := Open(path)
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
func newTensor(t *testing.T, e *Engine, shape []int64, data []float32) *Tensor {
	t.Helper()
	x, err := e.NewFloat32(shape, data)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := x.Close(); err != nil {
			t.Error(err)
		}
	})
	return x
}
func assertTensor(t *testing.T, x *Tensor, shape []int64, data []float32) {
	t.Helper()
	s, err := x.Shape()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(s, shape) {
		t.Fatalf("shape %v want %v", s, shape)
	}
	d, err := x.Data()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(d, data) {
		t.Fatalf("data %v want %v", d, data)
	}
}
func result(t *testing.T, x *Tensor, err error) *Tensor {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := x.Close(); err != nil {
			t.Error(err)
		}
	})
	return x
}
func TestNativeTensorOperations(t *testing.T) {
	e := nativeEngine(t)
	a := newTensor(t, e, []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b := newTensor(t, e, []int64{3, 2}, []float32{7, 8, 9, 10, 11, 12})
	x, err := e.MatMul(a, b)
	assertTensor(t, result(t, x, err), []int64{2, 2}, []float32{58, 64, 139, 154})
	x, err = e.Add(a, a)
	assertTensor(t, result(t, x, err), []int64{2, 3}, []float32{2, 4, 6, 8, 10, 12})
	x, err = e.Mul(a, a)
	assertTensor(t, result(t, x, err), []int64{2, 3}, []float32{1, 4, 9, 16, 25, 36})
	x, err = e.Transpose(a, 0, 1)
	view := result(t, x, err)
	assertTensor(t, view, []int64{3, 2}, []float32{1, 4, 2, 5, 3, 6})
	x, err = e.Contiguous(view)
	assertTensor(t, result(t, x, err), []int64{3, 2}, []float32{1, 4, 2, 5, 3, 6})
	x, err = e.Permute(a, []int64{1, 0})
	assertTensor(t, result(t, x, err), []int64{3, 2}, []float32{1, 4, 2, 5, 3, 6})
	x, err = e.Flip(a, []int64{1})
	assertTensor(t, result(t, x, err), []int64{2, 3}, []float32{3, 2, 1, 6, 5, 4})
	x, err = e.Narrow(a, 1, 1, 2)
	assertTensor(t, result(t, x, err), []int64{2, 2}, []float32{2, 3, 5, 6})
	x, err = e.Reshape(view, []int64{6})
	assertTensor(t, result(t, x, err), []int64{6}, []float32{1, 4, 2, 5, 3, 6})
	x, err = e.Concat([]*Tensor{a, a}, 0)
	assertTensor(t, result(t, x, err), []int64{4, 3}, []float32{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6})
	x, err = e.Stack([]*Tensor{a, a}, 0)
	assertTensor(t, result(t, x, err), []int64{2, 2, 3}, []float32{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6})
	x, err = e.AddScalar(a, .5)
	assertTensor(t, result(t, x, err), []int64{2, 3}, []float32{1.5, 2.5, 3.5, 4.5, 5.5, 6.5})
	x, err = e.MulScalar(a, 2)
	assertTensor(t, result(t, x, err), []int64{2, 3}, []float32{2, 4, 6, 8, 10, 12})
	x, err = e.Clamp(a, 2, 5)
	assertTensor(t, result(t, x, err), []int64{2, 3}, []float32{2, 2, 3, 4, 5, 5})
	z, err := e.Zeros([]int64{2, 3})
	zero := result(t, z, err)
	assertTensor(t, zero, []int64{2, 3}, make([]float32, 6))
	x, err = e.Tanh(zero)
	assertTensor(t, result(t, x, err), []int64{2, 3}, make([]float32, 6))
	x, err = e.SiLU(zero)
	assertTensor(t, result(t, x, err), []int64{2, 3}, make([]float32, 6))
	x, err = e.Sigmoid(zero)
	assertTensor(t, result(t, x, err), []int64{2, 3}, []float32{.5, .5, .5, .5, .5, .5})
	scalar := newTensor(t, e, []int64{}, []float32{3})
	assertTensor(t, scalar, []int64{}, []float32{3})
	empty := newTensor(t, e, []int64{0, 3}, []float32{})
	assertTensor(t, empty, []int64{0, 3}, []float32{})
	input := newTensor(t, e, []int64{1, 1, 4}, []float32{1, 2, 3, 4})
	weight := newTensor(t, e, []int64{1, 1, 2}, []float32{2, 3})
	bias := newTensor(t, e, []int64{1}, []float32{1})
	x, err = e.Conv1d(input, weight, bias, 1, 0, 1, 1)
	assertTensor(t, result(t, x, err), []int64{1, 1, 3}, []float32{9, 14, 19})
	x, err = e.Conv1d(input, weight, nil, 1, 0, 1, 1)
	assertTensor(t, result(t, x, err), []int64{1, 1, 3}, []float32{8, 13, 18})
	linearW := newTensor(t, e, []int64{1, 3}, []float32{1, 2, 3})
	x, err = e.Linear(a, linearW, bias)
	assertTensor(t, result(t, x, err), []int64{2, 1}, []float32{15, 33})
	x, err = e.Linear(a, linearW, nil)
	assertTensor(t, result(t, x, err), []int64{2, 1}, []float32{14, 32})
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	assertTensor(t, view, []int64{3, 2}, []float32{1, 4, 2, 5, 3, 6})
}
func TestNativeOwnershipAndCopies(t *testing.T) {
	e := nativeEngine(t)
	other := nativeEngine(t)
	input := []float32{1, 2}
	a := newTensor(t, e, []int64{2}, input)
	input[0] = 99
	d, err := a.Data()
	if err != nil {
		t.Fatal(err)
	}
	d[0] = 88
	s, err := a.Shape()
	if err != nil {
		t.Fatal(err)
	}
	s[0] = 22
	assertTensor(t, a, []int64{2}, []float32{1, 2})
	alias := *e
	x, err := alias.Add(a, a)
	assertTensor(t, result(t, x, err), []int64{2}, []float32{2, 4})
	b := newTensor(t, other, []int64{2}, []float32{3, 4})
	if _, err := e.Add(a, b); err == nil {
		t.Fatal("cross-engine tensor accepted")
	}
	if _, err := e.Add(a, nil); err == nil {
		t.Fatal("nil tensor accepted")
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Data(); !errors.Is(err, ErrClosed) {
		t.Fatalf("released tensor: %v", err)
	}
	if _, err := e.Add(a, a); !errors.Is(err, ErrClosed) {
		t.Fatalf("released operand: %v", err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := alias.Zeros([]int64{1}); !errors.Is(err, ErrClosed) {
		t.Fatalf("copied engine retained released native state: %v", err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := e.Zeros([]int64{1}); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed engine: %v", err)
	}
}
func TestNativeMalformedInputs(t *testing.T) {
	e := nativeEngine(t)
	a := newTensor(t, e, []int64{2}, []float32{1, 2})
	tests := []struct {
		name string
		run  func() error
	}{
		{"negative shape", func() error { _, err := e.NewFloat32([]int64{-1}, nil); return err }},
		{"data length", func() error { _, err := e.NewFloat32([]int64{3}, []float32{1}); return err }},
		{"matmul shape", func() error { _, err := e.MatMul(a, newTensor(t, e, []int64{3}, []float32{1, 2, 3})); return err }},
		{"narrow bounds", func() error { _, err := e.Narrow(a, 0, 0, 3); return err }},
		{"reshape size", func() error { _, err := e.Reshape(a, []int64{3}); return err }},
		{"transpose dims", func() error { _, err := e.Transpose(a, 0, 3); return err }},
		{"concat empty", func() error { _, err := e.Concat(nil, 0); return err }},
		{"clamp bounds", func() error { _, err := e.Clamp(a, 2, 1); return err }},
		{"load NUL", func() error { _, err := e.LoadWeight("x\x00y"); return err }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.run(); err == nil {
				t.Fatal("invalid input accepted")
			}
		})
	}
}
func TestNativeLoadWeight(t *testing.T) {
	e := nativeEngine(t)
	dir := filepath.Dir(os.Getenv("NANOPORE_TENSOR_BACKEND"))
	x, err := e.LoadWeight(filepath.Join(dir, "float32.tensor"))
	assertTensor(t, result(t, x, err), []int64{2, 3}, []float32{0, 1, 2, 3, 4, 5})
	for _, name := range []string{"int64.tensor", "multiple.tensor", "empty.tensor", "missing.tensor"} {
		t.Run(name, func(t *testing.T) {
			if _, err := e.LoadWeight(filepath.Join(dir, name)); err == nil {
				t.Fatal("invalid weight accepted")
			}
		})
	}
	if path := os.Getenv("NANOPORE_TENSOR_WEIGHT"); path != "" {
		x, err := e.LoadWeight(path)
		w := result(t, x, err)
		shape, err := w.Shape()
		if err != nil {
			t.Fatal(err)
		}
		data, err := w.Data()
		if err != nil {
			t.Fatal(err)
		}
		if len(data) == 0 {
			t.Fatal("empty real weight")
		}
		raw, err := os.ReadFile(filepath.Join(dir, "real-weight.json"))
		if err != nil {
			t.Fatal(err)
		}
		var expected struct {
			Shape []int64  `json:"shape"`
			Bits  []uint32 `json:"data_bits"`
		}
		if err := json.Unmarshal(raw, &expected); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(shape, expected.Shape) || len(data) != len(expected.Bits) {
			t.Fatal("real weight shape differs from independent native snapshot")
		}
		for i, value := range data {
			if math.Float32bits(value) != expected.Bits[i] {
				t.Fatalf("real weight differs at %d", i)
			}
		}
		manifestBytes, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
		if err != nil {
			t.Fatal(err)
		}
		var pins struct {
			Inputs map[string]string `json:"input_sha256"`
		}
		if err := json.Unmarshal(manifestBytes, &pins); err != nil {
			t.Fatal(err)
		}
		for _, p := range []string{path, filepath.Join(dir, "real-weight.json")} {
			raw, err := os.ReadFile(p)
			if err != nil {
				t.Fatal(err)
			}
			sum := sha256.Sum256(raw)
			if pins.Inputs[p] != hex.EncodeToString(sum[:]) {
				t.Fatalf("weight or native snapshot pin mismatch at %s", p)
			}
		}
		t.Logf("real Dorado weight %s shape=%v values=%d first=%v", path, shape, len(data), data[:min(8, len(data))])
	}
}
func TestNativeConcurrentClose(t *testing.T) {
	e := nativeEngine(t)
	x := newTensor(t, e, []int64{2}, []float32{1, 2})
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				_, err := x.Data()
				if err != nil && !errors.Is(err, ErrClosed) {
					t.Error(err)
				}
			}
		}()
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	wg.Wait()
}
func TestNativeMissingLibrary(t *testing.T) {
	if os.Getenv("NANOPORE_TENSOR_BACKEND") == "" {
		t.Skip("native gate not configured")
	}
	if _, err := Open(filepath.Join(t.TempDir(), "absent.so")); err == nil || !strings.Contains(err.Error(), "dlopen") {
		t.Fatalf("missing library: %v", err)
	}
}

func TestNativeLoaderABIRejections(t *testing.T) {
	if os.Getenv("NANOPORE_TENSOR_BACKEND") == "" {
		t.Skip("native gate not configured")
	}
	cases := []struct{ name, source, want string }{
		{"version", "unsigned int nt_version(void) { return 99; }", "unsupported tensor ABI version 99"},
		{"symbol", "unsigned int nt_version(void) { return 4; }", "missing ABI symbol nt_open"},
		{"attention symbol", "unsigned int nt_version(void) { return 4; }\n" +
			"void nt_open(void){} void nt_close(void){} void nt_new(void){} void nt_load(void){} void nt_release(void){}\n" +
			"void nt_shape(void){} void nt_data(void){} void nt_run(void){} void nt_new_int64(void){} void nt_data_int64(void){}\n", "missing ABI symbol nt_sdpa"},
	}
	symbols := []string{"nt_open", "nt_close", "nt_new", "nt_load", "nt_release", "nt_shape", "nt_data", "nt_run", "nt_new_int64", "nt_data_int64", "nt_sdpa", "nt_cuda_device_open", "nt_cuda_device_close", "nt_cuda_stream_open", "nt_cuda_stream_close", "nt_cuda_half_new", "nt_cuda_half_view", "nt_cuda_half_meta", "nt_cuda_half_bits", "nt_cuda_half_release", "nt_cuda_gemm_f16", "nt_cuda_storage_new", "nt_cuda_storage_view", "nt_cuda_storage_meta", "nt_cuda_storage_upload", "nt_cuda_storage_download", "nt_cuda_storage_copy", "nt_cuda_storage_fill", "nt_cuda_storage_scale", "nt_cuda_storage_release", "nt_cuda_conv_small_f16", "nt_cuda_weights_permute_cast_f16", "nt_cuda_conv_koi_f16", "nt_cuda_linear_f16", "nt_cuda_storage_reinterpret"}
	for _, missing := range symbols[11:] {
		source := "unsigned int nt_version(void){return 4;}\n"
		for _, symbol := range symbols {
			if symbol != missing {
				source += "void " + symbol + "(void){}\n"
			}
		}
		cases = append(cases, struct{ name, source, want string }{missing, source, "missing ABI symbol " + missing})
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			source := filepath.Join(dir, "backend.c")
			lib := filepath.Join(dir, "backend.so")
			if err := os.WriteFile(source, []byte(tt.source), 0600); err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command("cc", "-shared", "-fPIC", source, "-o", lib)
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("build rejection library: %s: %v", out, err)
			}
			if _, err := Open(lib); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("loader rejection = %v, want %s", err, tt.want)
			}
		})
	}
}
