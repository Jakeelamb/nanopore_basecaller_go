package tensor

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

type attentionSnapshot struct {
	Shape []int64  `json:"shape"`
	Bits  []uint32 `json:"data_bits"`
}

func (s attentionSnapshot) values() []float32 {
	data := make([]float32, len(s.Bits))
	for i, bits := range s.Bits {
		data[i] = math.Float32frombits(bits)
	}
	return data
}

func assertAttentionBits(t *testing.T, got *Tensor, want attentionSnapshot) {
	t.Helper()
	shape, err := got.Shape()
	if err != nil {
		t.Fatal(err)
	}
	data, err := got.Data()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(shape, want.Shape) || len(data) != len(want.Bits) {
		t.Fatalf("shape %v, values %d, want %v, values %d", shape, len(data), want.Shape, len(want.Bits))
	}
	for i, value := range data {
		if math.Float32bits(value) != want.Bits[i] {
			t.Fatalf("value %d bits %08x want %08x", i, math.Float32bits(value), want.Bits[i])
		}
	}
}

func TestNativeAttentionPrimitives(t *testing.T) {
	e := nativeEngine(t)
	dir := filepath.Dir(os.Getenv("NANOPORE_TENSOR_BACKEND"))
	path := filepath.Join(dir, "attention-reference.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	manifestRaw, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest struct {
		Inputs map[string]string `json:"input_sha256"`
	}
	if err := json.Unmarshal(manifestRaw, &manifest); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	if manifest.Inputs[path] != hex.EncodeToString(sum[:]) {
		t.Fatal("attention fixture differs from pinned independent producer output")
	}
	var snapshots map[string]attentionSnapshot
	if err := json.Unmarshal(raw, &snapshots); err != nil {
		t.Fatal(err)
	}
	x := newTensor(t, e, snapshots["input"].Shape, snapshots["input"].values())
	a, err := e.Transpose(x, 0, 1)
	transpose := result(t, a, err)
	a, err = e.Narrow(x, 0, 0, 1)
	row := result(t, a, err)
	for _, tt := range []struct {
		name string
		run  func() (*Tensor, error)
	}{
		{"mean", func() (*Tensor, error) { return e.Mean(x, -1, false) }},
		{"mean_keepdim", func() (*Tensor, error) { return e.Mean(x, -1, true) }},
		{"mean_transpose", func() (*Tensor, error) { return e.Mean(transpose, 1, true) }},
		{"rsqrt", func() (*Tensor, error) { return e.Rsqrt(x) }},
		{"reciprocal", func() (*Tensor, error) { return e.Reciprocal(x) }},
		{"sin", func() (*Tensor, error) { return e.Sin(x) }},
		{"cos", func() (*Tensor, error) { return e.Cos(x) }},
		{"sub", func() (*Tensor, error) { return e.Sub(x, row) }},
		{"square", func() (*Tensor, error) { return e.Square(x) }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a, err := tt.run()
			assertAttentionBits(t, result(t, a, err), snapshots[tt.name])
		})
	}
	qkv := newTensor(t, e, snapshots["qkv"].Shape, snapshots["qkv"].values())
	parts := make([]*Tensor, 3)
	for i := range parts {
		a, err := e.Narrow(qkv, 0, int64(i), 1)
		a = result(t, a, err)
		a, err = e.Reshape(a, []int64{2, 2, 9, 4})
		parts[i] = result(t, a, err)
	}
	a, err = e.Narrow(parts[0], -2, 2, 5)
	q := result(t, a, err)
	a, err = e.Narrow(parts[1], -2, 0, 7)
	k := result(t, a, err)
	a, err = e.Narrow(parts[2], -2, 0, 7)
	v := result(t, a, err)
	mask := make([]byte, 5*7)
	for qi := 0; qi < 5; qi++ {
		for ki := 0; ki < 7; ki++ {
			if delta := ki - (qi + 2); delta >= -2 && delta <= 3 {
				mask[qi*7+ki] = 1
			}
		}
	}
	for _, tt := range []struct {
		name  string
		shape []int64
		mask  []byte
	}{
		{"masked", []int64{5, 7}, mask},
		{"masked_contiguous", []int64{5, 7}, mask},
		{"broadcast", []int64{1, 1, 5, 7}, mask},
		{"unmasked", nil, nil},
		{"blocked", []int64{5, 7}, make([]byte, 5*7)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a, err := e.ScaledDotProductAttention(q, k, v, tt.shape, tt.mask)
			assertAttentionBits(t, result(t, a, err), snapshots[tt.name])
		})
	}
	shape := []int64{5, 7}
	a, err = e.ScaledDotProductAttention(q, k, v, shape, mask)
	copyResult := result(t, a, err)
	shape[0], mask[0] = 99, 0
	if err := qkv.Close(); err != nil {
		t.Fatal(err)
	}
	assertAttentionBits(t, copyResult, snapshots["masked"])
	if !reflect.DeepEqual(snapshots["masked"].Bits, snapshots["masked_contiguous"].Bits) {
		t.Fatal("source mask row stride changes CPU SDPA bits")
	}
}

func TestNativeAttentionRejections(t *testing.T) {
	e := nativeEngine(t)
	other := nativeEngine(t)
	x := newTensor(t, e, []int64{1, 1, 2, 2}, []float32{1, 2, 3, 4})
	foreign := newTensor(t, other, []int64{1, 1, 2, 2}, []float32{1, 2, 3, 4})
	for _, tt := range []struct {
		name string
		run  func() (*Tensor, error)
	}{
		{"mask count", func() (*Tensor, error) { return e.ScaledDotProductAttention(x, x, x, []int64{2, 2}, []byte{1}) }},
		{"mask byte", func() (*Tensor, error) {
			return e.ScaledDotProductAttention(x, x, x, []int64{2, 2}, []byte{1, 2, 0, 1})
		}},
		{"mask shape", func() (*Tensor, error) { return e.ScaledDotProductAttention(x, x, x, []int64{-1}, nil) }},
		{"mask rank", func() (*Tensor, error) { return e.ScaledDotProductAttention(x, x, x, make([]int64, 33), nil) }},
		{"mask overflow", func() (*Tensor, error) { return e.ScaledDotProductAttention(x, x, x, []int64{math.MaxInt64, 2}, nil) }},
		{"mask broadcast", func() (*Tensor, error) { return e.ScaledDotProductAttention(x, x, x, []int64{3}, []byte{1, 0, 1}) }},
		{"foreign q", func() (*Tensor, error) { return e.ScaledDotProductAttention(foreign, x, x, nil, nil) }},
		{"foreign k", func() (*Tensor, error) { return e.ScaledDotProductAttention(x, foreign, x, nil, nil) }},
		{"foreign v", func() (*Tensor, error) { return e.ScaledDotProductAttention(x, x, foreign, nil, nil) }},
		{"nil q", func() (*Tensor, error) { return e.ScaledDotProductAttention(nil, x, x, nil, nil) }},
		{"mean dimension", func() (*Tensor, error) { return e.Mean(x, 4, false) }},
		{"sub broadcast", func() (*Tensor, error) { return e.Sub(x, newTensor(t, e, []int64{3}, []float32{1, 2, 3})) }},
		{"sdpa dimension", func() (*Tensor, error) {
			return e.ScaledDotProductAttention(newTensor(t, e, []int64{2}, []float32{1, 2}), x, x, nil, nil)
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			a, err := tt.run()
			if a != nil || err == nil {
				t.Fatalf("invalid attention input returned %v, %v", a, err)
			}
		})
	}
	if err := x.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := e.ScaledDotProductAttention(x, x, x, nil, nil); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed attention input: %v", err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := e.Mean(x, -1, true); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed mean engine: %v", err)
	}
	if _, err := e.ScaledDotProductAttention(x, x, x, nil, nil); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed attention engine: %v", err)
	}
}
