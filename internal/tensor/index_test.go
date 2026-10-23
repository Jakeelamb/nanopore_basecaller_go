package tensor

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func newIndices(t *testing.T, e *Engine, shape, data []int64) *IndexTensor {
	t.Helper()
	x, err := e.NewInt64(shape, data)
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
func TestNativeDecoderPrimitives(t *testing.T) {
	e := nativeEngine(t)
	indicesData := []int64{2, 0, 2}
	indices := newIndices(t, e, []int64{3}, indicesData)
	indicesData[0] = 99
	got, err := indices.Data()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []int64{2, 0, 2}) {
		t.Fatalf("index input was retained: %v", got)
	}
	got[0] = 100
	got, err = indices.Data()
	if err != nil || got[0] != 2 {
		t.Fatalf("index copy aliases native memory: %v %v", got, err)
	}
	shape, err := indices.Shape()
	if err != nil || !reflect.DeepEqual(shape, []int64{3}) {
		t.Fatalf("index shape %v %v", shape, err)
	}
	tied := newIndices(t, e, []int64{6}, []int64{2, 1, 1, 0, 2, 0})
	order, err := e.ArgSort(tied, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := order.Close(); err != nil {
			t.Error(err)
		}
	})
	orderData, err := order.Data()
	if err != nil {
		t.Fatal(err)
	}
	orderRaw, err := os.ReadFile(filepath.Join(filepath.Dir(os.Getenv("NANOPORE_TENSOR_BACKEND")), "argsort-reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var expectedOrder []int64
	if err := json.Unmarshal(orderRaw, &expectedOrder); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(orderData, expectedOrder) {
		t.Fatalf("ATen tie order %v want %v", orderData, expectedOrder)
	}
	if _, err := e.ArgSort(tied, 2); err == nil {
		t.Fatal("invalid argsort dimension accepted")
	}
	a := newTensor(t, e, []int64{2, 3}, []float32{0, 1, 2, 3, 4, 5})
	stable := newTensor(t, e, []int64{2, 3}, []float32{1000, 1001, 999, -1000, -1000, -1000})
	raw, err := os.ReadFile(filepath.Join(filepath.Dir(os.Getenv("NANOPORE_TENSOR_BACKEND")), "primitive-reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	var snapshots map[string]struct {
		Shape []int64  `json:"shape"`
		Bits  []uint32 `json:"data_bits"`
	}
	if err := json.Unmarshal(raw, &snapshots); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name string
		run  func() (*Tensor, error)
	}{
		{"index_select", func() (*Tensor, error) { return e.IndexSelect(a, 1, indices) }},
		{"logsumexp", func() (*Tensor, error) { return e.LogSumExp(stable, 1, false) }},
		{"logsumexp_keepdim", func() (*Tensor, error) { return e.LogSumExp(stable, 1, true) }},
		{"softmax", func() (*Tensor, error) { return e.Softmax(stable, 1) }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			x, err := tt.run()
			x = result(t, x, err)
			expected := snapshots[tt.name]
			data := make([]float32, len(expected.Bits))
			for i, b := range expected.Bits {
				data[i] = math.Float32frombits(b)
			}
			assertTensor(t, x, expected.Shape, data)
		})
	}
	x, err := e.Unsqueeze(a, 1)
	assertTensor(t, result(t, x, err), []int64{2, 1, 3}, []float32{0, 1, 2, 3, 4, 5})
	invalid := newIndices(t, e, []int64{1}, []int64{3})
	if _, err := e.IndexSelect(a, 1, invalid); err == nil {
		t.Fatal("out-of-range index accepted")
	}
	negative := newIndices(t, e, []int64{1}, []int64{-1})
	if _, err := e.IndexSelect(a, 1, negative); err == nil {
		t.Fatal("negative index accepted")
	}
	other := nativeEngine(t)
	foreign := newIndices(t, other, []int64{1}, []int64{0})
	if _, err := e.IndexSelect(a, 1, foreign); err == nil {
		t.Fatal("foreign index accepted")
	}
	if _, err := e.IndexSelect(a, 1, nil); err == nil {
		t.Fatal("nil index accepted")
	}
	if _, err := e.NewInt64([]int64{-1}, nil); err == nil {
		t.Fatal("negative index shape accepted")
	}
	if _, err := e.NewInt64([]int64{2}, []int64{1}); err == nil {
		t.Fatal("mismatched index length accepted")
	}
	if err := indices.Close(); err != nil {
		t.Fatal(err)
	}
	if err := indices.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := indices.Data(); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed index data: %v", err)
	}
	if _, err := e.IndexSelect(a, 1, indices); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed index operand: %v", err)
	}
}
func TestNativeRejectPriorABI(t *testing.T) {
	path := os.Getenv("NANOPORE_TENSOR_OLD_BACKEND")
	if path == "" {
		t.Skip("set NANOPORE_TENSOR_OLD_BACKEND for retained ABI 2 rejection")
	}
	if _, err := Open(path); err == nil || !strings.Contains(err.Error(), "unsupported tensor ABI version 2") {
		t.Fatalf("prior ABI rejection: %v", err)
	}
}
