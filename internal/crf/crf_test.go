package crf

import (
	"context"
	"math"
	"reflect"
	"testing"
)

func TestDimensions(t *testing.T) {
	for _, shape := range [][]int64{nil, {1}, {1, 1}, {0, 1, 16}, {1, 2, 16}, {1, 0, 16}, {1, 1, 15}, {1, 1, 4}, {1, 1, 32}, {1, 1, 65537}, {1 << 30, 1, 16}, {1, 1, 4 * 65536}} {
		if _, _, err := dimensions(shape); err == nil {
			t.Fatalf("accepted shape %v", shape)
		}
	}
	for _, states := range []int64{4, 16, 64, 256, 1024, 4096, 16384} {
		blocks, s, err := dimensions([]int64{9, 1, 4 * states})
		if err != nil || blocks != 9 || s != int(states) {
			t.Fatalf("shape states %d: %d %d %v", states, blocks, s, err)
		}
	}
}
func TestValueBounds(t *testing.T) {
	for _, v := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), math.MaxFloat32} {
		if err := validateValues([]float32{v}, 2, 2); err == nil {
			t.Fatalf("accepted score %v", v)
		}
		if err := validateValues([]float32{0}, 2, v); err == nil {
			t.Fatalf("accepted blank %v", v)
		}
	}
	if err := validateValues([]float32{-1000, 0, 1000}, 32, 2); err != nil {
		t.Fatal(err)
	}
}
func TestDecodeInputBorrow(t *testing.T) {
	r := Result{Blocks: 1, States: 4, Scores: make([]float32, 16), Forward: make([]float32, 8), Backward: make([]float32, 8), Posts: make([]float32, 8)}
	in := r.DecodeInput()
	if in.Blocks != 1 || in.States != 4 || &in.Scores[0] != &r.Scores[0] || &in.BackGuides[0] != &r.Backward[0] || &in.Posts[0] != &r.Posts[0] {
		t.Fatal("decode handoff changed storage or shape")
	}
}
func TestCanceledBeforeNative(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r, err := Compute(ctx, nil, nil, 2)
	if err != context.Canceled || !reflect.DeepEqual(r, Result{}) {
		t.Fatalf("cancellation result %v %v", r, err)
	}
	if _, err := Compute(nil, nil, nil, 2); err == nil {
		t.Fatal("nil context accepted")
	}
}
