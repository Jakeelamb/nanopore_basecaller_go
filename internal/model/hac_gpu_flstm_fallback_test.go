package model

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

func TestHalfFLSTMCachesPreserveRolesAndIgnoreHiddenBias(t *testing.T) {
	p := FLSTMParams{C: 16, K: 32}
	sequence := func(n int, base float32) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = base + float32(i)/1024
		}
		return v
	}
	w := FLSTMWeightSet{DownIH: sequence(512, 1), DownHH: sequence(512, 2), UpIH: sequence(2048, 3), UpHH: sequence(2048, 6), UpBiasIH: sequence(64, 9), UpBiasHH: []float32{float32(math.NaN())}}
	cache, err := PackFLSTMHalfLayer(p, w)
	if err != nil {
		t.Fatal(err)
	}
	c := cache.Snapshot()
	for input := 0; input < 16; input++ {
		for inner := 0; inner < 32; inner++ {
			if c.DownIH[input*32+inner] != signal.Float16Bits(w.DownIH[inner*16+input]) || c.DownHH[input*32+inner] != signal.Float16Bits(w.DownHH[inner*16+input]) {
				t.Fatal("down transpose/role changed")
			}
		}
	}
	for joint := 0; joint < 64; joint++ {
		for gate := 0; gate < 64; gate++ {
			want := w.UpIH[gate*32+joint%32]
			if joint >= 32 {
				want = w.UpHH[gate*32+joint%32]
			}
			if c.Up[joint*64+gate] != signal.Float16Bits(want) {
				t.Fatal("concatenated up transpose changed")
			}
		}
	}
	for i := range c.Bias {
		if c.Bias[i] != signal.Float16Bits(w.UpBiasIH[i]) {
			t.Fatal("input bias changed")
		}
	}
	c.DownIH[0] ^= 1
	if cache.Snapshot().DownIH[0] == c.DownIH[0] {
		t.Fatal("snapshot mutates cache")
	}
	w.UpBiasIH[0] = float32(math.Inf(1))
	if _, err := PackFLSTMHalfLayer(p, w); err == nil {
		t.Fatal("nonfinite live bias accepted")
	}
}

type halfTrace struct {
	rows       []int64
	cells      []int
	calls      []halfFLSTMStep
	operations []string
	fail       string
	cancel     context.CancelFunc
}

func (x *halfTrace) zeroRow(r int64) error { x.rows = append(x.rows, r); return nil }
func (x *halfTrace) zeroCell(l int) error  { x.cells = append(x.cells, l); return nil }
func (x *halfTrace) record(name string, s halfFLSTMStep) error {
	x.operations = append(x.operations, name)
	if name == "step" {
		x.calls = append(x.calls, s)
	}
	if x.cancel != nil {
		x.cancel()
	}
	if name == x.fail {
		return errors.New("native failure")
	}
	return nil
}
func (x *halfTrace) downInput(s halfFLSTMStep) error  { return x.record("ih", s) }
func (x *halfTrace) downHidden(s halfFLSTMStep) error { return x.record("hh", s) }
func (x *halfTrace) up(s halfFLSTMStep) error         { return x.record("up", s) }
func (x *halfTrace) step(s halfFLSTMStep) error       { return x.record("step", s) }

func TestHalfFLSTMSourceTimestepOrder(t *testing.T) {
	p, err := PlanHACHalfStack(64, 3)
	if err != nil {
		t.Fatal(err)
	}
	x := new(halfTrace)
	if err = runHACHalfStack(context.Background(), p, x); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(x.rows, []int64{0, 5}) || !reflect.DeepEqual(x.cells, []int{0, 1, 2, 3, 4, 5, 6}) {
		t.Fatal("sentinel or per-layer cell initialization changed")
	}
	want := []halfFLSTMStep{{0, 3, 5, 4}, {0, 2, 4, 3}, {0, 1, 3, 2}, {1, 2, 0, 1}, {1, 3, 1, 2}, {1, 4, 2, 3}}
	if !reflect.DeepEqual(x.calls[:6], want) {
		t.Fatalf("rows %v", x.calls[:6])
	}
	if len(x.calls) != 21 || x.calls[20] != (halfFLSTMStep{6, 1, 3, 2}) {
		t.Fatal("seven-layer reverse output changed")
	}
	for i, name := range x.operations {
		if name != []string{"ih", "hh", "up", "step"}[i%4] {
			t.Fatal("three GEMMs before cell step required")
		}
	}
}

func TestHalfFLSTMStopsOnFailureAndCancellation(t *testing.T) {
	p, _ := PlanHACHalfStack(64, 2)
	x := &halfTrace{fail: "hh"}
	if err := runHACHalfStack(context.Background(), p, x); err == nil {
		t.Fatal("native failure lost")
	}
	if !reflect.DeepEqual(x.operations, []string{"ih", "hh"}) || len(x.calls) != 0 {
		t.Fatal("work continued after failure")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	x = &halfTrace{cancel: cancel}
	if err := runHACHalfStack(ctx, p, x); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(x.operations, []string{"ih"}) {
		t.Fatal("work continued after cancellation")
	}
}

func TestHalfFLSTMRejectsInvalidExtents(t *testing.T) {
	for _, shape := range [][2]int64{{0, 1}, {1, 0}, {-1, 3}, {1, math.MaxInt64}, {math.MaxInt64, 1}, {math.MaxInt32, 2}} {
		if _, err := PlanHACHalfStack(shape[0], shape[1]); err == nil {
			t.Fatal("invalid shape admitted", shape)
		}
	}
	if err := runHACHalfStack(context.Background(), HACHalfPlan{}, new(halfTrace)); err == nil {
		t.Fatal("zero plan admitted")
	}
	if _, err := PackFLSTMHalfLayer(FLSTMParams{16, 32}, FLSTMWeightSet{}); err == nil {
		t.Fatal("missing weights admitted")
	}
}

func TestHalfFLSTMClosedRuntimeRejectsBeforeStorageAccess(t *testing.T) {
	if _, err := NewHACHalfStack(nil, nil, nil); !errors.Is(err, tensor.ErrClosed) {
		t.Fatal(err)
	}
	s := &HACHalfStack{}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Run(context.Background(), nil, nil, nil, HACHalfPlan{}); !errors.Is(err, tensor.ErrClosed) {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}
