package model

import (
	"context"
	"errors"
	"math"
	"reflect"
	"slices"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
)

func outputTestWeights() HACOutputWeights {
	w := HACOutputWeights{
		down: outputWeight{receipt: WeightReceipt{Name: "11.linear.weight.tensor", Shape: []int64{128, 1024}}, values: make([]float32, 128*1024)},
		up:   outputWeight{receipt: WeightReceipt{Name: "12.linear.weight.tensor", Shape: []int64{1024, 128}}, values: make([]float32, 1024*128)},
	}
	for r := 0; r < 128; r++ {
		for c := 0; c < 1024; c++ {
			w.down.values[r*1024+c] = float32((r*13+c)%127 - 63)
		}
		w.down.values[r*1024] = float32(128 + r)
	}
	for i := range w.up.values {
		w.up.values[i] = signal.Float32FromHalf(uint16(0x2000 + i%0x4c00))
	}
	return w
}
func TestHACOutputCacheSourcePermutations(t *testing.T) {
	w := outputTestWeights()
	before := slices.Clone(w.down.values)
	c, err := PackHACOutput(w)
	if err != nil {
		t.Fatal(err)
	}
	q, scales, err := QuantizeFLSTMRows(CastFLSTMHalf(w.down.values), 128, 1024)
	if err != nil {
		t.Fatal(err)
	}
	snapshot := c.Snapshot()
	for r := 0; r < 128; r++ {
		dst := (r/16)*16 + ((r>>1)&1)*8 + ((r>>3)&1)*4 + ((r>>2)&1)*2 + (r & 1)
		if snapshot.DownScale[dst] != signal.Float16Bits(scales[r]) {
			t.Fatalf("down scale source row %d to %d", r, dst)
		}
		for col := 0; col < 1024; col++ {
			if snapshot.Down[dst*1024+col] != q[r*1024+col] {
				t.Fatalf("down source row %d column %d", r, col)
			}
		}
	}
	for r := 0; r < 1024; r++ {
		dst := (r/32)*32 + ((r>>2)&1)*16 + ((r>>1)&1)*8 + ((r>>4)&1)*4 + ((r>>3)&1)*2 + (r & 1)
		for k := 0; k < 128; k++ {
			if snapshot.Up[dst*128+k] != signal.Float16Bits(w.up.values[r*128+k]) {
				t.Fatalf("up source row %d column %d", r, k)
			}
		}
	}
	if !slices.Equal(w.down.values, before) {
		t.Fatal("parameter values mutated")
	}
	snapshot.Down[0]++
	snapshot.DownScale[0]++
	snapshot.Up[0]++
	if reflect.DeepEqual(snapshot, c.Snapshot()) {
		t.Fatal("snapshot aliases private cache")
	}
}
func TestHACOutputRejectsWeightRolesAndDomain(t *testing.T) {
	for _, change := range []func(*HACOutputWeights){
		func(w *HACOutputWeights) { w.down, w.up = w.up, w.down },
		func(w *HACOutputWeights) { w.down.receipt.Shape = []int64{1024, 128} },
		func(w *HACOutputWeights) { w.up.receipt.Name = "11.linear.weight.tensor" },
		func(w *HACOutputWeights) { w.down.values = w.down.values[:3] },
		func(w *HACOutputWeights) { clear(w.down.values[:1024]) },
		func(w *HACOutputWeights) { w.down.values[0] = float32(math.NaN()) },
		func(w *HACOutputWeights) { w.up.values[0] = 70000 },
	} {
		w := outputTestWeights()
		change(&w)
		if _, err := PackHACOutput(w); err == nil {
			t.Fatal("invalid named weight or quantization domain accepted")
		}
	}
	if _, err := NewHACOutput(nil, nil, HACOutputCache{}); err == nil {
		t.Fatal("zero cache accepted")
	}
}
func outputTestPlan(t *testing.T, n, steps int64) (HACOutputPlan, GPUMemoryReservation, GPUView, GPUView) {
	t.Helper()
	g, err := NewGPUStorageGeometry(GPUStorageCUTLASSI8, n, steps, 1024)
	if err != nil {
		t.Fatal(err)
	}
	elems, err := HACOutputTempElements(n, steps)
	if err != nil {
		t.Fatal(err)
	}
	tempGeometry, err := NewGPUScratchGeometry(GPUDtypeHalf, elems)
	if err != nil {
		t.Fatal(err)
	}
	r := GPUMemoryReservation{Bytes: g.Bytes + tempGeometry.Bytes}
	full := GPUView{Geometry: g, Orientation: GPUOrientationForward, Offset: tempGeometry.Bytes, End: r.Bytes}
	temp := GPUView{Scratch: true, Geometry: tempGeometry, End: tempGeometry.Bytes}
	p, err := PlanHACOutput(r, full, temp)
	if err != nil {
		t.Fatal(err)
	}
	return p, r, full, temp
}
func TestHACOutputReservationAndRuntimeGeometry(t *testing.T) {
	for _, tc := range []struct{ n, t, want int64 }{{64, 35, 2580480}, {64, 32, 2359296}, {1, 35, 40320}} {
		got, err := HACOutputTempElements(tc.n, tc.t)
		if err != nil || got != tc.want {
			t.Fatalf("temp %+v got %d %v", tc, got, err)
		}
	}
	for _, dims := range [][2]int64{{0, 32}, {64, 0}, {math.MaxInt64, 2}, {64, math.MaxInt64}} {
		if _, err := HACOutputTempElements(dims[0], dims[1]); err == nil {
			t.Fatalf("overflowing temp %v accepted", dims)
		}
	}
	for _, dims := range [][2]int64{{64, 32}, {128, 64}} {
		p, _, _, _ := outputTestPlan(t, dims[0], dims[1])
		if p.input.offset != p.full.Offset+2*dims[0]*1024 || p.down.bytes != dims[0]*dims[1]*128*2 || p.out.offset != p.down.offset+p.down.bytes {
			t.Fatal("source spans differ")
		}
	}
	_, r, full, temp := outputTestPlan(t, 64, 32)
	r.Bytes = 7782400
	full.Offset = 5357568
	full.End = 7782400
	p, err := PlanHACOutput(r, full, temp)
	if err != nil {
		t.Fatal(err)
	}
	if p.down.offset != 0 || p.down.bytes != 524288 || p.out.offset != 524288 || p.out.bytes != 4194304 {
		t.Fatalf("H01 split %+v", p)
	}
	for _, change := range []func(*GPUMemoryReservation, *GPUView, *GPUView){
		func(r *GPUMemoryReservation, f, t *GPUView) { f.Orientation = GPUOrientationReverse },
		func(r *GPUMemoryReservation, f, t *GPUView) { f.Geometry.T = 48 },
		func(r *GPUMemoryReservation, f, t *GPUView) { t.Scratch = false },
		func(r *GPUMemoryReservation, f, t *GPUView) { t.Offset = 1; t.End++ },
		func(r *GPUMemoryReservation, f, t *GPUView) { t.Offset = f.Offset; t.End = t.Offset + t.Geometry.Bytes },
		func(r *GPUMemoryReservation, f, t *GPUView) { t.End-- },
		func(r *GPUMemoryReservation, f, t *GPUView) { r.Bytes = 256 },
	} {
		rr, ff, tt := r, full, temp
		change(&rr, &ff, &tt)
		if _, err := PlanHACOutput(rr, ff, tt); err == nil {
			t.Fatal("malformed projection plan accepted")
		}
	}
}

type outputRecorder struct {
	events   []string
	fail     string
	cancel   context.CancelFunc
	cancelAt string
}

func (r *outputRecorder) record(name string) error {
	r.events = append(r.events, name)
	if name == r.cancelAt {
		r.cancel()
	}
	if name == r.fail {
		return errors.New("injected")
	}
	return nil
}
func (r *outputRecorder) validate() error { return r.record("validate") }
func (r *outputRecorder) zero() error     { return r.record("zero") }
func (r *outputRecorder) kernel() error   { return r.record("kernel") }
func TestHACOutputValidateZeroKernelOrder(t *testing.T) {
	for _, fail := range []string{"", "validate", "zero", "kernel"} {
		r := &outputRecorder{fail: fail}
		err := runHACOutput(context.Background(), r)
		if (fail == "") != (err == nil) {
			t.Fatal("failure lost")
		}
		want := []string{"validate", "zero", "kernel"}
		if fail == "validate" {
			want = want[:1]
		}
		if fail == "zero" {
			want = want[:2]
		}
		if !slices.Equal(r.events, want) {
			t.Fatalf("trace %v want %v", r.events, want)
		}
	}
	for _, at := range []string{"validate", "zero", "kernel"} {
		ctx, cancel := context.WithCancel(context.Background())
		r := &outputRecorder{cancel: cancel, cancelAt: at}
		err := runHACOutput(ctx, r)
		cancel()
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancel %s lost", at)
		}
	}
	if err := runHACOutput(nil, &outputRecorder{}); err == nil {
		t.Fatal("nil context accepted")
	}
	owner := &HACOutput{}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
	if err := owner.Close(); err != nil {
		t.Fatal(err)
	}
}
