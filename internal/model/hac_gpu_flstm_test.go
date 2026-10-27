package model

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

type fusedEvent struct {
	span  fusedSpan
	value byte
	call  fusedLayerCall
	layer bool
}
type fusedRecorder struct {
	trace    []fusedEvent
	fail     int
	cancel   context.CancelFunc
	cancelAt int
}

func (r *fusedRecorder) record(e fusedEvent) error {
	r.trace = append(r.trace, e)
	if r.cancel != nil && len(r.trace) == r.cancelAt {
		r.cancel()
	}
	if r.fail == len(r.trace) {
		return errors.New("injected")
	}
	return nil
}
func (r *fusedRecorder) fill(s fusedSpan, v byte) error {
	return r.record(fusedEvent{span: s, value: v})
}
func (r *fusedRecorder) layer(c fusedLayerCall) error {
	return r.record(fusedEvent{call: c, layer: true})
}
func fusedTestPlan(t *testing.T, n, steps int64) (HACFusedPlan, *HACLastConvPlan, GPUView) {
	t.Helper()
	g, e := NewGPUStorageGeometry(GPUStorageCUTLASSI8, n, steps, 1024)
	if e != nil {
		t.Fatal(e)
	}
	last := &HACLastConvPlan{Output: g, Layer: HACSmallConvLayer{TOut: steps}, InputToRevLSTM: true}
	bytes, e := HACFusedScratchBytes(n, steps)
	if e != nil {
		t.Fatal(e)
	}
	sg, e := NewGPUScratchGeometry(GPUDtypeInt8, bytes)
	if e != nil {
		t.Fatal(e)
	}
	scratch := GPUView{Scratch: true, Geometry: sg, Offset: g.Bytes, End: g.Bytes + sg.Bytes}
	p, e := PlanHACFusedStack(last, scratch)
	if e != nil {
		t.Fatal(e)
	}
	return p, last, scratch
}
func TestHACFusedPlanFixedContract(t *testing.T) {
	p, _, _ := fusedTestPlan(t, 64, 32)
	r := &fusedRecorder{}
	if e := runHACFused(context.Background(), p, r); e != nil {
		t.Fatal(e)
	}
	want := []fusedEvent{
		{span: fusedSpan{fusedInOut, 0, 65536}},
		{span: fusedSpan{fusedInOut, 2162688, 65536}},
		{span: fusedSpan{fusedInOut, 2228224, 65536}, value: 1},
		{span: fusedSpan{fusedScratch, 2424832, 1089920}},
		{layer: true, call: fusedLayerCall{0, 64, 32, -1, 1}},
		{layer: true, call: fusedLayerCall{1, 64, 32, 1, 0}},
		{layer: true, call: fusedLayerCall{2, 64, 32, -1, 1}},
		{layer: true, call: fusedLayerCall{3, 64, 32, 1, 0}},
		{layer: true, call: fusedLayerCall{4, 64, 32, -1, 1}},
		{layer: true, call: fusedLayerCall{5, 64, 32, 1, 0}},
		{layer: true, call: fusedLayerCall{6, 64, 32, -1, 1}},
	}
	if !reflect.DeepEqual(r.trace, want) {
		t.Fatalf("trace\n%+v\nwant\n%+v", r.trace, want)
	}
	r2 := &fusedRecorder{}
	if e := runHACFused(context.Background(), p, r2); e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(r.trace, r2.trace) {
		t.Fatal("replay changed")
	}
	v := p.OutputView()
	ntc, e := v.NTCView(GPUOrientationForward)
	if e != nil {
		t.Fatal(e)
	}
	if v.Geometry.Rows != 37 || ntc.Offset != 131072 || ntc.N != 64 || ntc.T != 32 || ntc.C != 1024 || ntc.StrideN != 1024 || ntc.StrideT != 65536 {
		t.Fatalf("final view %+v %+v", v, ntc)
	}
}
func TestHACFusedScratchBoundaries(t *testing.T) {
	for _, x := range []struct{ n, t, d, s, w int64 }{{64, 1, 65536, 512, 128}, {128, 1, 131072, 1024, 256}, {64, 32, 1081344, 8448, 128}, {128, 32, 2162688, 16896, 256}} {
		p, _, _ := fusedTestPlan(t, x.n, x.t)
		if p.down.bytes != x.d || p.scale.offset != p.down.offset+x.d || p.scale.bytes != x.s || p.workspace.offset != p.scale.offset+x.s || p.workspace.bytes != x.w || p.scratch.bytes != x.d+x.s+x.w {
			t.Fatalf("partition %+v", p)
		}
	}
	for _, x := range [][2]int64{{0, 1}, {63, 1}, {64, 0}, {64, -1}, {64, math.MaxInt32 + 1}, {math.MaxInt64, 1}, {2147483584, math.MaxInt32}} {
		if _, e := HACFusedScratchBytes(x[0], x[1]); e == nil {
			t.Fatalf("accepted %v", x)
		}
	}
}
func TestHACFusedRejectsInput(t *testing.T) {
	_, last, scratch := fusedTestPlan(t, 64, 32)
	for _, test := range []struct {
		name   string
		change func(*HACLastConvPlan, *GPUView)
	}{
		{"layout", func(l *HACLastConvPlan, s *GPUView) { l.Output.Layout = GPUStorageCUTLASSF16 }},
		{"channels", func(l *HACLastConvPlan, s *GPUView) { l.Output.C = 512 }},
		{"orientation", func(l *HACLastConvPlan, s *GPUView) { l.InputToRevLSTM = false }},
		{"batch", func(l *HACLastConvPlan, s *GPUView) { l.Output.N = 63 }},
		{"time", func(l *HACLastConvPlan, s *GPUView) { l.Output.T = 0 }},
		{"rows", func(l *HACLastConvPlan, s *GPUView) { l.Output.Rows++ }},
		{"short", func(l *HACLastConvPlan, s *GPUView) { s.Geometry.Bytes = 1089919; s.End = s.Offset + s.Geometry.Bytes }},
		{"overlap", func(l *HACLastConvPlan, s *GPUView) { s.Offset = 0; s.End = s.Geometry.Bytes }},
		{"unaligned", func(l *HACLastConvPlan, s *GPUView) { s.Offset++; s.End++ }},
		{"non-temp", func(l *HACLastConvPlan, s *GPUView) { s.Scratch = false }},
	} {
		t.Run(test.name, func(t *testing.T) {
			l, s := *last, scratch
			test.change(&l, &s)
			if _, e := PlanHACFusedStack(&l, s); e == nil {
				t.Fatal("accepted invalid plan")
			}
		})
	}
	if _, e := PlanHACFusedStack(nil, scratch); e == nil {
		t.Fatal("accepted nil")
	}
	r := &fusedRecorder{}
	if e := runHACFused(context.Background(), HACFusedPlan{}, r); e == nil || len(r.trace) != 0 {
		t.Fatal("zero plan executed")
	}
}
func TestHACFusedFailureAndCancellation(t *testing.T) {
	p, _, _ := fusedTestPlan(t, 64, 32)
	for i := 1; i <= 11; i++ {
		r := &fusedRecorder{fail: i}
		if e := runHACFused(context.Background(), p, r); e == nil || len(r.trace) != i {
			t.Fatalf("failure %d executed %d", i, len(r.trace))
		}
	}
	for i := 0; i <= 11; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		r := &fusedRecorder{cancel: cancel, cancelAt: i}
		if i == 0 {
			cancel()
		}
		if e := runHACFused(ctx, p, r); !errors.Is(e, context.Canceled) || len(r.trace) != i {
			t.Fatalf("cancel %d executed %d, %v", i, len(r.trace), e)
		}
		cancel()
	}
}
func sevenFusedCaches() []FLSTMCache {
	out := make([]FLSTMCache, 7)
	for i := range out {
		c := FLSTMCache{Params: FLSTMParams{1024, 128}, DownIH: make([]int8, 128*1024), DownHH: make([]int8, 128*1024), DownScaleIH: make([]uint16, 128), DownScaleHH: make([]uint16, 128), Up: make([]int8, 4096*256), UpScale: make([]uint16, 4096), UpBias: make([]uint16, 4096)}
		c.DownIH[0] = int8(i + 1)
		c.DownHH[0] = int8(i + 11)
		c.DownScaleIH[0] = uint16(i + 21)
		c.DownScaleHH[0] = uint16(i + 31)
		c.Up[0] = int8(i + 41)
		c.UpScale[0] = uint16(i + 51)
		c.UpBias[0] = uint16(i + 61)
		out[i] = c
	}
	return out
}
func TestHACFusedCacheBinding(t *testing.T) {
	caches := sevenFusedCaches()
	spec, e := fusedCacheSpecs(caches)
	if e != nil {
		t.Fatal(e)
	}
	for i, layer := range spec {
		for j, s := range layer {
			if s.data[0] != byte(1+i+j*10) {
				t.Fatalf("cache %d role %d got %d", i, j, s.data[0])
			}
		}
	}
	caches[0].DownIH[0] = 99
	caches[0].UpBias[0] = 99
	if spec[0][0].data[0] != 1 || spec[0][6].data[0] != 61 {
		t.Fatal("caller mutation changed snapshot")
	}
	if _, e := fusedCacheSpecs(caches[:6]); e == nil {
		t.Fatal("six caches accepted")
	}
	for i := 0; i < 7; i++ {
		bad := append([]FLSTMCache(nil), caches...)
		switch i {
		case 0:
			bad[0].DownIH = nil
		case 1:
			bad[0].DownHH = nil
		case 2:
			bad[0].DownScaleIH = nil
		case 3:
			bad[0].DownScaleHH = nil
		case 4:
			bad[0].Up = nil
		case 5:
			bad[0].UpScale = nil
		case 6:
			bad[0].UpBias = nil
		}
		if _, e := fusedCacheSpecs(bad); e == nil {
			t.Fatalf("bad role %d accepted", i)
		}
	}
}

// Reentrant recording interleaves invocations without sharing their trace or
// cursor. Concrete GPU invocations still require exclusive stack ownership.
type fusedInterleaved struct {
	fusedRecorder
	nested func()
	once   bool
}

func (r *fusedInterleaved) layer(c fusedLayerCall) error {
	if !r.once {
		r.once = true
		r.nested()
	}
	return r.fusedRecorder.layer(c)
}
func TestHACFusedInvocationIsolation(t *testing.T) {
	p, _, _ := fusedTestPlan(t, 64, 32)
	other, _, _ := fusedTestPlan(t, 128, 32)
	nested := &fusedRecorder{}
	outer := &fusedInterleaved{nested: func() {
		if e := runHACFused(context.Background(), other, nested); e != nil {
			t.Fatal(e)
		}
	}}
	if e := runHACFused(context.Background(), p, outer); e != nil {
		t.Fatal(e)
	}
	if len(outer.trace) != 11 || len(nested.trace) != 11 || outer.trace[4].call.n != 64 || nested.trace[4].call.n != 128 {
		t.Fatalf("crossed invocation trace %+v %+v", outer.trace, nested.trace)
	}
}
