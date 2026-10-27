package model

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

func TestHACGPUReserveUseGeometryAndLifetimes(t *testing.T) {
	p, err := PlanHACGPU(FixedHACGPUProfile())
	if err != nil {
		t.Fatal(err)
	}
	if p.reservation.Bytes != 7782400 || p.reservation.Steps != 6 || p.last.Output.T != 32 || p.last.Output.Rows != 37 || p.output.t != 32 || p.output.down.bytes != 524288 || p.output.out.bytes != 4194304 {
		t.Fatalf("source reserve/use geometry differs: %+v", p)
	}
	wantOffsets := []int64{0, 7389184, 0, 5357568, 0, 0}
	for i, v := range p.Views() {
		if v.Offset != wantOffsets[i] {
			t.Fatalf("step%d offset%d !=%d", i, v.Offset, wantOffsets[i])
		}
	}
	if p.reservation.steps[3].geometry.T != 35 || p.reservation.steps[5].geometry.LiveBytes != 5160960 || p.views[5].Geometry.LiveBytes != 4718592 {
		t.Fatal("padded reservation was collapsed into actual use")
	}
	u := p.reservation.Use()
	for _, v := range p.views {
		if v.Scratch {
			_, err = u.Temp([]int64{v.Geometry.Elems}, v.Geometry.Dtype)
		} else {
			_, err = u.NextTC(v.Geometry.T, v.Geometry.C, v.Geometry.Layout, v.Orientation)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if err = u.Alive(u.Views()[4]); !errors.Is(err, ErrGPUScratchExpired) {
		t.Fatalf("recurrent scratch outlived projection reuse: %v", err)
	}
	if err = u.Alive(u.Views()[3]); err != nil {
		t.Fatal("full recurrent allocation expired before projection", err)
	}
	views := p.Views()
	views[0].Offset = 99
	if p.Views()[0].Offset != 0 {
		t.Fatal("caller mutated sealed plan")
	}
}

func TestHACGPURejectsUnsupportedFullFallback(t *testing.T) {
	for _, p := range []HACGPUProfile{{64, 192, GPUStorageCUTLASSF16}, {64, 192, GPUStorageCUBLASTNC}, {1, 192, GPUStorageCUTLASSI8}, {64, 288, GPUStorageCUTLASSI8}, {0, 0, 0}} {
		if _, err := PlanHACGPU(p); err == nil {
			t.Fatalf("unsupported full graph admitted: %+v", p)
		}
	}
	if _, err := NewHACGPU(nil, nil, HACGPUWeights{}); err == nil {
		t.Fatal("unverified weights accepted")
	}
}

func TestHACGPUAuthoritativeRolesRejectForeignSameShapeWeights(t *testing.T) {
	manifest, err := WeightManifest(HAC)
	if err != nil {
		t.Fatal(err)
	}
	roles, err := hacGPURoles(manifest)
	if err != nil || len(roles) != 50 {
		t.Fatalf("authoritative roles: %d %v", len(roles), err)
	}
	changed := slices.Clone(manifest)
	changed[6].SHA256 = manifest[12].SHA256
	if _, err = hacGPURoles(changed); err == nil {
		t.Fatal("same-shaped foreign recurrent archive admitted")
	}
	changed = slices.Clone(manifest)
	changed[7] = changed[6]
	if _, err = hacGPURoles(changed); err == nil {
		t.Fatal("duplicate role admitted")
	}
	changed = slices.Clone(manifest)
	changed[48], changed[49] = changed[49], changed[48]
	if _, err = hacGPURoles(changed); err != nil {
		t.Fatal("role binding depends on manifest ordering", err)
	}
	w := HACGPUWeights{receipts: manifest}
	copy := w.Receipts()
	copy[0].Shape[0] = 999
	if w.receipts[0].Shape[0] == 999 {
		t.Fatal("receipt shape aliases private identity")
	}
}

func TestHACGPURecurrentBindingsKeepEveryLayerAndBiasRole(t *testing.T) {
	values := map[string][]float32{}
	for i := 0; i < 7; i++ {
		for j, name := range hacGPURecurrentNames(i) {
			values[name] = []float32{float32(i*6 + j + 1)}
		}
	}
	sets := hacGPURecurrentWeights(values)
	for i, w := range sets {
		for j, v := range [][]float32{w.DownIH, w.DownHH, w.UpIH, w.UpHH, w.UpBiasIH, w.UpBiasHH} {
			if len(v) != 1 || v[0] != float32(i*6+j+1) {
				t.Fatalf("layer%d role%d substituted: %v", i, j, v)
			}
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := LoadHACFLSTMWeights(ctx, nil, "absent"); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled loader reached native boundary: %v", err)
	}
	if _, err := LoadHACGPUWeights(context.Background(), nil, "absent"); !errors.Is(err, tensor.ErrClosed) {
		t.Fatalf("nil loader engine: %v", err)
	}
}

func TestHACGPUInputActiveSlotsAndOwnedMetadata(t *testing.T) {
	spec, _ := Pinned(HAC)
	scores := hacGPUScoreMetadata(spec)
	if scores.QScale != spec.QScale || scores.QBias != spec.QBias || scores.BlankScore != 2 || scores.DecoderClampValue != 0 || scores.ScoreMin != -5 || scores.ScoreMax != 5 || scores.StateLen != 4 || scores.MeanQStart != 60 {
		t.Fatalf("pinned Q/clamp metadata differs: %+v", scores)
	}
	meta := tensor.StorageMeta{Device: 7, DType: tensor.DTypeF16, Shape: []int64{64, 1, 192}, Strides: []int64{192, 192, 1}, Tags: []tensor.KoiTag{tensor.TagN, tensor.TagC, tensor.TagT}}
	if err := validateHACGPUInput(meta, HACGPUBatch{64}, 7, false, FixedHACGPUProfile()); err != nil {
		t.Fatal(err)
	}
	if err := validateHACGPUInput(meta, HACGPUBatch{1}, 7, false, FixedHACGPUProfile()); !errors.Is(err, ErrHACGPUInactiveSlotsUnqualified) {
		t.Fatalf("unqualified one-slot production path: %v", err)
	}
	if err := validateHACGPUInput(meta, HACGPUBatch{1}, 7, true, FixedHACGPUProfile()); err == nil {
		t.Fatal("extra contaminated inactive input accepted as logical one-slot input")
	}
	meta.Shape = []int64{1, 1, 192}
	if err := validateHACGPUInput(meta, HACGPUBatch{1}, 7, true, FixedHACGPUProfile()); err != nil {
		t.Fatal(err)
	}
	meta.Device = 8
	if err := validateHACGPUInput(meta, HACGPUBatch{1}, 7, true, FixedHACGPUProfile()); err == nil {
		t.Fatal("foreign device admitted")
	}
	meta.Device = 7
	meta.Strides = []int64{192, 192, 2}
	if err := validateHACGPUInput(meta, HACGPUBatch{1}, 7, true, FixedHACGPUProfile()); err == nil {
		t.Fatal("strided input admitted")
	}
}

type fakeHACGPUOps struct {
	calls  []string
	fail   string
	cancel context.CancelFunc
}

func (o *fakeHACGPUOps) step(name string) error {
	o.calls = append(o.calls, name)
	if name == o.fail {
		if o.cancel != nil {
			o.cancel()
		}
		return errors.New("injected native/observer failure")
	}
	return nil
}
func (o *fakeHACGPUOps) input() error     { return o.step("input") }
func (o *fakeHACGPUOps) conv(i int) error { return o.step(fmt.Sprintf("conv%d", i)) }
func (o *fakeHACGPUOps) recurrent(ctx context.Context) error {
	for i := 0; i < 7; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := o.step(fmt.Sprintf("flstm%d", i)); err != nil {
			return err
		}
	}
	return nil
}
func (o *fakeHACGPUOps) project(context.Context) error {
	if err := o.step("linear1"); err != nil {
		return err
	}
	return o.step("scores")
}

func TestHACGPUCompositionStopsAtEveryStageFailure(t *testing.T) {
	want := []string{"input", "conv0", "conv1", "conv2", "flstm0", "flstm1", "flstm2", "flstm3", "flstm4", "flstm5", "flstm6", "linear1", "scores"}
	good := &fakeHACGPUOps{}
	if err := runHACGPU(context.Background(), good); err != nil || !reflect.DeepEqual(good.calls, want) {
		t.Fatalf("composition: %v %v", good.calls, err)
	}
	for i, name := range want {
		o := &fakeHACGPUOps{fail: name}
		if err := runHACGPU(context.Background(), o); err == nil {
			t.Fatalf("partial graph success after%s", name)
		}
		if !reflect.DeepEqual(o.calls, want[:i+1]) {
			t.Fatalf("work continued after%s: %v", name, o.calls)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	o := &fakeHACGPUOps{}
	if err := runHACGPU(ctx, o); !errors.Is(err, context.Canceled) || len(o.calls) != 0 {
		t.Fatalf("cancelled graph ran: %v %v", o.calls, err)
	}
}

func TestHACGPUCloseAndCancellationCannotProducePartialResult(t *testing.T) {
	m := &HACGPU{}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if result, err := m.Forward(context.Background(), nil, nil, HACGPUPlan{}, HACGPUBatch{64}); result != nil || !errors.Is(err, ErrClosed) {
		t.Fatalf("closed model result=%v err=%v", result, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if result, err := (&HACGPU{}).ForwardObserved(ctx, nil, nil, HACGPUPlan{}, HACGPUBatch{64}, 0xa5, nil); result != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled model result=%v err=%v", result, err)
	}
	if err := (*HACGPUResult)(nil).Close(); err != nil {
		t.Fatal(err)
	}
}
