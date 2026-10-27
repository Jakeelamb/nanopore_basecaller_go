package model

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

func TestHACGPUAdmittedProfileReserveAndUse(t *testing.T) {
	for _, tc := range []struct {
		profile                                           HACGPUProfile
		t, padded, reserveT, reservation, recurrentOffset int64
	}{
		{FixedHACGPUProfile(), 32, 210, 35, 7782400, 5357568},
		{SourceShortHACGPUProfile(), 1024, 6162, 1027, 219070464, 151633920},
		{SourceDefaultHACGPUProfile(), 2048, 12306, 2051, 437174272, 302628864},
	} {
		t.Run(stringProfile(tc.profile), func(t *testing.T) {
			p, err := PlanHACGPU(tc.profile)
			if err != nil {
				t.Fatal(err)
			}
			if p.Profile() != tc.profile || p.reservation.Bytes != tc.reservation || p.reservation.Steps != 6 {
				t.Fatalf("profile or reservation differs: %+v", p)
			}
			if p.input.Geometry.T != tc.profile.Chunk || p.convs.Layers[0].TOut != tc.profile.Chunk || p.convs.Layers[1].TOut != tc.profile.Chunk || p.convs.Layers[1].Geometry.Rows != tc.padded {
				t.Fatal("input or small-convolution geometry differs")
			}
			if p.reservation.steps[2].geometry.T != tc.padded || p.reservation.steps[3].geometry.T != tc.reserveT || p.last.Output.T != tc.t || p.last.Output.Rows != tc.t+5 || p.last.OutputOffsetBytes != tc.recurrentOffset {
				t.Fatal("reservation padding was collapsed into actual conv2 use")
			}
			if p.fused.n != 64 || p.fused.t != tc.t || p.output.n != 64 || p.output.t != tc.t || p.output.down.bytes != 64*tc.t*128*2 || p.output.out.bytes != 64*tc.t*1024*2 {
				t.Fatal("recurrence or projection uses the wrong time extent")
			}
			reserveOutput, err := HACOutputTempElements(64, tc.reserveT)
			if err != nil || p.reservation.steps[5].geometry.LiveBytes != reserveOutput*2 || p.views[5].Geometry.LiveBytes != 64*tc.t*(128+1024)*2 {
				t.Fatal("reserved and used projection scratch differ from source traversals")
			}
		})
	}
}

func stringProfile(p HACGPUProfile) string {
	if p == FixedHACGPUProfile() {
		return "verification192"
	}
	if p == SourceShortHACGPUProfile() {
		return "source6144"
	}
	return "source12288"
}

func TestHACGPUAdmittedProfileViewsAndLiveness(t *testing.T) {
	for _, profile := range []HACGPUProfile{FixedHACGPUProfile(), SourceDefaultHACGPUProfile(), SourceShortHACGPUProfile()} {
		t.Run(stringProfile(profile), func(t *testing.T) {
			p, err := PlanHACGPU(profile)
			if err != nil {
				t.Fatal(err)
			}
			u := p.reservation.Use()
			for _, v := range p.Views() {
				var got GPUView
				if v.Scratch {
					got, err = u.Temp([]int64{v.Geometry.Elems}, v.Geometry.Dtype)
				} else {
					got, err = u.NextTC(v.Geometry.T, v.Geometry.C, v.Geometry.Layout, v.Orientation)
				}
				if err != nil || got != v || v.Offset < 0 || v.End > p.reservation.Bytes {
					t.Fatalf("view escaped or changed during use: %+v %v", got, err)
				}
			}
			if err = u.Finish(); err != nil {
				t.Fatal(err)
			}
			if err = u.Alive(u.Views()[4]); !errors.Is(err, ErrGPUScratchExpired) {
				t.Fatalf("recurrence scratch survived projection reuse: %v", err)
			}
			if err = u.Alive(u.Views()[3]); err != nil {
				t.Fatal("recurrent owner expired before projection", err)
			}
			crop := p.convs.Layers[1]
			if !slices.Equal(crop.CropShape(), []int64{64, profile.Chunk, 16}) || !slices.Equal(crop.CropStrides(), []int64{(profile.Chunk + 18) * 16, 16, 1}) || crop.CropOffsetElements() != 9*16 {
				t.Fatal("conv1 crop lost source padding or batch stride")
			}
			if !slices.Equal(p.last.OutputViewShape(), []int64{64, profile.Chunk / 6, 1024}) || !slices.Equal(p.last.OutputViewStrides(), []int64{1024, 65536, 1}) || p.last.OutputViewOffsetElements() != 65536 {
				t.Fatal("reverse conv2 view lost parent row offset or TNC strides")
			}
			view, err := p.fused.OutputView().NTCView(GPUOrientationForward)
			if err != nil || view.T != profile.Chunk/6 || view.N != 64 || view.C != 1024 || view.Offset != p.last.OutputOffsetBytes+2*65536 || view.StrideN != 1024 || view.StrideT != 65536 || p.output.input.offset != view.Offset {
				t.Fatalf("projection did not select the final forward row2 view: %+v %v", view, err)
			}
			if p.output.out.offset != p.output.down.offset+p.output.down.bytes || p.output.out.offset+p.output.out.bytes > p.output.full.Offset {
				t.Fatal("projection overlaps its recurrent input")
			}
		})
	}
}

func TestHACGPUSourceDefaultRecurrenceHostCalls(t *testing.T) {
	for _, profile := range []HACGPUProfile{SourceDefaultHACGPUProfile(), SourceShortHACGPUProfile()} {
		t.Run(stringProfile(profile), func(t *testing.T) {
			p, err := PlanHACGPU(profile)
			if err != nil {
				t.Fatal(err)
			}
			r := &fusedRecorder{}
			if err = runHACFused(context.Background(), p.fused, r); err != nil {
				t.Fatal(err)
			}
			if len(r.trace) != 11 {
				t.Fatalf("got %d host operations", len(r.trace))
			}
			frames := profile.Chunk / 6
			for i, row := range []int64{0, frames + 1, frames + 2} {
				want := fusedSpan{fusedInOut, p.last.OutputOffsetBytes + row*65536, 65536}
				if r.trace[i].span != want || r.trace[i].value != byte(i/2) {
					t.Fatalf("sentinel %d uses wrong full parent extent: %+v", i, r.trace[i])
				}
			}
			if r.trace[3].span != p.fused.scratch || p.fused.scratch.bytes != (frames+1)*64*516+128 {
				t.Fatal("source-default scratch is not fully initialized")
			}
			for i, event := range r.trace[4:] {
				want := fusedLayerCall{i, 64, int32(frames), -1, 1}
				if i%2 == 1 {
					want.direction, want.parity = 1, 0
				}
				if !event.layer || event.call != want {
					t.Fatalf("layer%d host invocation geometry: %+v", i, event)
				}
			}
		})
	}
}

func TestHACGPUProfileInputAndQuarantine(t *testing.T) {
	for _, profile := range []HACGPUProfile{FixedHACGPUProfile(), SourceDefaultHACGPUProfile(), SourceShortHACGPUProfile()} {
		t.Run(stringProfile(profile), func(t *testing.T) {
			meta := tensor.StorageMeta{Device: 7, DType: tensor.DTypeF16, Shape: []int64{64, 1, profile.Chunk}, Strides: []int64{profile.Chunk, profile.Chunk, 1}, Tags: []tensor.KoiTag{tensor.TagN, tensor.TagC, tensor.TagT}}
			if err := validateHACGPUInput(meta, HACGPUBatch{64}, 7, false, profile); err != nil {
				t.Fatal(err)
			}
			other := SourceDefaultHACGPUProfile()
			if other == profile {
				other = FixedHACGPUProfile()
			}
			if err := validateHACGPUInput(meta, HACGPUBatch{64}, 7, false, other); err == nil {
				t.Fatal("input geometry for another admitted profile was accepted")
			}
			if err := validateHACGPUInput(meta, HACGPUBatch{1}, 7, false, profile); !errors.Is(err, ErrHACGPUInactiveSlotsUnqualified) {
				t.Fatal("one-slot production quarantine was bypassed", err)
			}
			if err := validateHACGPUInput(meta, HACGPUBatch{1}, 7, true, profile); err == nil {
				t.Fatal("probe admitted contaminated inactive input slots")
			}
			meta.Shape[0] = 1
			if err := validateHACGPUInput(meta, HACGPUBatch{1}, 7, true, profile); err != nil {
				t.Fatal(err)
			}
			for _, mutate := range []func(*tensor.StorageMeta){
				func(m *tensor.StorageMeta) { m.Device++ },
				func(m *tensor.StorageMeta) { m.DType = tensor.DTypeI8 },
				func(m *tensor.StorageMeta) { m.ScalePresent = true },
				func(m *tensor.StorageMeta) { m.Strides = []int64{profile.Chunk, profile.Chunk, 2} },
				func(m *tensor.StorageMeta) { m.Tags = []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC} },
			} {
				bad := meta
				mutate(&bad)
				if err := validateHACGPUInput(bad, HACGPUBatch{1}, 7, true, profile); err == nil {
					t.Fatal("invalid profile-bound input accepted")
				}
			}
		})
	}
}

func TestHACGPUShortSourceConstructorAndRejections(t *testing.T) {
	if _, err := NewHACGPU(nil, nil, HACGPUWeights{}); !errors.Is(err, ErrGPUWeights) {
		t.Fatalf("short profile did not reach weight admission: %v", err)
	}
	for _, mutate := range []func(*HACGPUProfile){func(p *HACGPUProfile) { p.Batch = 32 }, func(p *HACGPUProfile) { p.Chunk = 6143 }, func(p *HACGPUProfile) { p.Chunk = 6150 }, func(p *HACGPUProfile) { p.Layout = GPUStorageCUTLASSF16 }} {
		profile := SourceShortHACGPUProfile()
		mutate(&profile)
		if _, err := PlanHACGPU(profile); !errors.Is(err, ErrGPUPlanLayout) {
			t.Fatalf("foreign short profile accepted: %+v %v", profile, err)
		}
	}
}

func TestHACGPUFiniteProfilesRejectNegativeAndOverflow(t *testing.T) {
	for _, p := range []HACGPUProfile{
		{64, -1, GPUStorageCUTLASSI8}, {64, 0, GPUStorageCUTLASSI8},
		{-64, 12288, GPUStorageCUTLASSI8}, {128, 12288, GPUStorageCUTLASSI8},
		{64, 12480, GPUStorageCUTLASSI8}, {64, math.MaxInt64, GPUStorageCUTLASSI8},
		{math.MaxInt64, 12288, GPUStorageCUTLASSI8}, {64, 12288, GPUStorageCUTLASSF16},
	} {
		if _, err := PlanHACGPU(p); !errors.Is(err, ErrGPUPlanLayout) {
			t.Fatalf("unadmitted profile reached planner arithmetic: %+v %v", p, err)
		}
	}
	if _, err := HACOutputTempElements(64, math.MaxInt64); !errors.Is(err, ErrGPUPlanOverflow) {
		t.Fatalf("projection scratch overflow not rejected: %v", err)
	}
	if _, err := NewGPUStorageGeometry(GPUStorageCUTLASSI8, 64, math.MaxInt64, 1024); !errors.Is(err, ErrGPUPlanOverflow) {
		t.Fatalf("recurrent parent overflow not rejected: %v", err)
	}
}
