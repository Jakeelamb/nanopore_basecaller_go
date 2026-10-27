package model

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
)

// gpuPlanStep describes one reservation call in a differential case.
type gpuPlanStep struct {
	scratch     bool
	layout      GPUStorageLayout
	t, c        int64
	sizes       []int64
	dtype       GPUDtype
	orientation GPUOrientation
}

type gpuPlanCase struct {
	name  string
	batch int64
	steps []gpuPlanStep
	// wantReservation, wantPeakLive and wantOffsets are hand-computed anchors
	// from WorkingMemory.cpp:38-58, independent of the oracle.
	wantReservation int64
	wantPeakLive    int64
	wantOffsets     []int64
}

func gpuCurrent(layout GPUStorageLayout, t, c int64, orientation GPUOrientation) gpuPlanStep {
	return gpuPlanStep{layout: layout, t: t, c: c, orientation: orientation}
}

func gpuScratch(dtype GPUDtype, sizes ...int64) gpuPlanStep {
	return gpuPlanStep{scratch: true, sizes: sizes, dtype: dtype}
}

func oracleLayout(layout GPUStorageLayout) hacgpu.Layout {
	switch layout {
	case GPUStorageNTC:
		return hacgpu.NTC
	case GPUStorageTNC:
		return hacgpu.TNC
	case GPUStorageCUTLASSF16:
		return hacgpu.CUTLASSTNCF16
	case GPUStorageCUTLASSI8:
		return hacgpu.CUTLASSTNCI8
	case GPUStorageCUBLASExt2C:
		return hacgpu.CUBLASTN2C
	case GPUStorageCUBLASTNC:
		return hacgpu.CUBLASTNC
	default:
		panic("oracleLayout: " + layout.String())
	}
}

func oracleDtype(dtype GPUDtype) hacgpu.Dtype {
	if dtype == GPUDtypeInt8 {
		return hacgpu.Int8
	}
	return hacgpu.Half
}

func oracleOps(c gpuPlanCase) []hacgpu.Op {
	ops := make([]hacgpu.Op, 0, len(c.steps))
	for _, step := range c.steps {
		if step.scratch {
			ops = append(ops, hacgpu.Op{Scratch: true, Sizes: step.sizes, Dtype: oracleDtype(step.dtype)})
			continue
		}
		ops = append(ops, hacgpu.Op{
			Layout: oracleLayout(step.layout),
			T:      step.t,
			C:      step.c,
		})
	}
	return ops
}

// runPlan reserves, admits a native extent, replays the use phase and closes it.
func runPlan(t *testing.T, c gpuPlanCase) (GPUMemoryReservation, *GPUMemoryUse) {
	t.Helper()
	plan, err := NewGPUMemoryPlan(c.batch)
	if err != nil {
		t.Fatal(err)
	}
	for _, step := range c.steps {
		if step.scratch {
			if err := plan.Temp(step.sizes, step.dtype); err != nil {
				t.Fatal(err)
			}
			continue
		}
		if err := plan.NextTC(step.t, step.c, step.layout, step.orientation); err != nil {
			t.Fatal(err)
		}
	}
	reservation, err := plan.Reserve()
	if err != nil {
		t.Fatal(err)
	}
	if err := reservation.AdmitNative(GPUNativeAllocation{
		Bytes:     reservation.Bytes,
		Alignment: GPUWorkingMemoryAlignment,
	}); err != nil {
		t.Fatal(err)
	}
	use := reservation.Use()
	for _, step := range c.steps {
		if step.scratch {
			if _, err := use.Temp(step.sizes, step.dtype); err != nil {
				t.Fatal(err)
			}
			continue
		}
		if _, err := use.NextTC(step.t, step.c, step.layout, step.orientation); err != nil {
			t.Fatal(err)
		}
	}
	if err := use.Finish(); err != nil {
		t.Fatal(err)
	}
	return reservation, use
}

// gpuPlanCases covers the HAC conv/LSTM sequences the pinned source reserves:
// ConvStack.cpp:191-211 and FLSTMStack.cpp:97-109, in CUTLASS, CUBLAS and
// CUBLAS_TN2C forms.
func gpuPlanCases() []gpuPlanCase {
	return []gpuPlanCase{
		{
			// HAC: three convs (1->16, 16->16, 16->1024 stride 6) then a
			// CUTLASS int8 FLSTM with the source scratch formula.
			name:  "hac-conv-cutlass-lstm",
			batch: 1,
			steps: []gpuPlanStep{
				gpuCurrent(GPUStorageNTC, 768, 1, GPUOrientationForward),
				gpuCurrent(GPUStorageNTC, 768, 16, GPUOrientationForward),
				gpuCurrent(GPUStorageNTC, 768, 16, GPUOrientationForward),
				gpuCurrent(GPUStorageCUTLASSI8, 128, 1024, GPUOrientationForward),
				gpuScratch(GPUDtypeInt8, 2*(128+1)*1*(2*128)+4*((128+1)*1)+4*(1/2)),
				gpuCurrent(GPUStorageNTC, 128, 128, GPUOrientationForward),
			},
			wantReservation: 203008,
			wantPeakLive:    202756,
			wantOffsets:     []int64{0, 178432, 0, 66816, 0, 0},
		},
		{
			// CUBLAS_TNC fallback with the FLSTM half scratch.
			name:  "hac-cublas-tnc-flstm",
			batch: 4,
			steps: []gpuPlanStep{
				gpuCurrent(GPUStorageTNC, 256, 1024, GPUOrientationForward),
				gpuCurrent(GPUStorageCUBLASTNC, 256, 1024, GPUOrientationForward),
				gpuScratch(GPUDtypeHalf, 4*((2*128)+(4*1024))),
				gpuCurrent(GPUStorageNTC, 256, 128, GPUOrientationForward),
			},
		},
		{
			// CUBLAS_TN2C interleaved storage, odd T and reverse orientation.
			name:  "hac-cublas-tn2c",
			batch: 3,
			steps: []gpuPlanStep{
				gpuCurrent(GPUStorageTNC, 97, 1024, GPUOrientationForward),
				gpuCurrent(GPUStorageCUBLASExt2C, 97, 1024, GPUOrientationReverse),
				gpuScratch(GPUDtypeHalf, 3, 4*1024),
				gpuCurrent(GPUStorageNTC, 97, 128, GPUOrientationForward),
			},
		},
		{
			// ConvStack.cpp:330,344-348: window TNC, int8 conversion then the
			// CUTLASS int8 input, with the variable-chunk FLSTM scratch.
			name:  "conv-window-cutlass-i8",
			batch: 2,
			steps: []gpuPlanStep{
				gpuCurrent(GPUStorageNTC, 64, 16, GPUOrientationForward),
				gpuCurrent(GPUStorageTNC, 64, 80, GPUOrientationForward),
				gpuCurrent(GPUStorageTNC, 64, 128, GPUOrientationForward),
				gpuCurrent(GPUStorageCUTLASSI8, 64, 128, GPUOrientationReverse),
				gpuScratch(GPUDtypeInt8, 2*(64+1)*2*(2*128)+4*((64+1)*2)+4*(2/2)),
			},
		},
		{
			// Minimum shapes: every extent pads to one 256-byte block.
			name:  "minimum-shapes",
			batch: 1,
			steps: []gpuPlanStep{
				gpuCurrent(GPUStorageNTC, 1, 1, GPUOrientationForward),
				gpuCurrent(GPUStorageTNC, 1, 1, GPUOrientationForward),
				gpuCurrent(GPUStorageCUTLASSF16, 1, 1, GPUOrientationReverse),
				gpuCurrent(GPUStorageCUBLASTNC, 1, 1, GPUOrientationReverse),
				gpuCurrent(GPUStorageCUBLASExt2C, 1, 1, GPUOrientationReverse),
				gpuScratch(GPUDtypeInt8, 1),
			},
			wantReservation: 512,
			wantPeakLive:    20,
			wantOffsets:     []int64{0, 256, 0, 256, 0, 256},
		},
	}
}

func TestHACGPUMemory(t *testing.T) {
	t.Run("SourcePlannerDifferential", func(t *testing.T) {
		for _, c := range gpuPlanCases() {
			t.Run(c.name, func(t *testing.T) {
				trace, err := hacgpu.Plan(c.batch, oracleOps(c))
				if err != nil {
					t.Fatal(err)
				}
				reservation, use := runPlan(t, c)
				if reservation.Bytes != trace.Reservation {
					t.Fatalf("reservation %d, oracle %d", reservation.Bytes, trace.Reservation)
				}
				if reservation.PeakLiveBytes != trace.PeakLive {
					t.Fatalf("peak live %d, oracle %d", reservation.PeakLiveBytes, trace.PeakLive)
				}
				if use.PeakLiveBytes() != trace.PeakLive {
					t.Fatalf("use peak live %d, oracle %d", use.PeakLiveBytes(), trace.PeakLive)
				}
				views := use.Views()
				if len(views) != len(trace.Steps) {
					t.Fatalf("%d views, oracle %d steps", len(views), len(trace.Steps))
				}
				for i, view := range views {
					want := trace.Steps[i]
					got := view.Geometry
					if got.Bytes != want.Bytes || got.Elems != want.Elems ||
						got.LiveBytes != want.LiveBytes || got.Rows != want.Rows {
						t.Fatalf("step %d geometry %+v, oracle %+v", i, got, want)
					}
					if view.Offset != want.Offset {
						t.Fatalf("step %d offset %d, oracle %d", i, view.Offset, want.Offset)
					}
					if view.End != want.Offset+want.Bytes {
						t.Fatalf("step %d end %d, oracle %d", i, view.End, want.Offset+want.Bytes)
					}
					if (view.Offset == 0) != want.Front {
						t.Fatalf("step %d front %v, oracle %v", i, view.Offset == 0, want.Front)
					}
					if view.Scratch != want.Scratch {
						t.Fatalf("step %d scratch %v, oracle %v", i, view.Scratch, want.Scratch)
					}
					if got.Bytes%GPUWorkingMemoryAlignment != 0 {
						t.Fatalf("step %d bytes %d not 256-aligned", i, got.Bytes)
					}
				}
				if c.wantReservation != 0 && reservation.Bytes != c.wantReservation {
					t.Fatalf("anchor reservation %d, want %d", reservation.Bytes, c.wantReservation)
				}
				if c.wantPeakLive != 0 && reservation.PeakLiveBytes != c.wantPeakLive {
					t.Fatalf("anchor peak live %d, want %d", reservation.PeakLiveBytes, c.wantPeakLive)
				}
				if c.wantOffsets != nil {
					offsets := make([]int64, len(views))
					for i, view := range views {
						offsets[i] = view.Offset
					}
					if !slices.Equal(offsets, c.wantOffsets) {
						t.Fatalf("anchor offsets %v, want %v", offsets, c.wantOffsets)
					}
				}
			})
		}
	})

	t.Run("ReservationVersusUse", func(t *testing.T) {
		reservation, use := runPlan(t, gpuPlanCases()[0])
		if reservation.Bytes%GPUWorkingMemoryAlignment != 0 {
			t.Fatalf("reservation %d is not 256-aligned", reservation.Bytes)
		}
		if reservation.Bytes < reservation.PeakLiveBytes {
			t.Fatalf("reservation %d below live peak %d", reservation.Bytes, reservation.PeakLiveBytes)
		}
		if reservation.Bytes == reservation.PeakLiveBytes {
			t.Fatal("aligned reservation should exceed the exact live peak for this case")
		}
		for i, view := range use.Views() {
			if view.End != view.Offset+view.Geometry.Bytes || view.End > reservation.Bytes {
				t.Fatalf("step %d extent [%d,%d) outside reservation %d", i, view.Offset, view.End, reservation.Bytes)
			}
			// A current view stays live as the input of the next step; a scratch
			// view is only live until a later view reuses its memory.
			if !view.Scratch {
				if err := use.Alive(view); err != nil {
					t.Fatalf("step %d not alive: %v", i, err)
				}
			}
		}
	})

	t.Run("Overflow", func(t *testing.T) {
		if _, err := NewGPUStorageGeometry(GPUStorageCUTLASSF16, 1, math.MaxInt64, 1); !errors.Is(err, ErrGPUPlanOverflow) {
			t.Fatalf("shape overflow: %v", err)
		}
		plan, err := NewGPUMemoryPlan(1)
		if err != nil {
			t.Fatal(err)
		}
		if err := plan.NextTC(math.MaxInt64, 1, GPUStorageNTC, GPUOrientationForward); !errors.Is(err, ErrGPUPlanOverflow) {
			t.Fatalf("next_TC overflow: %v", err)
		}
		if err := plan.Temp([]int64{math.MaxInt64, 2}, GPUDtypeHalf); !errors.Is(err, ErrGPUPlanOverflow) {
			t.Fatalf("temp overflow: %v", err)
		}
	})

	t.Run("UnderReservation", func(t *testing.T) {
		plan, err := NewGPUMemoryPlan(1)
		if err != nil {
			t.Fatal(err)
		}
		if err := plan.NextTC(64, 16, GPUStorageNTC, GPUOrientationForward); err != nil {
			t.Fatal(err)
		}
		if err := plan.NextTC(64, 16, GPUStorageNTC, GPUOrientationForward); err != nil {
			t.Fatal(err)
		}
		reservation, err := plan.Reserve()
		if err != nil {
			t.Fatal(err)
		}
		if reservation.Bytes != 4096 {
			t.Fatalf("reservation %d, want 4096", reservation.Bytes)
		}
		use := reservation.Use()
		if _, err := use.NextTC(64, 16, GPUStorageNTC, GPUOrientationForward); err != nil {
			t.Fatal(err)
		}
		// A later step larger than the reserved pair is the source's
		// "overlap detected" case (WorkingMemory.cpp:42-44).
		if _, err := use.NextTC(128, 16, GPUStorageNTC, GPUOrientationForward); !errors.Is(err, ErrGPUPlanOverlap) {
			t.Fatalf("under-reservation accepted: %v", err)
		}
	})

	t.Run("UnderReservationBeforeCurrent", func(t *testing.T) {
		for _, c := range []struct {
			name           string
			scratch        bool
			leadingScratch bool
		}{
			{name: "current"},
			{name: "scratch", scratch: true},
			{name: "current-after-scratch", leadingScratch: true},
			{name: "scratch-after-scratch", scratch: true, leadingScratch: true},
		} {
			t.Run(c.name, func(t *testing.T) {
				plan, err := NewGPUMemoryPlan(1)
				if err != nil {
					t.Fatal(err)
				}
				if c.leadingScratch {
					if err := plan.Temp([]int64{256}, GPUDtypeInt8); err != nil {
						t.Fatal(err)
					}
				}
				if c.scratch {
					err = plan.Temp([]int64{256}, GPUDtypeInt8)
				} else {
					err = plan.NextTC(1, 128, GPUStorageNTC, GPUOrientationForward)
				}
				if err != nil {
					t.Fatal(err)
				}
				reservation, err := plan.Reserve()
				if err != nil {
					t.Fatal(err)
				}
				use := reservation.Use()
				if c.leadingScratch {
					if _, err := use.Temp([]int64{256}, GPUDtypeInt8); err != nil {
						t.Fatal(err)
					}
				}
				before, peakBefore := use.Views(), use.PeakLiveBytes()
				if c.scratch {
					_, err = use.Temp([]int64{257}, GPUDtypeInt8)
				} else {
					_, err = use.NextTC(1, 129, GPUStorageNTC, GPUOrientationForward)
				}
				if !errors.Is(err, ErrGPUPlanOverlap) {
					t.Fatalf("oversized extent before current accepted: %v", err)
				}
				if !slices.Equal(use.Views(), before) || use.PeakLiveBytes() != peakBefore {
					t.Fatal("rejected extent changed the use traversal")
				}
				for _, view := range before {
					if err := use.Alive(view); err != nil {
						t.Fatalf("rejected extent invalidated scratch: %v", err)
					}
				}
				if c.scratch {
					_, err = use.Temp([]int64{256}, GPUDtypeInt8)
				} else {
					_, err = use.NextTC(1, 128, GPUStorageNTC, GPUOrientationForward)
				}
				if err != nil {
					t.Fatalf("retry within reservation: %v", err)
				}
				if err := use.Finish(); err != nil {
					t.Fatal(err)
				}
			})
		}
	})

	t.Run("Overlap", func(t *testing.T) {
		plan, err := NewGPUMemoryPlan(1)
		if err != nil {
			t.Fatal(err)
		}
		for range 2 {
			if err := plan.NextTC(64, 16, GPUStorageNTC, GPUOrientationForward); err != nil {
				t.Fatal(err)
			}
		}
		reservation, err := plan.Reserve()
		if err != nil {
			t.Fatal(err)
		}
		use := reservation.Use()
		if _, err := use.NextTC(64, 16, GPUStorageNTC, GPUOrientationForward); err != nil {
			t.Fatal(err)
		}
		// 64*17*2 = 2176 live bytes pad to 2304; 2048+2304 > 4096.
		if _, err := use.NextTC(64, 17, GPUStorageNTC, GPUOrientationForward); !errors.Is(err, ErrGPUPlanOverlap) {
			t.Fatalf("overlapping pair accepted: %v", err)
		}
	})

	t.Run("UndersizedPlan", func(t *testing.T) {
		empty, err := NewGPUMemoryPlan(1)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := empty.Reserve(); !errors.Is(err, ErrGPUPlanUndersized) {
			t.Fatalf("empty plan accepted: %v", err)
		}

		plan, err := NewGPUMemoryPlan(1)
		if err != nil {
			t.Fatal(err)
		}
		for range 2 {
			if err := plan.NextTC(64, 16, GPUStorageNTC, GPUOrientationForward); err != nil {
				t.Fatal(err)
			}
		}
		reservation, err := plan.Reserve()
		if err != nil {
			t.Fatal(err)
		}
		if err := reservation.AdmitNative(GPUNativeAllocation{Bytes: reservation.Bytes - GPUWorkingMemoryAlignment, Alignment: GPUWorkingMemoryAlignment}); !errors.Is(err, ErrGPUPlanUndersized) {
			t.Fatalf("undersized native extent accepted: %v", err)
		}
		if err := reservation.AdmitNative(GPUNativeAllocation{Bytes: reservation.Bytes, Alignment: 128}); !errors.Is(err, ErrGPUPlanUndersized) {
			t.Fatalf("under-aligned native extent accepted: %v", err)
		}

		use := reservation.Use()
		if _, err := use.NextTC(64, 16, GPUStorageNTC, GPUOrientationForward); err != nil {
			t.Fatal(err)
		}
		if err := use.Finish(); !errors.Is(err, ErrGPUPlanUndersized) {
			t.Fatalf("short use accepted: %v", err)
		}
		if _, err := use.NextTC(64, 16, GPUStorageNTC, GPUOrientationForward); err != nil {
			t.Fatal(err)
		}
		if _, err := use.NextTC(64, 16, GPUStorageNTC, GPUOrientationForward); !errors.Is(err, ErrGPUPlanUndersized) {
			t.Fatalf("extra use step accepted: %v", err)
		}
	})

	t.Run("CUTLASSStorage", func(t *testing.T) {
		cutlass, err := NewGPUStorageGeometry(GPUStorageCUTLASSF16, 2, 128, 1024)
		if err != nil {
			t.Fatal(err)
		}
		cublas, err := NewGPUStorageGeometry(GPUStorageCUBLASTNC, 2, 128, 1024)
		if err != nil {
			t.Fatal(err)
		}
		// WorkingMemory.cpp:91-98 fixes CUTLASS at T+5 and CUBLAS_TNC at T+3.
		if cutlass.Rows != 133 || cutlass.Bytes != 544768 {
			t.Fatalf("CUTLASS storage rows %d bytes %d, want 133 and 544768", cutlass.Rows, cutlass.Bytes)
		}
		if cublas.Rows != 131 || cublas.Bytes != 536576 {
			t.Fatalf("CUBLAS_TNC storage rows %d bytes %d, want 131 and 536576", cublas.Rows, cublas.Bytes)
		}
		if cutlass.Bytes <= cublas.Bytes {
			t.Fatal("CUTLASS must reserve the extra two rows")
		}
	})

	t.Run("CUTLASSSubstitution", func(t *testing.T) {
		plan, err := NewGPUMemoryPlan(1)
		if err != nil {
			t.Fatal(err)
		}
		if err := plan.NextTC(64, 1024, GPUStorageCUTLASSF16, GPUOrientationForward); err != nil {
			t.Fatal(err)
		}
		reservation, err := plan.Reserve()
		if err != nil {
			t.Fatal(err)
		}
		use := reservation.Use()
		if _, err := use.NextTC(64, 1024, GPUStorageCUBLASTNC, GPUOrientationForward); !errors.Is(err, ErrGPUPlanMismatch) {
			t.Fatalf("CUTLASS reservation replayed as CUBLAS_TNC: %v", err)
		}
	})

	t.Run("WrongReverseView", func(t *testing.T) {
		plan, err := NewGPUMemoryPlan(1)
		if err != nil {
			t.Fatal(err)
		}
		if err := plan.NextTC(64, 1024, GPUStorageCUTLASSF16, GPUOrientationForward); err != nil {
			t.Fatal(err)
		}
		reservation, err := plan.Reserve()
		if err != nil {
			t.Fatal(err)
		}
		use := reservation.Use()
		view, err := use.NextTC(64, 1024, GPUStorageCUTLASSF16, GPUOrientationForward)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := view.NTCView(GPUOrientationReverse); !errors.Is(err, ErrGPUViewOrientation) {
			t.Fatalf("reverse view of forward storage accepted: %v", err)
		}
		forward, err := view.NTCView(GPUOrientationForward)
		if err != nil {
			t.Fatal(err)
		}
		if forward.Offset != view.Offset+2*1024*2 {
			t.Fatalf("forward view offset %d, want %d", forward.Offset, view.Offset+2*1024*2)
		}
		if forward.StrideN != 1024 || forward.StrideT != 1024 || forward.StrideC != 1 {
			t.Fatalf("forward strides %d/%d/%d", forward.StrideN, forward.StrideT, forward.StrideC)
		}
	})

	t.Run("ReverseViewOffset", func(t *testing.T) {
		plan, err := NewGPUMemoryPlan(2)
		if err != nil {
			t.Fatal(err)
		}
		if err := plan.NextTC(64, 128, GPUStorageCUBLASExt2C, GPUOrientationReverse); err != nil {
			t.Fatal(err)
		}
		reservation, err := plan.Reserve()
		if err != nil {
			t.Fatal(err)
		}
		view, err := reservation.Use().NextTC(64, 128, GPUStorageCUBLASExt2C, GPUOrientationReverse)
		if err != nil {
			t.Fatal(err)
		}
		reverse, err := view.NTCView(GPUOrientationReverse)
		if err != nil {
			t.Fatal(err)
		}
		// narrow(0, 1, T).transpose(1, 0).select(2, 1): row 1, plane 1.
		want := view.Offset + (1*2*2+1)*128*2
		if reverse.Offset != want {
			t.Fatalf("reverse view offset %d, want %d", reverse.Offset, want)
		}
		if reverse.StrideN != 2*128 || reverse.StrideT != 2*2*128 {
			t.Fatalf("interleaved strides %d/%d", reverse.StrideN, reverse.StrideT)
		}
	})

	t.Run("ScratchReuse", func(t *testing.T) {
		plan, err := NewGPUMemoryPlan(1)
		if err != nil {
			t.Fatal(err)
		}
		if err := plan.NextTC(64, 16, GPUStorageNTC, GPUOrientationForward); err != nil {
			t.Fatal(err)
		}
		if err := plan.Temp([]int64{2048}, GPUDtypeInt8); err != nil {
			t.Fatal(err)
		}
		if err := plan.NextTC(64, 16, GPUStorageNTC, GPUOrientationForward); err != nil {
			t.Fatal(err)
		}
		reservation, err := plan.Reserve()
		if err != nil {
			t.Fatal(err)
		}
		use := reservation.Use()
		first, err := use.NextTC(64, 16, GPUStorageNTC, GPUOrientationForward)
		if err != nil {
			t.Fatal(err)
		}
		scratch, err := use.Temp([]int64{2048}, GPUDtypeInt8)
		if err != nil {
			t.Fatal(err)
		}
		if err := use.Alive(scratch); err != nil {
			t.Fatalf("fresh scratch rejected: %v", err)
		}
		if _, err := use.NextTC(64, 16, GPUStorageNTC, GPUOrientationForward); err != nil {
			t.Fatal(err)
		}
		if err := use.Alive(scratch); !errors.Is(err, ErrGPUScratchExpired) {
			t.Fatalf("reused scratch accepted: %v", err)
		}
		if err := use.Alive(first); err != nil {
			t.Fatalf("live current rejected: %v", err)
		}
	})

	t.Run("SentinelRows", func(t *testing.T) {
		cases := []struct {
			name        string
			layout      GPUStorageLayout
			orientation GPUOrientation
			want        GPURowLayout
		}{
			{
				name: "cutlass-forward", layout: GPUStorageCUTLASSF16, orientation: GPUOrientationForward,
				want: GPURowLayout{ValidStart: 2, ValidRows: 64, Sentinels: []int64{0, 66}, Unused: []int64{67, 68}},
			},
			{
				name: "cutlass-reverse", layout: GPUStorageCUTLASSI8, orientation: GPUOrientationReverse,
				want: GPURowLayout{ValidStart: 1, ValidRows: 64, Sentinels: []int64{0, 66}, Unused: []int64{67, 68}},
			},
			{
				name: "cublas-tnc-forward", layout: GPUStorageCUBLASTNC, orientation: GPUOrientationForward,
				want: GPURowLayout{ValidStart: 2, ValidRows: 64, Sentinels: []int64{0, 66}},
			},
			{
				name: "cublas-tn2c-reverse", layout: GPUStorageCUBLASExt2C, orientation: GPUOrientationReverse,
				want: GPURowLayout{ValidStart: 1, ValidRows: 64, Sentinels: []int64{0, 64}},
			},
			{
				name: "ntc", layout: GPUStorageNTC, orientation: GPUOrientationForward,
				want: GPURowLayout{ValidStart: 0, ValidRows: 64},
			},
		}
		for _, c := range cases {
			t.Run(c.name, func(t *testing.T) {
				plan, err := NewGPUMemoryPlan(1)
				if err != nil {
					t.Fatal(err)
				}
				if err := plan.NextTC(64, 128, c.layout, c.orientation); err != nil {
					t.Fatal(err)
				}
				reservation, err := plan.Reserve()
				if err != nil {
					t.Fatal(err)
				}
				view, err := reservation.Use().NextTC(64, 128, c.layout, c.orientation)
				if err != nil {
					t.Fatal(err)
				}
				got, err := view.RowLayout()
				if err != nil {
					t.Fatal(err)
				}
				if got.ValidStart != c.want.ValidStart || got.ValidRows != c.want.ValidRows {
					t.Fatalf("valid range %d+%d, want %d+%d", got.ValidStart, got.ValidRows, c.want.ValidStart, c.want.ValidRows)
				}
				if !slices.Equal(got.Sentinels, c.want.Sentinels) {
					t.Fatalf("sentinels %v, want %v", got.Sentinels, c.want.Sentinels)
				}
				if !slices.Equal(got.Unused, c.want.Unused) {
					t.Fatalf("unused rows %v, want %v", got.Unused, c.want.Unused)
				}
			})
		}
	})

	t.Run("Alignment", func(t *testing.T) {
		for _, layout := range []GPUStorageLayout{
			GPUStorageNTC, GPUStorageTNC, GPUStorageCUTLASSF16,
			GPUStorageCUTLASSI8, GPUStorageCUBLASExt2C, GPUStorageCUBLASTNC,
		} {
			geometry, err := NewGPUStorageGeometry(layout, 3, 7, 5)
			if err != nil {
				t.Fatal(err)
			}
			if geometry.Bytes%GPUWorkingMemoryAlignment != 0 {
				t.Fatalf("%s bytes %d not aligned", layout, geometry.Bytes)
			}
			if geometry.Bytes < geometry.LiveBytes || geometry.Bytes-geometry.LiveBytes >= GPUWorkingMemoryAlignment {
				t.Fatalf("%s bytes %d live %d", layout, geometry.Bytes, geometry.LiveBytes)
			}
			elemSize := int64(2)
			if layout == GPUStorageCUTLASSI8 {
				elemSize = 1
			}
			if geometry.LiveBytes != geometry.Elems*elemSize {
				t.Fatalf("%s live %d elems %d", layout, geometry.LiveBytes, geometry.Elems)
			}
		}
		scratch, err := NewGPUScratchGeometry(GPUDtypeInt8, 1)
		if err != nil {
			t.Fatal(err)
		}
		if scratch.Bytes != GPUWorkingMemoryAlignment || scratch.LiveBytes != 1 {
			t.Fatalf("minimum scratch bytes %d live %d", scratch.Bytes, scratch.LiveBytes)
		}
	})
}

func FuzzHACGPUMemoryInitialBounds(f *testing.F) {
	for layout := uint8(0); layout <= uint8(GPUStorageRaw); layout++ {
		f.Add(layout, uint8(1), uint16(1), uint16(128), uint16(129))
		f.Add(layout, uint8(3), uint16(17), uint16(129), uint16(128))
	}
	f.Fuzz(func(t *testing.T, layoutID, batchID uint8, timeID, reservedC, usedC uint16) {
		layout := GPUStorageLayout(layoutID % uint8(GPUStorageRaw+1))
		batch, time := int64(batchID%8)+1, int64(timeID%256)+1
		reserveChannels, useChannels := int64(reservedC%2048)+1, int64(usedC%2048)+1
		plan, err := NewGPUMemoryPlan(batch)
		if err != nil {
			t.Fatal(err)
		}
		var geometry GPUStorageGeometry
		var geometryErr error
		if layout == GPUStorageRaw {
			err = plan.Temp([]int64{reserveChannels}, GPUDtypeInt8)
			geometry, geometryErr = NewGPUScratchGeometry(GPUDtypeInt8, useChannels)
		} else {
			err = plan.NextTC(time, reserveChannels, layout, GPUOrientationReverse)
			geometry, geometryErr = NewGPUStorageGeometry(layout, batch, time, useChannels)
		}
		if err != nil {
			t.Fatal(err)
		}
		if geometryErr != nil {
			t.Fatal(geometryErr)
		}
		reservation, err := plan.Reserve()
		if err != nil {
			t.Fatal(err)
		}
		use := reservation.Use()
		var view GPUView
		if layout == GPUStorageRaw {
			view, err = use.Temp([]int64{useChannels}, GPUDtypeInt8)
		} else {
			view, err = use.NextTC(time, useChannels, layout, GPUOrientationReverse)
		}
		if geometry.Bytes > reservation.Bytes {
			if !errors.Is(err, ErrGPUPlanOverlap) {
				t.Fatalf("oversized first extent: geometry=%+v reservation=%d err=%v", geometry, reservation.Bytes, err)
			}
			if len(use.Views()) != 0 {
				t.Fatal("rejected first extent became a bound view")
			}
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		if view.Offset != 0 || view.End > reservation.Bytes || view.End != geometry.Bytes {
			t.Fatalf("first extent outside reservation %d: %+v", reservation.Bytes, view)
		}
		if err := use.Finish(); err != nil {
			t.Fatal(err)
		}
	})
}
