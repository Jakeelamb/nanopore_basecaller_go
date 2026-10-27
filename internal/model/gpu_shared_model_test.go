package model

import (
	"context"
	"errors"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

func TestGPUForwardRequiresAnUnchangedExplicitPlan(t *testing.T) {
	ctx := context.Background()
	for _, profile := range []HACGPUProfile{FixedHACGPUProfile(), SourceShortHACGPUProfile(), SourceDefaultHACGPUProfile()} {
		p, err := PlanHACGPU(profile)
		if err != nil {
			t.Fatal(err)
		}
		m := &HACGPU{}
		if _, err = m.Forward(ctx, nil, nil, p, HACGPUBatch{64}); !errors.Is(err, tensor.ErrClosed) {
			t.Fatalf("valid HAC plan rejected before input boundary: %v", err)
		}
		p.output.t++
		if _, err = m.Forward(ctx, nil, nil, p, HACGPUBatch{64}); !errors.Is(err, ErrGPUPlanState) {
			t.Fatalf("mutated HAC plan admitted: %v", err)
		}
	}
	if _, err := (&HACGPU{}).Forward(ctx, nil, nil, HACGPUPlan{}, HACGPUBatch{64}); !errors.Is(err, ErrGPUPlanState) {
		t.Fatalf("zero HAC plan admitted: %v", err)
	}
	for _, profile := range []SUPGPUProfile{FixedSUPGPUProfile(), ShortFixedSUPGPUProfile(), FullFixedSUPGPUProfile()} {
		p, err := PlanSUPGPU(profile)
		if err != nil {
			t.Fatal(err)
		}
		m := &SUPGPU{}
		if _, err = m.Forward(ctx, nil, nil, p); !errors.Is(err, tensor.ErrClosed) {
			t.Fatalf("valid SUP plan rejected before input boundary: %v", err)
		}
		p.dimensions.EncoderT++
		if _, err = m.Forward(ctx, nil, nil, p); !errors.Is(err, ErrGPUPlanState) {
			t.Fatalf("mutated SUP plan admitted: %v", err)
		}
	}
	if _, err := (&SUPGPU{}).Forward(ctx, nil, nil, SUPGPUPlan{}); !errors.Is(err, ErrGPUPlanState) {
		t.Fatalf("zero SUP plan admitted: %v", err)
	}
}
