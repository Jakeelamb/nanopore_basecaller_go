package gpudecode

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
)

func TestPosteriorPTXPreservesSelectedNumericInstructionOrder(t *testing.T) {
	for _, tc := range []struct {
		file    string
		ordered []string
	}{
		{"forward_ptx.h", []string{"shr.u32 %r4,%r2,2;", "shr.u32 %r5,%r0,2;", "ld.global.nc.u16 %h,[%a6];", "ld.global.nc.u16 %h,[%a6+2];", "ld.global.nc.u16 %h,[%a6+4];", "ld.global.nc.u16 %h,[%a6+6];", "max.ftz.f32 %f14,%f4,%f10;", "lg2.approx.ftz.f32 %f17,%f16;", "fma.rn.ftz.f32 %f18,%f17,0f3F317218,%f14;"}},
		{"posterior_ptx.h", []string{"mov.f32 %f3,0f00000000;", "posterior_max:", "add.u32 %r6,%r6,32;", "mov.u32 %r8,16;", "shfl.sync.bfly.b32", "shr.u32 %r8,%r8,1;", "mov.f32 %f5,0f00000000;", "posterior_sum:", "add.ftz.f32 %f5,%f5,%f4;", "posterior_sum_shuffle:", "lg2.approx.ftz.f32 %f6,%f5;", "fma.rn.ftz.f32 %f7,%f6,0f3F317218,%f3;", "sub.ftz.f32 %f8,%f2,%f7;", "ex2.approx.ftz.f32 %f8,%f8;", "cvt.rn.f16.f32 %h,%f8;"}},
		{"quality_ptx.h", []string{"group_sum:", "add.ftz.f32 %f0,%f0,%f1;", "cvt.rn.f32.s32 %f2,%r6;", "div.approx.ftz.f32 %f3,%f0,%f2;", "sub.ftz.f32 %f4,0f3F800000,%f3;", "lg2.approx.ftz.f32 %f5,%f4;", "mul.ftz.f32 %f6,%f5,0fBE9A209B;", "mul.ftz.f32 %f7,%f6,0f41200000;", "fma.rn.ftz.f32 %f8,%f20,%f7,%f21;", "min.ftz.f32 %f9,%f8,0f42480000;", "max.ftz.f32 %f10,%f9,0f3F800000;", "add.ftz.f32 %f11,%f10,0f42060000;", "cvt.rzi.ftz.s32.f32 %r8,%f11;"}},
	} {
		t.Run(tc.file, func(t *testing.T) {
			data, e := os.ReadFile("../../verification/tensor/decoder/" + tc.file)
			if e != nil {
				t.Fatal(e)
			}
			remaining := string(data)
			for _, want := range tc.ordered {
				_, tail, ok := strings.Cut(remaining, want)
				if !ok {
					t.Fatalf("missing/out-of-order selected instruction %q", want)
				}
				remaining = tail
			}
		})
	}
}

func TestPosteriorScheduleCancelsAfterFinalStore(t *testing.T) {
	p, _ := PlanGuides(64, 2, 256, 0, 2)
	ctx, cancel := context.WithCancel(context.Background())
	e := posteriorSchedule(ctx, p, func(int64, int, int) error { return nil }, func(t int64, _ int) error {
		if t == 2 {
			cancel()
		}
		return nil
	})
	if !errors.Is(e, context.Canceled) {
		t.Fatal("final cancellation escaped", e)
	}
}
