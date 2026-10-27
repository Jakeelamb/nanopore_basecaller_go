package h05

import (
	"context"
	"fmt"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
)

func TestUnadmittedRunDoesNotReadInputs(t *testing.T) {
	r, err := Run(context.Background(), Options{Capture: "missing", Caches: "missing", Backend: "missing", Out: "missing"})
	if err == nil || !strings.Contains(err.Error(), "explicit coordinator admission") || r.Status != "failed" {
		t.Fatalf("unadmitted result %+v, %v", r, err)
	}
}
func TestStageComparisonUsesProspectiveRowsAndPins(t *testing.T) {
	dir := t.TempDir()
	row := 64 * 1024
	payload := make([]byte, 37*row)
	stage := hacgpu.Stage{Name: "flstm0", DataFile: "stage.bin", DataSHA: sum(payload), UnwrittenT: [][2]int{{1, 1}, {35, 2}}}
	c := hacgpu.Capture{Batch: 64, TOut: 32, Stages: []hacgpu.Stage{stage}}
	if err := os.WriteFile(filepath.Join(dir, "stage.bin"), payload, 0600); err != nil {
		t.Fatal(err)
	}
	observed := append([]byte(nil), payload...)
	observed[row] = 99
	observed[35*row] = 99
	rec, err := compare(prepared{capture: c}, dir, "flstm0", observed)
	if err != nil || !rec.Equal || rec.ComparedBytes != 34*row {
		t.Fatalf("excluded rows compared: %+v %v", rec, err)
	}
	observed[2*row] = 99
	rec, err = compare(prepared{capture: c}, dir, "flstm0", observed)
	if err != nil || rec.Equal {
		t.Fatalf("written row ignored %+v %v", rec, err)
	}
	payload[0] = 1
	if err := os.WriteFile(filepath.Join(dir, "stage.bin"), payload, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := compare(prepared{capture: c}, dir, "flstm0", observed); err == nil {
		t.Fatal("post-import mutation accepted")
	}
}

func TestNativeArgumentGuardUsesBoundaryCUDAError(t *testing.T) {
	actual := &tensor.CUDAError{Op: "factorised_lstm", Subsystem: "boundary", Reason: nativeScalarGuardReason}
	if !isNativeScalarGuard(actual) || !isNativeScalarGuard(fmt.Errorf("probe: %w", actual)) {
		t.Fatal("ABI status1 boundary error rejected")
	}
	for _, err := range []error{nil, &tensor.DeviceError{Op: "factorised_lstm", Reason: nativeScalarGuardReason}, &tensor.CUDAError{Op: "factorised_lstm", Subsystem: "cuda", Reason: nativeScalarGuardReason}, &tensor.CUDAError{Op: "factorised_lstm", Subsystem: "boundary", Reason: "another boundary failure"}, &tensor.CUDAError{Op: "factorised_lstm", Subsystem: "boundary", Reason: nativeScalarGuardReason, Detail: tensor.CUDADetail{Stage: 4}}, &tensor.CUDAError{Op: "another operation", Subsystem: "boundary", Reason: nativeScalarGuardReason}} {
		if isNativeScalarGuard(err) {
			t.Fatalf("unrelated error accepted: %v", err)
		}
	}
}
