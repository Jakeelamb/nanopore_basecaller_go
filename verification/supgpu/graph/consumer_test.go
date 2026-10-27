package graph

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
)

func TestSUPGraphRuntimeHoldPrecedesAnyPathOrLoader(t *testing.T) {
	o := Options{Capture: supgpu.Pin{Path: "/must/not/open"}, Backend: "/must/not/dlopen", Model: "/must/not/load", Out: filepath.Join(t.TempDir(), "must-not-create")}
	r, err := Consume(context.Background(), o)
	if !errors.Is(err, ErrRuntimeHeld) || r.Status != "structural_hold" {
		t.Fatalf("runtime escaped hold: %+v %v", r, err)
	}
	if _, err = os.Stat(o.Out); !os.IsNotExist(err) {
		t.Fatal("held runtime wrote output")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = Consume(ctx, o); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
func TestSUPGraphDifferentialRejectsDataBackingDescriptorsAndOrder(t *testing.T) {
	dir := t.TempDir()
	data := []byte{0, 0x3c, 0, 0x40}
	file := filepath.Join(dir, "a.bin")
	if err := os.WriteFile(file, data, 0600); err != nil {
		t.Fatal(err)
	}
	pin := supgpu.Pin{Path: "a.bin", Bytes: 4, SHA256: digest(data)}
	want := supgpu.Event{Name: "scores", Layer: -1, Tensors: []supgpu.Tensor{{Name: "output", DType: "Half", Shape: []int64{2}, Strides: []int64{1}, Tags: []int{}, Data: pin, Backing: pin}}}
	got := model.SUPGPUObservation{Name: "scores", Layer: -1, Tensors: []model.SUPGPUBufferSnapshot{{Name: "output", DType: tensor.DTypeF16, Shape: []int64{2}, Strides: []int64{1}, Data: append([]byte(nil), data...)}}}
	comparisons, err := CompareObservation(dir, want, got)
	if err != nil || len(comparisons) != 1 || !comparisons[0].Equal {
		t.Fatal(comparisons, err)
	}
	got.Origin = model.SUPGPUPreparedCache
	if _, err = CompareObservation(dir, want, got); err == nil {
		t.Fatal("prepared host bytes mislabeled as uploaded scores")
	}
	got.Origin = model.SUPGPUUploaded
	got.Tensors[0].Data[0] = 1
	if _, err = CompareObservation(dir, want, got); err == nil {
		t.Fatal("different numerical bytes accepted")
	}
	got.Tensors[0].Data[0] = 0
	got.Tensors[0].Strides[0] = 2
	if _, err = CompareObservation(dir, want, got); err == nil {
		t.Fatal("wrong layout accepted")
	}
	got.Tensors[0].Strides[0] = 1
	got.Name = "upsample"
	if _, err = CompareObservation(dir, want, got); err == nil {
		t.Fatal("future stage accepted")
	}
	got.Name = "scores"
	if err = os.WriteFile(filepath.Join(dir, "back.bin"), []byte{0, 0x3c, 0, 0x41}, 0600); err != nil {
		t.Fatal(err)
	}
	want.Tensors[0].Backing = supgpu.Pin{Path: "back.bin", Bytes: 4, SHA256: digest([]byte{0, 0x3c, 0, 0x41})}
	if _, err = CompareObservation(dir, want, got); err == nil {
		t.Fatal("different full backing accepted")
	}
}
