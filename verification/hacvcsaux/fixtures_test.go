package hacvcsaux

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpubatch"
)

func TestFrozenMatrixMatchesSourceBatchMembership(t *testing.T) {
	m, err := ReadMatrix()
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, f := range m.Cases {
		t.Run(f.ID, func(t *testing.T) {
			if seen[f.ID] {
				t.Fatalf("duplicate case %q", f.ID)
			}
			seen[f.ID] = true
			a, err := f.Aux(m.InputStride)
			if err != nil {
				t.Fatal(err)
			}
			cfg := gpubatch.Config{Lane: gpubatch.HAC, ChunkSize: m.InputStride * f.FullChunkSizeOut, BatchSize: f.BatchSize, Stride: m.InputStride, InnerStride: m.InputStride, Granularity: m.InputStride, VariableChunks: true}
			b, err := gpubatch.NewBatch(cfg)
			if err != nil {
				t.Fatal(err)
			}
			for _, n := range a.Sizes() {
				flushed, err := b.Add(gpubatch.Chunk{Length: n})
				if err != nil || flushed {
					t.Fatalf("fixture exceeds one source batch: flushed=%v error=%v", flushed, err)
				}
			}
			if b.Real() != len(a.SizesOut()) {
				t.Fatal("source membership changed")
			}
			packing, err := a.PackLSTM()
			if err != nil {
				t.Fatal(err)
			}
			if len(packing.InLayout) != a.Total() || len(packing.OutLayout) != f.BatchSize*a.TLSTM() || packing.OutLayoutCapacity != len(packing.BackwardEncoding) {
				t.Fatal("frozen fixture packing span differs")
			}
			if f.ID == "signed-id" {
				found := false
				for _, word := range packing.ForwardEncoding {
					found = found || word < 0
				}
				if !found {
					t.Fatal("signed root identifier was lost")
				}
			}
		})
	}
}

func TestMatrixPinRejectsChangedBytesBeforeOutput(t *testing.T) {
	original := fixtureJSON
	fixtureJSON = append(append([]byte(nil), original...), '\n')
	defer func() { fixtureJSON = original }()
	output := filepath.Join(t.TempDir(), "inputs.txt")
	if err := PrepareInputs(output); err == nil {
		t.Fatal("changed matrix pin accepted")
	}
	if _, err := os.Stat(output); !os.IsNotExist(err) {
		t.Fatalf("changed input pin reached output creation: %v", err)
	}
}

func TestPrepareInputsPreservesExistingArtifact(t *testing.T) {
	output := filepath.Join(t.TempDir(), "inputs.txt")
	if err := os.WriteFile(output, []byte("retained artifact\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := PrepareInputs(output); err == nil {
		t.Fatal("existing input artifact replaced")
	}
	body, err := os.ReadFile(output)
	if err != nil || string(body) != "retained artifact\n" {
		t.Fatal("existing input artifact changed", err)
	}
}

func TestFirstFixtureUploadGeometry(t *testing.T) {
	m, err := ReadMatrix()
	if err != nil {
		t.Fatal(err)
	}
	a, err := m.Cases[0].Aux(m.InputStride)
	if err != nil {
		t.Fatal(err)
	}
	sum, outCapacity, forward, backward, err := a.LSTMLayout()
	if err != nil || sum != 2432 || outCapacity != 131264 || forward != 131200 || backward != 131264 {
		t.Fatalf("unexpected first fixture geometry: %d %d %d %d %v", sum, outCapacity, forward, backward, err)
	}
	if outCapacity-forward != 64 {
		t.Fatal("OutLayout unwritten tail changed")
	}
}
