package model

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	modeloracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/model"
)

// TestRetainedHACModelScores consumes the admitted capture without regenerating
// its native producer or the 55-fixture HAC suite.
func TestRetainedHACModelScores(t *testing.T) {
	library, directory, capture := os.Getenv("NANOPORE_TENSOR_BACKEND"), os.Getenv("HAC_MODEL_DIR"), os.Getenv("HAC_STOCK_CAPTURE")
	if library == "" || directory == "" || capture == "" {
		t.Skip("set NANOPORE_TENSOR_BACKEND, HAC_MODEL_DIR and HAC_STOCK_CAPTURE for retained HAC scores")
	}
	const scoreSHA = "36229b983d66c9db8db986ea968f4fc679f81e6da8cf9bff5f75178128c35657"
	fixtures, err := modeloracle.Fixtures(capture)
	if err != nil || len(fixtures) != 1 {
		t.Fatalf("HAC fixture %v", err)
	}
	f := fixtures[0]
	if f.Name != "stock_hac_n1_l96" {
		t.Fatal("wrong HAC capture")
	}
	scores := f.Stages[len(f.Stages)-1]
	h, err := modeloracle.Hash(filepath.Join(capture, scores.Path+".bin"))
	if err != nil || h != scoreSHA {
		t.Fatalf("HAC score pin %s %v", h, err)
	}
	e, err := tensor.Open(library)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	m, err := Open(e, directory)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	x, err := e.LoadWeight(filepath.Join(capture, f.Input.Path+".tensor"))
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	if _, err = checkBits(x, f.Input, capture); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	out, err := m.graph.forward(ctx, e, x, func(name string, _ *tensor.Tensor) error {
		if name == "rnn0" {
			cancel()
		}
		return nil
	})
	if out != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("midgraph HAC cancellation", err)
	}
	out, err = m.Forward(context.Background(), x)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	if n, err := checkBits(out, scores, capture); err != nil || n != 16384 {
		t.Fatalf("HAC score replay %d %v", n, err)
	}
	if err = m.Close(); err != nil {
		t.Fatal(err)
	}
	if err = m.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = out.Data(); err != nil {
		t.Fatal("model closed caller scores", err)
	}
	if _, err = x.Data(); err != nil {
		t.Fatal("model closed caller input", err)
	}
	t.Log("retained HAC public Model.Forward matched all 16384 F32 score bits on current backend")
}
