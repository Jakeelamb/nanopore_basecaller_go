package crf

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/decode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	crforacle "github.com/Jakeelamb/nanopore_basecaller_go/verification/crf"
)

func engine(t *testing.T) *tensor.Engine {
	t.Helper()
	path := os.Getenv("NANOPORE_TENSOR_BACKEND")
	if path == "" {
		t.Skip("native CRF gate skipped; set NANOPORE_TENSOR_BACKEND")
	}
	e, err := tensor.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := e.Close(); err != nil {
			t.Error(err)
		}
	})
	return e
}
func inputTensor(t *testing.T, e *tensor.Engine, shape []int64, data []float32) *tensor.Tensor {
	t.Helper()
	x, err := e.NewFloat32(shape, data)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := x.Close(); err != nil {
			t.Error(err)
		}
	})
	return x
}
func saveJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0644)
}
func compareBits(got []float32, path string) (int, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	if len(raw) != 4*len(got) {
		return 0, fmt.Errorf("byte length %d want %d", len(raw), 4*len(got))
	}
	for i, v := range got {
		expected := binary.LittleEndian.Uint32(raw[4*i:])
		if math.Float32bits(v) != expected {
			return i, fmt.Errorf("element %d bits %08x want %08x (%.9g vs %.9g)", i, math.Float32bits(v), expected, v, math.Float32frombits(expected))
		}
	}
	return len(got), nil
}
func saveFloats(path string, values []float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	writeErr := binary.Write(f, binary.LittleEndian, values)
	closeErr := f.Close()
	return errors.Join(writeErr, closeErr)
}
func sourceHashes(root string) (map[string]string, error) {
	pins := map[string]string{}
	for _, dir := range []string{"internal/crf", "verification/crf"} {
		err := filepath.WalkDir(filepath.Join(root, dir), func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			h, err := crforacle.Hash(path)
			if err != nil {
				return err
			}
			relative, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			pins[relative] = h
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return pins, nil
}
func TestNativeScanParity(t *testing.T) {
	if os.Getenv("CRF_ORACLE_MODE") != "native-produce" {
		t.Skip("native oracle production requires CRF_ORACLE_MODE=native-produce; use retained replay otherwise")
	}
	library, source, root := os.Getenv("NANOPORE_TENSOR_BACKEND"), os.Getenv("DORADO_SOURCE"), os.Getenv("CRF_ARTIFACT_DIR")
	if library == "" || source == "" || root == "" {
		t.Skip("source-linked CPU CRF parity skipped; set NANOPORE_TENSOR_BACKEND, DORADO_SOURCE and CRF_ARTIFACT_DIR")
	}
	for _, p := range []*string{&library, &source, &root} {
		v, err := filepath.Abs(*p)
		if err != nil {
			t.Fatal(err)
		}
		*p = v
	}
	if err := os.MkdirAll(root, 0755); err != nil {
		t.Fatal(err)
	}
	attempt := filepath.Join(root, time.Now().UTC().Format("attempt-20060102T150405.000000000Z"))
	if err := os.Mkdir(attempt, 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OMP_NUM_THREADS", "1")
	t.Setenv("MKL_NUM_THREADS", "1")
	stats := struct {
		Status                        string `json:"status"`
		Fixtures, Stages, Floats      int
		Head                          string
		Source, Inputs, GoOutputs     map[string]string
		FailedCase                    string `json:"failed_case,omitempty"`
		DecodeSequence, DecodeQuality string
	}{Status: "failed", GoOutputs: map[string]string{}}
	defer func() {
		if !t.Failed() {
			stats.Status = "passed"
		}
		if err := saveJSON(filepath.Join(attempt, "parity.json"), stats); err != nil {
			t.Error(err)
		}
	}()
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	head, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		t.Fatal(err)
	}
	stats.Head = strings.TrimSpace(string(head))
	stats.Source, err = sourceHashes(repo)
	if err != nil {
		t.Fatal(err)
	}
	captured, pins, err := crforacle.Captured(repo)
	if err != nil {
		t.Fatal(err)
	}
	stats.Inputs = pins
	fixtures := append(crforacle.Synthetic(), captured)
	exe, err := crforacle.Build(source, library, filepath.Join(attempt, "oracle"))
	if err != nil {
		t.Fatal(err)
	}
	native := filepath.Join(attempt, "native")
	if err := crforacle.Run(exe, native, fixtures); err != nil {
		t.Fatal(err)
	}
	e := engine(t)
	for i, c := range fixtures {
		t.Run(c.Name, func(t *testing.T) {
			x := inputTensor(t, e, []int64{int64(c.Blocks), 1, int64(4 * c.States)}, c.Scores)
			got, err := Compute(context.Background(), e, x, c.Blank)
			if err != nil {
				stats.FailedCase = c.Name
				t.Fatal(err)
			}
			dir := filepath.Join(attempt, "go", fmt.Sprint(i))
			if err := os.MkdirAll(dir, 0755); err != nil {
				t.Fatal(err)
			}
			for _, stage := range []struct {
				name string
				data []float32
			}{{"scores", got.Scores}, {"forward", got.Forward}, {"backward", got.Backward}, {"posts", got.Posts}} {
				n, err := compareBits(stage.data, filepath.Join(native, fmt.Sprint(i), stage.name+".bin"))
				if err != nil {
					stats.FailedCase = c.Name + "/" + stage.name
					t.Fatal(err)
				}
				stats.Stages++
				stats.Floats += n
				path := filepath.Join(dir, stage.name+".bin")
				if err := saveFloats(path, stage.data); err != nil {
					t.Fatal(err)
				}
				stats.GoOutputs[path], err = crforacle.Hash(path)
				if err != nil {
					t.Fatal(err)
				}
			}
			after, err := x.Data()
			if err != nil {
				t.Fatal(err)
			}
			for j, v := range after {
				if math.Float32bits(v) != math.Float32bits(c.Scores[j]) {
					t.Fatalf("borrowed input mutated at %d", j)
				}
			}
			if c.Name == captured.Name {
				decoded, err := decode.Decode(got.DecodeInput(), decode.Options{BeamWidth: 32, BeamCut: 100, BlankScore: c.Blank, QShift: 0, QScale: 1})
				if err != nil {
					t.Fatal(err)
				}
				stats.DecodeSequence = decoded.Sequence
				stats.DecodeQuality = decoded.Qstring
				if len(decoded.Moves) != c.Blocks {
					t.Fatal("decoder handoff lost blocks")
				}
			}
			stats.Fixtures++
		})
	}
	t.Run("source_mutation", func(t *testing.T) {
		mutated := filepath.Join(attempt, "mutated-source")
		for _, path := range []string{crforacle.SourcePath, "dorado/torch_utils/torch_utils.cpp"} {
			data, err := os.ReadFile(filepath.Join(source, path))
			if err != nil {
				t.Fatal(err)
			}
			if path == crforacle.SourcePath {
				data[len(data)/2] ^= 1
			}
			target := filepath.Join(mutated, path)
			if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(target, data, 0644); err != nil {
				t.Fatal(err)
			}
		}
		if _, err := crforacle.Build(mutated, library, filepath.Join(attempt, "rejected-mutation")); err == nil {
			t.Fatal("mutated source reached compilation")
		}
	})
	t.Logf("%d fixtures, %d stages, %d exact float32 comparisons; artifacts %s", stats.Fixtures, stats.Stages, stats.Floats, attempt)
}

type cancelAfter struct {
	context.Context
	cancel    context.CancelFunc
	remaining atomic.Int64
}

func (c *cancelAfter) Err() error {
	if c.remaining.Add(-1) == 0 {
		c.cancel()
	}
	return c.Context.Err()
}
func TestNativeValidationAndCancellation(t *testing.T) {
	e := engine(t)
	valid := inputTensor(t, e, []int64{9, 1, 16}, make([]float32, 9*16))
	for _, tt := range []struct {
		name  string
		x     *tensor.Tensor
		blank float32
	}{
		{"batch", inputTensor(t, e, []int64{1, 2, 16}, make([]float32, 32)), 2},
		{"rank", inputTensor(t, e, []int64{1, 16}, make([]float32, 16)), 2},
		{"state_count", inputTensor(t, e, []int64{1, 1, 32}, make([]float32, 32)), 2},
		{"nan", inputTensor(t, e, []int64{1, 1, 16}, append([]float32{float32(math.NaN())}, make([]float32, 15)...)), 2},
		{"infinite", inputTensor(t, e, []int64{1, 1, 16}, append([]float32{float32(math.Inf(1))}, make([]float32, 15)...)), 2},
		{"unsafe", inputTensor(t, e, []int64{1, 1, 16}, append([]float32{math.MaxFloat32}, make([]float32, 15)...)), 2},
		{"blank_nan", valid, float32(math.NaN())},
		{"blank_unsafe", valid, math.MaxFloat32},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Compute(context.Background(), e, tt.x, tt.blank)
			if err == nil || !reflect.DeepEqual(got, Result{}) {
				t.Fatalf("invalid input accepted or partial result: %v %v", got, err)
			}
		})
	}
	other := engine(t)
	foreign := inputTensor(t, other, []int64{1, 1, 16}, make([]float32, 16))
	if _, err := Compute(context.Background(), e, foreign, 2); err == nil {
		t.Fatal("foreign input accepted")
	}
	closed := inputTensor(t, e, []int64{1, 1, 16}, make([]float32, 16))
	if err := closed.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := Compute(context.Background(), e, closed, 2); !errors.Is(err, tensor.ErrClosed) {
		t.Fatalf("closed input: %v", err)
	}
	if _, err := Compute(context.Background(), nil, valid, 2); !errors.Is(err, tensor.ErrClosed) {
		t.Fatalf("nil engine: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	mid := &cancelAfter{Context: ctx, cancel: cancel}
	mid.remaining.Store(5)
	got, err := Compute(mid, e, valid, 2)
	if !errors.Is(err, context.Canceled) || !reflect.DeepEqual(got, Result{}) {
		t.Fatalf("mid-scan cancellation: %v %v", got, err)
	}
	if _, err := Compute(context.Background(), e, valid, 2); err != nil {
		t.Fatalf("engine unusable after rejection/cancellation: %v", err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := Compute(context.Background(), e, valid, 2); !errors.Is(err, tensor.ErrClosed) {
		t.Fatalf("closed engine: %v", err)
	}
}

// TestRetainedBatchParity compares lane composition with previously captured
// original-source N=1 outputs. No new native batch capture occurs here.
func TestRetainedBatchParity(t *testing.T) {
	if os.Getenv("CRF_ORACLE_MODE") != "replay" {
		t.Skip("retained replay requires CRF_ORACLE_MODE=replay")
	}
	source, library, retained, root := os.Getenv("DORADO_SOURCE"), os.Getenv("NANOPORE_TENSOR_BACKEND"), os.Getenv("CRF_RETAINED_ATTEMPT"), os.Getenv("CRF_ARTIFACT_DIR")
	if source == "" || library == "" || retained == "" || root == "" {
		t.Fatal("replay requires DORADO_SOURCE, NANOPORE_TENSOR_BACKEND, CRF_RETAINED_ATTEMPT, CRF_ARTIFACT_DIR")
	}
	if err := os.MkdirAll(root, 0755); err != nil {
		t.Fatal(err)
	}
	attempt := filepath.Join(root, time.Now().UTC().Format("replay-20060102T150405.000000000Z"))
	if err := os.Mkdir(attempt, 0755); err != nil {
		t.Fatal(err)
	}
	stats := struct {
		Status                                   string
		Qualification                            string
		Fixtures, Batches, Lanes, Stages, Floats int
		Inputs, Source                           map[string]string
	}{Status: "failed", Qualification: "retained original-source N=1 evidence reused for N=1..4 repeated-lane composition; no new native capture"}
	defer func() {
		if !t.Failed() {
			stats.Status = "passed"
		}
		if err := saveJSON(filepath.Join(attempt, "replay.json"), stats); err != nil {
			t.Error(err)
		}
	}()
	fixtures, pins, err := crforacle.Replay(source, library, retained)
	if err != nil {
		t.Fatal(err)
	}
	stats.Inputs = pins
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	stats.Source, err = sourceHashes(repo)
	if err != nil {
		t.Fatal(err)
	}
	e := engine(t)
	for _, f := range fixtures {
		t.Run(f.Name, func(t *testing.T) {
			for n := 1; n <= 4; n++ {
				data := make([]float32, len(f.Scores)*n)
				channels := 4 * f.States
				for time := 0; time < f.Blocks; time++ {
					for lane := 0; lane < n; lane++ {
						copy(data[(time*n+lane)*channels:], f.Scores[time*channels:(time+1)*channels])
					}
				}
				x, err := e.NewFloat32([]int64{int64(f.Blocks), int64(n), int64(channels)}, data)
				if err != nil {
					t.Fatal(err)
				}
				results, err := ComputeBatch(context.Background(), e, x, f.Blank)
				closeErr := x.Close()
				if err := errors.Join(err, closeErr); err != nil {
					t.Fatal(err)
				}
				if len(results) != n {
					t.Fatalf("N=%d got %d results", n, len(results))
				}
				want := Result{Blocks: f.Blocks, States: f.States, Scores: f.Expected["scores"], Forward: f.Expected["forward"], Backward: f.Expected["backward"], Posts: f.Expected["posts"]}
				for _, got := range results {
					assertResultBits(t, got, want)
					stats.Lanes++
					stats.Stages += 4
					stats.Floats += len(got.Scores) + len(got.Forward) + len(got.Backward) + len(got.Posts)
				}
				stats.Batches++
			}
			stats.Fixtures++
		})
	}
}
