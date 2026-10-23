package model

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	modeloracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/model"
)

type supFixture struct {
	modeloracle.Fixture
	Heads, Hidden, Upper, Lower int
	MaxSequence                 int64 `json:"max_sequence"`
	Alpha                       float32
}

func supFixtures(directory string) ([]supFixture, error) {
	data, err := os.ReadFile(filepath.Join(directory, "fixtures.json"))
	if err != nil {
		return nil, err
	}
	var fixtures []supFixture
	err = json.Unmarshal(data, &fixtures)
	return fixtures, err
}

func supFixtureWeights(t *testing.T, e *tensor.Engine, f supFixture, directory string) []*tensor.Tensor {
	t.Helper()
	var weights []*tensor.Tensor
	for _, r := range f.Weights {
		w, err := e.LoadWeight(filepath.Join(directory, r.Path+".tensor"))
		if err != nil {
			t.Fatal(err)
		}
		if _, err = checkBits(w, r, directory); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if _, err := checkBits(w, r, directory); err != nil {
				t.Error("weight mutated", err)
			}
			if err := w.Close(); err != nil {
				t.Error(err)
			}
		})
		weights = append(weights, w)
	}
	return weights
}

func supFixtureRotary(t *testing.T, e *tensor.Engine, f supFixture) rotaryEmbedding {
	t.Helper()
	dim := int64(f.Channels / f.Heads)
	if f.Kind == "rope" {
		dim = int64(f.Channels)
	}
	r, err := newRotary(e, dim, f.MaxSequence, 10000)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.cos.Close(); r.sin.Close() })
	return r
}

func supLayers(f supFixture, weights []*tensor.Tensor) []txEncoderLayer {
	var layers []txEncoderLayer
	for i := 0; i < f.Layers; i++ {
		w := weights[7*i : 7*i+7]
		layers = append(layers, txEncoderLayer{w[0], w[1], w[2], w[3], w[4], w[5], w[6], int64(f.Channels), int64(f.Heads), int64(f.Hidden), int64(f.Upper), int64(f.Lower)})
	}
	return layers
}

func syntheticSUPGraph(t *testing.T, e *tensor.Engine, f supFixture, weights []*tensor.Tensor) *supGraph {
	t.Helper()
	g := &supGraph{rotary: supFixtureRotary(t, e, f)}
	for i, p := range [][2]int64{{1, 2}, {1, 2}, {3, 4}, {2, 4}, {2, 2}} {
		g.convs = append(g.convs, convLayer{weights[2*i], weights[2*i+1], p[0], p[1], silu})
	}
	g.encoders = supLayers(f, weights[10:])
	offset := 10 + 7*f.Layers
	g.upsampleWeight, g.upsampleBias = weights[offset], weights[offset+1]
	var err error
	g.crf, err = e.MulScalar(weights[offset+2], 5)
	if err != nil {
		t.Fatal(err)
	}
	g.alpha, err = e.NewFloat32(nil, []float32{f.Alpha})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { g.crf.Close(); g.alpha.Close() })
	return g
}

func TestNativeSUPParity(t *testing.T) {
	library, source, root := os.Getenv("NANOPORE_TENSOR_BACKEND"), os.Getenv("DORADO_SOURCE"), os.Getenv("SUP_MODEL_ARTIFACT_DIR")
	if library == "" || source == "" || root == "" {
		t.Skip("set NANOPORE_TENSOR_BACKEND, DORADO_SOURCE and SUP_MODEL_ARTIFACT_DIR for SUP CPU parity")
	}
	var err error
	library, err = filepath.Abs(library)
	if err != nil {
		t.Fatal(err)
	}
	source, err = filepath.Abs(source)
	if err != nil {
		t.Fatal(err)
	}
	root, err = filepath.Abs(root)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.MkdirAll(root, 0755); err != nil {
		t.Fatal(err)
	}
	attempt := filepath.Join(root, time.Now().UTC().Format("attempt-20060102T150405.000000000Z"))
	if err = os.Mkdir(attempt, 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OMP_NUM_THREADS", "1")
	t.Setenv("MKL_NUM_THREADS", "1")
	stats := struct {
		Status, NativeAttempt, Backend, Error                  string
		StockCapture, TorchNNLinearHeader, TorchNNLinearSHA256 string
		Fixtures, Stages, Floats                               int
		GoSources                                              map[string]string
	}{Status: "failed", Backend: library}
	stats.TorchNNLinearHeader, stats.TorchNNLinearSHA256, err = modeloracle.TorchNNLinearHeader(library)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if !t.Failed() {
			stats.Status = "passed"
		}
		stats.GoSources, err = modeloracle.GoSourceHashes()
		if err != nil {
			t.Error(err)
		}
		data, er := json.MarshalIndent(stats, "", "  ")
		if er == nil {
			er = os.WriteFile(filepath.Join(attempt, "parity.json"), append(data, '\n'), 0644)
		}
		if er != nil {
			t.Error(er)
		}
	}()
	nativeAttempt := os.Getenv("SUP_ORACLE_ATTEMPT")
	var exe string
	if nativeAttempt == "" {
		nativeAttempt = filepath.Join(attempt, "oracle")
		exe, err = modeloracle.BuildSUP(source, library, nativeAttempt)
	} else {
		nativeAttempt, err = filepath.Abs(nativeAttempt)
		if err == nil {
			exe, err = modeloracle.VerifySUPCapture(source, library, nativeAttempt)
		}
	}
	stats.NativeAttempt = nativeAttempt
	if err != nil {
		stats.Error = err.Error()
		t.Fatal(err)
	}
	e, err := tensor.Open(library)
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	checkFixture := func(t *testing.T, f supFixture, dir string, g *supGraph) {
		t.Helper()
		x, err := e.LoadWeight(filepath.Join(dir, f.Input.Path+".tensor"))
		if err != nil {
			t.Fatal(err)
		}
		defer x.Close()
		byName := map[string]modeloracle.Ref{}
		for _, r := range f.Stages {
			byName[r.Name] = r
		}
		seen := map[string]bool{}
		observe := func(name string, x *tensor.Tensor) error {
			r, ok := byName[name]
			if !ok {
				return fmt.Errorf("unexpected stage %s", name)
			}
			n, err := checkBits(x, r, dir)
			if err == nil {
				stats.Stages++
				stats.Floats += n
				seen[name] = true
			}
			return err
		}
		weights := supFixtureWeights(t, e, f, dir)
		var out *tensor.Tensor
		switch f.Kind {
		case "rope":
			r := supFixtureRotary(t, e, f)
			if err = observe("cos", r.cos); err != nil {
				t.Fatal(err)
			}
			if err = observe("sin", r.sin); err != nil {
				t.Fatal(err)
			}
			powers := make([]float32, f.Channels/2)
			for j := range powers {
				powers[j] = float32(math.Pow(10000, float64(float32(2*j)/float32(f.Channels))))
			}
			p, er := e.NewFloat32([]int64{int64(len(powers))}, powers)
			if er != nil {
				t.Fatal(er)
			}
			inv, er := e.Reciprocal(p)
			p.Close()
			if er != nil {
				t.Fatal(er)
			}
			err = observe("inv_freq", inv)
			inv.Close()
			if err != nil {
				t.Fatal(err)
			}
			out, err = r.forward(e, x)
			if err == nil {
				err = observe("rope", out)
			}
		case "rms":
			out, err = rmsNorm(e, x, weights[0])
			if err == nil {
				err = observe("rms", out)
			}
		case "mlp":
			out, err = gatedMLP(e, x, weights[0], weights[1], int64(f.Hidden), nil)
			if err == nil {
				err = observe("mlp", out)
			}
		case "upsample":
			out, err = linearUpsample(e, x, weights[0], weights[1], 2)
			if err == nil {
				err = observe("upsample", out)
			}
		case "scaled_crf":
			var scaled *tensor.Tensor
			scaled, err = e.MulScalar(weights[0], 5)
			if err == nil {
				err = observe("scaled_weight", scaled)
			}
			if err == nil {
				out, err = e.Linear(x, scaled, nil)
			}
			if err == nil {
				err = observe("scores", out)
			}
			if err == nil {
				var repeat *tensor.Tensor
				repeat, err = e.Linear(x, scaled, nil)
				if err == nil {
					err = observe("scores_repeat", repeat)
				}
				repeat.Close()
			}
			scaled.Close()
		case "encoder":
			r := supFixtureRotary(t, e, f)
			alpha, er := e.NewFloat32(nil, []float32{f.Alpha})
			if er != nil {
				t.Fatal(er)
			}
			defer alpha.Close()
			out = x
			for i, l := range supLayers(f, weights) {
				next, er := l.forward(context.Background(), e, out, r, alpha, func(name string, x *tensor.Tensor) error { return observe(fmt.Sprintf("encoder%d.%s", i, name), x) })
				if out != x {
					out.Close()
				}
				out = next
				if er != nil {
					err = er
					break
				}
			}
			if err == nil {
				err = observe("stack", out)
			}
		case "sup_graph":
			if g == nil {
				g = syntheticSUPGraph(t, e, f, weights)
			}
			out, err = g.forward(context.Background(), e, x, observe)
			if err == nil {
				var repeat *tensor.Tensor
				repeat, err = g.forward(context.Background(), e, x, nil)
				if err == nil {
					err = observe("scores_repeat", repeat)
				}
				repeat.Close()
			}
		default:
			t.Fatal("unknown fixture", f.Kind)
		}
		if out != nil {
			defer out.Close()
		}
		if err != nil {
			t.Fatal(err)
		}
		if len(seen) != len(byName) {
			t.Fatalf("checked %d stages want %d", len(seen), len(byName))
		}
		if _, err = checkBits(x, f.Input, dir); err != nil {
			t.Fatal("input mutated", err)
		}
		stats.Fixtures++
	}
	dir := filepath.Join(nativeAttempt, "synthetic")
	fixtures, err := supFixtures(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range fixtures {
		t.Run(f.Name, func(t *testing.T) { checkFixture(t, f, dir, nil) })
	}
	if t.Failed() {
		return
	}
	if modelDir := os.Getenv("SUP_MODEL_DIR"); modelDir != "" {
		t.Run("stock_sup_n1_l768", func(t *testing.T) {
			m, err := Open(e, modelDir)
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			if len(m.Weights()) != 139 {
				t.Fatal("wrong consumed weight count")
			}
			stock := os.Getenv("SUP_STOCK_CAPTURE")
			if stock == "" {
				stock = filepath.Join(attempt, "stock")
				err = modeloracle.RunStock(exe, modelDir, stock)
			} else {
				stock, err = filepath.Abs(stock)
				if err == nil {
					err = modeloracle.VerifyStockCapture(exe, modelDir, stock)
				}
			}
			stats.StockCapture = stock
			if err != nil {
				t.Fatal(err)
			}
			fs, err := supFixtures(stock)
			if err != nil || len(fs) != 1 {
				t.Fatalf("stock fixture %v", err)
			}
			checkFixture(t, fs[0], stock, m.graph.(*supGraph))
			data, err := json.MarshalIndent(m.Weights(), "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			if err = os.WriteFile(filepath.Join(attempt, "weight-receipts.json"), append(data, '\n'), 0644); err != nil {
				t.Fatal(err)
			}
			x, err := e.LoadWeight(filepath.Join(stock, fs[0].Input.Path+".tensor"))
			if err != nil {
				t.Fatal(err)
			}
			defer x.Close()
			y, err := m.Forward(context.Background(), x)
			if err != nil {
				t.Fatal(err)
			}
			defer y.Close()
			if _, err = checkBits(y, fs[0].Stages[len(fs[0].Stages)-1], stock); err != nil {
				t.Fatal(err)
			}
			if err = m.Close(); err != nil {
				t.Fatal(err)
			}
			if err = m.Close(); err != nil {
				t.Fatal(err)
			}
			if _, err = m.Forward(context.Background(), x); !errors.Is(err, ErrClosed) {
				t.Fatal(err)
			}
			if _, err = y.Data(); err != nil {
				t.Fatal("model closed output", err)
			}
			if _, err = x.Data(); err != nil {
				t.Fatal("model closed input", err)
			}
		})
	}
	t.Run("sup_rejections", func(t *testing.T) { testSUPRejections(t, e, library, source, fixtures, dir) })
	t.Logf("CPU13 SUP exact F32 parity %d fixtures %d stages %d floats; receipt %s", stats.Fixtures, stats.Stages, stats.Floats, attempt)
}

func testSUPRejections(t *testing.T, e *tensor.Engine, library, source string, fixtures []supFixture, dir string) {
	t.Helper()
	var f supFixture
	for _, candidate := range fixtures {
		if candidate.Kind == "sup_graph" {
			f = candidate
			break
		}
	}
	w := supFixtureWeights(t, e, f, dir)
	g := syntheticSUPGraph(t, e, f, w)
	x, err := e.LoadWeight(filepath.Join(dir, f.Input.Path+".tensor"))
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	ctx, cancel := context.WithCancel(context.Background())
	y, err := g.forward(ctx, e, x, func(name string, _ *tensor.Tensor) error {
		if name == "encoder0.norm1" {
			cancel()
		}
		return nil
	})
	if y != nil || !errors.Is(err, context.Canceled) {
		t.Fatal("midgraph cancellation", err)
	}
	y, err = g.forward(context.Background(), e, x, nil)
	if err != nil {
		t.Fatal("reuse after cancellation", err)
	}
	y.Close()
	r := g.rotary
	for _, shape := range [][]int64{{1, 4097, 3, 2, r.dim}, {1, 2, 2, 2, r.dim}, {1, 2, 3, 2, r.dim + 2}} {
		bad, err := e.Zeros(shape)
		if err != nil {
			t.Fatal(err)
		}
		if out, err := r.forward(e, bad); err == nil {
			out.Close()
			t.Fatal("invalid rotary accepted", shape)
		}
		bad.Close()
	}
	m := &Model{engine: e, spec: Spec{Architecture: SUP, Granularity: 768, InnerStride: 12}, graph: g}
	for _, shape := range [][]int64{{1, 1, 769}, {1, 1, 49920}, {1, 2, 768}, {0, 1, 768}} {
		bad, err := e.Zeros(shape)
		if err != nil {
			t.Fatal(err)
		}
		if out, err := m.Forward(context.Background(), bad); err == nil {
			out.Close()
			t.Fatal("invalid SUP input accepted", shape)
		}
		bad.Close()
	}
	other, err := tensor.Open(library)
	if err != nil {
		t.Fatal(err)
	}
	defer other.Close()
	foreign, err := other.Zeros([]int64{1, 1, 768})
	if err != nil {
		t.Fatal(err)
	}
	defer foreign.Close()
	if out, err := m.Forward(context.Background(), foreign); err == nil {
		out.Close()
		t.Fatal("foreign SUP input accepted")
	}
	data := make([]float32, 768)
	data[0] = float32(math.Inf(1))
	bad, err := e.NewFloat32([]int64{1, 1, 768}, data)
	if err != nil {
		t.Fatal(err)
	}
	defer bad.Close()
	if out, err := m.Forward(context.Background(), bad); err == nil {
		out.Close()
		t.Fatal("nonfinite SUP input accepted")
	}
	fake := t.TempDir()
	sources, err := modeloracle.ExtractSUP(source)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range sources {
		b, err := os.ReadFile(filepath.Join(source, s.Path))
		if err != nil {
			t.Fatal(err)
		}
		dest := filepath.Join(fake, s.Path)
		if err = os.MkdirAll(filepath.Dir(dest), 0755); err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(dest, b, 0644); err != nil {
			t.Fatal(err)
		}
	}
	if _, err = modeloracle.ExtractSUP(fake); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(fake, "dorado/nn/TxModules.cpp")
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(p, append(slices.Clone(b), '\n'), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err = modeloracle.ExtractSUP(fake); err == nil {
		t.Fatal("changed source accepted")
	}
}
