package model

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	modeloracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/model"
)

func checkBits(x *tensor.Tensor, r modeloracle.Ref, root string) (int, error) {
	shape, err := x.Shape()
	if err != nil {
		return 0, err
	}
	if !slices.Equal(shape, r.Shape) {
		return 0, fmt.Errorf("%s shape %v want %v", r.Name, shape, r.Shape)
	}
	data, err := x.Data()
	if err != nil {
		return 0, err
	}
	raw, err := os.ReadFile(filepath.Join(root, r.Path+".bin"))
	if err != nil {
		return 0, err
	}
	if len(raw) != 4*len(data) {
		return 0, fmt.Errorf("%s byte count %d want %d", r.Name, len(raw), 4*len(data))
	}
	for i, v := range data {
		got, want := math.Float32bits(v), binary.LittleEndian.Uint32(raw[4*i:])
		if got != want {
			return i, fmt.Errorf("%s element %d bits %08x want %08x (%.9g vs %.9g)", r.Name, i, got, want, v, math.Float32frombits(want))
		}
	}
	return len(data), nil
}

func fixtureGraph(t *testing.T, e *tensor.Engine, f modeloracle.Fixture, root string) *hacGraph {
	t.Helper()
	var weights []*tensor.Tensor
	for _, r := range f.Weights {
		w, err := e.LoadWeight(filepath.Join(root, r.Path+".tensor"))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if _, err := checkBits(w, r, root); err != nil {
				t.Errorf("weight mutated: %v", err)
			}
			if err := w.Close(); err != nil {
				t.Error(err)
			}
		})
		shape, err := w.Shape()
		if err != nil || !slices.Equal(shape, r.Shape) {
			t.Fatalf("fixture weight shape %v %v", shape, err)
		}
		weights = append(weights, w)
	}
	g := &hacGraph{firstReverse: f.Reverse}
	offset := 0
	if f.Kind == "graph" {
		for i := 0; i < 3; i++ {
			stride, pad, a := int64(1), int64(2), silu
			if i == 2 {
				stride, pad, a = 6, 9, tanh
			}
			g.convs = append(g.convs, convLayer{weights[2*i], weights[2*i+1], stride, pad, a})
		}
		offset = 6
	}
	for i := 0; i < f.Layers; i++ {
		w := weights[offset+6*i : offset+6*i+6]
		g.rnns = append(g.rnns, flstmLayer{w[0], w[1], w[2], w[3], w[4], w[5], int64(f.Channels)})
	}
	if f.Kind == "graph" {
		g.bottleneck, g.crf = weights[len(weights)-2], weights[len(weights)-1]
	}
	return g
}

func TestNativeModelParity(t *testing.T) {
	library, source, root := os.Getenv("NANOPORE_TENSOR_BACKEND"), os.Getenv("DORADO_SOURCE"), os.Getenv("MODEL_ARTIFACT_DIR")
	if library == "" || source == "" || root == "" {
		t.Skip("CPU model parity gate skipped; set NANOPORE_TENSOR_BACKEND, DORADO_SOURCE and MODEL_ARTIFACT_DIR")
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
	// Both processes use one CPU thread so the same ATen algorithms reduce in the
	// same order. This setting is part of the receipt, not a performance claim.
	t.Setenv("OMP_NUM_THREADS", "1")
	t.Setenv("MKL_NUM_THREADS", "1")
	exe, err := modeloracle.Build(source, library, attempt)
	if err != nil {
		t.Fatal(err)
	}
	e, err := tensor.Open(library)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := e.Close(); err != nil {
			t.Error(err)
		}
	})
	stats := struct {
		Status                   string
		Fixtures, Stages, Floats int
		Backend, Threads         string
	}{"failed", 0, 0, 0, library, "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"}
	defer func() {
		if !t.Failed() {
			stats.Status = "passed"
		}
		data, er := json.MarshalIndent(stats, "", "  ")
		if er == nil {
			er = os.WriteFile(filepath.Join(attempt, "parity.json"), append(data, '\n'), 0644)
		}
		if er != nil {
			t.Error(er)
		}
	}()
	checkFixture := func(t *testing.T, f modeloracle.Fixture, dir string, g *hacGraph) {
		x, err := e.LoadWeight(filepath.Join(dir, f.Input.Path+".tensor"))
		if err != nil {
			t.Fatal(err)
		}
		defer x.Close()
		if _, err = checkBits(x, f.Input, dir); err != nil {
			t.Fatal(err)
		}
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
				stats.Floats += n
				stats.Stages++
				seen[name] = true
			}
			return err
		}
		var y *tensor.Tensor
		if f.Kind == "rnn" {
			y, err = flstmStackForward(context.Background(), e, g.rnns, g.firstReverse, x, observe)
		} else {
			y, err = g.forward(context.Background(), e, x, observe)
		}
		if err != nil {
			t.Fatal(err)
		}
		defer y.Close()
		if len(seen) != len(byName) {
			t.Fatalf("stages seen %d want %d", len(seen), len(byName))
		}
		if _, err = checkBits(x, f.Input, dir); err != nil {
			t.Fatalf("input mutated: %v", err)
		}
		stats.Fixtures++
	}
	dir := filepath.Join(attempt, "synthetic")
	fixtures, err := modeloracle.Fixtures(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range fixtures {
		t.Run(f.Name, func(t *testing.T) { g := fixtureGraph(t, e, f, dir); checkFixture(t, f, dir, g) })
	}
	if t.Failed() {
		return
	}
	modelDir := filepath.Join(filepath.Dir(filepath.Dir(source)), "models", "dna_r10.4.1_e8.2_400bps_hac@v6.0.0")
	t.Run("stock_hac_n1_l96", func(t *testing.T) {
		m, err := Open(e, modelDir)
		if err != nil {
			t.Fatal(err)
		}
		defer m.Close()
		if len(m.Weights()) != 50 {
			t.Fatal("missing receipts")
		}
		r := m.Weights()
		r[0].Shape[0] = 999
		if m.Weights()[0].Shape[0] != 16 {
			t.Fatal("receipt shape aliases internal storage")
		}
		stock := filepath.Join(attempt, "stock")
		if err = modeloracle.RunStock(exe, modelDir, stock); err != nil {
			t.Fatal(err)
		}
		fs, err := modeloracle.Fixtures(stock)
		if err != nil || len(fs) != 1 {
			t.Fatalf("stock fixture %v", err)
		}
		checkFixture(t, fs[0], stock, m.graph.(*hacGraph))
		b, err := json.MarshalIndent(m.Weights(), "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(stock, "weight-receipts.json"), append(b, '\n'), 0644); err != nil {
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
			t.Fatalf("closed model: %v", err)
		}
		if _, err = x.Data(); err != nil {
			t.Fatalf("model closed caller input: %v", err)
		}
		if _, err = y.Data(); err != nil {
			t.Fatalf("model closed caller result: %v", err)
		}
		if err = y.Close(); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("rejections", func(t *testing.T) { testRejections(t, e, library, source, modelDir) })
	t.Logf("CPU13 exact F32 parity: %d fixtures, %d stages, %d floats; receipts %s", stats.Fixtures, stats.Stages, stats.Floats, attempt)
}

func testRejections(t *testing.T, e *tensor.Engine, library, source, modelDir string) {
	t.Helper()
	// Test the loader's dtype and shape seam independently of stock checksum rejection.
	for _, tc := range []struct {
		name  string
		shape []int64
	}{{"int64.tensor", []int64{2, 3}}, {"multiple.tensor", []int64{2, 3}}, {"empty.tensor", []int64{2, 3}}, {"float32.tensor", []int64{999}}} {
		p := filepath.Join(filepath.Dir(library), tc.name)
		h, err := fileSHA(p)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = loadWeight(e, p, WeightReceipt{tc.name, h, tc.shape}); err == nil {
			t.Fatalf("accepted %s", tc.name)
		}
	}
	config, err := os.ReadFile(filepath.Join(modelDir, "config.toml"))
	if err != nil {
		t.Fatal(err)
	}
	d := t.TempDir()
	if err = os.WriteFile(filepath.Join(d, "config.toml"), append(slices.Clone(config), '\n'), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err = Open(e, d); err == nil {
		t.Fatal("mutated config accepted")
	}
	if err = os.WriteFile(filepath.Join(d, "config.toml"), config, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err = Open(e, d); err == nil {
		t.Fatal("missing weights accepted")
	}
	var rs []WeightReceipt
	if err = json.Unmarshal(hacWeightManifest, &rs); err != nil {
		t.Fatal(err)
	}
	for _, r := range rs {
		if err = os.Symlink(filepath.Join(modelDir, r.Name), filepath.Join(d, r.Name)); err != nil {
			t.Fatal(err)
		}
	}
	p := filepath.Join(d, rs[7].Name)
	if err = os.Remove(p); err != nil {
		t.Fatal(err)
	}
	if _, err = Open(e, d); err == nil {
		t.Fatal("partially missing weights accepted")
	}
	raw, err := os.ReadFile(filepath.Join(modelDir, rs[7].Name))
	if err != nil {
		t.Fatal(err)
	}
	raw[len(raw)-1] ^= 1
	if err = os.WriteFile(p, raw, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err = Open(e, d); err == nil || !strings.Contains(err.Error(), "SHA256 mismatch") {
		t.Fatalf("mutated weight: %v", err)
	}
	m, err := Open(e, modelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	x, err := e.Zeros([]int64{1, 1, 6})
	if err != nil {
		t.Fatal(err)
	}
	defer x.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = m.Forward(ctx, x); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel: %v", err)
	}
	ctx, cancel = context.WithCancel(context.Background())
	y, err := m.graph.forward(ctx, e, x, func(name string, _ *tensor.Tensor) error {
		if name == "rnn0" {
			cancel()
		}
		return nil
	})
	if y != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("midgraph cancellation: %v %v", y, err)
	}
	if _, err = x.Data(); err != nil {
		t.Fatalf("cancellation closed input: %v", err)
	}
	y, err = m.Forward(context.Background(), x)
	if err != nil {
		t.Fatalf("forward after cancellation: %v", err)
	}
	y.Close()
	for _, shape := range [][]int64{{1, 6}, {1, 2, 6}, {0, 1, 6}, {1, 1, 7}} {
		bad, err := e.Zeros(shape)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = m.Forward(context.Background(), bad); err == nil {
			t.Fatalf("accepted shape %v", shape)
		}
		bad.Close()
	}
	bad, err := e.NewFloat32([]int64{1, 1, 6}, []float32{0, 0, 0, 0, 0, float32(math.NaN())})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = m.Forward(context.Background(), bad); err == nil {
		t.Fatal("accepted nonfinite input")
	}
	bad.Close()
	other, err := tensor.Open(library)
	if err != nil {
		t.Fatal(err)
	}
	foreign, err := other.Zeros([]int64{1, 1, 6})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = m.Forward(context.Background(), foreign); err == nil {
		t.Fatal("accepted foreign input")
	}
	other.Close()
	x.Close()
	if _, err = m.Forward(context.Background(), x); err == nil {
		t.Fatal("accepted closed input")
	}
	supDir := filepath.Join(filepath.Dir(modelDir), "dna_r10.4.1_e8.2_400bps_sup@v5.2.0")
	spec, err := Inspect(supDir)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Architecture != SUP {
		t.Fatal(spec)
	}
	// A source-byte change invalidates extraction before any C++ compilation.
	fake := t.TempDir()
	for _, p := range []string{"dorado/nn/FLSTMStack.cpp", "dorado/nn/ConvStack.cpp", "dorado/nn/CRFModules.cpp", "dorado/config/include/config/common.h", "dorado/torch_utils/torch_utils.cpp", "dorado/basecall/model/CRFModel.cpp"} {
		b, err := os.ReadFile(filepath.Join(source, p))
		if err != nil {
			t.Fatal(err)
		}
		dest := filepath.Join(fake, p)
		if err = os.MkdirAll(filepath.Dir(dest), 0755); err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(dest, b, 0644); err != nil {
			t.Fatal(err)
		}
	}
	if _, err = modeloracle.Extract(fake); err != nil {
		t.Fatalf("unchanged source fixture: %v", err)
	}
	p = filepath.Join(fake, "dorado/nn/FLSTMStack.cpp")
	f, err := os.OpenFile(p, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, err = f.WriteString("\n")
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if _, err = modeloracle.Extract(fake); err == nil {
		t.Fatal("mutated source accepted")
	}
}
