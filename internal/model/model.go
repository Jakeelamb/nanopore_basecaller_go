// Package model owns the source-pinned model graph and its immutable weights.
// Tensor arithmetic is provided by the separately qualified CPU ATen backend.
package model

import (
	"context"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sync"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

const HACConfigSHA256 = "e550ccbedb68ed74d6ee88e2f510adf33c52a1c868bd5e3a2bfbb7dfe6154c3f"
const SUPConfigSHA256 = "ad82d7c369deddc0d1e69367302b8e002c306a9770ae4304a3cbb70715fcce49"

var ErrUnsupported = errors.New("model graph is not implemented")
var ErrClosed = errors.New("model is closed")

type Architecture string

const (
	HAC Architecture = "hac-flstm-v6"
	SUP Architecture = "sup-transformer-v5.2"
)

// Spec contains the interpreted values of an admitted exact config. All floating
// fields have the float32 precision used by Dorado's config loader.
type Spec struct {
	Name, ConfigSHA256                                     string
	Architecture                                           Architecture
	Features, SampleRate, Stride, InnerStride, Granularity int64
	ChunkSize, Overlap, StateLen, OutputChannels           int64
	Mean, Stdev, QScale, QBias, BlankScore                 float32
	MeanQStart                                             int
}

func Identify(config []byte) (Spec, error) {
	h := sha256.Sum256(config)
	pin := hex.EncodeToString(h[:])
	s := Spec{ConfigSHA256: pin, Features: 1, SampleRate: 5000, Stride: 6,
		ChunkSize: 12288, Overlap: 600, BlankScore: 2, MeanQStart: 60}
	switch pin {
	case HACConfigSHA256:
		s.Name = "dna_r10.4.1_e8.2_400bps_hac@v6.0.0"
		s.Architecture, s.InnerStride, s.Granularity = HAC, 6, 6
		s.StateLen, s.OutputChannels = 4, 1024
		s.Mean, s.Stdev, s.QScale, s.QBias = 94, 24, 1.02, -0.3
	case SUPConfigSHA256:
		s.Name = "dna_r10.4.1_e8.2_400bps_sup@v5.2.0"
		s.Architecture, s.InnerStride, s.Granularity = SUP, 12, 768
		s.StateLen, s.OutputChannels = 5, 4096
		s.Mean, s.Stdev, s.QScale, s.QBias = 93.69239463939118, 23.506745239082388, 1.05, .3
	default:
		return Spec{}, fmt.Errorf("unknown model config SHA256 %s", pin)
	}
	return s, nil
}

type WeightReceipt struct {
	Name   string  `json:"name"`
	SHA256 string  `json:"sha256"`
	Shape  []int64 `json:"shape"`
}

//go:embed hac_weights.json
var hacWeightManifest []byte

// Model borrows its Engine. Close and Forward serialize; Close never closes the
// Engine or caller input/output handles.
type Model struct {
	mu       sync.Mutex
	engine   *tensor.Engine
	spec     Spec
	graph    *hacGraph
	weights  []*tensor.Tensor
	receipts []WeightReceipt
	closed   bool
}

func Open(engine *tensor.Engine, directory string) (*Model, error) {
	if engine == nil {
		return nil, errors.New("nil tensor engine")
	}
	spec, err := Inspect(directory)
	if err != nil {
		return nil, err
	}
	m := &Model{engine: engine, spec: spec}
	if spec.Architecture == SUP {
		return nil, fmt.Errorf("%s: %w", spec.Name, ErrUnsupported)
	}
	var manifest []WeightReceipt
	if err := json.Unmarshal(hacWeightManifest, &manifest); err != nil {
		return nil, err
	}
	var loadErr error
	for _, r := range manifest {
		w, err := loadWeight(engine, filepath.Join(directory, r.Name), r)
		if err != nil {
			loadErr = fmt.Errorf("weight %s: %w", r.Name, err)
			break
		}
		m.weights = append(m.weights, w)
		m.receipts = append(m.receipts, r)
	}
	if loadErr != nil {
		return nil, errors.Join(loadErr, m.Close())
	}
	m.graph = bindHAC(m.weights)
	return m, nil
}

// Inspect recognizes a known config without loading weights or promising that
// its graph is implemented. Open succeeds only for usable inference models.
func Inspect(directory string) (Spec, error) {
	config, err := os.ReadFile(filepath.Join(directory, "config.toml"))
	if err != nil {
		return Spec{}, err
	}
	return Identify(config)
}

func fileSHA(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() {
		return "", errors.New("weight must be a regular file")
	}
	h := sha256.New()
	if _, err = io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func loadWeight(e *tensor.Engine, path string, r WeightReceipt) (*tensor.Tensor, error) {
	before, err := fileSHA(path)
	if err != nil {
		return nil, err
	}
	if before != r.SHA256 {
		return nil, fmt.Errorf("SHA256 mismatch: got %s want %s", before, r.SHA256)
	}
	w, err := e.LoadWeight(path)
	if err != nil {
		return nil, err
	}
	shape, err := w.Shape()
	if err == nil && !slices.Equal(shape, r.Shape) {
		err = fmt.Errorf("shape %v want %v", shape, r.Shape)
	}
	if err == nil {
		var after string
		after, err = fileSHA(path)
		if err == nil && after != before {
			err = errors.New("weight changed while loading")
		}
	}
	if err != nil {
		return nil, errors.Join(err, w.Close())
	}
	return w, nil
}

func (m *Model) Spec() Spec { return m.spec }

func (m *Model) Weights() []WeightReceipt {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]WeightReceipt, len(m.receipts))
	copy(out, m.receipts)
	for i := range out {
		out[i].Shape = slices.Clone(out[i].Shape)
	}
	return out
}

func (m *Model) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	var err error
	for _, w := range m.weights {
		err = errors.Join(err, w.Close())
	}
	m.weights = nil
	m.graph = nil
	return err
}

func (m *Model) Forward(ctx context.Context, input *tensor.Tensor) (*tensor.Tensor, error) {
	if m == nil {
		return nil, ErrClosed
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, ErrClosed
	}
	if m.spec.Architecture != HAC {
		return nil, fmt.Errorf("%s: %w", m.spec.Name, ErrUnsupported)
	}
	if ctx == nil {
		return nil, errors.New("nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if input == nil {
		return nil, errors.New("nil model input")
	}
	shape, err := input.Shape()
	if err != nil {
		return nil, err
	}
	if len(shape) != 3 || shape[0] < 1 || shape[1] != 1 || shape[2] < 1 || shape[2]%m.spec.Granularity != 0 {
		return nil, fmt.Errorf("input must be positive [N,1,L] with L divisible by %d: %v", m.spec.Granularity, shape)
	}
	data, err := input.Data()
	if err != nil {
		return nil, err
	}
	for _, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, errors.New("nonfinite model input")
		}
	}
	return m.graph.forward(ctx, m.engine, input, nil)
}
