package hacgpu

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
)

// FLSTMCacheSchema identifies a native FLSTM weight-cache capture.
const FLSTMCacheSchema = "nanopore-hacgpu-flstm-cache/v1"

// FLSTMCacheTensor is one tensor the cache oracle published.
type FLSTMCacheTensor struct {
	Role       string  `json:"role"`
	File       string  `json:"file"`
	Shape      []int64 `json:"shape"`
	DType      string  `json:"dtype"`
	Contiguous bool    `json:"contiguous"`
	SHA256     string  `json:"sha256"`
}

// FLSTMCacheLayer is one recurrent layer's published tensors and the measured
// quantization domain of the intermediate scales.
type FLSTMCacheLayer struct {
	Layer              int                `json:"layer"`
	Reverse            bool               `json:"reverse"`
	ScaleMin           float32            `json:"scale_min"`
	ScaleMax           float32            `json:"scale_max"`
	ZeroScaleRows      int                `json:"zero_scale_rows"`
	NonfiniteScaleRows int                `json:"nonfinite_scale_rows"`
	Tensors            []FLSTMCacheTensor `json:"tensors"`
}

// FLSTMQuantizeContract is the quantizer the capture is evidence for. Every field
// is compared with the port's constants, so a capture produced by a different
// quantizer cannot be consumed as if it were this one.
type FLSTMQuantizeContract struct {
	Levels      int    `json:"levels"`
	Dim         int    `json:"dim"`
	QuantMax    int    `json:"quant_max"`
	QuantMin    int    `json:"quant_min"`
	WeightDType string `json:"weight_dtype"`
	ScaleDType  string `json:"scale_dtype"`
	InputDType  string `json:"input_dtype"`
	Round       string `json:"round"`
}

// FLSTMCaches is the typed view of caches.json.
type FLSTMCaches struct {
	Schema       string                `json:"schema"`
	Status       string                `json:"status"`
	Case         string                `json:"case"`
	ModelDir     string                `json:"model_dir"`
	ModelPackage string                `json:"model_package"`
	SourceCommit string                `json:"source_commit"`
	Ordinal      int                   `json:"ordinal"`
	Batch        int                   `json:"batch"`
	Chunk        int                   `json:"chunk"`
	Features     int                   `json:"features"`
	LSTMLayers   int                   `json:"lstm_layers"`
	LSTMSize     int                   `json:"lstm_size"`
	LSTMInnerDim int                   `json:"lstm_inner_dim"`
	C            int64                 `json:"C"`
	K            int64                 `json:"K"`
	FirstReverse bool                  `json:"first_reverse"`
	InputSHA256  string                `json:"input_sha256"`
	Reservation  int64                 `json:"reservation_bytes"`
	Quantize     FLSTMQuantizeContract `json:"quantize"`
	Layers       []FLSTMCacheLayer     `json:"layers"`
}

// flstmRoleSpec is one tensor the oracle must publish, in the order it publishes
// them. The shapes are the ones `FLSTMStack.cpp:165-217` produces:
//
//	down quantized   [K, C]      down scale   [K]
//	up quantized     [4C, 2K]    up scale     [4C]
//	bias f16         [4C]
//	down caches      [K, C]      down scales  [K]
//	up caches        [4C, 2K]    up scales    [4C]   up bias  [4C]
//
// A role published in a different order, at a different dtype or at a different
// shape is a rejection: the whole point of the capture is that it is the exact
// tensor the kernel reads.
type flstmRoleSpec struct {
	role  string
	dtype string
	shape func(c, k int64) []int64
}

var flstmRoleSpecs = []flstmRoleSpec{
	{"dn-ih-quantized", "Char", func(c, k int64) []int64 { return []int64{k, c} }},
	{"dn-ih-quant-scale", "Float", func(c, k int64) []int64 { return []int64{k} }},
	{"dn-hh-quantized", "Char", func(c, k int64) []int64 { return []int64{k, c} }},
	{"dn-hh-quant-scale", "Float", func(c, k int64) []int64 { return []int64{k} }},
	{"up-quantized", "Char", func(c, k int64) []int64 { return []int64{4 * c, 2 * k} }},
	{"up-quant-scale", "Float", func(c, k int64) []int64 { return []int64{4 * c} }},
	{"up-bias-f16", "Half", func(c, k int64) []int64 { return []int64{4 * c} }},
	{"dn-ih-cache", "Char", func(c, k int64) []int64 { return []int64{k, c} }},
	{"dn-ih-cache-scale", "Half", func(c, k int64) []int64 { return []int64{k} }},
	{"dn-hh-cache", "Char", func(c, k int64) []int64 { return []int64{k, c} }},
	{"dn-hh-cache-scale", "Half", func(c, k int64) []int64 { return []int64{k} }},
	{"up-cache", "Char", func(c, k int64) []int64 { return []int64{4 * c, 2 * k} }},
	{"up-cache-scale", "Half", func(c, k int64) []int64 { return []int64{4 * c} }},
	{"up-cache-bias", "Half", func(c, k int64) []int64 { return []int64{4 * c} }},
}

// FLSTMCacheRoleCount is how many tensors each layer publishes.
func FLSTMCacheRoleCount() int { return len(flstmRoleSpecs) }

// FLSTMCacheRoleNames returns the published roles in order.
func FLSTMCacheRoleNames() []string {
	out := make([]string, len(flstmRoleSpecs))
	for i, s := range flstmRoleSpecs {
		out[i] = s.role
	}
	return out
}

// ImportFLSTMCaches reads and fully verifies a FLSTM cache capture. Every tensor
// is re-hashed, its shape and dtype are checked against the declared geometry and
// its byte count against its shape, so a corrupt or mislabelled capture is a hard
// rejection. This is the only way a consumer obtains the native cache bytes.
func ImportFLSTMCaches(dir string) (FLSTMCaches, error) {
	var c FLSTMCaches
	raw, err := os.ReadFile(filepath.Join(dir, "caches.json"))
	if err != nil {
		return c, err
	}
	if err = json.Unmarshal(raw, &c); err != nil {
		return c, err
	}
	if c.Schema != FLSTMCacheSchema {
		return c, fmt.Errorf("cache schema %q is not %q", c.Schema, FLSTMCacheSchema)
	}
	if c.Status != "captured" {
		return c, fmt.Errorf("cache status %q is not captured", c.Status)
	}
	if c.SourceCommit != SourceCommit {
		return c, fmt.Errorf("cache source commit %q is not the pinned revision", c.SourceCommit)
	}
	if c.C <= 0 || c.K <= 0 || c.C%16 != 0 || c.K%32 != 0 {
		return c, fmt.Errorf("cache geometry C=%d K=%d is not the declared factorisation", c.C, c.K)
	}
	if c.C > math.MaxInt64/8/c.K {
		return c, fmt.Errorf("cache geometry overflows its packed extent")
	}
	if int64(c.LSTMSize) != c.C || int64(c.LSTMInnerDim) != c.K {
		return c, fmt.Errorf("cache geometry C=%d K=%d does not match lstm_size %d / lstm_inner_dim %d",
			c.C, c.K, c.LSTMSize, c.LSTMInnerDim)
	}
	if c.LSTMLayers <= 0 || len(c.Layers) != c.LSTMLayers {
		return c, fmt.Errorf("cache declares %d recurrent layers but publishes %d", c.LSTMLayers, len(c.Layers))
	}
	if c.Features <= 0 || c.Batch <= 0 || c.Chunk <= 0 {
		return c, fmt.Errorf("cache case geometry is not positive")
	}
	if c.InputSHA256 == "" {
		return c, fmt.Errorf("cache does not pin its input digest")
	}
	if err = c.Quantize.verify(); err != nil {
		return c, err
	}
	for i, layer := range c.Layers {
		if layer.Layer != i {
			return c, fmt.Errorf("cache layer %d is labelled %d", i, layer.Layer)
		}
		if layer.Reverse != (c.FirstReverse != (i%2 != 0)) {
			return c, fmt.Errorf("cache layer %d has the wrong direction", i)
		}
		if layer.ScaleMin <= 0 || layer.ScaleMax < layer.ScaleMin || math.IsNaN(float64(layer.ScaleMin)) ||
			math.IsNaN(float64(layer.ScaleMax)) || math.IsInf(float64(layer.ScaleMax), 0) {
			return c, fmt.Errorf("cache layer %d has an invalid scale range", i)
		}
		if layer.ZeroScaleRows != 0 || layer.NonfiniteScaleRows != 0 {
			return c, fmt.Errorf("cache layer %d leaves the admitted quantization domain (%d zero, %d non-finite rows)",
				i, layer.ZeroScaleRows, layer.NonfiniteScaleRows)
		}
		if err = verifyFLSTMCacheLayer(dir, &c, layer); err != nil {
			return c, fmt.Errorf("cache layer %d: %w", i, err)
		}
	}
	return c, nil
}

func (q FLSTMQuantizeContract) verify() error {
	want := FLSTMQuantizeContract{
		Levels: 256, Dim: 1, QuantMax: 127, QuantMin: -127,
		WeightDType: "Char", ScaleDType: "Float", InputDType: "Half", Round: "half_to_even",
	}
	if q != want {
		return fmt.Errorf("cache quantizer contract %+v is not the pinned %+v", q, want)
	}
	return nil
}

func verifyFLSTMCacheLayer(dir string, c *FLSTMCaches, layer FLSTMCacheLayer) error {
	if len(layer.Tensors) != len(flstmRoleSpecs) {
		return fmt.Errorf("publishes %d tensors, want %d", len(layer.Tensors), len(flstmRoleSpecs))
	}
	for i, spec := range flstmRoleSpecs {
		t := layer.Tensors[i]
		if t.Role != spec.role {
			return fmt.Errorf("tensor %d is %q, want %q", i, t.Role, spec.role)
		}
		if t.DType != spec.dtype {
			return fmt.Errorf("tensor %q is %s, want %s", t.Role, t.DType, spec.dtype)
		}
		if want := spec.shape(c.C, c.K); !slices.Equal(t.Shape, want) {
			return fmt.Errorf("tensor %q has shape %v, want %v", t.Role, t.Shape, want)
		}
		if !t.Contiguous {
			return fmt.Errorf("tensor %q is not contiguous", t.Role)
		}
		if t.File == "" || filepath.Base(t.File) != t.File {
			return fmt.Errorf("tensor %q has file %q", t.Role, t.File)
		}
		elem, ok := dtypeBytes[t.DType]
		if !ok {
			return fmt.Errorf("tensor %q has unknown dtype %q", t.Role, t.DType)
		}
		body, err := os.ReadFile(filepath.Join(dir, t.File))
		if err != nil {
			return err
		}
		count := int64(1)
		for _, d := range t.Shape {
			if d <= 0 || count > math.MaxInt64/elem/d {
				return fmt.Errorf("tensor %q has an invalid byte extent", t.Role)
			}
			count *= d
		}
		if int64(len(body)) != count*elem {
			return fmt.Errorf("tensor %q payload is %d bytes, shape %v at %d bytes/element needs %d",
				t.Role, len(body), t.Shape, elem, count*elem)
		}
		got, err := bytesSHA(body)
		if err != nil {
			return err
		}
		if got != t.SHA256 {
			return fmt.Errorf("tensor %q digest %s does not match the recorded %s", t.Role, got, t.SHA256)
		}
	}
	return nil
}

// Tensor returns a verified tensor by role.
func (l FLSTMCacheLayer) Tensor(role string) (FLSTMCacheTensor, bool) {
	for _, t := range l.Tensors {
		if t.Role == role {
			return t, true
		}
	}
	return FLSTMCacheTensor{}, false
}

// Bytes reads a verified tensor payload.
func (l FLSTMCacheLayer) Bytes(dir, role string) ([]byte, error) {
	t, ok := l.Tensor(role)
	if !ok {
		return nil, fmt.Errorf("layer %d has no tensor %q", l.Layer, role)
	}
	if t.File == "" || filepath.Base(t.File) != t.File {
		return nil, fmt.Errorf("tensor %q has file %q", role, t.File)
	}
	body, err := os.ReadFile(filepath.Join(dir, t.File))
	if err != nil {
		return nil, err
	}
	got, err := bytesSHA(body)
	if err != nil {
		return nil, err
	}
	if got != t.SHA256 {
		return nil, fmt.Errorf("tensor %q changed after import", role)
	}
	return body, nil
}
