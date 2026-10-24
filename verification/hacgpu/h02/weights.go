package h02

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

// WeightManifestPath is the caller's own weight pin file. A consumer reads the
// same file the caller loads against, so the replay cannot be pinned to a
// different checkpoint than the port.
const WeightManifestPath = "internal/model/hac_weights.json"

type weightReceipt struct {
	Name   string  `json:"name"`
	SHA256 string  `json:"sha256"`
	Shape  []int64 `json:"shape"`
}

// ConvWeight is one loaded convolution's float32 parameters and their pins.
type ConvWeight struct {
	Index     int
	Weight    []float32
	Bias      []float32
	WeightSHA string
	BiasSHA   string
	Shape     []int64
}

// ConvWeights is the loaded stack plus the manifest digest it was checked
// against.
type ConvWeights struct {
	ManifestSHA string
	Layers      []ConvWeight
}

// LoadConvWeights loads every convolution weight and bias through the CPU ATen
// engine and requires the file digest and shape to match the caller's manifest
// before a single byte reaches the device. It is exported so a later consumer
// (H03) reuses the same pin checks instead of re-implementing them.
func LoadConvWeights(e *tensor.Engine, modelDir string, convs []ConvSpec) (ConvWeights, error) {
	var out ConvWeights
	raw, err := os.ReadFile(WeightManifestPath)
	if err != nil {
		return out, err
	}
	if out.ManifestSHA, err = fileSHA(WeightManifestPath); err != nil {
		return out, err
	}
	var manifest []weightReceipt
	if err = json.Unmarshal(raw, &manifest); err != nil {
		return out, err
	}
	byName := map[string]weightReceipt{}
	for _, r := range manifest {
		byName[r.Name] = r
	}
	for _, c := range convs {
		prefix := fmt.Sprintf("%d.conv.", c.Index)
		w, err := loadPinnedWeight(e, modelDir, byName, prefix+"weight.tensor")
		if err != nil {
			return out, err
		}
		b, err := loadPinnedWeight(e, modelDir, byName, prefix+"bias.tensor")
		if err != nil {
			return out, err
		}
		shape, err := w.Shape()
		if err != nil {
			return out, err
		}
		if !slices.Equal(shape, []int64{c.COut, c.CIn, c.WinLen}) {
			return out, fmt.Errorf("weight %s shape %v is not the declared [C_out,C_in,W] %v",
				prefix+"weight.tensor", shape, []int64{c.COut, c.CIn, c.WinLen})
		}
		biasShape, err := b.Shape()
		if err != nil {
			return out, err
		}
		if !slices.Equal(biasShape, []int64{c.COut}) {
			return out, fmt.Errorf("bias %s shape %v is not the declared [C_out] %d",
				prefix+"bias.tensor", biasShape, c.COut)
		}
		weight, err := w.Data()
		if err != nil {
			return out, err
		}
		bias, err := b.Data()
		if err != nil {
			return out, err
		}
		if err = w.Close(); err != nil {
			return out, err
		}
		if err = b.Close(); err != nil {
			return out, err
		}
		out.Layers = append(out.Layers, ConvWeight{
			Index: c.Index, Weight: weight, Bias: bias,
			WeightSHA: byName[prefix+"weight.tensor"].SHA256,
			BiasSHA:   byName[prefix+"bias.tensor"].SHA256,
			Shape:     shape,
		})
	}
	return out, nil
}

func loadPinnedWeight(e *tensor.Engine, modelDir string, manifest map[string]weightReceipt, name string) (*tensor.Tensor, error) {
	want, ok := manifest[name]
	if !ok {
		return nil, fmt.Errorf("weight manifest has no %q", name)
	}
	path := filepath.Join(modelDir, name)
	got, err := fileSHA(path)
	if err != nil {
		return nil, err
	}
	if got != want.SHA256 {
		return nil, fmt.Errorf("weight %s digest %s does not match the pinned %s", name, got, want.SHA256)
	}
	w, err := e.LoadWeight(path)
	if err != nil {
		return nil, err
	}
	shape, err := w.Shape()
	if err == nil && !slices.Equal(shape, want.Shape) {
		err = fmt.Errorf("weight %s shape %v does not match the pinned %v", name, shape, want.Shape)
	}
	if err != nil {
		return nil, err
	}
	return w, nil
}
