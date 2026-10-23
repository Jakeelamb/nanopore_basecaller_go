package modeloracle

import (
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
)

// WriteSUPWeightManifest pins only the 139 files consumed by load_tx_model_weights.
// Native loading separately validates each archive's tensor count, dtype and shape.
func WriteSUPWeightManifest(directory string, out io.Writer) error {
	type receipt struct {
		Name   string  `json:"name"`
		SHA256 string  `json:"sha256"`
		Shape  []int64 `json:"shape"`
	}
	var receipts []receipt
	add := func(name string, shape ...int64) error {
		h, err := Hash(filepath.Join(directory, name))
		if err != nil {
			return err
		}
		receipts = append(receipts, receipt{name, h, shape})
		return nil
	}
	for i, shape := range [][]int64{{64, 1, 5}, {64, 64, 5}, {128, 64, 9}, {128, 128, 9}, {512, 128, 5}} {
		if err := add(fmt.Sprintf("conv.%d.conv.weight.tensor", i), shape...); err != nil {
			return err
		}
		if err := add(fmt.Sprintf("conv.%d.conv.bias.tensor", i), shape[0]); err != nil {
			return err
		}
	}
	for i := 0; i < 18; i++ {
		for _, w := range []struct {
			name  string
			shape []int64
		}{
			{"self_attn.Wqkv.weight", []int64{1536, 512}},
			{"self_attn.out_proj.weight", []int64{512, 512}},
			{"self_attn.out_proj.bias", []int64{512}},
			{"ff.fc1.weight", []int64{4096, 512}},
			{"ff.fc2.weight", []int64{512, 2048}},
			{"norm1.weight", []int64{512}},
			{"norm2.weight", []int64{512}},
		} {
			if err := add(fmt.Sprintf("transformer_encoder.%d.%s.tensor", i, w.name), w.shape...); err != nil {
				return err
			}
		}
	}
	for _, w := range []struct {
		name  string
		shape []int64
	}{
		{"upsample.linear.weight.tensor", []int64{1024, 512}},
		{"upsample.linear.bias.tensor", []int64{1024}},
		{"crf.linear.weight.tensor", []int64{4096, 512}},
	} {
		if err := add(w.name, w.shape...); err != nil {
			return err
		}
	}
	e := json.NewEncoder(out)
	e.SetIndent("", "  ")
	return e.Encode(receipts)
}
