package h02

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

// packAndVerify packs every executed layer's weights in Go and then proves the
// Go bytes against the verification-only native packing oracle that reproduces
//
//	conv->weight.permute({2, 1, 0}).contiguous().flatten(0, 1).to(kF16)
//	conv->bias.to(kF16)
//
// (ConvStack.cpp:230-234). The oracle is never on the caller path; it exists so
// the pure-Go packer can be checked bit for bit.
func packAndVerify(d *tensor.Device, s *tensor.Stream, executed []model.HACSmallConvLayer, weights ConvWeights) ([]Packing, error) {
	var out []Packing
	for i, l := range executed {
		w := weights.Layers[i]
		if w.Index != l.Index {
			return nil, fmt.Errorf("weight layer %d does not match the plan's conv%d", w.Index, l.Index)
		}
		goWeight, err := model.PackSmallConvWeights(w.Weight, l.Params.COut, l.Params.CIn, l.Params.WinLen)
		if err != nil {
			return nil, err
		}
		goBias, err := model.PackSmallConvBias(w.Bias)
		if err != nil {
			return nil, err
		}

		src, err := d.NewStorage(s, tensor.DTypeF32,
			[]int64{l.Params.COut, l.Params.CIn, l.Params.WinLen},
			[]tensor.KoiTag{tensor.TagW, tensor.TagC, tensor.TagD}, float32Bytes(w.Weight))
		if err != nil {
			return nil, err
		}
		defer src.Close()
		biasSrc, err := d.NewStorage(s, tensor.DTypeF32, []int64{l.Params.COut},
			[]tensor.KoiTag{tensor.TagC}, float32Bytes(w.Bias))
		if err != nil {
			return nil, err
		}
		defer biasSrc.Close()
		packed, err := d.NewStorage(s, tensor.DTypeF16,
			[]int64{l.Params.WinLen * l.Params.CIn, l.Params.COut},
			[]tensor.KoiTag{tensor.TagW, tensor.TagC}, make([]byte, len(goWeight)*2))
		if err != nil {
			return nil, err
		}
		defer packed.Close()
		packedBias, err := d.NewStorage(s, tensor.DTypeF16, []int64{l.Params.COut},
			[]tensor.KoiTag{tensor.TagC}, make([]byte, len(goBias)*2))
		if err != nil {
			return nil, err
		}
		defer packedBias.Close()

		if err = d.ReferencePackSmallConvF16(s, src, biasSrc, packed, packedBias); err != nil {
			return nil, fmt.Errorf("conv%d reference packing: %w", l.Index, err)
		}
		refWeight, err := packed.Download(s)
		if err != nil {
			return nil, err
		}
		refBias, err := packedBias.Download(s)
		if err != nil {
			return nil, err
		}
		goWeightBytes := model.HalfBytes(goWeight)
		goBiasBytes := model.HalfBytes(goBias)
		rec := Packing{
			Index:         l.Index,
			WeightFileSHA: w.WeightSHA, WeightShape: w.Shape, BiasFileSHA: w.BiasSHA,
			GoWeightSHA: hashBytes(goWeightBytes), ReferenceSHA: hashBytes(refWeight),
			GoBiasSHA: hashBytes(goBiasBytes), ReferenceBiasSHA: hashBytes(refBias),
			Equal: bytes.Equal(goWeightBytes, refWeight) && bytes.Equal(goBiasBytes, refBias),
		}
		if !rec.Equal {
			return nil, fmt.Errorf("conv%d Go packing differs from the native reference", l.Index)
		}
		out = append(out, rec)
	}
	return out, nil
}

func float32Bytes(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}
