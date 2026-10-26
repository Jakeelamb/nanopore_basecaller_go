package model

import (
	"context"
	"fmt"
	"slices"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

type supEncoderStage struct {
	name string
	op   tensor.SUPPrimitive
}

var supEncoderStages = [...]supEncoderStage{{"qkv", tensor.SUPQKVRotary}, {"attention", tensor.SUPMaskedAttention}, {"projection", tensor.SUPProjection}, {"norm1", tensor.SUPNorm1}, {"swiglu", tensor.SUPSwiGLU}, {"fc2", tensor.SUPFC2}, {"norm2", tensor.SUPNorm2}}

func runSUPEncoderStages(ctx context.Context, layer int, run func(supEncoderStage) error) error {
	for _, stage := range supEncoderStages {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := run(stage); err != nil {
			return fmt.Errorf("encoder%d.%s: %w", layer, stage.name, err)
		}
	}
	return ctx.Err()
}

type SUPGPUProfile struct {
	Batch, Chunk, EncoderT   int64
	UseI8, UseF8, UseUSwiGLU bool
	RemoveBits               int
}

func FixedSUPGPUProfile() SUPGPUProfile { return SUPGPUProfile{32, 768, 64, true, true, false, 4} }
func FullFixedSUPGPUProfile() SUPGPUProfile {
	return SUPGPUProfile{32, 12288, 1024, true, true, false, 4}
}

// ShortFixedSUPGPUProfile is the source-default simplex half-size fixed queue.
// It does not select the variable transformer path.
func ShortFixedSUPGPUProfile() SUPGPUProfile {
	return SUPGPUProfile{32, 6144, 512, true, true, false, 4}
}

type SUPGPUCall struct {
	Name      string
	Layer     int
	Primitive tensor.SUPPrimitive
	Selector  int
	Operands  []tensor.SUPOperandSpec
}
type SUPGPUPlan struct {
	profile    SUPGPUProfile
	geometry   tensor.SUPGeometry
	dimensions tensor.SUPDimensions
	calls      []SUPGPUCall
}

func (p SUPGPUPlan) Profile() SUPGPUProfile       { return p.profile }
func (p SUPGPUPlan) Geometry() tensor.SUPGeometry { return p.geometry }
func (p SUPGPUPlan) Calls() []SUPGPUCall {
	out := slices.Clone(p.calls)
	for i := range out {
		out[i].Operands = slices.Clone(out[i].Operands)
		for j := range out[i].Operands {
			out[i].Operands[j].Shape = slices.Clone(out[i].Operands[j].Shape)
			out[i].Operands[j].Tags = slices.Clone(out[i].Operands[j].Tags)
		}
	}
	return out
}
func PlanSUPGPU(profile SUPGPUProfile) (SUPGPUPlan, error) {
	var geometry tensor.SUPGeometry
	switch profile {
	case FixedSUPGPUProfile():
		geometry = tensor.SUPGeometryTiny
	case FullFixedSUPGPUProfile():
		geometry = tensor.SUPGeometryFull
	case ShortFixedSUPGPUProfile():
		geometry = tensor.SUPGeometryShort
	default:
		return SUPGPUPlan{}, fmt.Errorf("%w: SUP admits only fixed N32/chunk768/T64, chunk6144/T512 or chunk12288/T1024, F8/I8, remove_bits4 and default SwiGLU", ErrGPUPlanLayout)
	}
	dimensions, err := geometry.Dimensions()
	if err != nil {
		return SUPGPUPlan{}, err
	}
	p := SUPGPUPlan{profile: profile, geometry: geometry, dimensions: dimensions}
	add := func(name string, layer int, op tensor.SUPPrimitive, selector int) error {
		specs, err := tensor.SUPPrimitiveSpecs(geometry, op, selector)
		if err != nil {
			return err
		}
		p.calls = append(p.calls, SUPGPUCall{name, layer, op, selector, specs})
		return nil
	}
	for i := 0; i < 5; i++ {
		if err := add(fmt.Sprintf("conv%d", i), -1, tensor.SUPConvSiLU, i); err != nil {
			return SUPGPUPlan{}, err
		}
	}
	if err := add("encoder.quantize", -1, tensor.SUPQuantizeRows, 0); err != nil {
		return SUPGPUPlan{}, err
	}
	for i := 0; i < 18; i++ {
		for _, stage := range supEncoderStages {
			if err := add(fmt.Sprintf("encoder.%02d.%s", i, stage.name), i, stage.op, 0); err != nil {
				return SUPGPUPlan{}, err
			}
		}
	}
	if err := add("upsample", -1, tensor.SUPLinear, 0); err != nil {
		return SUPGPUPlan{}, err
	}
	if err := add("scores", -1, tensor.SUPLinear, 1); err != nil {
		return SUPGPUPlan{}, err
	}
	return p, nil
}

func supContiguous(shape []int64) []int64 {
	out := make([]int64, len(shape))
	stride := int64(1)
	for i := len(shape) - 1; i >= 0; i-- {
		out[i] = stride
		stride *= shape[i]
	}
	return out
}

// SUPNTCTileStrides describes a view of contiguous NTC as N,T/t,C/c,t,c.
func SUPNTCTileStrides(geometry tensor.SUPGeometry, lane int64) ([]int64, []int64, error) {
	d, err := geometry.Dimensions()
	if err != nil {
		return nil, nil, err
	}
	if lane != 8 && lane != 16 {
		return nil, nil, fmt.Errorf("%w: fixed SUP tile lane required", ErrGPUPlanLayout)
	}
	const c = int64(512)
	return []int64{d.Batch, d.EncoderT / 16, c / lane, 16, lane}, []int64{d.EncoderT * c, 16 * c, lane, c, 1}, nil
}
