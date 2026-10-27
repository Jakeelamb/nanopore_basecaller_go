package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
)

// This file is the host-side port of the HAC FLSTM device weight caches
// (nn/FLSTMStack.cpp:165-217) and of the quantizer they are built from
// (torch_utils/tensor_utils.cpp:291-297).
//
// The source builds the caches on the first forward pass:
//
//	scaled = quantize_tensor(param.to(f16), 1)
//	device_dn_weights_ih  = scaled.t.view({-1,2,2,2,2,2,C}).permute({2,0,4,1,3,5,6}).contiguous().view({K,C})
//	device_dn_weights_scale_ih = scaled.scale.view({-1,2,2,2,2,2}).permute({2,0,4,1,3,5}).contiguous().view({K}).to(f16)
//	up = cat({up_weight_ih, up_weight_hh}, 1)
//	scaled_up = quantize_tensor(up, 1)
//	device_up_weights = scaled_up.t.view({4,-1,2,2,2,2,2K}).permute({1,5,0,2,3,4,6}).contiguous().view({4C,2K})
//	device_up_weights_scale = scaled_up.scale.to(f16).view({4,-1,2,2,2,2}).permute({1,5,2,3,4,0}).contiguous()
//	device_up_bias = up_bias_ih.to(f16).view({4,-1,2,2,2,2}).permute({1,5,2,3,4,0}).contiguous()
//
// Four details are load-bearing and are why this is not a generic quantizer:
//
//   - the cast to half happens *before* the row reduction, so the reduction is
//     over half values and the scale is a half;
//   - the reduction is over dimension 1 only (`dim = 1`), which for the down
//     weights is the input-channel axis and for the concatenated up weights is
//     the joint input/hidden axis;
//   - the quantized value is `round_half_even(t * scale)` evaluated in half, so
//     the product is rounded to half once and then to an integer, and only then
//     clipped to +-127 and cast to Char;
//   - `up_bias_hh` is never read: the Koi path only ever uses `up_bias_ih`
//     (FLSTMStack.cpp:212-216). Carrying the hidden bias into the cache would
//     add a recurrent bias the source does not add.

// ErrGPUWeights marks a rejected weight-packing request: a parameter shape, a
// layer geometry or a quantization domain the pinned source would not produce.
// It is separate from the working-memory plan errors because a rejected cache is
// not a placement problem.
var ErrGPUWeights = errors.New("model: GPU weight packing rejected")

// The quantizer constants, verbatim from `quantize_tensor`.
const (
	FLSTMQuantLevels = 256
	FLSTMQuantMax    = 127
	FLSTMQuantMin    = -127
)

// FLSTMMutant names a deliberate, verification-only deviation from the source's
// cache construction. Every production call passes FLSTMMutantNone; the H04
// differential packs each layer with each mutant and requires the native cache to
// reject it, which is how each step is shown to be load-bearing rather than
// decorative. A mutant that stops changing the bytes is a failed control.
type FLSTMMutant int

const (
	// FLSTMMutantNone is the source's construction.
	FLSTMMutantNone FLSTMMutant = iota
	// FLSTMMutantReductionAxis0 reduces over dimension 0 instead of 1.
	FLSTMMutantReductionAxis0
	// FLSTMMutantFloat32Quantize skips the `to(opts_f16)` cast, quantizing the
	// float32 parameters directly.
	FLSTMMutantFloat32Quantize
	// FLSTMMutantEpsilon scales the row maximum by (1 + 1e-3) before the
	// reciprocal, the way a "guard against division by zero" fudge would. The
	// source divides by the exact maximum and has no such guard.
	FLSTMMutantEpsilon
	// FLSTMMutantClip128 clips to +-128 instead of +-127, so the extreme values
	// wrap in the Char cast.
	FLSTMMutantClip128
	// FLSTMMutantTruncate truncates instead of rounding half to even.
	FLSTMMutantTruncate
	// FLSTMMutantIdentityPermute skips the gate/block permutation.
	FLSTMMutantIdentityPermute
	// FLSTMMutantHiddenBias packs `up_bias_hh` instead of `up_bias_ih`.
	FLSTMMutantHiddenBias
)

func (m FLSTMMutant) String() string {
	switch m {
	case FLSTMMutantNone:
		return "none"
	case FLSTMMutantReductionAxis0:
		return "reduction-axis-0"
	case FLSTMMutantFloat32Quantize:
		return "float32-quantize"
	case FLSTMMutantEpsilon:
		return "epsilon"
	case FLSTMMutantClip128:
		return "clip-128"
	case FLSTMMutantTruncate:
		return "truncate"
	case FLSTMMutantIdentityPermute:
		return "identity-permute"
	case FLSTMMutantHiddenBias:
		return "hidden-bias"
	}
	return fmt.Sprintf("mutant(%d)", int(m))
}

// FLSTMParams is one factorised LSTM layer's geometry: the hidden size C and the
// decomposition K (`lstm_size` and `lstm_inner_dim` in the model config).
type FLSTMParams struct {
	C int64
	K int64
}

// UpRows is the up-projection's row count, `4 * C` (the four gates).
func (p FLSTMParams) UpRows() int64 { return 4 * p.C }

// UpCols is the up-projection's column count after the ih/hh concatenation,
// `2 * K` (FLSTMStack.cpp:204-205).
func (p FLSTMParams) UpCols() int64 { return 2 * p.K }

func (p FLSTMParams) validate() error {
	if p.C < 1 || p.K < 1 {
		return fmt.Errorf("%w: FLSTM geometry C=%d K=%d", ErrGPUWeights, p.C, p.K)
	}
	// The `-1` dim of `view({-1,2,2,2,2,2,C})` is K/32, and the `-1` dim of
	// `view({4,-1,2,2,2,2,2K})` and `view({4,-1,2,2,2,2})` is C/16
	// (FLSTMStack.cpp:170-215). A geometry that does not divide is refused
	// instead of being silently reshaped.
	if p.C%16 != 0 {
		return fmt.Errorf("%w: FLSTM C=%d is not a multiple of 16", ErrGPUWeights, p.C)
	}
	if p.K%32 != 0 {
		return fmt.Errorf("%w: FLSTM K=%d is not a multiple of 32", ErrGPUWeights, p.K)
	}
	if _, err := flstmElementCount(4, p.C, 2, p.K); err != nil {
		return err
	}
	return nil
}

func flstmElementCount(shape ...int64) (int64, error) {
	const maxElements = int64(int(^uint(0)>>1)) / 8
	total := int64(1)
	for _, d := range shape {
		if d < 1 || total > maxElements/d {
			return 0, fmt.Errorf("%w: invalid or oversized shape %v", ErrGPUWeights, shape)
		}
		total *= d
	}
	return total, nil
}

// FLSTMWeightSet is one layer's pinned float32 parameters, in the shapes
// FLSTMLayerImpl registers (FLSTMStack.cpp:19-25).
type FLSTMWeightSet struct {
	DownIH   []float32 // [K, C]
	DownHH   []float32 // [K, C]
	UpIH     []float32 // [4C, K]
	UpHH     []float32 // [4C, K]
	UpBiasIH []float32 // [4C]
	// UpBiasHH is loaded and pinned but never packed; it is carried so the
	// consumer can prove the cache does not contain it.
	UpBiasHH []float32 // [4C]
}

// FLSTMCache is the seven device tensors the Koi factorised kernel reads
// (FLSTMStack.cpp:186-217), in the packed form the kernel expects.
type FLSTMCache struct {
	Params FLSTMParams
	// DownIH and DownHH are int8 [K, C]; their scales are binary16 [K].
	DownIH, DownHH           []int8
	DownScaleIH, DownScaleHH []uint16
	// Up is int8 [4C, 2K]; its scale and bias are binary16 [4C].
	Up      []int8
	UpScale []uint16
	UpBias  []uint16
}

// DownShape is the packed down-projection shape.
func (c FLSTMCache) DownShape() []int64 { return []int64{c.Params.K, c.Params.C} }

// UpShape is the packed up-projection shape.
func (c FLSTMCache) UpShape() []int64 { return []int64{c.Params.UpRows(), c.Params.UpCols()} }

// HalfFromFloat32 is the `to(opts_f16)` cast, round to nearest even.
func HalfFromFloat32(v float32) uint16 { return signal.Float16Bits(v) }

// CastFLSTMHalf casts a float32 parameter tensor to binary16 bits.
func CastFLSTMHalf(v []float32) []uint16 {
	out := make([]uint16, len(v))
	for i, x := range v {
		out[i] = signal.Float16Bits(x)
	}
	return out
}

// quantOpts is the resolved form of a mutant: exactly one field differs from the
// source's construction.
type quantOpts struct {
	axis           int
	castToHalf     bool
	epsilon        float64
	clip           int
	truncate       bool
	biasFromHidden bool
}

func optsFor(m FLSTMMutant) quantOpts {
	o := quantOpts{axis: 1, castToHalf: true, clip: FLSTMQuantMax}
	switch m {
	case FLSTMMutantReductionAxis0:
		o.axis = 0
	case FLSTMMutantFloat32Quantize:
		o.castToHalf = false
	case FLSTMMutantEpsilon:
		o.epsilon = 1e-3
	case FLSTMMutantClip128:
		o.clip = 128
	case FLSTMMutantTruncate:
		o.truncate = true
	case FLSTMMutantHiddenBias:
		o.biasFromHidden = true
	}
	return o
}

// quantizeFLSTM is `utils::quantize_tensor(src, dim = o.axis)` for a contiguous
// [rows, cols] tensor: an int8 tensor of the same shape and one float32 scale per
// reduced slice.
//
// Every step is the source's, in the source's dtype:
//
//	fp_range    = amax(|t|, dim)                  (half)
//	quant_scale = (256/2) / fp_range              (half, one rounding)
//	t_quant     = round_half_even(t * quant_scale) (half product, half round)
//	                .clip(-127, 127)              (half)
//	                .to(Char)
//
// The returned scale is `quant_scale.to(Float)`: the exact widening of the half
// scale, not the unrounded quotient.
func quantizeFLSTM(src []float32, rows, cols int64, o quantOpts) ([]int8, []float32, error) {
	elements, err := flstmElementCount(rows, cols)
	if err != nil {
		return nil, nil, err
	}
	if int64(len(src)) != elements {
		return nil, nil, fmt.Errorf("%w: quantize input has %d elements, want %d", ErrGPUWeights, len(src), elements)
	}
	// The value the tensor holds after `to(opts_f16)`: the cast is a rounding
	// step, so the reduction and the product see the half value.
	value := func(i int64) float32 {
		if o.castToHalf {
			return signal.Float32FromHalf(signal.Float16Bits(src[i]))
		}
		return src[i]
	}
	slices := rows
	if o.axis == 0 {
		slices = cols
	}
	q := make([]int8, len(src))
	scale := make([]float32, slices)
	amax := make([]float32, slices)
	for at := range src {
		i := int64(at)
		var s int64
		if o.axis == 0 {
			s = i % cols
		} else {
			s = i / cols
		}
		v := value(i)
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, nil, fmt.Errorf("%w: quantize slice %d holds a non-finite weight", ErrGPUWeights, s)
		}
		if a := float32(math.Abs(float64(v))); a > amax[s] {
			amax[s] = a
		}
	}
	for s := int64(0); s < slices; s++ {
		// `quant_scale = 128 / fp_range`. A zero slice divides by zero and the
		// following Char cast is undefined, so the admitted domain excludes it
		// instead of emitting garbage. Pinned-model native comparisons qualify
		// this division and conversion sequence; they do not prove equivalence
		// over every possible half input.
		if amax[s] == 0 {
			return nil, nil, fmt.Errorf("%w: quantize slice %d is all zeros, so its scale is undefined", ErrGPUWeights, s)
		}
		scaleBits := signal.Float16Bits(float32(128 / (float64(amax[s]) * (1 + o.epsilon))))
		scaleValue := signal.Float32FromHalf(scaleBits)
		if math.IsInf(float64(scaleValue), 0) || math.IsNaN(float64(scaleValue)) {
			return nil, nil, fmt.Errorf("%w: quantize slice %d scale overflows binary16", ErrGPUWeights, s)
		}
		scale[s] = scaleValue
	}
	for i := int64(0); i < int64(len(src)); i++ {
		var s int64
		if o.axis == 0 {
			s = i % cols
		} else {
			s = i / cols
		}
		product := value(i) * scale[s]
		if o.castToHalf {
			// The half product is exact in float32 (11 + 11 significant bits),
			// so rounding it once to half reproduces the tensor product; the
			// integer rounding then applies to the half value, exactly like
			// `.round()` on a half tensor.
			product = signal.Float32FromHalf(signal.Float16Bits(product))
		}
		var rounded float64
		if o.truncate {
			rounded = math.Trunc(float64(product))
		} else {
			rounded = math.RoundToEven(float64(product))
		}
		if rounded > float64(o.clip) {
			rounded = float64(o.clip)
		}
		if rounded < -float64(o.clip) {
			rounded = -float64(o.clip)
		}
		q[i] = int8(rounded)
	}
	return q, scale, nil
}

// QuantizeFLSTMRows is `utils::quantize_tensor(half, dim = 1)` for a contiguous
// [rows, cols] half tensor.
func QuantizeFLSTMRows(half []uint16, rows, cols int64) ([]int8, []float32, error) {
	elements, err := flstmElementCount(rows, cols)
	if err != nil {
		return nil, nil, err
	}
	if int64(len(half)) != elements {
		return nil, nil, fmt.Errorf("%w: quantize input has %d elements, want %d", ErrGPUWeights, len(half), elements)
	}
	values := make([]float32, len(half))
	for i, bits := range half {
		values[i] = signal.Float32FromHalf(bits)
	}
	return quantizeFLSTM(values, rows, cols, optsFor(FLSTMMutantNone))
}

// permuteContiguous is `view(shape).permute(perm).contiguous()` for a row-major
// source of the given shape. The result is row-major in the permuted shape.
func permuteContiguous[T any](src []T, shape []int64, perm []int) ([]T, error) {
	if len(shape) != len(perm) {
		return nil, fmt.Errorf("%w: permute rank %d with %d axes", ErrGPUWeights, len(shape), len(perm))
	}
	total, err := flstmElementCount(shape...)
	if err != nil {
		return nil, err
	}
	if int64(len(src)) != total {
		return nil, fmt.Errorf("%w: permute source has %d elements, shape %v needs %d", ErrGPUWeights, len(src), shape, total)
	}
	strides := make([]int64, len(shape))
	stride := int64(1)
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	seen := make([]bool, len(perm))
	for _, p := range perm {
		if p < 0 || p >= len(perm) || seen[p] {
			return nil, fmt.Errorf("%w: permute %v is not a permutation", ErrGPUWeights, perm)
		}
		seen[p] = true
	}
	out := make([]T, total)
	index := make([]int64, len(shape))
	for flat := int64(0); flat < total; flat++ {
		// Decompose the destination index, map it onto the source, and read.
		rem := flat
		for i := len(perm) - 1; i >= 0; i-- {
			d := shape[perm[i]]
			index[perm[i]] = rem % d
			rem /= d
		}
		var at int64
		for i := range index {
			at += index[i] * strides[i]
		}
		out[flat] = src[at]
	}
	return out, nil
}

// resolveMinusOne fills a single `-1` in a view shape the way ATen does.
func resolveMinusOne(shape []int64, total int64) ([]int64, error) {
	out := make([]int64, len(shape))
	var known int64 = 1
	minus := -1
	for i, d := range shape {
		if d == -1 {
			if minus >= 0 {
				return nil, fmt.Errorf("%w: view shape %v has two inferred dims", ErrGPUWeights, shape)
			}
			minus = i
			continue
		}
		if d < 1 {
			return nil, fmt.Errorf("%w: view shape %v", ErrGPUWeights, shape)
		}
		var err error
		known, err = flstmElementCount(known, d)
		if err != nil {
			return nil, err
		}
	}
	if minus < 0 {
		if known != total {
			return nil, fmt.Errorf("%w: view shape %v holds %d elements, want %d", ErrGPUWeights, shape, known, total)
		}
		copy(out, shape)
		return out, nil
	}
	if total%known != 0 {
		return nil, fmt.Errorf("%w: view shape %v does not divide %d elements", ErrGPUWeights, shape, total)
	}
	copy(out, shape)
	out[minus] = total / known
	return out, nil
}

// concatRows is `torch::cat({a, b}, 1)` for two contiguous [rows, cols] tensors.
func concatRows[T any](a, b []T, rows, cols int64) []T {
	out := make([]T, 0, 2*rows*cols)
	for r := int64(0); r < rows; r++ {
		out = append(out, a[r*cols:(r+1)*cols]...)
		out = append(out, b[r*cols:(r+1)*cols]...)
	}
	return out
}

// permuteForPack resolves a view shape and applies the source's permute, except
// under the identity-permute mutant, which leaves the elements where they are.
func permuteForPack[T any](m FLSTMMutant, src []T, shape []int64, perm []int) ([]T, error) {
	view, err := resolveMinusOne(shape, int64(len(src)))
	if err != nil {
		return nil, err
	}
	if m == FLSTMMutantIdentityPermute {
		return append([]T(nil), src...), nil
	}
	return permuteContiguous(src, view, perm)
}

// FLSTMQuantized is one layer's `utils::quantize_tensor` output before any view
// or permute: exactly the value the source holds in `scaled` and `scaled_up`
// (FLSTMStack.cpp:168-205). The device caches are permutations of these bytes, so
// a differential can compare the quantizer itself rather than only the cache
// derived from it.
type FLSTMQuantized struct {
	// DownIH and DownHH are `Char` [K, C]; their scales are the `Float` tensor
	// `quantize_tensor` returns, one value per row.
	DownIH, DownHH           []int8
	DownScaleIH, DownScaleHH []float32
	// Up is the quantized `cat({up_weight_ih, up_weight_hh}, 1)`, `Char` [4C, 2K].
	Up      []int8
	UpScale []float32
	// UpBias is `up_bias_ih.to(f16)`; `up_bias_hh` is never used.
	UpBias []uint16
}

// QuantizeFLSTMLayer is the source's quantization for one layer, with no
// permutation applied.
func QuantizeFLSTMLayer(p FLSTMParams, w FLSTMWeightSet) (FLSTMQuantized, error) {
	return quantizeFLSTMLayer(p, w, optsFor(FLSTMMutantNone))
}

// quantizeFLSTMLayer is the mutant-aware form. The bias source and every
// quantizer knob come from `o`, so a control cannot silently skip the real code.
func quantizeFLSTMLayer(p FLSTMParams, w FLSTMWeightSet, o quantOpts) (FLSTMQuantized, error) {
	var q FLSTMQuantized
	if err := p.validate(); err != nil {
		return q, err
	}
	down := p.K * p.C
	upInput := p.UpRows() * p.K
	for _, f := range []struct {
		name string
		got  int
		want int64
	}{
		{"dn_weight_ih", len(w.DownIH), down},
		{"dn_weight_hh", len(w.DownHH), down},
		{"up_weight_ih", len(w.UpIH), upInput},
		{"up_weight_hh", len(w.UpHH), upInput},
		{"up_bias_ih", len(w.UpBiasIH), p.UpRows()},
		{"up_bias_hh", len(w.UpBiasHH), p.UpRows()},
	} {
		if int64(f.got) != f.want {
			return q, fmt.Errorf("%w: %s has %d elements, want %d", ErrGPUWeights, f.name, f.got, f.want)
		}
	}
	var err error
	if q.DownIH, q.DownScaleIH, err = quantizeFLSTM(w.DownIH, p.K, p.C, o); err != nil {
		return FLSTMQuantized{}, err
	}
	if q.DownHH, q.DownScaleHH, err = quantizeFLSTM(w.DownHH, p.K, p.C, o); err != nil {
		return FLSTMQuantized{}, err
	}
	joined := concatRows(w.UpIH, w.UpHH, p.UpRows(), p.K)
	if q.Up, q.UpScale, err = quantizeFLSTM(joined, p.UpRows(), p.UpCols(), o); err != nil {
		return FLSTMQuantized{}, err
	}
	biasSource := w.UpBiasIH
	if o.biasFromHidden {
		biasSource = w.UpBiasHH
	}
	q.UpBias = CastFLSTMHalf(biasSource)
	return q, nil
}

// halfs is `scale.to(opts_f16)`.
func halfs(v []float32) []uint16 {
	out := make([]uint16, len(v))
	for i, x := range v {
		out[i] = signal.Float16Bits(x)
	}
	return out
}

// PackFLSTMLayer builds the seven device caches for one layer.
func PackFLSTMLayer(p FLSTMParams, w FLSTMWeightSet) (FLSTMCache, error) {
	return PackFLSTMLayerMutant(p, w, FLSTMMutantNone)
}

// PackFLSTMLayerMutant is PackFLSTMLayer with one source step deliberately
// changed. It exists for the H04 rejection controls; production code always
// passes FLSTMMutantNone.
func PackFLSTMLayerMutant(p FLSTMParams, w FLSTMWeightSet, m FLSTMMutant) (FLSTMCache, error) {
	var c FLSTMCache
	q, err := quantizeFLSTMLayer(p, w, optsFor(m))
	if err != nil {
		return c, err
	}
	c.Params = p
	down := p.K * p.C
	upPacked := p.UpRows() * p.UpCols()

	// --- down projections ---
	downShape := []int64{-1, 2, 2, 2, 2, 2, p.C}
	for i, side := range []struct {
		quantized []int8
		scale     []float32
	}{
		{q.DownIH, q.DownScaleIH},
		{q.DownHH, q.DownScaleHH},
	} {
		packed, err := permuteForPack(m, side.quantized, downShape, []int{2, 0, 4, 1, 3, 5, 6})
		if err != nil {
			return c, err
		}
		packedScale, err := permuteForPack(m, halfs(side.scale), downShape[:6], []int{2, 0, 4, 1, 3, 5})
		if err != nil {
			return c, err
		}
		if i == 0 {
			c.DownIH, c.DownScaleIH = packed, packedScale
		} else {
			c.DownHH, c.DownScaleHH = packed, packedScale
		}
	}

	// --- up projection: already the quantized ih/hh concatenation ---
	if c.Up, err = permuteForPack(m, q.Up, []int64{4, -1, 2, 2, 2, 2, p.UpCols()}, []int{1, 5, 0, 2, 3, 4, 6}); err != nil {
		return c, err
	}
	if c.UpScale, err = permuteForPack(m, halfs(q.UpScale), []int64{4, -1, 2, 2, 2, 2}, []int{1, 5, 2, 3, 4, 0}); err != nil {
		return c, err
	}

	// --- input bias only, cast then permuted with the up scale's map ---
	if c.UpBias, err = permuteForPack(m, q.UpBias, []int64{4, -1, 2, 2, 2, 2}, []int{1, 5, 2, 3, 4, 0}); err != nil {
		return c, err
	}

	// The declared packed extents, checked rather than assumed.
	if int64(len(c.DownIH)) != down || int64(len(c.DownHH)) != down ||
		int64(len(c.DownScaleIH)) != p.K || int64(len(c.DownScaleHH)) != p.K ||
		int64(len(c.Up)) != upPacked || int64(len(c.UpScale)) != p.UpRows() ||
		int64(len(c.UpBias)) != p.UpRows() {
		return FLSTMCache{}, fmt.Errorf("%w: packed FLSTM cache extents are not the declared ones", ErrGPUWeights)
	}
	return c, nil
}

// PackFLSTMStack builds every layer's caches in order.
func PackFLSTMStack(p FLSTMParams, layers []FLSTMWeightSet) ([]FLSTMCache, error) {
	if len(layers) == 0 {
		return nil, fmt.Errorf("%w: no FLSTM layers", ErrGPUWeights)
	}
	out := make([]FLSTMCache, len(layers))
	for i, w := range layers {
		c, err := PackFLSTMLayer(p, w)
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
		out[i] = c
	}
	return out, nil
}
