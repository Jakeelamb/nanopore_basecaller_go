package model

import (
	"fmt"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

// This file is the host-side port of the HAC last-convolution path
// (nn/ConvStack.cpp:264-322, the `cutlass_conv` branch). Go owns the input
// padding writes, the weight transpose into [C_out, W*C_in], the CUTLASS output
// view and its two strides, the dev-option dispatch, and the decision to fall
// back to host_linear. The only native work is one named primitive per attempt.
//
// The `!cutlass_conv` window/half-GEMM branch (ConvStack.cpp:323-373) is a
// different selection the pinned config never takes (cutlass_conv defaults to
// true). It is refused explicitly rather than approximated, and H01 captured no
// stage for it.

// LastConvDevOptions carries the source developer options that decide the
// CUTLASS branch's dispatch (ConvStack.cpp:143,280-288). The zero value is not
// useful: use DefaultLastConvDevOptions, which is what the pinned source does.
type LastConvDevOptions struct {
	// CutlassConv is `cutlass_conv` (ConvStack.cpp:143).
	CutlassConv bool
	// KoiConv is `koi_conv`: try host_convolution first.
	KoiConv bool
	// KoiConvF32 is `koi_conv_f32`: accumulate in F32 instead of F16.
	KoiConvF32 bool
}

// DefaultLastConvDevOptions returns the pinned source's defaults:
// cutlass_conv=true, koi_conv=true, koi_conv_f32=false.
func DefaultLastConvDevOptions() LastConvDevOptions {
	return LastConvDevOptions{CutlassConv: true, KoiConv: true}
}

func (o LastConvDevOptions) String() string {
	return fmt.Sprintf("cutlass_conv=%t koi_conv=%t koi_conv_f32=%t", o.CutlassConv, o.KoiConv, o.KoiConvF32)
}

// LastConvPath names which primitive produced the result. A fallback success is
// not fast-path support, so the path is recorded, never inferred.
type LastConvPath string

const (
	// LastConvFast is `host_convolution` (ConvStack.cpp:289-292).
	LastConvFast LastConvPath = "host_convolution"
	// LastConvFallback is `host_linear` (ConvStack.cpp:293-300).
	LastConvFallback LastConvPath = "host_linear"
)

// HACLastConvPlan binds the final convolution to the allocation it reads and the
// allocation it writes. Both are views of the same whole-model reservation.
type HACLastConvPlan struct {
	Layer HACSmallConvLayer
	// Input is the previous layer's allocation, which is `wm.current` when the
	// last layer runs (ConvStack.cpp:216).
	Input            GPUStorageGeometry
	InputOffsetBytes int64
	// Output is this layer's own allocation, `wm.next_TC(T_out, C_out, layout)`.
	Output            GPUStorageGeometry
	OutputOffsetBytes int64
	// InputToRevLSTM selects the `narrow(0, 1, T)` branch of
	// get_current_NTC_view (WorkingMemory.cpp:62-80).
	InputToRevLSTM bool
}

// PlanHACLastConv derives the last-convolution plan from the conv-stack plan.
// The last layer's input length is `wm.T - 2*padding`, not the previous layer's
// T_out: those agree here because the previous layer has stride 1, and the plan
// checks that they do rather than assuming it.
func PlanHACLastConv(plan *HACSmallConvPlan, inputToRevLSTM bool) (*HACLastConvPlan, error) {
	if plan == nil || len(plan.Layers) < 2 {
		return nil, fmt.Errorf("%w: the last convolution needs a preceding layer", ErrGPUPlanLayout)
	}
	last := plan.Layers[len(plan.Layers)-1]
	prev := plan.Layers[len(plan.Layers)-2]
	if last.Params.OutputLayout != GPUStorageCUTLASSI8 && last.Params.OutputLayout != GPUStorageCUTLASSF16 {
		return nil, fmt.Errorf("%w: last layer layout %s is not a CUTLASS layout", ErrGPUPlanLayout, last.Params.OutputLayout)
	}
	// ConvStack.cpp:221: T_in = wm.T - 2 * padding. wm.T is the previous
	// allocation's row count.
	padding := last.Params.Padding()
	if last.TIn != prev.Geometry.Rows-2*padding {
		return nil, fmt.Errorf("%w: last layer T_in %d is not wm.T %d minus 2*padding %d",
			ErrGPUPlanLayout, last.TIn, prev.Geometry.Rows, padding)
	}
	if prev.Geometry.C != last.Params.CIn {
		return nil, fmt.Errorf("%w: last layer C_in %d does not follow the previous allocation's %d channels",
			ErrGPUPlanLayout, last.Params.CIn, prev.Geometry.C)
	}
	if prev.Geometry.N != last.Geometry.N {
		return nil, fmt.Errorf("%w: last layer batch %d differs from the input's %d",
			ErrGPUPlanLayout, last.Geometry.N, prev.Geometry.N)
	}
	return &HACLastConvPlan{
		Layer: last, Input: prev.Geometry, InputOffsetBytes: prev.OffsetBytes,
		Output: last.Geometry, OutputOffsetBytes: last.OffsetBytes,
		InputToRevLSTM: inputToRevLSTM,
	}, nil
}

// startRow is the `narrow(0, start, T)` offset of the CUTLASS NTC view:
// `is_input_to_rev_lstm ? 1 : 2` (WorkingMemory.cpp:68-70).
func (p *HACLastConvPlan) startRow() int64 {
	if p.InputToRevLSTM {
		return 1
	}
	return 2
}

// OutputViewOffsetElements is where the transposed view starts inside the
// output allocation: startRow rows of N*C_out elements.
func (p *HACLastConvPlan) OutputViewOffsetElements() uint64 {
	return uint64(p.startRow() * p.Output.N * p.Output.C)
}

// OutputViewShape is [N, T_out, C_out] after the transpose.
func (p *HACLastConvPlan) OutputViewShape() []int64 {
	return []int64{p.Output.N, p.Layer.TOut, p.Output.C}
}

// OutputViewStrides is the transposed parent's stride pair: the N stride is
// C_out and the T stride is N*C_out, so the view is not contiguous.
func (p *HACLastConvPlan) OutputViewStrides() []int64 {
	return []int64{p.Output.C, p.Output.N * p.Output.C, 1}
}

// UnwrittenRows is the set of allocation time rows the view never covers. It is
// derived from the source layout before execution and must agree with the
// capture's declared `unwritten_t`; it is never inferred from observed bytes.
func (p *HACLastConvPlan) UnwrittenRows() [][2]int {
	start := int(p.startRow())
	rows := int(p.Output.Rows)
	end := start + int(p.Layer.TOut)
	var out [][2]int
	if start > 0 {
		out = append(out, [2]int{0, start})
	}
	if end < rows {
		out = append(out, [2]int{end, rows - end})
	}
	return out
}

// Geometry is the exact argument set both primitives take.
func (p *HACLastConvPlan) Geometry(dev LastConvDevOptions) (tensor.LastConvGeometry, error) {
	act, err := p.Layer.Params.Activation.KoiActivation()
	if err != nil {
		return tensor.LastConvGeometry{}, err
	}
	outType := tensor.KoiI8
	if p.Output.Dtype == GPUDtypeHalf {
		outType = tensor.KoiF16
	}
	accum := tensor.KoiF16
	if dev.KoiConvF32 {
		accum = tensor.KoiF32
	}
	return tensor.LastConvGeometry{
		N: int32(p.Output.N), TIn: int32(p.Layer.TIn), CIn: int32(p.Layer.Params.CIn),
		COut: int32(p.Layer.Params.COut), Window: int32(p.Layer.Params.WinLen),
		Stride: int32(p.Layer.Params.Stride), Padding: int32(p.Layer.Params.Padding()),
		InNStride: int32(p.Input.Rows * p.Input.C), OutNStride: int32(p.Output.C),
		OutTStride: int32(p.Output.N * p.Output.C), Activation: act,
		InType: tensor.KoiF16, OutType: outType, AccumType: accum,
	}, nil
}

// PackLastConvWeights is the Go port of
//
//	conv->weight.permute({2, 1, 0}).contiguous().flatten(0, 1).to(kF16)
//	.then(transpose(0, 1).contiguous())   (ConvStack.cpp:230-236)
//
// weight is the pinned [C_out, C_in, W] row-major float32 tensor. The result is
// the [C_out, W*C_in] row-major half tensor `host_convolution` and `host_linear`
// both expect, so element [co][w][ci] is written at co*(W*C_in)+w*C_in+ci.
func PackLastConvWeights(weight []float32, cOut, cIn, win int64) ([]uint16, error) {
	if cOut < 1 || cIn < 1 || win < 1 {
		return nil, fmt.Errorf("%w: weight geometry C_out=%d C_in=%d W=%d", ErrGPUPlanLayout, cOut, cIn, win)
	}
	if int64(len(weight)) != cOut*cIn*win {
		return nil, fmt.Errorf("%w: weight has %d elements, want %d", ErrGPUPlanLayout, len(weight), cOut*cIn*win)
	}
	out := make([]uint16, cOut*cIn*win)
	for co := int64(0); co < cOut; co++ {
		for ci := int64(0); ci < cIn; ci++ {
			for w := int64(0); w < win; w++ {
				src := (co*cIn+ci)*win + w
				dst := co*(win*cIn) + w*cIn + ci
				out[dst] = signal.Float16Bits(weight[src])
			}
		}
	}
	return out, nil
}

// ZeroInputPadding reproduces ConvStack.cpp:265-266:
//
//	in.slice(1, 0, padding) = 0;
//	in.slice(1, -padding, None) = 0;
//
// Only one batch's rows are contiguous in an NTC allocation, so each band is
// zeroed through its own contiguous [padding, C] view rather than one strided
// view. Nothing else in the allocation is touched.
func ZeroInputPadding(s *tensor.Stream, alloc *tensor.Storage, batch, rows, c, padding int64) error {
	if padding <= 0 {
		return nil
	}
	if batch < 1 || rows < 2*padding || c < 1 {
		return fmt.Errorf("%w: padding %d does not fit %d rows", ErrGPUPlanLayout, padding, rows)
	}
	zero := []byte{0, 0}
	for _, band := range inputPaddingBands(batch, rows, c, padding) {
		v, err := alloc.View(band[0], []int64{padding, c}, []int64{c, 1},
			[]tensor.KoiTag{tensor.TagT, tensor.TagC})
		if err != nil {
			return err
		}
		if err = v.Fill(s, zero); err != nil {
			_ = v.Close()
			return err
		}
		if err = v.Close(); err != nil {
			return err
		}
	}
	return nil
}

// inputPaddingBands returns the element offset and element count of each
// contiguous band ZeroInputPadding zeroes, in batch-major order.
func inputPaddingBands(batch, rows, c, padding int64) [][2]uint64 {
	out := make([][2]uint64, 0, 2*batch)
	for n := int64(0); n < batch; n++ {
		for _, start := range []int64{0, rows - padding} {
			out = append(out, [2]uint64{uint64((n*rows + start) * c), uint64(padding * c)})
		}
	}
	return out
}

// HACLastConv owns the final layer's device weight cache, packed once and
// reused like ConvLayer::w_device/b_device (ConvStack.cpp:228-236).
type HACLastConv struct {
	Params  SmallConvParams
	Dev     LastConvDevOptions
	device  *tensor.Device
	Weights *tensor.Storage
	Bias    *tensor.Storage
}

// NewHACLastConv packs the pinned float32 weight/bias into device F16 storage.
func NewHACLastConv(d *tensor.Device, s *tensor.Stream, p SmallConvParams, dev LastConvDevOptions, weight, bias []float32) (*HACLastConv, error) {
	if !dev.CutlassConv {
		return nil, fmt.Errorf("%w: cutlass_conv=false selects the window/half-GEMM path, which this card does not port", ErrGPUPlanMismatch)
	}
	packed, err := PackLastConvWeights(weight, p.COut, p.CIn, p.WinLen)
	if err != nil {
		return nil, err
	}
	packedBias, err := PackSmallConvBias(bias)
	if err != nil {
		return nil, err
	}
	if int64(len(bias)) != p.COut {
		return nil, fmt.Errorf("%w: bias has %d elements, want C_out %d", ErrGPUPlanLayout, len(bias), p.COut)
	}
	w, err := d.NewStorage(s, tensor.DTypeF16, []int64{p.COut, p.WinLen * p.CIn},
		[]tensor.KoiTag{tensor.TagC, tensor.TagW}, HalfBytes(packed))
	if err != nil {
		return nil, err
	}
	b, err := d.NewStorage(s, tensor.DTypeF16, []int64{p.COut},
		[]tensor.KoiTag{tensor.TagC}, HalfBytes(packedBias))
	if err != nil {
		_ = w.Close()
		return nil, err
	}
	return &HACLastConv{Params: p, Dev: dev, device: d, Weights: w, Bias: b}, nil
}

// Close releases the device weight cache.
func (c *HACLastConv) Close() error {
	if c == nil {
		return nil
	}
	var err error
	if c.Bias != nil {
		err = c.Bias.Close()
	}
	if c.Weights != nil {
		if e := c.Weights.Close(); e != nil {
			err = e
		}
	}
	return err
}

// LastConvDispatch records both attempts of the source's dispatch, not just the
// winner. A fallback success is not fast-path support, so which primitive ran
// and what each returned are kept apart.
type LastConvDispatch struct {
	FastAttempted     bool
	FastResult        tensor.KoiResult
	FallbackAttempted bool
	FallbackResult    tensor.KoiResult
	Path              LastConvPath
}

// Run executes the CUTLASS branch with the source's dispatch: try
// host_convolution, and only on a non-success KoiResult try host_linear. A
// non-success fast result is not an error; a non-success fallback result is,
// exactly like the source's throw (ConvStack.cpp:301-303). The returned
// dispatch says which primitive actually produced the output.
func (c *HACLastConv) Run(s *tensor.Stream, in, out *tensor.Storage, p *HACLastConvPlan) (LastConvDispatch, error) {
	var d LastConvDispatch
	d.FastResult = tensor.KoiResult(-1)
	d.FallbackResult = tensor.KoiResult(-1)
	if c == nil {
		return d, fmt.Errorf("model: nil last convolution")
	}
	if !c.Dev.CutlassConv {
		return d, fmt.Errorf("%w: cutlass_conv=false selects the window/half-GEMM path, which this card does not port", ErrGPUPlanMismatch)
	}
	if p.Layer.Index < 0 || p.Layer.Params.CIn != c.Params.CIn ||
		p.Layer.Params.COut != c.Params.COut || p.Layer.Params.WinLen != c.Params.WinLen ||
		p.Layer.Params.Stride != c.Params.Stride {
		return d, fmt.Errorf("%w: layer %d does not match the packed weights", ErrGPUPlanMismatch, p.Layer.Index)
	}
	g, err := p.Geometry(c.Dev)
	if err != nil {
		return d, err
	}
	if c.Dev.KoiConv {
		d.FastAttempted = true
		d.FastResult, err = c.device.LastConvKoi(s, in, c.Weights, c.Bias, out, g)
		if err != nil {
			return d, err
		}
	}
	if !d.FastAttempted || d.FastResult != tensor.KoiSuccess {
		d.FallbackAttempted = true
		d.FallbackResult, err = c.device.LastConvLinear(s, in, c.Weights, c.Bias, out, g)
		if err != nil {
			return d, err
		}
		if d.FallbackResult != tensor.KoiSuccess {
			return d, fmt.Errorf("model: Koi convolution (host_linear) failed with in size %d: %s",
				c.Params.CIn*c.Params.WinLen, d.FallbackResult)
		}
		d.Path = LastConvFallback
		return d, nil
	}
	d.Path = LastConvFast
	return d, nil
}
