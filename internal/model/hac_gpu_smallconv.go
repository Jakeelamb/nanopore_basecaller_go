package model

import (
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

// This file is the host-side port of the HAC small-convolution path
// (nn/ConvStack.cpp:143-148,191-263). Go owns the weight packing, the working
// memory views, the crop that keeps the nine rows of last-convolution padding,
// and the single host_convolution_f16 call per layer. The only native work is
// the named primitive behind the typed boundary.
//
// It plans and runs two convolutions. It does not execute the final CUTLASS
// convolution or any recurrent layer, and it is not end-to-end basecalling.

// SmallConvActivation is the pinned config activation of a convolution layer
// (config/common.h). Only the two values the HAC graph uses are carried.
type SmallConvActivation uint8

const (
	SmallConvSwish SmallConvActivation = iota + 1
	SmallConvTanh
)

func (a SmallConvActivation) String() string {
	switch a {
	case SmallConvSwish:
		return "swish"
	case SmallConvTanh:
		return "tanh"
	}
	return fmt.Sprintf("activation(%d)", uint8(a))
}

// KoiActivation maps the config activation the way get_koi_activation does
// (ConvStack.cpp:21-31). An unknown activation is a hard error, never a default.
func (a SmallConvActivation) KoiActivation() (tensor.KoiActivation, error) {
	switch a {
	case SmallConvSwish:
		return tensor.KoiSwish, nil
	case SmallConvTanh:
		return tensor.KoiTanh, nil
	}
	return 0, fmt.Errorf("model: unrecognised convolution activation %d", uint8(a))
}

// SmallConvParams is one convolution sublayer of the pinned config.
type SmallConvParams struct {
	CIn, COut, WinLen, Stride int64
	Activation                SmallConvActivation
	// OutputLayout is the layout reserve_working_memory assigns to this layer
	// (ConvStack.cpp:138-148). HAC v6 uses NTC for the two width-5 layers and
	// CUTLASS_TNC_I8 for the final width-19 layer.
	OutputLayout GPUStorageLayout
	// OutputTPadding is the number of unwritten rows on each side of the
	// allocation. The stack sets it only on the layer before a CUTLASS layer
	// (ConvStack.cpp:147).
	OutputTPadding int64
}

// Padding is the input padding the source passes to the kernel
// (`params.winlen / 2`, ConvStack.cpp:223).
func (p SmallConvParams) Padding() int64 { return p.WinLen / 2 }

// HACSmallConvLayer is one planned convolution. Geometry is the allocation
// `wm.next_TC(T_out + 2*OutputTPadding, C_out, layout)` reserves; Crop is the
// `narrow(1, OutputTPadding, T_out)` view the kernel writes.
type HACSmallConvLayer struct {
	Index    int
	Params   SmallConvParams
	TIn      int64
	TOut     int64
	Geometry GPUStorageGeometry
	// OffsetBytes is where next() places this allocation in the backing:
	// 0 at the front, reservation-bytes at the back (WorkingMemory.cpp:46-49).
	OffsetBytes int64
	// Executes is true for the two layers H02 runs.
	Executes bool
}

// CropOffsetElements is the offset of the cropped view inside the layer
// allocation: OutputTPadding rows of C_out elements.
func (l HACSmallConvLayer) CropOffsetElements() uint64 {
	return uint64(l.Params.OutputTPadding * l.Params.COut)
}

// CropShape is the cropped view shape [N, T_out, C_out].
func (l HACSmallConvLayer) CropShape() []int64 {
	return []int64{l.Geometry.N, l.TOut, l.Params.COut}
}

// CropStrides keeps the allocation's batch stride so the nine padding rows stay
// between batches (ConvStack.cpp:241-242). It is not contiguous when
// OutputTPadding is nonzero.
func (l HACSmallConvLayer) CropStrides() []int64 {
	return []int64{l.Geometry.Rows * l.Params.COut, l.Params.COut, 1}
}

// BatchStride is `out.stride(0)` of the cropped view, the value the kernel is
// given as out_N_stride.
func (l HACSmallConvLayer) BatchStride() int64 { return l.Geometry.Rows * l.Params.COut }

// SmallConvGeometry is the exact host_convolution_f16 argument list for this
// layer.
func (l HACSmallConvLayer) SmallConvGeometry() (tensor.SmallConvGeometry, error) {
	act, err := l.Params.Activation.KoiActivation()
	if err != nil {
		return tensor.SmallConvGeometry{}, err
	}
	return tensor.SmallConvGeometry{
		N: int32(l.Geometry.N), CIn: int32(l.Params.CIn), COut: int32(l.Params.COut),
		TIn: int32(l.TIn), Window: int32(l.Params.WinLen), Stride: int32(l.Params.Stride),
		Padding: int32(l.Params.Padding()), OutNStride: int32(l.BatchStride()),
		Activation: act,
	}, nil
}

// HACSmallConvPlan is the conv-stack portion of the model's WorkingMemory
// traversal. ReservationBytes is the whole-model reservation (H01 records it);
// the front/back offsets are derived from it exactly as WorkingMemory::next
// does, so the conv views land where the native graph puts them.
type HACSmallConvPlan struct {
	Batch            int64
	Chunk            int64
	Features         int64
	ReservationBytes int64
	Input            GPUStorageGeometry
	InputOffsetBytes int64
	Layers           []HACSmallConvLayer
}

// PlanHACSmallConvs derives the conv-stack plan from the config geometry and the
// whole-model reservation. The final layer is planned but not executed: its
// layout is what gives the second layer its nine rows of padding.
func PlanHACSmallConvs(batch, chunk, features, reservation int64, layers []SmallConvParams) (*HACSmallConvPlan, error) {
	if batch < 1 || chunk < 1 || features < 1 || reservation < 1 {
		return nil, fmt.Errorf("%w: batch %d chunk %d features %d reservation %d", ErrGPUPlanLayout, batch, chunk, features, reservation)
	}
	if len(layers) < 2 {
		return nil, fmt.Errorf("%w: the HAC conv stack needs at least two layers", ErrGPUPlanLayout)
	}
	for i := range layers {
		if layers[i].CIn < 1 || layers[i].COut < 1 || layers[i].WinLen < 1 || layers[i].Stride < 1 {
			return nil, fmt.Errorf("%w: conv layer %d geometry", ErrGPUPlanLayout, i)
		}
		if layers[i].OutputTPadding < 0 {
			return nil, fmt.Errorf("%w: conv layer %d negative output padding", ErrGPUPlanLayout, i)
		}
	}
	last := &layers[len(layers)-1]
	cutlass := last.OutputLayout == GPUStorageCUTLASSI8 || last.OutputLayout == GPUStorageCUTLASSF16
	// ConvStack.cpp:147: only the layer before a CUTLASS layer is padded, and it
	// is padded by the last layer's window/2.
	if cutlass {
		layers[len(layers)-2].OutputTPadding = last.WinLen / 2
	}

	input, err := NewGPUStorageGeometry(GPUStorageNTC, batch, chunk, features)
	if err != nil {
		return nil, err
	}
	if input.Bytes > reservation {
		return nil, fmt.Errorf("%w: input extent %d exceeds reservation %d", ErrGPUPlanUndersized, input.Bytes, reservation)
	}
	plan := &HACSmallConvPlan{Batch: batch, Chunk: chunk, Features: features, ReservationBytes: reservation, Input: input, InputOffsetBytes: 0}

	// WorkingMemory::next alternates ends. The input is the first view (front),
	// so the first convolution lands at the back.
	currentAtFront := true
	tIn := chunk
	for i := range layers {
		p := layers[i]
		if p.CIn != plan.LayersC(i) {
			return nil, fmt.Errorf("%w: conv layer %d C_in %d does not follow the previous layer", ErrGPUPlanLayout, i, p.CIn)
		}
		if tIn%p.Stride != 0 {
			return nil, fmt.Errorf("%w: conv layer %d input %d is not a stride multiple", ErrGPUPlanLayout, i, tIn)
		}
		tOut := tIn / p.Stride
		allocRows := tOut + 2*p.OutputTPadding
		geom, err := NewGPUStorageGeometry(p.OutputLayout, batch, allocRows, p.COut)
		if err != nil {
			return nil, err
		}
		atFront := !currentAtFront
		offset := int64(0)
		if !atFront {
			offset = reservation - geom.Bytes
			if offset < 0 {
				return nil, fmt.Errorf("%w: conv layer %d extent %d exceeds reservation %d", ErrGPUPlanUndersized, i, geom.Bytes, reservation)
			}
		}
		if geom.Bytes > reservation {
			return nil, fmt.Errorf("%w: conv layer %d extent %d exceeds reservation %d", ErrGPUPlanUndersized, i, geom.Bytes, reservation)
		}
		plan.Layers = append(plan.Layers, HACSmallConvLayer{
			Index: i, Params: p, TIn: tIn, TOut: tOut, Geometry: geom,
			OffsetBytes: offset, Executes: i < 2,
		})
		currentAtFront = atFront
		tIn = tOut
	}
	return plan, nil
}

// LayersC is the channel count the previous layer leaves current. The input
// layer has Features channels.
func (p *HACSmallConvPlan) LayersC(i int) int64 {
	if i == 0 {
		return p.Features
	}
	return p.Layers[i-1].Params.COut
}

// PackSmallConvWeights is the Go port of
//
//	conv->weight.permute({2, 1, 0}).contiguous().flatten(0, 1).to(kF16)
//	(ConvStack.cpp:230-233)
//
// weight is the pinned [C_out, C_in, W] row-major float32 tensor. The result is
// the [W*C_in, C_out] row-major half tensor the kernel expects, so element
// [w][ci][co] is written at (w*C_in+ci)*C_out+co.
func PackSmallConvWeights(weight []float32, cOut, cIn, win int64) ([]uint16, error) {
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
				dst := (w*cIn+ci)*cOut + co
				out[dst] = signal.Float16Bits(weight[src])
			}
		}
	}
	return out, nil
}

// PackSmallConvBias is `conv->bias.to(kF16)` (ConvStack.cpp:234).
func PackSmallConvBias(bias []float32) ([]uint16, error) {
	if len(bias) == 0 {
		return nil, fmt.Errorf("%w: empty bias", ErrGPUPlanLayout)
	}
	out := make([]uint16, len(bias))
	for i, v := range bias {
		out[i] = signal.Float16Bits(v)
	}
	return out, nil
}

// HalfBytes encodes half bit patterns as little-endian bytes, the storage
// upload form.
func HalfBytes(bits []uint16) []byte {
	out := make([]byte, len(bits)*2)
	for i, v := range bits {
		binary.LittleEndian.PutUint16(out[i*2:], v)
	}
	return out
}

// HACSmallConv owns one layer's device weight cache. The cache is packed once
// and reused, exactly like ConvLayer::w_device/b_device
// (ConvStack.cpp:228-235).
type HACSmallConv struct {
	Params  SmallConvParams
	device  *tensor.Device
	Weights *tensor.Storage
	Bias    *tensor.Storage
}

// NewHACSmallConv packs the pinned float32 weight/bias into device F16 storage.
func NewHACSmallConv(d *tensor.Device, s *tensor.Stream, p SmallConvParams, weight, bias []float32) (*HACSmallConv, error) {
	packed, err := PackSmallConvWeights(weight, p.COut, p.CIn, p.WinLen)
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
	w, err := d.NewStorage(s, tensor.DTypeF16, []int64{p.WinLen * p.CIn, p.COut},
		[]tensor.KoiTag{tensor.TagW, tensor.TagC}, HalfBytes(packed))
	if err != nil {
		return nil, err
	}
	b, err := d.NewStorage(s, tensor.DTypeF16, []int64{p.COut},
		[]tensor.KoiTag{tensor.TagC}, HalfBytes(packedBias))
	cache := &HACSmallConv{Params: p, device: d, Weights: w, Bias: b}
	if err != nil {
		cleanup := cache.Close()
		if cleanup == nil {
			cache = nil
		}
		return cache, errors.Join(err, cleanup)
	}
	return cache, nil
}

// Close releases the device weight cache.
func (c *HACSmallConv) Close() error {
	if c == nil {
		return nil
	}
	var err error
	for _, slot := range []**tensor.Storage{&c.Bias, &c.Weights} {
		if *slot != nil {
			if e := (*slot).Close(); e != nil {
				err = errors.Join(err, e)
			} else {
				*slot = nil
			}
		}
	}
	return err
}

// Run executes one convolution into the cropped output view. The view must have
// the plan's crop shape and batch stride; both Go and the native boundary check
// it.
func (c *HACSmallConv) Run(s *tensor.Stream, in, out *tensor.Storage, l HACSmallConvLayer) error {
	if c == nil {
		return fmt.Errorf("model: nil small convolution")
	}
	if l.Index < 0 || l.Params.CIn != c.Params.CIn || l.Params.COut != c.Params.COut ||
		l.Params.WinLen != c.Params.WinLen || l.Params.Stride != c.Params.Stride {
		return fmt.Errorf("%w: layer %d does not match the packed weights", ErrGPUPlanMismatch, l.Index)
	}
	g, err := l.SmallConvGeometry()
	if err != nil {
		return err
	}
	return c.device.SmallConvF16(s, in, c.Weights, c.Bias, out, g)
}
