package tensor

import "fmt"

// KoiActivation mirrors the pinned KoiActivation enum (koi.h:4). The HAC config
// loader maps SWISH and TANH only (ConvStack.cpp:21-31); the other values are
// carried so a rejection control can prove the boundary checks the value.
type KoiActivation uint32

const (
	KoiIdentity   KoiActivation = 0
	KoiSwish      KoiActivation = 1
	KoiSwishClamp KoiActivation = 2
	KoiTanh       KoiActivation = 3
	KoiTanhX5     KoiActivation = 4
)

func (a KoiActivation) valid() bool {
	switch a {
	case KoiIdentity, KoiSwish, KoiSwishClamp, KoiTanh:
		return true
	}
	return false
}

func (a KoiActivation) String() string {
	switch a {
	case KoiIdentity:
		return "identity"
	case KoiSwish:
		return "swish"
	case KoiSwishClamp:
		return "swish_clamp"
	case KoiTanh:
		return "tanh"
	case KoiTanhX5:
		return "tanh_x5"
	}
	return fmt.Sprintf("activation(%d)", uint32(a))
}

// SmallConvGeometry is the exact argument list of the pinned
// host_convolution_f16 call (koi.h:438-451, ConvStack.cpp:243-246). OutNStride
// is `out.stride(0)` of the cropped output view: the parent's batch stride, not
// the cropped extent, which is what keeps the nine rows of last-convolution
// padding in place.
type SmallConvGeometry struct {
	N, CIn, COut, TIn, Window, Stride, Padding int32
	OutNStride                                 int32
	Activation                                 KoiActivation
}

func (g SmallConvGeometry) validate() error {
	if g.N <= 0 || g.CIn <= 0 || g.COut <= 0 || g.TIn <= 0 || g.Window <= 0 || g.Stride <= 0 {
		return &LayoutError{Op: "conv_small_f16", Reason: "geometry must be positive"}
	}
	if g.Padding != g.Window/2 {
		return &LayoutError{Op: "conv_small_f16", Reason: "padding must be window/2"}
	}
	if g.TIn%g.Stride != 0 {
		return &LayoutError{Op: "conv_small_f16", Reason: "input length is not a stride multiple"}
	}
	if g.OutNStride <= 0 {
		return &LayoutError{Op: "conv_small_f16", Reason: "output batch stride must be positive"}
	}
	if !g.Activation.valid() {
		return &LayoutError{Op: "conv_small_f16", Reason: fmt.Sprintf("unsupported Koi activation %d", uint32(g.Activation))}
	}
	return nil
}

func (g SmallConvGeometry) tOut() int32 { return g.TIn / g.Stride }

func storageShape(m StorageMeta) []int64 { return m.Shape }

func equalShape(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// SmallConvF16 runs one pinned host_convolution_f16 call over typed storage.
// Every argument is checked against the source contract before the native call;
// the native boundary repeats the checks so a Go-side gap cannot reach the
// kernel.
func (d *Device) SmallConvF16(s *Stream, in, weights, bias, out *Storage, g SmallConvGeometry) error {
	if d == nil || d.state == nil {
		return ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.state.valid(); err != nil {
		return err
	}
	if err := g.validate(); err != nil {
		return err
	}
	for _, h := range []*Storage{in, weights, bias, out} {
		if h == nil || h.state == nil {
			return fmt.Errorf("conv_small_f16: %w", ErrClosed)
		}
		if err := h.state.check(s, "conv_small_f16"); err != nil {
			return err
		}
		if h.state.dtype != DTypeF16 {
			return &DTypeError{Op: "conv_small_f16", Reason: "small convolution requires F16 storage", Detail: CUDADetail{Handle: h.state.id, ActualDType: ScalarType(h.state.dtype), ExpectedDType: ScalarType(DTypeF16)}}
		}
	}
	meta := func(h *Storage, role string) (StorageMeta, error) {
		m, err := e.backend.cuda(cudaRequest{op: "storage_meta", storage: h.state.id})
		if err != nil {
			return StorageMeta{}, fmt.Errorf("%s: %w", role, err)
		}
		return m.meta, nil
	}
	im, err := meta(in, "conv input")
	if err != nil {
		return err
	}
	wm, err := meta(weights, "conv weights")
	if err != nil {
		return err
	}
	bm, err := meta(bias, "conv bias")
	if err != nil {
		return err
	}
	om, err := meta(out, "conv output")
	if err != nil {
		return err
	}
	if !equalShape(storageShape(im), []int64{int64(g.N), int64(g.TIn), int64(g.CIn)}) {
		return &LayoutError{Op: "conv_small_f16", Reason: fmt.Sprintf("input shape %v is not N,T_in,C_in", im.Shape)}
	}
	if !equalShape(storageShape(wm), []int64{int64(g.Window) * int64(g.CIn), int64(g.COut)}) {
		return &LayoutError{Op: "conv_small_f16", Reason: fmt.Sprintf("weight shape %v is not W*C_in,C_out", wm.Shape)}
	}
	if !equalShape(storageShape(bm), []int64{int64(g.COut)}) {
		return &LayoutError{Op: "conv_small_f16", Reason: fmt.Sprintf("bias shape %v is not C_out", bm.Shape)}
	}
	if !equalShape(storageShape(om), []int64{int64(g.N), int64(g.tOut()), int64(g.COut)}) {
		return &LayoutError{Op: "conv_small_f16", Reason: fmt.Sprintf("output shape %v is not N,T_out,C_out", om.Shape)}
	}
	if len(om.Strides) != 3 || om.Strides[0] != int64(g.OutNStride) || om.Strides[1] != int64(g.COut) || om.Strides[2] != 1 {
		return &LayoutError{Op: "conv_small_f16", Reason: fmt.Sprintf("output strides %v do not match batch stride %d and unit final stride", om.Strides, g.OutNStride)}
	}
	return d.smallConvF16(s, in, weights, bias, out, g)
}

// SmallConvF16Unchecked forwards the exact geometry to the native boundary
// without Go-side validation. It exists only so a rejection control can prove
// the native guard itself fires; production code must use SmallConvF16.
func (d *Device) SmallConvF16Unchecked(s *Stream, in, weights, bias, out *Storage, g SmallConvGeometry) error {
	if d == nil || d.state == nil {
		return ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.state.valid(); err != nil {
		return err
	}
	return d.smallConvF16(s, in, weights, bias, out, g)
}

func (d *Device) smallConvF16(s *Stream, in, weights, bias, out *Storage, g SmallConvGeometry) error {
	if s == nil || s.state == nil {
		return fmt.Errorf("conv_small_f16: %w", ErrClosed)
	}
	if err := s.state.valid("conv_small_f16"); err != nil {
		return err
	}
	_, err := d.state.engine.backend.cuda(cudaRequest{
		op: "conv_small_f16", stream: s.state.id,
		in: in.state.id, weights: weights.state.id, bias: bias.state.id, out: out.state.id,
		n: g.N, cIn: g.CIn, cOut: g.COut, tIn: g.TIn,
		window: g.Window, stride: g.Stride, padding: g.Padding, outNStride: g.OutNStride,
		activation: uint32(g.Activation),
	})
	return err
}

// ReferencePackSmallConvF16 is the verification-only packing oracle. It
// reproduces ConvStack.cpp:230-233 exactly:
//
//	conv->weight.permute({2, 1, 0}).contiguous().flatten(0, 1).to(kF16)
//	conv->bias.to(kF16)
//
// The Go caller packs weights itself; the consumer compares the two byte for
// byte. This is never part of the caller path.
func (d *Device) ReferencePackSmallConvF16(s *Stream, weight, bias, packed, packedBias *Storage) error {
	if d == nil || d.state == nil {
		return ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.state.valid(); err != nil {
		return err
	}
	if s == nil || s.state == nil {
		return fmt.Errorf("weights_permute_cast_f16: %w", ErrClosed)
	}
	if err := s.state.valid("weights_permute_cast_f16"); err != nil {
		return err
	}
	for _, h := range []*Storage{weight, bias, packed, packedBias} {
		if h == nil || h.state == nil {
			return fmt.Errorf("weights_permute_cast_f16: %w", ErrClosed)
		}
		if err := h.state.check(s, "weights_permute_cast_f16"); err != nil {
			return err
		}
	}
	_, err := e.backend.cuda(cudaRequest{
		op: "weights_permute_cast_f16", stream: s.state.id,
		weights: weight.state.id, bias: bias.state.id, out: packed.state.id, storage: packedBias.state.id,
	})
	return err
}
