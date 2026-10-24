package tensor

import "fmt"

// KoiResult mirrors the pinned KoiResult enum (koi.h:5-10). The CUTLASS branch
// of ConvLayer::run_koi dispatches on it: a non-success result from
// `host_convolution` selects the `host_linear` fallback (ConvStack.cpp:291-300).
type KoiResult int32

const (
	KoiSuccess       KoiResult = 0
	KoiNotSupported  KoiResult = 1
	KoiInvalidValue  KoiResult = 2
	KoiInternalError KoiResult = 3
)

func (r KoiResult) String() string {
	switch r {
	case KoiSuccess:
		return "KOI_SUCCESS"
	case KoiNotSupported:
		return "KOI_NOT_SUPPORTED"
	case KoiInvalidValue:
		return "KOI_INVALID_VALUE"
	case KoiInternalError:
		return "KOI_INTERNAL_ERROR"
	}
	return fmt.Sprintf("KoiResult(%d)", int32(r))
}

// LastConvGeometry is the exact argument set shared by the CUTLASS branch's two
// primitives: `host_convolution` (the fast path) and `host_linear` (its
// fallback) at ConvStack.cpp:289-300.
//
// InNStride is `in.stride(0)` of the contiguous [N, T_in + 2*padding, C_in]
// allocation; OutNStride and OutTStride are the two strides of the transposed
// CUTLASS view, which are (C_out, T_out * N * C_out) and not the contiguous
// pair. Both primitives write through those strides, which is why the output
// view is deliberately non-contiguous.
type LastConvGeometry struct {
	N, TIn, CIn, COut, Window, Stride, Padding int32
	InNStride, OutNStride, OutTStride          int32
	Activation                                 KoiActivation
	InType, OutType, AccumType                 KoiDType
}

func (g LastConvGeometry) tOut() int32 { return g.TIn / g.Stride }

// k is the flattened window width both primitives take as their K axis
// (`C_in * winlen`, ConvStack.cpp:295).
func (g LastConvGeometry) k() int32 { return g.Window * g.CIn }

// inStride1 is the window stride in elements: `stride * C_in`
// (ConvStack.cpp:296).
func (g LastConvGeometry) inStride1() int32 { return g.Stride * g.CIn }

func (g LastConvGeometry) validate() error {
	if g.N <= 0 || g.TIn <= 0 || g.CIn <= 0 || g.COut <= 0 || g.Window <= 0 || g.Stride <= 0 {
		return &LayoutError{Op: "conv_last", Reason: "geometry must be positive"}
	}
	if g.Padding != g.Window/2 {
		return &LayoutError{Op: "conv_last", Reason: "padding must be window/2"}
	}
	if g.TIn%g.Stride != 0 {
		return &LayoutError{Op: "conv_last", Reason: "input length is not a stride multiple"}
	}
	// get_koi_activation (ConvStack.cpp:21-31) can only return these three.
	// IDENTITY and TANH_X5 are not reachable from a config activation, so they
	// are refused here as well as at the boundary.
	switch g.Activation {
	case KoiSwish, KoiSwishClamp, KoiTanh:
	default:
		return &LayoutError{Op: "conv_last", Reason: fmt.Sprintf("activation %s is not reachable from get_koi_activation", g.Activation)}
	}
	for _, t := range []struct {
		name string
		v    KoiDType
	}{{"in", g.InType}, {"out", g.OutType}, {"accum", g.AccumType}} {
		switch t.v {
		case KoiI8, KoiI32, KoiF16, KoiF32, KoiE4M3:
		default:
			return &LayoutError{Op: "conv_last", Reason: fmt.Sprintf("unsupported Koi %s type %d", t.name, uint32(t.v))}
		}
	}
	if g.OutNStride <= 0 || g.OutTStride <= 0 || g.InNStride <= 0 {
		return &LayoutError{Op: "conv_last", Reason: "strides must be positive"}
	}
	return nil
}

// koiStorageDType is the storage dtype a declared Koi type must match. A request
// that names a type the storage does not hold would reinterpret bytes.
func koiStorageDType(t KoiDType) (DType, error) {
	switch t {
	case KoiI8:
		return DTypeI8, nil
	case KoiI32:
		return DTypeI32, nil
	case KoiF16:
		return DTypeF16, nil
	case KoiF32:
		return DTypeF32, nil
	case KoiE4M3:
		return DTypeE4M3, nil
	}
	return 0, fmt.Errorf("unsupported Koi type %d", uint32(t))
}

// checkLastConv validates the request against the source contract. It is shared
// by the fast path and the fallback because both consume the same views.
func (d *Device) checkLastConv(s *Stream, in, weights, bias, out *Storage, g LastConvGeometry) error {
	e := d.state.engine
	if err := g.validate(); err != nil {
		return err
	}
	for _, h := range []*Storage{in, weights, bias, out} {
		if h == nil || h.state == nil {
			return fmt.Errorf("conv_last: %w", ErrClosed)
		}
		if err := h.state.check(s, "conv_last"); err != nil {
			return err
		}
	}
	for _, c := range []struct {
		role string
		h    *Storage
		want KoiDType
	}{
		{"input", in, g.InType},
		{"weights", weights, KoiF16},
		{"bias", bias, KoiF16},
		{"output", out, g.OutType},
	} {
		dtype, err := koiStorageDType(c.want)
		if err != nil {
			return err
		}
		if c.h.state.dtype != dtype {
			return &DTypeError{Op: "conv_last", Reason: fmt.Sprintf("last convolution %s is %s but the request declares %s", c.role, c.h.state.dtype, c.want),
				Detail: CUDADetail{Handle: c.h.state.id, ActualDType: ScalarType(c.h.state.dtype), ExpectedDType: ScalarType(dtype)}}
		}
	}
	// The engine lock is already held here, so the descriptor is read through
	// the backend directly; Storage.Meta would take the same lock again.
	meta := func(h *Storage, role string) (StorageMeta, error) {
		r, err := e.backend.cuda(cudaRequest{op: "storage_meta", storage: h.state.id})
		if err != nil {
			return StorageMeta{}, fmt.Errorf("%s: %w", role, err)
		}
		return r.meta, nil
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
	// The source slices `padding` rows off the contiguous allocation, so the
	// allocation shape (not just T_in) is part of the contract.
	if !equalShape(im.Shape, []int64{int64(g.N), int64(g.TIn) + 2*int64(g.Padding), int64(g.CIn)}) {
		return &LayoutError{Op: "conv_last", Reason: fmt.Sprintf("input shape %v is not N,T_in+2*padding,C_in", im.Shape)}
	}
	if !equalShape(im.Strides, []int64{int64(g.InNStride), int64(g.CIn), 1}) {
		return &LayoutError{Op: "conv_last", Reason: fmt.Sprintf("input strides %v do not describe the contiguous NTC allocation", im.Strides)}
	}
	if !equalShape(wm.Shape, []int64{int64(g.COut), int64(g.k())}) {
		return &LayoutError{Op: "conv_last", Reason: fmt.Sprintf("weight shape %v is not C_out,W*C_in", wm.Shape)}
	}
	if !equalShape(bm.Shape, []int64{int64(g.COut)}) {
		return &LayoutError{Op: "conv_last", Reason: fmt.Sprintf("bias shape %v is not C_out", bm.Shape)}
	}
	if !equalShape(om.Shape, []int64{int64(g.N), int64(g.tOut()), int64(g.COut)}) {
		return &LayoutError{Op: "conv_last", Reason: fmt.Sprintf("output shape %v is not N,T_out,C_out", om.Shape)}
	}
	if !equalShape(om.Strides, []int64{int64(g.OutNStride), int64(g.OutTStride), 1}) {
		return &LayoutError{Op: "conv_last", Reason: fmt.Sprintf("output strides %v do not describe the CUTLASS NTC view", om.Strides)}
	}
	return nil
}

// LastConvKoi runs the CUTLASS branch's fast path, `host_convolution`
// (ConvStack.cpp:289-292), and returns the primitive's own KoiResult. A
// non-success result is not an error: the source falls back to
// LastConvLinear, so the caller decides.
func (d *Device) LastConvKoi(s *Stream, in, weights, bias, out *Storage, g LastConvGeometry) (KoiResult, error) {
	return d.lastConv("conv_koi_f16", s, in, weights, bias, out, g, false)
}

// LastConvKoiUnchecked is LastConvKoi without the Go-side checks. It exists so a
// rejection control can prove the native guard itself fires.
func (d *Device) LastConvKoiUnchecked(s *Stream, in, weights, bias, out *Storage, g LastConvGeometry) (KoiResult, error) {
	return d.lastConv("conv_koi_f16", s, in, weights, bias, out, g, true)
}

// LastConvLinear runs the source fallback, `host_linear`
// (ConvStack.cpp:293-300), and returns the primitive's own KoiResult.
func (d *Device) LastConvLinear(s *Stream, in, weights, bias, out *Storage, g LastConvGeometry) (KoiResult, error) {
	return d.lastConv("linear_f16", s, in, weights, bias, out, g, false)
}

// LastConvLinearUnchecked is LastConvLinear without the Go-side checks.
func (d *Device) LastConvLinearUnchecked(s *Stream, in, weights, bias, out *Storage, g LastConvGeometry) (KoiResult, error) {
	return d.lastConv("linear_f16", s, in, weights, bias, out, g, true)
}

func (d *Device) lastConv(op string, s *Stream, in, weights, bias, out *Storage, g LastConvGeometry, unchecked bool) (KoiResult, error) {
	if d == nil || d.state == nil {
		return 0, ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.state.valid(); err != nil {
		return 0, err
	}
	if !unchecked {
		if err := d.checkLastConv(s, in, weights, bias, out, g); err != nil {
			return 0, err
		}
	}
	if s == nil || s.state == nil {
		return 0, fmt.Errorf("%s: %w", op, ErrClosed)
	}
	if err := s.state.valid(op); err != nil {
		return 0, err
	}
	for _, h := range []*Storage{in, weights, bias, out} {
		if h == nil || h.state == nil {
			return 0, fmt.Errorf("%s: %w", op, ErrClosed)
		}
	}
	r, err := e.backend.cuda(cudaRequest{
		op: op, stream: s.state.id,
		in: in.state.id, weights: weights.state.id, bias: bias.state.id, out: out.state.id,
		n: g.N, tIn: g.TIn, cIn: g.CIn, cOut: g.COut, tOut: g.tOut(), k: g.k(),
		window: g.Window, stride: g.Stride, padding: g.Padding,
		inNStride: g.InNStride, inStride1: g.inStride1(),
		outNStride: g.OutNStride, outTStride: g.OutTStride,
		activation: uint32(g.Activation),
		inType:     uint32(g.InType), outType: uint32(g.OutType), accumType: uint32(g.AccumType),
	})
	if err != nil {
		return 0, err
	}
	return KoiResult(r.koiResult), nil
}
