package tensor

import (
	"fmt"
	"math"
	"slices"
)

// SUPPrimitive selects one mathematical operation, never a model or encoder.
type SUPPrimitive uint32

const (
	SUPCopyStrided SUPPrimitive = 1 + iota
	SUPConvSiLU
	SUPQuantizeRows
	SUPSinCos
	SUPLinear
	SUPQKVRotary
	SUPMaskedAttention
	SUPProjection
	SUPNorm1
	SUPSwiGLU
	SUPFC2
	SUPNorm2
)

// SUPGeometry selects a complete fixed aux=nullptr source geometry.
type SUPGeometry uint32

const (
	SUPGeometryTiny SUPGeometry = 1 + iota
	SUPGeometryFull
	SUPGeometryShort
)

type SUPDimensions struct {
	Batch, Chunk, EncoderT int64
}

func (g SUPGeometry) Dimensions() (SUPDimensions, error) {
	switch g {
	case SUPGeometryTiny:
		return SUPDimensions{32, 768, 64}, nil
	case SUPGeometryFull:
		return SUPDimensions{32, 12288, 1024}, nil
	case SUPGeometryShort:
		return SUPDimensions{32, 6144, 512}, nil
	default:
		return SUPDimensions{}, &LayoutError{Op: "sup_primitive", Reason: "unknown fixed SUP geometry"}
	}
}

const SUPDeepNormAlpha float32 = 2.4494897

type SUPOperandSpec struct {
	DType         DType
	Shape         []int64
	Tags          []KoiTag
	ScaleTag      KoiTag
	Scaled, Write bool
}

func supOperand(t DType, shape []int64, tags string, write bool) SUPOperandSpec {
	x := SUPOperandSpec{DType: t, Shape: shape, Write: write}
	for _, c := range tags {
		x.Tags = append(x.Tags, KoiTag(c))
	}
	return x
}

// SUPPrimitiveSpecs returns the selected closed N32/C512 source contract. selector
// chooses a convolution (0..4) or the upsample/CRF linear (0..1).
func SUPPrimitiveSpecs(geometry SUPGeometry, op SUPPrimitive, selector int) ([]SUPOperandSpec, error) {
	dims, err := geometry.Dimensions()
	if err != nil {
		return nil, err
	}
	n, t := dims.Batch, dims.EncoderT
	tiles, rows := t/16, n*t/16
	h := func(shape []int64, tags string, w bool) SUPOperandSpec { return supOperand(DTypeF16, shape, tags, w) }
	f8 := func(shape []int64, tags string, w bool) SUPOperandSpec { return supOperand(DTypeE4M3, shape, tags, w) }
	i8 := func(shape []int64, tags string, w bool) SUPOperandSpec {
		x := supOperand(DTypeI8, shape, tags, w)
		x.Scaled = true
		x.ScaleTag = TagC
		return x
	}
	f32 := func(shape []int64) SUPOperandSpec {
		return supOperand(DTypeF32, shape, string(make([]byte, len(shape))), false)
	}
	ctr := func() SUPOperandSpec { return supOperand(DTypeI32, []int64{1}, "C", true) }
	half := func(w bool) SUPOperandSpec { return h([]int64{n, tiles, 64, 16, 8}, "NTCtc", w) }
	if op != SUPConvSiLU && op != SUPLinear && selector != 0 {
		return nil, &LayoutError{Op: "sup_primitive", Reason: "unexpected selector"}
	}
	switch op {
	case SUPConvSiLU:
		if selector < 0 || selector >= 5 {
			break
		}
		ci := []int64{1, 64, 64, 128, 128}[selector]
		co := []int64{64, 64, 128, 128, 512}[selector]
		convT := []int64{dims.Chunk, dims.Chunk, dims.Chunk, dims.Chunk / 3, dims.Chunk / 6}[selector]
		stride := []int64{1, 1, 3, 2, 2}[selector]
		win := []int64{5, 5, 9, 9, 5}[selector]
		return []SUPOperandSpec{h([]int64{n, ci, convT}, "NCT", false), h([]int64{co, ci, win}, "NCW", false), h([]int64{co}, "C", false), h([]int64{n, co, convT / stride}, "NCT", true)}, nil
	case SUPQuantizeRows:
		s := f32([]int64{n, t})
		s.Write = true
		return []SUPOperandSpec{h([]int64{n, t, 512}, "NTC", false), supOperand(DTypeI8, []int64{n, t, 512}, "NTC", true), s}, nil
	case SUPSinCos:
		return []SUPOperandSpec{h([]int64{4096, 32}, "TD", false), h([]int64{4096, 32}, "TD", true), h([]int64{4096, 32}, "TD", true)}, nil
	case SUPLinear:
		if selector == 0 {
			return []SUPOperandSpec{h([]int64{n, t, 512}, "NTC", false), h([]int64{1024, 512}, "NC", false), h([]int64{1024}, "C", false), h([]int64{n, t, 1024}, "NTC", true)}, nil
		}
		if selector == 1 {
			return []SUPOperandSpec{h([]int64{n, 2 * t, 512}, "NTC", false), h([]int64{4096, 512}, "NC", false), h([]int64{n, 2 * t, 4096}, "NTC", true)}, nil
		}
	case SUPQKVRotary:
		w := i8([]int64{3, 8, 4, 32, 16, 16}, "\x00HDCdc", false)
		return []SUPOperandSpec{i8([]int64{n, tiles, 32, 16, 16}, "NTCtc", false), f32([]int64{n, t}), w, f32([]int64{3, 8, 64}), h([]int64{tiles, 8, 16, 8}, "TDtd", false), h([]int64{n, tiles, 3, 8, 8, 16, 8}, "NT\x00HDtd", true), ctr()}, nil
	case SUPMaskedAttention:
		return []SUPOperandSpec{h([]int64{n, tiles, 3, 8, 8, 16, 8}, "NT\x00HDtd", false), h([]int64{n, tiles, 8, 8, 16, 8}, "NTHDtd", true)}, nil
	case SUPProjection:
		return []SUPOperandSpec{h([]int64{rows, 64, 16, 8}, "MKmk", false), h([]int64{32, 64, 16, 8}, "NKnk", false), h([]int64{512}, "N", false), h([]int64{rows, 64, 16, 8}, "MNmn", true), ctr()}, nil
	case SUPNorm1:
		return []SUPOperandSpec{half(false), half(true), h([]int64{512}, "C", false), f8([]int64{n, tiles, 32, 16, 16}, "NTCtc", true)}, nil
	case SUPSwiGLU:
		return []SUPOperandSpec{f8([]int64{rows, 32, 16, 16}, "MKmk", false), f8([]int64{256, 32, 16, 16}, "NKnk", false), f8([]int64{rows, 128, 16, 16}, "MNmn", true), ctr()}, nil
	case SUPFC2:
		return []SUPOperandSpec{f8([]int64{rows, 128, 16, 16}, "MKmk", false), f8([]int64{32, 128, 16, 16}, "NKnk", false), h([]int64{rows, 64, 16, 8}, "MNmn", true), ctr()}, nil
	case SUPNorm2:
		s := f32([]int64{n, t})
		s.Write = true
		return []SUPOperandSpec{half(false), half(true), h([]int64{512}, "C", false), i8([]int64{n, tiles, 32, 16, 16}, "NTCtc", true), s}, nil
	}
	return nil, &LayoutError{Op: "sup_primitive", Reason: "operation or selector outside fixed SUP contract"}
}

func supByteInterval(m StorageMeta) (uint64, uint64, error) {
	size := uint64(m.DType.Size())
	if size == 0 {
		return 0, 0, &DTypeError{Op: "sup_primitive"}
	}
	end, e := storageSpan(m.Shape, m.Strides, m.OffsetElements)
	if e != nil {
		return 0, 0, e
	}
	if m.StorageElements == 0 || end > m.StorageElements || end > math.MaxUint64/size {
		return 0, 0, &LayoutError{Op: "sup_primitive", Reason: "operand byte interval exceeds allocation or overflows"}
	}
	return m.OffsetElements * size, end * size, nil
}

// ValidateSUPMetadata is also used by host rejection controls. Read/write
// alias checks compare byte intervals across all dtypes and all strided spans.
func ValidateSUPMetadata(geometry SUPGeometry, op SUPPrimitive, selector int, metas []StorageMeta) error {
	if _, err := geometry.Dimensions(); err != nil {
		return err
	}
	var specs []SUPOperandSpec
	var err error
	if op == SUPCopyStrided {
		if selector != 0 || len(metas) != 2 {
			return &LayoutError{Op: "sup_primitive", Reason: "copy requires two operands and selector zero"}
		}
		if !slices.Equal(metas[0].Shape, metas[1].Shape) || metas[0].DType != metas[1].DType {
			return &LayoutError{Op: "sup_primitive", Reason: "strided copy shape or dtype differs"}
		}
		specs = []SUPOperandSpec{{DType: metas[0].DType, Shape: metas[0].Shape, Tags: metas[0].Tags}, {DType: metas[1].DType, Shape: metas[1].Shape, Tags: metas[1].Tags, Write: true}}
	} else if specs, err = SUPPrimitiveSpecs(geometry, op, selector); err != nil {
		return err
	}
	if len(metas) != len(specs) {
		return &LayoutError{Op: "sup_primitive", Reason: "operand count differs"}
	}
	starts := make([]uint64, len(metas))
	ends := make([]uint64, len(metas))
	for i, m := range metas {
		p := specs[i]
		if m.DType != p.DType {
			return &DTypeError{Op: "sup_primitive", Reason: fmt.Sprintf("operand %d dtype differs", i)}
		}
		if m.Device == 0 || m.Device != metas[0].Device || m.Ordinal != metas[0].Ordinal {
			return &DeviceError{Op: "sup_primitive", Reason: "operand device differs"}
		}
		if !slices.Equal(m.Shape, p.Shape) || !slices.Equal(m.Tags, p.Tags) || len(m.Strides) != len(m.Shape) || m.ScalePresent != p.Scaled || (p.Scaled && (m.ScaleDType != DTypeF32 || m.ScaleTag != p.ScaleTag)) {
			return &LayoutError{Op: "sup_primitive", Reason: fmt.Sprintf("operand %d shape/tags/scale differ", i)}
		}
		if err = storageTags(m.Tags, len(m.Shape)); err != nil {
			return err
		}
		stride := int64(1)
		for j := len(m.Shape) - 1; j >= 0; j-- {
			if op != SUPCopyStrided && m.Strides[j] != stride {
				return &LayoutError{Op: "sup_primitive", Reason: "operand must be contiguous"}
			}
			if m.Shape[j] < 1 || stride > math.MaxInt64/m.Shape[j] {
				return &LayoutError{Op: "sup_primitive", Reason: "shape overflow"}
			}
			stride *= m.Shape[j]
		}
		if starts[i], ends[i], err = supByteInterval(m); err != nil {
			return err
		}
		if p.Write && op == SUPCopyStrided {
			dims := make([]int, len(m.Shape))
			for j := range dims {
				dims[j] = j
			}
			slices.SortFunc(dims, func(a, b int) int {
				if m.Strides[a] < m.Strides[b] {
					return -1
				}
				if m.Strides[a] > m.Strides[b] {
					return 1
				}
				return 0
			})
			span := int64(1)
			for _, j := range dims {
				if m.Shape[j] > 1 {
					if m.Strides[j] < span {
						return &LayoutError{Op: "sup_primitive", Reason: "strided output overlaps itself"}
					}
					if m.Strides[j] > (math.MaxInt64-span)/(m.Shape[j]-1) {
						return &LayoutError{Op: "sup_primitive", Reason: "strided output span overflow"}
					}
					span += (m.Shape[j] - 1) * m.Strides[j]
				}
			}
		}
	}
	for i, p := range specs {
		if p.Write {
			for j := range metas {
				if i != j && metas[i].Allocation == metas[j].Allocation && starts[i] < ends[j] && starts[j] < ends[i] {
					return &LayoutError{Op: "sup_primitive", Reason: fmt.Sprintf("writable operand %d overlaps operand %d", i, j)}
				}
			}
		}
	}
	return nil
}

// SUPMath invokes one fixed source operation. Every handle must belong to the
// same live Device and Stream. Native validation independently repeats this.
func (d *Device) SUPMath(s *Stream, geometry SUPGeometry, op SUPPrimitive, selector int, operands ...*Storage) error {
	if _, err := geometry.Dimensions(); err != nil {
		return err
	}
	if d == nil || d.state == nil {
		return ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.state.valid(); err != nil {
		return err
	}
	if len(operands) > 8 {
		return &LayoutError{Op: "sup_primitive", Reason: "too many operands"}
	}
	var ids [8]uint64
	metas := make([]StorageMeta, len(operands))
	for i, h := range operands {
		if h == nil || h.state == nil {
			return ErrClosed
		}
		if h.state.device != d.state {
			return &DeviceError{Op: "sup_primitive", Reason: "operand owner differs"}
		}
		if err := h.state.check(s, "sup_primitive"); err != nil {
			return err
		}
		ids[i] = h.state.id
		r, err := e.backend.cuda(cudaRequest{op: "storage_meta", storage: ids[i]})
		if err != nil {
			return err
		}
		metas[i] = r.meta
	}
	if err := ValidateSUPMetadata(geometry, op, selector, metas); err != nil {
		return err
	}
	alpha := float32(0)
	if op == SUPNorm1 || op == SUPNorm2 {
		alpha = SUPDeepNormAlpha
	}
	_, err := e.backend.cuda(cudaRequest{supAlpha: alpha, op: "sup_primitive", stream: s.state.id, sup: ids, supGeometry: uint32(geometry), supOp: uint32(op), supSelector: int32(selector), count: len(operands)})
	return err
}
