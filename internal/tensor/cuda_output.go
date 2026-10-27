package tensor

import (
	"fmt"
	"math"
)

type FactorisedLinearGeometry struct{ N, T int64 }
type FactorisedLinearOperands struct {
	Input, DownWeight, DownScale, Down, UpWeight, Output *Storage
}

func (g FactorisedLinearGeometry) validate() error {
	if g.N <= 0 || g.T <= 0 || g.N%64 != 0 || g.T%32 != 0 || g.N > math.MaxInt32 || g.T > math.MaxInt32-5 ||
		g.T+5 > math.MaxInt32/1024/g.N {
		return &LayoutError{Op: "factorised_linear", Reason: "requires positive N64/T32 tiles and checked native int extents"}
	}
	return nil
}
func (o FactorisedLinearOperands) storages() []*Storage {
	return []*Storage{o.Input, o.DownWeight, o.DownScale, o.Down, o.UpWeight, o.Output}
}
func (d *Device) checkFactorisedLinear(s *Stream, o FactorisedLinearOperands, g FactorisedLinearGeometry) error {
	if err := g.validate(); err != nil {
		return err
	}
	shapes := [][]int64{{g.T, g.N, 1024}, {128, 1024}, {128}, {g.N * g.T * 128}, {1024, 128}, {g.N, g.T, 1024}}
	types := []DType{DTypeI8, DTypeI8, DTypeF16, DTypeF16, DTypeF16, DTypeF16}
	tags := [][]KoiTag{{TagT, TagN, TagC}, {'K', TagC}, {'K'}, {TagC}, {TagC, 'K'}, {TagN, TagT, TagC}}
	metas := make([]StorageMeta, 6)
	var starts, ends [6]uint64
	for i, h := range o.storages() {
		if h == nil || h.state == nil {
			return ErrClosed
		}
		if h.state.device != d.state {
			return &DeviceError{Op: "factorised_linear", Reason: "operand belongs to another device"}
		}
		if err := h.state.check(s, "factorised_linear"); err != nil {
			return err
		}
		if h.state.dtype != types[i] {
			return &DTypeError{Op: "factorised_linear", Reason: fmt.Sprintf("operand %d requires %s", i, types[i])}
		}
		r, err := d.state.engine.backend.cuda(cudaRequest{op: "storage_meta", storage: h.state.id})
		if err != nil {
			return err
		}
		m := r.meta
		metas[i] = m
		if m.DType != types[i] || !equalShape(m.Shape, shapes[i]) || len(m.Strides) != len(m.Shape) || len(m.Tags) != len(tags[i]) || m.ScalePresent {
			return &LayoutError{Op: "factorised_linear", Reason: fmt.Sprintf("operand %d shape, dtype, tags, or scale metadata mismatch", i)}
		}
		stride := int64(1)
		for j := len(m.Shape) - 1; j >= 0; j-- {
			if m.Strides[j] != stride || m.Tags[j] != tags[i][j] {
				return &LayoutError{Op: "factorised_linear", Reason: "operand must have source tags and contiguous strides"}
			}
			stride *= m.Shape[j]
		}
		size := uint64(types[i].Size())
		if m.OffsetElements > math.MaxUint64/size {
			return &LayoutError{Op: "factorised_linear", Reason: "operand byte offset overflow"}
		}
		starts[i] = m.OffsetElements * size
		if uint64(stride) > (math.MaxUint64-starts[i])/size {
			return &LayoutError{Op: "factorised_linear", Reason: "operand byte end overflow"}
		}
		ends[i] = starts[i] + uint64(stride)*size
		if starts[i]%16 != 0 {
			return &LayoutError{Op: "factorised_linear", Reason: "operand offset must be 16-byte aligned"}
		}
	}
	for _, i := range []int{3, 5} {
		for j := range metas {
			if i == j {
				continue
			}
			if metas[i].Allocation == metas[j].Allocation && starts[i] < ends[j] && starts[j] < ends[i] {
				return &LayoutError{Op: "factorised_linear", Reason: "writable operand byte intervals overlap"}
			}
		}
	}
	return nil
}

func (d *Device) ValidateFactorisedLinear(s *Stream, o FactorisedLinearOperands, g FactorisedLinearGeometry) error {
	if d == nil || d.state == nil {
		return ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.state.valid(); err != nil {
		return err
	}
	return d.checkFactorisedLinear(s, o, g)
}
func (d *Device) FactorisedLinear(s *Stream, o FactorisedLinearOperands, g FactorisedLinearGeometry) error {
	return d.factorisedLinear(s, o, g, 1, 5, false)
}

// FactorisedLinearUnchecked bypasses Go validation for native rejection controls.
func (d *Device) FactorisedLinearUnchecked(s *Stream, o FactorisedLinearOperands, g FactorisedLinearGeometry, parity int32, scale float32) error {
	return d.factorisedLinear(s, o, g, parity, scale, true)
}
func (d *Device) factorisedLinear(s *Stream, o FactorisedLinearOperands, g FactorisedLinearGeometry, parity int32, scale float32, unchecked bool) error {
	if d == nil || d.state == nil {
		return ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.state.valid(); err != nil {
		return err
	}
	if !unchecked {
		if err := d.checkFactorisedLinear(s, o, g); err != nil {
			return err
		}
	}
	if s == nil || s.state == nil {
		return ErrClosed
	}
	if err := s.state.valid("factorised_linear"); err != nil {
		return err
	}
	var ids [6]uint64
	for i, h := range o.storages() {
		if h == nil || h.state == nil {
			return ErrClosed
		}
		ids[i] = h.state.id
	}
	_, err := e.backend.cuda(cudaRequest{op: "factorised_linear", stream: s.state.id, output: ids, outputN: g.N, outputT: g.T, parity: parity, outputScale: scale})
	return err
}
