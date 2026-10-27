package tensor

import "math"

func (d *Device) halfStorageOperands(s *Stream, op string, handles ...*Storage) ([]StorageMeta, error) {
	if err := d.state.valid(); err != nil {
		return nil, err
	}
	metas := make([]StorageMeta, len(handles))
	for i, h := range handles {
		if h == nil || h.state == nil {
			return nil, ErrClosed
		}
		if h.state.device != d.state {
			return nil, &DeviceError{Op: op, Reason: "foreign device operand"}
		}
		if err := h.state.check(s, op); err != nil {
			return nil, err
		}
		if h.state.dtype != DTypeF16 {
			return nil, &DTypeError{Op: op, Reason: "half operands required"}
		}
		r, err := d.state.engine.backend.cuda(cudaRequest{op: "storage_meta", storage: h.state.id})
		if err != nil {
			return nil, err
		}
		if r.meta.ScalePresent {
			return nil, &LayoutError{Op: op, Reason: "half operation does not accept scales"}
		}
		metas[i] = r.meta
	}
	return metas, nil
}

func halfWritableDisjoint(op string, a, b StorageMeta) error {
	if a.Allocation != b.Allocation {
		return nil
	}
	aEnd, err := storageSpan(a.Shape, a.Strides, a.OffsetElements)
	if err != nil {
		return err
	}
	bEnd, err := storageSpan(b.Shape, b.Strides, b.OffsetElements)
	if err != nil {
		return err
	}
	if a.OffsetElements < bEnd && b.OffsetElements < aEnd {
		return &LayoutError{Op: op, Reason: "writable operand intervals overlap"}
	}
	return nil
}

func (d *Device) validateStorageGEMMHalf(s *Stream, a, b, out *Storage) error {
	metas, err := d.halfStorageOperands(s, "storage_gemm_f16", a, b, out)
	if err != nil {
		return err
	}
	for _, m := range metas {
		if len(m.Shape) != 2 || len(m.Strides) != 2 || m.Shape[0] > math.MaxInt32 || m.Shape[1] > math.MaxInt32 || m.Strides[1] != 1 || m.Strides[0] < m.Shape[1] || m.Strides[0] > math.MaxInt32 {
			return &LayoutError{Op: "storage_gemm_f16", Reason: "rank two with int32 row-leading dimensions required"}
		}
	}
	if metas[0].Shape[1] != metas[1].Shape[0] || metas[2].Shape[0] != metas[0].Shape[0] || metas[2].Shape[1] != metas[1].Shape[1] {
		return &LayoutError{Op: "storage_gemm_f16", Reason: "matrix dimensions differ"}
	}
	for _, m := range metas[:2] {
		if err = halfWritableDisjoint("storage_gemm_f16", m, metas[2]); err != nil {
			return err
		}
	}
	return nil
}

func (d *Device) ValidateStorageGEMMHalf(s *Stream, a, b, out *Storage) error {
	if d == nil || d.state == nil {
		return ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	return d.validateStorageGEMMHalf(s, a, b, out)
}

// StorageGEMMHalf uses the same source half cuBLAS operation as GEMMHalf.
func (d *Device) StorageGEMMHalf(s *Stream, a, b, out *Storage) error {
	if d == nil || d.state == nil {
		return ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.validateStorageGEMMHalf(s, a, b, out); err != nil {
		return err
	}
	_, err := e.backend.cuda(cudaRequest{op: "storage_gemm_f16", stream: s.state.id, a: a.state.id, b: b.state.id, out: out.state.id})
	return err
}

type HalfLSTMStepOperands struct{ Bias, Gates, Cell, Out *Storage }

func (d *Device) validateHalfLSTMStep(s *Stream, o HalfLSTMStepOperands, n int32) error {
	if n <= 0 || n > math.MaxInt32/4096 {
		return &LayoutError{Op: "lstm_step_f16", Reason: "positive checked batch required"}
	}
	metas, err := d.halfStorageOperands(s, "lstm_step_f16", o.Bias, o.Gates, o.Cell, o.Out)
	if err != nil {
		return err
	}
	shapes := [][]int64{{4096}, {int64(n), 4096}, {int64(n), 1024}, {int64(n), 1024}}
	for i, m := range metas {
		if !equalShape(m.Shape, shapes[i]) || len(m.Strides) != len(m.Shape) {
			return &LayoutError{Op: "lstm_step_f16", Reason: "fixed C1024 operand shape required"}
		}
		stride := int64(1)
		for j := len(m.Shape) - 1; j >= 0; j-- {
			if m.Strides[j] != stride {
				return &LayoutError{Op: "lstm_step_f16", Reason: "contiguous operand required"}
			}
			stride *= m.Shape[j]
		}
	}
	for _, w := range []int{2, 3} {
		for r := range metas {
			if w != r {
				if err = halfWritableDisjoint("lstm_step_f16", metas[w], metas[r]); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (d *Device) ValidateHalfLSTMStep(s *Stream, o HalfLSTMStepOperands, n int32) error {
	if d == nil || d.state == nil {
		return ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	return d.validateHalfLSTMStep(s, o, n)
}

// HalfLSTMStep applies one source C1024 cell update with hard gates and only
// the input bias. The caller selects the timestep and resets each layer's cell.
func (d *Device) HalfLSTMStep(s *Stream, o HalfLSTMStepOperands, n int32) error {
	if d == nil || d.state == nil {
		return ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.validateHalfLSTMStep(s, o, n); err != nil {
		return err
	}
	_, err := e.backend.cuda(cudaRequest{op: "lstm_step_f16", stream: s.state.id, bias: o.Bias.state.id, a: o.Gates.state.id, b: o.Cell.state.id, out: o.Out.state.id, n: n})
	return err
}
