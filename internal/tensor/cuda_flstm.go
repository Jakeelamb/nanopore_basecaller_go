package tensor

import (
	"fmt"
	"math"
)

// FactorisedLSTMGeometry describes one fixed C1024 K128 layer. Encoding is
// always null. Variable chunk inference is outside this operation's contract.
type FactorisedLSTMGeometry struct{ N, T, Direction, Parity int32 }
type FactorisedLSTMOperands struct {
	InOut, Workspace, Down, DownScale        *Storage
	DownIH, DownHH, DownScaleIH, DownScaleHH *Storage
	Up, UpScale, UpBias                      *Storage
}

func (o FactorisedLSTMOperands) storages() []*Storage {
	return []*Storage{o.InOut, o.Workspace, o.Down, o.DownScale, o.DownIH, o.DownHH, o.DownScaleIH, o.DownScaleHH, o.Up, o.UpScale, o.UpBias}
}
func (g FactorisedLSTMGeometry) validate() error {
	if g.N <= 0 || g.N%64 != 0 || g.T <= 0 || int64(g.T)+1 > (math.MaxInt64-2*int64(g.N))/516/int64(g.N) || int64(g.T)+5 > math.MaxInt64/1024/int64(g.N) || (g.Direction != -1 && g.Direction != 1) || (g.Parity != 0 && g.Parity != 1) {
		return &LayoutError{Op: "factorised_lstm", Reason: "requires N64, positive T, direction +/-1 and parity 0/1"}
	}
	if (g.Direction == -1 && g.Parity != 1) || (g.Direction == 1 && g.Parity != 0) {
		return &CUDAUnsupportedError{Op: "factorised_lstm", Reason: "fixed H05 requires source direction/parity pair"}
	}
	return nil
}
func (d *Device) checkFactorisedLSTM(s *Stream, o FactorisedLSTMOperands, g FactorisedLSTMGeometry) error {
	if err := g.validate(); err != nil {
		return err
	}
	n, t := int64(g.N), int64(g.T)
	shapes := [][]int64{{t + 5, n, 1024}, {2 * n}, {512 * (t + 1) * n}, {4 * (t + 1) * n}, {128, 1024}, {128, 1024}, {128}, {128}, {4096, 256}, {4096}, {4096}}
	types := []DType{DTypeI8, DTypeU8, DTypeU8, DTypeU8, DTypeI8, DTypeI8, DTypeF16, DTypeF16, DTypeI8, DTypeF16, DTypeF16}
	handles := o.storages()
	metas := make([]StorageMeta, len(handles))
	for i, h := range handles {
		if h == nil || h.state == nil {
			return ErrClosed
		}
		if h.state.device != d.state {
			return &DeviceError{Op: "factorised_lstm", Reason: "operand belongs to another device"}
		}
		if err := h.state.check(s, "factorised_lstm"); err != nil {
			return err
		}
		if h.state.dtype != types[i] {
			return &DTypeError{Op: "factorised_lstm", Reason: fmt.Sprintf("operand %d requires %s", i, types[i])}
		}
		r, err := d.state.engine.backend.cuda(cudaRequest{op: "storage_meta", storage: h.state.id})
		if err != nil {
			return err
		}
		m := r.meta
		metas[i] = m
		if !equalShape(m.Shape, shapes[i]) {
			return &LayoutError{Op: "factorised_lstm", Reason: fmt.Sprintf("operand %d shape %v, want %v", i, m.Shape, shapes[i])}
		}
		stride := int64(1)
		for j := len(m.Shape) - 1; j >= 0; j-- {
			if m.Strides[j] != stride {
				return &LayoutError{Op: "factorised_lstm", Reason: "all operands must be contiguous"}
			}
			stride *= m.Shape[j]
		}
		if (m.OffsetElements*uint64(types[i].Size()))%16 != 0 {
			return &LayoutError{Op: "factorised_lstm", Reason: "operand offset must be 16-byte aligned"}
		}
	}
	for i, m := range metas {
		for j := i + 1; j < len(metas); j++ {
			if i >= 4 && j >= 4 {
				continue
			}
			b := metas[j]
			size := uint64(1)
			for _, v := range m.Shape {
				size *= uint64(v)
			}
			size *= uint64(m.DType.Size())
			bs := uint64(1)
			for _, v := range b.Shape {
				bs *= uint64(v)
			}
			bs *= uint64(b.DType.Size())
			a0, b0 := m.OffsetElements*uint64(m.DType.Size()), b.OffsetElements*uint64(b.DType.Size())
			if m.Allocation == b.Allocation && a0 < b0+bs && b0 < a0+size {
				return &LayoutError{Op: "factorised_lstm", Reason: "writable operand intervals overlap"}
			}
		}
	}
	return nil
}

// ValidateFactorisedLSTM checks all concrete owners without enqueueing work.
// The model uses this before its first sentinel write.
func (d *Device) ValidateFactorisedLSTM(s *Stream, o FactorisedLSTMOperands, g FactorisedLSTMGeometry) error {
	if d == nil || d.state == nil {
		return ErrClosed
	}
	e := d.state.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := d.state.valid(); err != nil {
		return err
	}
	return d.checkFactorisedLSTM(s, o, g)
}
func (d *Device) FactorisedLSTM(s *Stream, o FactorisedLSTMOperands, g FactorisedLSTMGeometry) error {
	return d.factorisedLSTM(s, o, g, false)
}

// FactorisedLSTMUnchecked bypasses the Go guard for native rejection controls.
func (d *Device) FactorisedLSTMUnchecked(s *Stream, o FactorisedLSTMOperands, g FactorisedLSTMGeometry) error {
	return d.factorisedLSTM(s, o, g, true)
}
func (d *Device) factorisedLSTM(s *Stream, o FactorisedLSTMOperands, g FactorisedLSTMGeometry, unchecked bool) error {
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
		if err := d.checkFactorisedLSTM(s, o, g); err != nil {
			return err
		}
	}
	if s == nil || s.state == nil {
		return ErrClosed
	}
	if err := s.state.valid("factorised_lstm"); err != nil {
		return err
	}
	var ids [11]uint64
	for i, h := range o.storages() {
		if h == nil || h.state == nil {
			return ErrClosed
		}
		ids[i] = h.state.id
	}
	_, err := e.backend.cuda(cudaRequest{op: "factorised_lstm", stream: s.state.id, flstm: ids, n: g.N, tIn: g.T, direction: g.Direction, parity: g.Parity})
	return err
}
