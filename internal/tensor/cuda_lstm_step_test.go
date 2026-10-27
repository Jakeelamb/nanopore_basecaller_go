//go:build cgo && linux

package tensor

import (
	"encoding/binary"
	"testing"
)

func halfStepDevice(t *testing.T) (*Device, *Stream, func([]int64) *Storage) {
	t.Helper()
	e := fakeStorageEngine(t)
	d, err := e.NewCUDADevice(0)
	if err != nil {
		t.Fatal(err)
	}
	s, err := d.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
		if err := d.Close(); err != nil {
			t.Error(err)
		}
	})
	newHalf := func(shape []int64) *Storage {
		count := 1
		tags := make([]KoiTag, len(shape))
		for i, n := range shape {
			count *= int(n)
			tags[i] = TagC
		}
		h, err := d.NewStorage(s, DTypeF16, shape, tags, make([]byte, count*2))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := h.Close(); err != nil {
				t.Error(err)
			}
		})
		return h
	}
	return d, s, newHalf
}

func TestStorageHalfGEMMStridedOutputAndABI(t *testing.T) {
	d, s, newHalf := halfStepDevice(t)
	a, b := newHalf([]int64{2, 4}), newHalf([]int64{4, 8})
	parent := newHalf([]int64{2, 16})
	out, err := parent.View(0, []int64{2, 8}, []int64{16, 1}, []KoiTag{TagM, TagN})
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	if err = d.StorageGEMMHalf(s, a, b, out); err != nil {
		t.Fatal(err)
	}
	data, err := parent.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	if binary.LittleEndian.Uint64(data[:8]) != a.state.id || binary.LittleEndian.Uint64(data[8:16]) != b.state.id {
		t.Fatal("GEMM ABI operand order changed")
	}
	if err = d.ValidateStorageGEMMHalf(s, a, b, a); err == nil {
		t.Fatal("wrong output shape accepted")
	}
	square := newHalf([]int64{4, 4})
	if err = d.ValidateStorageGEMMHalf(s, square, square, square); err == nil {
		t.Fatal("input/output overlap accepted")
	}
	if err = a.Close(); err != nil {
		t.Fatal(err)
	}
	if err = d.StorageGEMMHalf(s, a, b, out); err == nil {
		t.Fatal("closed matrix accepted")
	}
}

func TestHalfLSTMStepOwnsFourTypedOperands(t *testing.T) {
	d, s, newHalf := halfStepDevice(t)
	o := HalfLSTMStepOperands{newHalf([]int64{4096}), newHalf([]int64{2, 4096}), newHalf([]int64{2, 1024}), newHalf([]int64{2, 1024})}
	if err := d.HalfLSTMStep(s, o, 2); err != nil {
		t.Fatal(err)
	}
	data, err := o.Out.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	if binary.LittleEndian.Uint64(data[:8]) != o.Bias.state.id || binary.LittleEndian.Uint64(data[8:16]) != o.Gates.state.id || binary.LittleEndian.Uint64(data[16:24]) != o.Cell.state.id || binary.LittleEndian.Uint32(data[24:28]) != 2 {
		t.Fatal("cell ABI operands changed")
	}
	bad := o
	bad.Out = bad.Cell
	if err = d.ValidateHalfLSTMStep(s, bad, 2); err == nil {
		t.Fatal("in-place cell/output overlap accepted")
	}
	if err = d.ValidateHalfLSTMStep(s, o, 0); err == nil {
		t.Fatal("zero batch accepted")
	}
	bad = o
	bad.Gates = o.Out
	if err = d.ValidateHalfLSTMStep(s, bad, 2); err == nil {
		t.Fatal("gate shape accepted")
	}
}
