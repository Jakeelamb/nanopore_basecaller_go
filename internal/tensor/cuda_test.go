package tensor

import (
	"errors"
	"math"
	"testing"
)

func TestHalfLayoutValidation(t *testing.T) {
	for _, shape := range [][]int64{nil, {}, make([]int64, 9), {0}, {-1}, {math.MaxInt64}, {1 << 40, 1 << 40}} {
		if _, err := halfCount(shape); err == nil {
			t.Fatalf("accepted shape %v", shape)
		}
	}
	if n, err := halfCount([]int64{3, 5}); err != nil || n != 15 {
		t.Fatal(n, err)
	}
	for _, v := range []struct {
		s, st    []int64
		off, lim uint64
	}{
		{[]int64{2}, []int64{-1}, 0, 10}, {[]int64{2}, nil, 0, 10}, {[]int64{3}, []int64{2}, 0, 4}, {[]int64{2}, []int64{1}, math.MaxUint64, math.MaxUint64},
	} {
		if err := halfSpan(v.s, v.st, v.off, v.lim); err == nil {
			t.Fatalf("accepted span %+v", v)
		}
	}
	if err := halfSpan([]int64{2, 3}, []int64{8, 1}, 2, 13); err != nil {
		t.Fatal(err)
	}
}
func TestCUDAErrorMapping(t *testing.T) {
	d := CUDADetail{Handle: 13, NativeCode: 42, ExpectedDType: ScalarHalf, ActualDType: ScalarFloat32}
	for _, status := range []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} {
		err := cudaStatus("test", status, d, "reason")
		if err == nil {
			t.Fatal(status)
		}
		switch status {
		case 2:
			if !errors.Is(err, ErrClosed) {
				t.Fatal(err)
			}
		case 8:
			if !errors.Is(err, ErrCUDAUnavailable) {
				t.Fatal(err)
			}
		case 4:
			var typed *DTypeError
			if !errors.As(err, &typed) || typed.Detail.ActualDType != ScalarFloat32 {
				t.Fatal(err)
			}
		}
	}
	if !errors.Is((&StreamError{Closed: true}), ErrClosed) {
		t.Fatal("closed stream sentinel")
	}
}
func TestCUDAClosedAndCopiedOwners(t *testing.T) {
	es := &engineState{closed: true}
	ds := &deviceState{engine: es}
	ss := &streamState{device: ds}
	hs := &halfState{device: ds}
	d := Device{ds}
	copyD := d
	s := Stream{ss}
	copyS := s
	h := HalfBuffer{hs}
	copyH := h
	if err := copyD.Close(); err != nil || !ds.closed {
		t.Fatal(err)
	}
	if err := copyS.Close(); err != nil || !ss.closed {
		t.Fatal(err)
	}
	if err := copyH.Close(); err != nil || !hs.closed {
		t.Fatal(err)
	}
	if _, err := s.NewHalf([]int64{1}, []uint16{0}); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	if _, err := h.Layout(); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	var nilD *Device
	var nilS *Stream
	var nilH *HalfBuffer
	if nilD.Close() != nil || nilS.Close() != nil || nilH.Close() != nil {
		t.Fatal("nil Close")
	}
	if _, err := nilD.NewStream(); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
}
