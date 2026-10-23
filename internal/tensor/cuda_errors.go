package tensor

import (
	"errors"
	"fmt"
)

var ErrCUDAUnavailable = errors.New("tensor backend was built without CUDA")

type ScalarType uint32

const (
	ScalarNone ScalarType = iota
	ScalarHalf
	ScalarFloat32
	ScalarInt64
)

type CUDADetail struct {
	Stage                                              uint32
	NativeCode                                         int32
	Handle, ActualEngine, ExpectedDevice, ActualDevice uint64
	ExpectedOrdinal, ActualOrdinal                     int32
	ExpectedDType, ActualDType                         ScalarType
}
type DTypeError struct {
	Op     string
	Detail CUDADetail
}

func (e *DTypeError) Error() string {
	return fmt.Sprintf("%s: handle %d dtype %d requires %d", e.Op, e.Detail.Handle, e.Detail.ActualDType, e.Detail.ExpectedDType)
}

type DeviceError struct {
	Op, Reason     string
	RequiredEngine uint64
	Detail         CUDADetail
}

func (e *DeviceError) Error() string {
	return fmt.Sprintf("%s: %s (engine %d device %d ordinal %d requires engine %d device %d ordinal %d)", e.Op, e.Reason, e.Detail.ActualEngine, e.Detail.ActualDevice, e.Detail.ActualOrdinal, e.RequiredEngine, e.Detail.ExpectedDevice, e.Detail.ExpectedOrdinal)
}

type StreamError struct {
	Op, Reason string
	Detail     CUDADetail
	Closed     bool
}

func (e *StreamError) Error() string {
	return fmt.Sprintf("%s: stream %d: %s (native code %d)", e.Op, e.Detail.Handle, e.Reason, e.Detail.NativeCode)
}
func (e *StreamError) Unwrap() error {
	if e.Closed {
		return ErrClosed
	}
	return nil
}

type LayoutError struct {
	Op, Reason string
	Handle     uint64
}

func (e *LayoutError) Error() string {
	return fmt.Sprintf("%s: half layout %d: %s", e.Op, e.Handle, e.Reason)
}

type CUDAError struct {
	Op, Subsystem, Reason string
	Detail                CUDADetail
}

func (e *CUDAError) Error() string {
	return fmt.Sprintf("%s: %s stage %d code %d: %s", e.Op, e.Subsystem, e.Detail.Stage, e.Detail.NativeCode, e.Reason)
}

type CUDAUnsupportedError struct{ Op, Reason string }

func (e *CUDAUnsupportedError) Error() string { return e.Op + ": " + e.Reason }
func (e *CUDAUnsupportedError) Unwrap() error { return ErrCUDAUnavailable }
func cudaStatus(op string, code int, d CUDADetail, reason string) error {
	switch code {
	case 0:
		return nil
	case 2:
		return fmt.Errorf("%s: handle %d: %s: %w", op, d.Handle, reason, ErrClosed)
	case 3, 5:
		return &DeviceError{Op: op, Reason: reason, Detail: d}
	case 4:
		return &DTypeError{Op: op, Detail: d}
	case 6:
		return &StreamError{Op: op, Reason: reason, Detail: d}
	case 7:
		return &LayoutError{Op: op, Reason: reason, Handle: d.Handle}
	case 8:
		return &CUDAUnsupportedError{Op: op, Reason: reason}
	case 9, 10:
		sub := "cuda"
		if code == 10 {
			sub = "cublas"
		}
		return &CUDAError{Op: op, Subsystem: sub, Reason: reason, Detail: d}
	default:
		return &CUDAError{Op: op, Subsystem: "boundary", Reason: reason, Detail: d}
	}
}
