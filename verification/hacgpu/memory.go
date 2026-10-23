// Package hacgpu holds source-pinned host oracles for the HAC GPU caller port.
// It extracts the pinned Dorado planner arithmetic and replays it literally. It
// is not a producer: it imports no CUDA, allocates no device memory and
// executes no model.
package hacgpu

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// SourceCommit is the pinned Dorado revision used by every other oracle in this
// repository.
const SourceCommit = "8b8fc5d36a9c0baab262a743cb175b5878e38ca6"

// Alignment is WorkingMemory::ALIGNMENT (WorkingMemory.h:56-58).
const Alignment int64 = 256

// Source is one pinned file and the exact lines this oracle was extracted from.
type Source struct {
	Path   string
	SHA256 string
	Lines  string
}

// Sources records the extraction pins. The planner below is a literal
// transliteration of these ranges; the surrounding comments are not used as
// arithmetic.
var Sources = []Source{
	{
		Path:   "dorado/nn/WorkingMemory.cpp",
		SHA256: "4b512195a11e418ce35f8a60cf050dc499cec1b3c5504fa6ce8e21d6986e2b0d",
		Lines:  "30-33 tensor_bytes; 35-59 next; 61-79 get_current_NTC_view; 83-102 next_TC; 104-106 temp; 108-114 allocate_backing_tensor",
	},
	{
		Path:   "dorado/nn/include/nn/WorkingMemory.h",
		SHA256: "271f1c1e4451832eacfb8553a1f72902fa8795040c1e36aa282f08d214f1aa6e",
		Lines:  "55-82 class WorkingMemory: ALIGNMENT, next/next_TC/temp, reservation_bytes, layout, is_input_to_rev_lstm, N/T/C",
	},
	{
		Path:   "dorado/nn/ConvStack.cpp",
		SHA256: "d71cbdb1100010991753e4b3b80b8866265d97a08855f0b8ff1e41c00c91f6b2",
		Lines:  "191-211 ConvLayer::reserve_working_memory",
	},
	{
		Path:   "dorado/nn/FLSTMStack.cpp",
		SHA256: "97c2df7d81b7e8bb7c1d3c807ff4e80f39973df0bbc63cd2e2aa38d203a20160",
		Lines:  "97-109 FLSTMStackImpl::reserve_working_memory",
	},
	{
		Path:   "dorado/basecall/model/CRFModel.cpp",
		SHA256: "37f1f9712c3e61957ff17099d4cffb606c6cc784d2973c040bcc77129b5f9964",
		Lines:  "88-106 one WorkingMemory per invocation: reserve traversal, then allocate_backing_tensor",
	},
}

// VerifySources checks the pinned checkout before the oracle is trusted. It is
// the source-pin gate for a maintained verification run; it performs no native
// build and executes no model.
func VerifySources(root string) error {
	for _, source := range Sources {
		data, err := os.ReadFile(filepath.Join(root, source.Path))
		if err != nil {
			return err
		}
		sum := sha256.Sum256(data)
		if got := hex.EncodeToString(sum[:]); got != source.SHA256 {
			return fmt.Errorf("Dorado source hash mismatch for %s: %s != %s", source.Path, got, source.SHA256)
		}
	}
	return nil
}

// Layout mirrors dorado::nn::TensorLayout (WorkingMemory.h:9).
type Layout uint8

const (
	NTC Layout = iota
	TNC
	CUTLASSTNCF16
	CUTLASSTNCI8
	CUBLASTN2C
	CUBLASTNC
)

// Dtype is the source element type of one extent.
type Dtype uint8

const (
	Half Dtype = iota
	Int8
)

func (d Dtype) elementSize() (int64, error) {
	switch d {
	case Half:
		return 2, nil
	case Int8:
		return 1, nil
	default:
		return 0, fmt.Errorf("hacgpu: unknown element type %d", uint8(d))
	}
}

// Op is one reservation-phase call: next_TC(T, C, layout) when Scratch is
// false, temp(Sizes, Dtype) when it is true. Orientation is not part of the
// reservation: next_TC never reads is_input_to_rev_lstm
// (WorkingMemory.cpp:83-102), it only selects rows inside the storage.
type Op struct {
	Scratch bool
	Layout  Layout
	T       int64
	C       int64
	Sizes   []int64
	Dtype   Dtype
}

// Step is one bound extent, in bytes and elements.
type Step struct {
	Bytes     int64
	Elems     int64
	LiveBytes int64
	Rows      int64
	Offset    int64
	Front     bool
	Scratch   bool
}

// Trace is the reservation maximum plus the use-phase extents.
type Trace struct {
	Reservation int64
	PeakLive    int64
	Steps       []Step
}

// padTo is utils::pad_to (utils/math_utils.h:23-26), the padding
// WorkingMemory::tensor_bytes applies.
func padTo(a, b int64) (int64, error) {
	if b <= 0 || a < 0 || a > math.MaxInt64-(b-1) {
		return 0, fmt.Errorf("hacgpu: padding overflows int64")
	}
	return ((a + b - 1) / b) * b, nil
}

func multiply(values []int64) (int64, error) {
	product := int64(1)
	for _, value := range values {
		if value < 0 || (product != 0 && value > math.MaxInt64/product) {
			return 0, fmt.Errorf("hacgpu: element count overflows int64")
		}
		product *= value
	}
	return product, nil
}

// sizes is the IntArrayRef next_TC passes to next (WorkingMemory.cpp:87-98),
// or the sizes temp passes through unchanged (WorkingMemory.cpp:104-106).
func sizes(op Op, n int64) ([]int64, error) {
	if op.Scratch {
		if len(op.Sizes) == 0 {
			return nil, fmt.Errorf("hacgpu: temp needs at least one size")
		}
		for _, size := range op.Sizes {
			if size < 1 {
				return nil, fmt.Errorf("hacgpu: temp size %d", size)
			}
		}
		return op.Sizes, nil
	}
	if op.T < 1 || op.C < 1 || n < 1 {
		return nil, fmt.Errorf("hacgpu: invalid shape N=%d T=%d C=%d", n, op.T, op.C)
	}
	switch op.Layout {
	case NTC:
		return []int64{n, op.T, op.C}, nil
	case TNC:
		return []int64{op.T, n, op.C}, nil
	case CUTLASSTNCF16, CUTLASSTNCI8:
		return []int64{op.T + 5, n, op.C}, nil
	case CUBLASTN2C:
		return []int64{op.T + 1, n, 2, op.C}, nil
	case CUBLASTNC:
		return []int64{op.T + 3, n, op.C}, nil
	default:
		return nil, fmt.Errorf("hacgpu: unknown layout %d", uint8(op.Layout))
	}
}

// layoutDtype is the element type next_TC passes to next.
func layoutDtype(layout Layout) Dtype {
	if layout == CUTLASSTNCI8 {
		return Int8
	}
	return Half
}

// storageRows is the first dimension next_TC reserves: the time rows plus the
// layout slack (WorkingMemory.cpp:87-98).
func storageRows(layout Layout, t int64) (int64, error) {
	switch layout {
	case NTC, TNC:
		return t, nil
	case CUTLASSTNCF16, CUTLASSTNCI8:
		return t + 5, nil
	case CUBLASTN2C:
		return t + 1, nil
	case CUBLASTNC:
		return t + 3, nil
	default:
		return 0, fmt.Errorf("hacgpu: unknown layout %d", uint8(layout))
	}
}

type extent struct {
	bytes     int64
	elems     int64
	liveBytes int64
	rows      int64
}

// Plan replays the reservation traversal and then the use phase exactly as
// WorkingMemory does: reservation_bytes is the maximum of current_bytes plus
// the new extent (WorkingMemory.cpp:38-40), and the use phase alternates ends
// from the 256-byte-padded reservation (WorkingMemory.cpp:42-58).
func Plan(n int64, ops []Op) (Trace, error) {
	if n < 1 {
		return Trace{}, fmt.Errorf("hacgpu: batch size %d", n)
	}
	extents := make([]extent, len(ops))

	// Reservation phase: no backing tensor is defined.
	var reservation, current, peakLive, currentLive int64
	for i, op := range ops {
		shape, err := sizes(op, n)
		if err != nil {
			return Trace{}, err
		}
		elems, err := multiply(shape)
		if err != nil {
			return Trace{}, err
		}
		dtype := op.Dtype
		if !op.Scratch {
			dtype = layoutDtype(op.Layout)
		}
		elemSize, err := dtype.elementSize()
		if err != nil {
			return Trace{}, err
		}
		if elems > math.MaxInt64/elemSize {
			return Trace{}, fmt.Errorf("hacgpu: byte count overflows int64")
		}
		live := elems * elemSize
		bytes, err := padTo(live, Alignment)
		if err != nil {
			return Trace{}, err
		}
		rows := int64(1)
		if !op.Scratch {
			rows, err = storageRows(op.Layout, op.T)
			if err != nil {
				return Trace{}, err
			}
		}
		extents[i] = extent{bytes: bytes, elems: elems, liveBytes: live, rows: rows}
		if current > math.MaxInt64-bytes {
			return Trace{}, fmt.Errorf("hacgpu: reservation overflows int64")
		}
		if pair := current + bytes; pair > reservation {
			reservation = pair
		}
		if currentLive > math.MaxInt64-live {
			return Trace{}, fmt.Errorf("hacgpu: live pair overflows int64")
		}
		if pair := currentLive + live; pair > peakLive {
			peakLive = pair
		}
		if !op.Scratch {
			current, currentLive = bytes, live
		}
	}

	// Use phase: backing_tensor is defined and current_bytes starts at zero
	// (WorkingMemory.cpp:108-114).
	trace := Trace{Reservation: reservation, PeakLive: peakLive, Steps: make([]Step, 0, len(ops))}
	var boundBytes int64
	defined := false
	front := false
	for i, op := range ops {
		e := extents[i]
		if defined && boundBytes+e.bytes > reservation {
			return Trace{}, fmt.Errorf("WorkingMemory: overlap detected at step %d", i)
		}
		offset := int64(0)
		if defined && front {
			offset = reservation - e.bytes
		}
		trace.Steps = append(trace.Steps, Step{
			Bytes:     e.bytes,
			Elems:     e.elems,
			LiveBytes: e.liveBytes,
			Rows:      e.rows,
			Offset:    offset,
			Front:     offset == 0,
			Scratch:   op.Scratch,
		})
		if !op.Scratch {
			boundBytes = e.bytes
			defined = true
			front = offset == 0
		}
	}
	return trace, nil
}
