// Package readtrim implements pinned Dorado post-call read trimming.
package readtrim

import (
	"fmt"
	"math"
)

// MuxRead borrows its move table and normalized binary16 signal. Signal is not
// raw POD5 int16 data. Metadata not represented here remains caller-owned.
type MuxRead struct {
	Sequence              string
	Qstring               string
	Moves                 []uint8
	Signal                []uint16
	Stride                uint32
	IsMuxChange           bool
	NumMinknowEvents      uint64
	PreTrimSequenceLength uint64
}

// MuxResult contains half-open prefix ends into the input. Read borrows the
// input's storage, including on no-op. Neither this function nor its result
// grants exclusive ownership. Callers must copy before mutating shared slices.
// NumSamples is the resulting current signal length, without leading trims.
type MuxResult struct {
	Read        MuxRead
	Trimmed     bool
	SequenceEnd int
	MoveEnd     int
	SignalEnd   int
	NumSamples  uint64
}

// MuxChangeTrim preserves Dorado 8b8fc5d's mux tail algorithm, including its
// exclusive i-1 trim end and binary32 excessive-trim threshold. It validates
// even no-op reads and never modifies input storage or metadata.
func MuxChangeTrim(read MuxRead) (MuxResult, error) {
	if err := validate(read); err != nil {
		return MuxResult{}, err
	}
	result := MuxResult{Read: read, SequenceEnd: len(read.Sequence), MoveEnd: len(read.Moves), SignalEnd: len(read.Signal), NumSamples: uint64(len(read.Signal))}
	n := len(read.Qstring)
	if !read.IsMuxChange || n < 100 {
		return result, nil
	}
	end := muxIndex(read.Qstring)
	// The multiplication must round to binary32 before floor, as in C++.
	limit := int(math.Floor(float64(float32(n) * float32(0.3))))
	if end < limit || end >= n-5 {
		return result, nil
	}
	moveEnd := moveIndex(read.Moves, end, n)
	if moveEnd < 0 {
		return result, nil
	}
	signalEnd := int(uint64(moveEnd) * uint64(read.Stride))
	if signalEnd > len(read.Signal) {
		signalEnd = len(read.Signal)
	}
	result.Read.Sequence = read.Sequence[:end]
	result.Read.Qstring = read.Qstring[:end]
	result.Read.Moves = read.Moves[:moveEnd:moveEnd]
	result.Read.Signal = read.Signal[:signalEnd:signalEnd]
	result.Read.PreTrimSequenceLength = uint64(end)
	result.Read.NumMinknowEvents = 0
	result.Trimmed = true
	result.SequenceEnd, result.MoveEnd, result.SignalEnd = end, moveEnd, signalEnd
	result.NumSamples = uint64(signalEnd)
	return result, nil
}

func validate(read MuxRead) error {
	n := len(read.Sequence)
	if n != len(read.Qstring) {
		return fmt.Errorf("sequence/quality length mismatch")
	}
	// Native string slicing uses int and its quality accumulator uses int.
	if n > math.MaxInt32-10 || len(read.Moves) > math.MaxInt32 {
		return fmt.Errorf("native count overflow")
	}
	if read.Stride == 0 || read.Stride > math.MaxInt32 {
		return fmt.Errorf("stride outside native positive int domain")
	}
	for i := range read.Sequence {
		switch read.Sequence[i] {
		case 'A', 'C', 'G', 'T':
		default:
			return fmt.Errorf("invalid DNA base at %d", i)
		}
		if read.Qstring[i] < 33 || read.Qstring[i] > 126 {
			return fmt.Errorf("invalid quality at %d", i)
		}
	}
	bases := 0
	for i, m := range read.Moves {
		if m > 1 {
			return fmt.Errorf("nonbinary move at %d", i)
		}
		bases += int(m)
	}
	if bases != n {
		return fmt.Errorf("move base count %d differs from sequence length %d", bases, n)
	}
	samples, stride, moves := uint64(len(read.Signal)), uint64(read.Stride), uint64(len(read.Moves))
	if moves > math.MaxInt64/stride {
		return fmt.Errorf("sample count overflow")
	}
	floor := samples / stride
	ceil := floor
	if samples%stride != 0 {
		ceil++
	}
	if moves != floor && moves != ceil {
		return fmt.Errorf("signal/move stride mismatch")
	}
	for i, v := range read.Signal {
		if v&0x7c00 == 0x7c00 {
			return fmt.Errorf("nonfinite binary16 signal at %d", i)
		}
	}
	return nil
}

func muxIndex(q string) int {
	end, cumulative, minimum := len(q)-1, 0, -1
	for i := len(q) - 1; i >= 0; i-- {
		switch {
		case q[i] <= 40:
			cumulative--
		case q[i] <= 45:
			cumulative++
		default:
			cumulative += 10
		}
		if cumulative <= minimum {
			minimum = cumulative
			end = i - 1
		}
		if cumulative > i {
			break
		}
	}
	return end
}

func moveIndex(moves []uint8, index, size int) int {
	if len(moves) == 0 || index >= len(moves) || index >= size || size > len(moves) {
		return -1
	}
	if index <= size/2 {
		position := -1
		for i, m := range moves {
			if m == 1 {
				position++
				if position == index {
					return i
				}
			}
		}
	} else {
		position := size
		for i := len(moves) - 1; i >= 0; i-- {
			if moves[i] == 1 {
				position--
				if position == index {
					return i
				}
			}
		}
	}
	return -1
}
