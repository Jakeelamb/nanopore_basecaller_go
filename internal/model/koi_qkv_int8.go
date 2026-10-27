package model

import (
	"errors"
	"fmt"
)

const (
	SUPQKVRows = 3 * 8 * 64
	SUPQKVCols = 512
)

var ErrSUPQKVInt8 = errors.New("model: SUP QKV int8 cache rejected")

type QKVRowRejection string

const (
	QKVNonfiniteRow     QKVRowRejection = "nonfinite"
	QKVZeroRow          QKVRowRejection = "zero"
	QKVScaleOverflowRow QKVRowRejection = "scale-overflow"
)

type QKVRowError struct {
	Row    int
	Reason QKVRowRejection
}

func (e *QKVRowError) Error() string {
	return fmt.Sprintf("%v: row %d is %s", ErrSUPQKVInt8, e.Row, e.Reason)
}

func (e *QKVRowError) Unwrap() error { return ErrSUPQKVInt8 }

// QuantizeSUPQKVRows quantizes contiguous 512-value half rows and returns one
// float32 inverse scale per row. Every value must be finite, and each absolute
// row maximum must have half bits 0x1801 through 0x7bff.
func QuantizeSUPQKVRows(half []uint16) ([]int8, []float32, error) {
	if len(half) == 0 || len(half)%SUPQKVCols != 0 {
		return nil, nil, fmt.Errorf("%w: input must contain complete nonempty 512-value rows", ErrSUPQKVInt8)
	}
	rows := len(half) / SUPQKVCols
	for row := 0; row < rows; row++ {
		var maximum uint16
		for _, bits := range half[row*SUPQKVCols : (row+1)*SUPQKVCols] {
			magnitude := bits & 0x7fff
			if magnitude >= 0x7c00 {
				return nil, nil, &QKVRowError{row, QKVNonfiniteRow}
			}
			if magnitude > maximum {
				maximum = magnitude
			}
		}
		if maximum == 0 {
			return nil, nil, &QKVRowError{row, QKVZeroRow}
		}
		if maximum < 0x1801 {
			return nil, nil, &QKVRowError{row, QKVScaleOverflowRow}
		}
	}
	q, inverseScale, err := QuantizeFLSTMRows(half, int64(rows), SUPQKVCols)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: %w", ErrSUPQKVInt8, err)
	}
	for i := range inverseScale {
		inverseScale[i] = 1 / inverseScale[i]
	}
	return q, inverseScale, nil
}

// SUPQKVInt8 holds contiguous int8 [3,8,4,32,16,16] weights and float32
// [3,8,64] inverse scales from TxModules.cpp:455-461.
type SUPQKVInt8 struct {
	Tiled        []int8
	InverseScale []float32
}

// BuildSUPQKVInt8 consumes already interleaved, unrounded half [3,8,64,512].
func BuildSUPQKVInt8(half []uint16) (SUPQKVInt8, error) {
	if len(half) != SUPQKVRows*SUPQKVCols {
		return SUPQKVInt8{}, fmt.Errorf("%w: want half[3,8,64,512], got %d elements", ErrSUPQKVInt8, len(half))
	}
	q, inverse, err := QuantizeSUPQKVRows(half)
	if err != nil {
		return SUPQKVInt8{}, err
	}
	tiled, err := permuteContiguous(q, []int64{3, 8, 4, 16, 32, 16}, []int{0, 1, 2, 4, 3, 5})
	if err != nil {
		return SUPQKVInt8{}, err
	}
	return SUPQKVInt8{Tiled: tiled, InverseScale: inverse}, nil
}
