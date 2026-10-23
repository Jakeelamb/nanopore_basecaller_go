// Package chunk ports Dorado v2.1.2 chunk coordinates and repeat padding.
package chunk

import (
	"fmt"
	"math"
)

// Limits bound coordinates and allocations at this package's external boundary.
const (
	MaxCoordinate    uint64 = 1<<53 - 1
	MaxChunks        uint64 = 1_000_000
	MaxPaddedSamples uint64 = 1 << 26
)

type Config struct{ ChunkSize, Stride, Overlap, Granularity uint64 }
type Interval struct{ Start, End uint64 }

func validate(n uint64, c Config) error {
	if n == 0 {
		return fmt.Errorf("empty read")
	}
	if n > MaxCoordinate || c.ChunkSize > MaxCoordinate || c.Overlap > MaxCoordinate || c.Stride > MaxCoordinate || c.Granularity > MaxCoordinate {
		return fmt.Errorf("coordinate exceeds MaxCoordinate (%d)", MaxCoordinate)
	}
	if c.Stride == 0 {
		return fmt.Errorf("invalid stride")
	}
	if c.ChunkSize == 0 || c.ChunkSize%c.Stride != 0 || c.ChunkSize <= c.Overlap {
		return fmt.Errorf("invalid chunk size or overlap")
	}
	return nil
}

// Fixed returns equal-size chunk starts, shifting the last start to a stride boundary.
func Fixed(n uint64, c Config) ([]uint64, error) {
	if err := validate(n, c); err != nil {
		return nil, err
	}
	if c.Overlap%c.Stride != 0 {
		return nil, fmt.Errorf("invalid overlap")
	}
	last := uint64(0)
	if n > c.ChunkSize {
		last = n - c.ChunkSize
	}
	if m := last % c.Stride; m != 0 {
		last += c.Stride - m
	}
	step := c.ChunkSize - c.Overlap
	count := uint64(1)
	if last > 0 {
		count += (last + step - 1) / step
	}
	if count > MaxChunks {
		return nil, fmt.Errorf("chunk count exceeds MaxChunks (%d)", MaxChunks)
	}
	offsets := make([]uint64, 0, int(count))
	offsets = append(offsets, 0)
	for offset := uint64(0); offset+c.ChunkSize < n; {
		offset = min(offset+step, last)
		offsets = append(offsets, offset)
	}
	return offsets, nil
}

// Variable balances signal across chunks, then adjusts boundaries to the stride.
func Variable(n uint64, c Config) ([]Interval, error) {
	if err := validate(n, c); err != nil {
		return nil, err
	}
	if c.ChunkSize == c.Stride || c.Overlap%c.Stride != 0 || (c.Stride != 1 && c.Overlap == 0) {
		return nil, fmt.Errorf("invalid variable chunk configuration")
	}
	count := uint64(1)
	if n > c.ChunkSize {
		count += uint64(math.Ceil(float64(n-c.ChunkSize) / float64(c.ChunkSize-c.Overlap)))
	}
	if count > MaxChunks {
		return nil, fmt.Errorf("chunk count exceeds MaxChunks (%d)", MaxChunks)
	}
	if c.Overlap != 0 && count-1 > (math.MaxUint64-n)/c.Overlap {
		return nil, fmt.Errorf("overlap total overflow")
	}
	total := n + (count-1)*c.Overlap
	longer, size := total%count, total/count
	intervals := make([]Interval, int(count))
	start := uint64(0)
	for i := range intervals {
		end := start + size
		if uint64(i) < longer {
			end++
		}
		intervals[i] = Interval{start, end}
		// The final subtraction is unused and can underflow for one short chunk.
		if i+1 < len(intervals) {
			start = end - c.Overlap
		}
	}
	for i := 1; i < len(intervals); i++ {
		if m := intervals[i].Start % c.Stride; m != 0 {
			intervals[i].Start += c.Stride - m
		}
	}
	for i := 0; i+1 < len(intervals); i++ {
		intervals[i].End -= intervals[i].End % c.Stride
	}
	return intervals, nil
}

// Transformer uses fixed steps with a variable final end. Granularity constrains the maximum size.
func Transformer(n uint64, c Config) ([]Interval, error) {
	if err := validate(n, c); err != nil {
		return nil, err
	}
	if c.Granularity == 0 || c.Granularity%c.Stride != 0 || c.ChunkSize%c.Granularity != 0 {
		return nil, fmt.Errorf("invalid chunk granularity")
	}
	step := c.ChunkSize - c.Overlap
	count := uint64(1)
	if n > c.ChunkSize {
		count += (n - c.ChunkSize + step - 1) / step
	}
	if count > MaxChunks {
		return nil, fmt.Errorf("chunk count exceeds MaxChunks (%d)", MaxChunks)
	}
	intervals := make([]Interval, 0, int(count))
	offset := uint64(0)
	for offset+c.ChunkSize < n {
		intervals = append(intervals, Interval{offset, offset + c.ChunkSize})
		offset += step
	}
	if offset != n {
		intervals = append(intervals, Interval{offset, n})
	}
	return intervals, nil
}

// RepeatPad repeats half-precision sample bits to exactly target samples.
// It also represents variable-chunk padding when target is the rounded-up size.
func RepeatPad(signal []uint16, target uint64) ([]uint16, error) {
	if len(signal) == 0 {
		return nil, fmt.Errorf("empty signal")
	}
	if target < uint64(len(signal)) {
		return nil, fmt.Errorf("padding target shorter than signal")
	}
	if target > MaxPaddedSamples {
		return nil, fmt.Errorf("padding target exceeds MaxPaddedSamples (%d)", MaxPaddedSamples)
	}
	out := make([]uint16, int(target))
	for i := range out {
		out[i] = signal[i%len(signal)]
	}
	return out, nil
}
