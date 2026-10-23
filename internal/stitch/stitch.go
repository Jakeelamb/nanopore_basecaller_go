// Package stitch ports Dorado v2.1.2 decoded-chunk assembly.
package stitch

import "fmt"

// Config describes the signal after trimming and the model's sample stride.
type Config struct {
	RawSamples uint64
	Stride     uint64
	Variable   bool
}

// Chunk contains decoder output in input-offset order. RawSize excludes repeat
// padding for variable chunks; Granularity describes that padding's alignment.
type Chunk struct {
	Offset, RawSize, Granularity uint64
	Sequence, Qstring            string
	Moves                        []uint8
}

type Result struct {
	Sequence, Qstring string
	Moves             []uint8
}

// Dorado casts signal coordinates, sequence lengths and move counts to int.
// Limit inputs to its signed 32-bit domain rather than reproduce overflow.
const maxNativeInt uint64 = 1<<31 - 1

// Stitch joins decoded chunks without mutating or retaining their backing data.
// The accepted domain has binary moves and complete, stride-aligned coverage.
// Empty chunks produce an empty result, matching Dorado's early return.
func Stitch(config Config, chunks []Chunk) (Result, error) {
	if len(chunks) == 0 {
		return Result{}, nil
	}
	if config.RawSamples == 0 || config.RawSamples > maxNativeInt || config.Stride == 0 || config.Stride > maxNativeInt {
		return Result{}, fmt.Errorf("invalid signal size or model stride")
	}
	if uint64(len(chunks)) > maxNativeInt {
		return Result{}, fmt.Errorf("chunk count exceeds native integer domain")
	}
	for i, c := range chunks {
		if c.RawSize == 0 || c.RawSize > maxNativeInt || c.Granularity == 0 || c.Granularity > maxNativeInt || c.Granularity%config.Stride != 0 {
			return Result{}, fmt.Errorf("chunk %d: invalid raw size or granularity", i)
		}
		if c.Offset >= config.RawSamples || c.RawSize > maxNativeInt-c.Offset {
			return Result{}, fmt.Errorf("chunk %d: coordinate overflow or offset outside signal", i)
		}
		padded := c.RawSize
		if remainder := padded % c.Granularity; remainder != 0 {
			increment := c.Granularity - remainder
			if increment > maxNativeInt-padded {
				return Result{}, fmt.Errorf("chunk %d: padded size overflow", i)
			}
			padded += increment
		}
		if uint64(len(c.Moves)) != padded/config.Stride {
			return Result{}, fmt.Errorf("chunk %d: moves do not match padded model length", i)
		}
		if len(c.Sequence) != len(c.Qstring) || uint64(len(c.Sequence)) > maxNativeInt {
			return Result{}, fmt.Errorf("chunk %d: sequence and quality lengths differ or overflow", i)
		}
		bases := uint64(0)
		for _, move := range c.Moves {
			if move > 1 {
				return Result{}, fmt.Errorf("chunk %d: non-binary move", i)
			}
			bases += uint64(move)
		}
		if bases != uint64(len(c.Sequence)) {
			return Result{}, fmt.Errorf("chunk %d: move count differs from sequence length", i)
		}
		if i == 0 && c.Offset != 0 {
			return Result{}, fmt.Errorf("first chunk must start at signal origin")
		}
		if i > 0 {
			previous := chunks[i-1]
			end := previous.Offset + previous.RawSize
			if c.Offset <= previous.Offset || c.Offset > end || (end-c.Offset)%config.Stride != 0 {
				return Result{}, fmt.Errorf("chunk %d: unordered, gapped or unaligned overlap", i)
			}
		}
		if config.Variable && c.Offset+c.RawSize > config.RawSamples {
			return Result{}, fmt.Errorf("chunk %d: variable interval exceeds signal", i)
		}
	}
	last := chunks[len(chunks)-1]
	if last.Offset+last.RawSize < config.RawSamples {
		return Result{}, fmt.Errorf("last chunk does not cover signal end")
	}

	// Store only cut positions; these are the same asymmetric midpoint cuts as
	// stitch.cpp. Validate all cuts before allocating the output.
	starts := make([]int, len(chunks))
	ends := make([]int, len(chunks))
	var moveCount, baseCount uint64
	for i, c := range chunks {
		ends[i] = len(c.Moves)
		if i+1 < len(chunks) {
			overlap := (c.Offset + c.RawSize - chunks[i+1].Offset) / config.Stride
			rear := overlap / 2
			front := overlap - rear
			if rear > uint64(len(c.Moves)) || front > uint64(len(chunks[i+1].Moves)) {
				return Result{}, fmt.Errorf("chunk %d: overlap exceeds decoded moves", i)
			}
			ends[i] -= int(rear)
			starts[i+1] = int(front)
		} else if len(chunks) == 1 || config.Variable {
			signalSize := min(config.RawSamples, c.RawSize)
			ends[i] = int((signalSize + config.Stride - 1) / config.Stride)
		}
		if starts[i] > ends[i] || ends[i] > len(c.Moves) {
			return Result{}, fmt.Errorf("chunk %d: overlap cuts cross or exceed decoded moves", i)
		}
		moveCount += uint64(ends[i] - starts[i])
		for _, move := range c.Moves[starts[i]:ends[i]] {
			baseCount += uint64(move)
		}
		if moveCount > maxNativeInt || baseCount > maxNativeInt {
			return Result{}, fmt.Errorf("stitched output exceeds native integer domain")
		}
	}
	// Native removes at most one partial-stride move. Reject malformed coverage
	// that would need a broader repair instead of silently accepting it.
	wantMoves := config.RawSamples / config.Stride
	if moveCount < wantMoves || moveCount > wantMoves+1 {
		return Result{}, fmt.Errorf("stitched moves do not cover the signal")
	}
	moves := make([]uint8, 0, int(moveCount))
	sequence := make([]byte, 0, int(baseCount))
	quality := make([]byte, 0, int(baseCount))
	for i, c := range chunks {
		startBase := 0
		for _, move := range c.Moves[:starts[i]] {
			startBase += int(move)
		}
		keptBases := 0
		for _, move := range c.Moves[starts[i]:ends[i]] {
			keptBases += int(move)
		}
		sequence = append(sequence, c.Sequence[startBase:startBase+keptBases]...)
		quality = append(quality, c.Qstring[startBase:startBase+keptBases]...)
		moves = append(moves, c.Moves[starts[i]:ends[i]]...)
	}
	if moveCount > wantMoves {
		if moves[len(moves)-1] == 1 {
			sequence = sequence[:len(sequence)-1]
			quality = quality[:len(quality)-1]
		}
		moves = moves[:len(moves)-1]
	}
	return Result{Sequence: string(sequence), Qstring: string(quality), Moves: moves}, nil
}
