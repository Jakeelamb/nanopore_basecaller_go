package model

import "fmt"

const supQKVElements = 3 * 8 * 64 * 512

type supQKVF16Weights struct {
	interleavedHalf  []uint16
	tiledRoundedHalf []uint16
}

type qkvWeightShapeError struct {
	gotElements, wantElements int
}

func (e *qkvWeightShapeError) Error() string {
	return fmt.Sprintf("SUP QKV half weights: got %d elements, want %d", e.gotElements, e.wantElements)
}

func prepareSUPQKVF16Weights(rawHalf []uint16) (supQKVF16Weights, error) {
	if len(rawHalf) != supQKVElements {
		return supQKVF16Weights{}, &qkvWeightShapeError{len(rawHalf), supQKVElements}
	}
	out := supQKVF16Weights{make([]uint16, supQKVElements), make([]uint16, supQKVElements)}
	for q := 0; q < 3; q++ {
		for h := 0; h < 8; h++ {
			for d := 0; d < 64; d++ {
				sourceD := d
				if q < 2 {
					sourceD = d/2 + (d%2)*32
				}
				dst := ((q*8+h)*64 + d) * 512
				src := ((q*8+h)*64 + sourceD) * 512
				copy(out.interleavedHalf[dst:dst+512], rawHalf[src:src+512])
				for c := 0; c < 512; c++ {
					tiled := ((((q*8+h)*4+d/16)*64+c/8)*16+d%16)*8 + c%8
					out.tiledRoundedHalf[tiled] = (out.interleavedHalf[dst+c] + 8) & 0xfff0
				}
			}
		}
	}
	return out, nil
}
