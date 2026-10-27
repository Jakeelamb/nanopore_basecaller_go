package model

import (
	"fmt"
	"math"
	"slices"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
)

// HalfToSUPFP8 converts the already-cast half weights to Torch E4M3FN bits.
// Its integer rounding and denormal addition follow the pinned Torch header;
// nonfinite and overflow encodings are preserved here for differential checks.
func HalfToSUPFP8(half uint16) byte {
	bits := math.Float32bits(signal.Float32FromHalf(half))
	sign := bits & 0x80000000
	bits ^= sign
	var out byte
	switch {
	case bits >= 1087<<20:
		out = 0x7f
	case bits < 121<<23:
		const mask uint32 = 141 << 23
		rounded := float32(math.Float32frombits(bits) + math.Float32frombits(mask))
		out = byte(math.Float32bits(rounded) - mask)
	default:
		odd := (bits >> 20) & 1
		bits += uint32(0xc407ffff) + odd
		out = byte(bits >> 20)
	}
	return out | byte(sign>>24)
}

type SUPFeedForwardFP8Cache struct{ fc1, fc2 []byte }
type SUPFeedForwardFP8Snapshot struct {
	FC1, FC2           []byte
	FC1Shape, FC2Shape []int64
}

func (c SUPFeedForwardFP8Cache) Snapshot() SUPFeedForwardFP8Snapshot {
	return SUPFeedForwardFP8Snapshot{slices.Clone(c.fc1), slices.Clone(c.fc2), []int64{256, 32, 16, 16}, []int64{32, 128, 16, 16}}
}

// PackSUPFeedForwardFP8 implements TxModules.cpp:488-506 for C512/E4096.
// Inputs are the source's half-cast parameter arrays, before remove_bits.
// This cache is selected only after independently observed E4M3 support.
func PackSUPFeedForwardFP8(fc1Half, fc2Half []uint16) (SUPFeedForwardFP8Cache, error) {
	const c, e = 512, 4096
	if len(fc1Half) != e*c || len(fc2Half) != c*(e/2) {
		return SUPFeedForwardFP8Cache{}, fmt.Errorf("SUP FP8 requires fc1[4096,512] and fc2[512,2048]")
	}
	for role, values := range [][]uint16{fc1Half, fc2Half} {
		for i, v := range values {
			if v&0x7c00 == 0x7c00 || HalfToSUPFP8(v)&0x7f == 0x7f {
				return SUPFeedForwardFP8Cache{}, fmt.Errorf("SUP FP8 role%d element%d is outside finite E4M3 cache domain", role, i)
			}
		}
	}
	out := SUPFeedForwardFP8Cache{make([]byte, len(fc1Half)), make([]byte, len(fc2Half))}
	shuffle := [16]int{0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15}
	for block := 0; block < e/16; block++ {
		for colBlock := 0; colBlock < c/16; colBlock++ {
			for lane := 0; lane < 16; lane++ {
				for colLane := 0; colLane < 16; colLane++ {
					originalRow := (block%2)*(e/2) + (block/2)*16 + shuffle[lane]
					source := originalRow*c + colBlock*16 + colLane
					target := ((block*(c/16)+colBlock)*16+lane)*16 + colLane
					out.fc1[target] = HalfToSUPFP8(fc1Half[source])
				}
			}
		}
	}
	for rowBlock := 0; rowBlock < c/16; rowBlock++ {
		for colBlock := 0; colBlock < e/32; colBlock++ {
			for rowLane := 0; rowLane < 16; rowLane++ {
				for colLane := 0; colLane < 16; colLane++ {
					source := (rowBlock*16+rowLane)*(e/2) + colBlock*16 + colLane
					target := ((rowBlock*(e/32)+colBlock)*16+rowLane)*16 + colLane
					out.fc2[target] = HalfToSUPFP8(fc2Half[source])
				}
			}
		}
	}
	return out, nil
}
