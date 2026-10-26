package s02

import (
	"math"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
)

func mutations(half []uint16, q []int8, reciprocal, scale []float32, rows []int, char, inverse, tile []byte) []MutationResult {
	result := []MutationResult{{Name: "reciprocal-direction"}, {Name: "float32-product"}, {Name: "truncate"}, {Name: "clip-128"}, {Name: "tile-axis"}, {Name: "early-input-bit-rounding"}}
	packedRow := make(map[int]int, len(rows))
	for p, row := range rows {
		packedRow[row] = p
	}
	for p, row := range rows {
		if math.Float32bits(scale[p]) != math.Float32bits(reciprocal[p]) {
			result[0].DifferentInverseBits++
		}
		for c := 0; c < 512; c++ {
			i := p*512 + c
			product := signal.Float32FromHalf(half[i]) * scale[p]
			halfProduct := signal.Float32FromHalf(signal.Float16Bits(product))
			for mutant, value := range []float64{math.RoundToEven(float64(product)), math.Trunc(float64(halfProduct)), math.RoundToEven(float64(halfProduct))} {
				limit := float64(127)
				if mutant == 2 {
					limit = 128
				}
				if value > limit {
					value = limit
				}
				if value < -limit {
					value = -limit
				}
				if byte(int8(value)) != char[row*512+c] {
					result[mutant+1].DifferentI8++
				}
			}
			at := tileAt(row, c)
			var identity int8
			if packed, ok := packedRow[at/512]; ok {
				identity = q[packed*512+at%512]
			} else if at%512 == 0 {
				identity = 127
			}
			if byte(identity) != tile[at] {
				result[4].DifferentI8++
			}
		}
		rounded := append([]uint16(nil), half[p*512:(p+1)*512]...)
		for i, v := range rounded {
			rounded[i] = (v + 8) & 0xfff0
		}
		wrong, inv, err := model.QuantizeSUPQKVRows(rounded)
		if err != nil {
			result[5].RejectedRows++
			continue
		}
		for c, v := range wrong {
			if byte(v) != char[row*512+c] {
				result[5].DifferentI8++
			}
		}
		if math.Float32bits(inv[0]) != math.Float32bits(reciprocal[p]) {
			result[5].DifferentInverseBits++
		}
	}
	return result
}
