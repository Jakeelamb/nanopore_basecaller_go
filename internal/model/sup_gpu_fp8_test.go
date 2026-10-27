package model

import (
	"bytes"
	"math"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
)

func fp8ReferenceValues() [128]float64 {
	var values [128]float64
	for code := 0; code < 128; code++ {
		exponent, fraction := code>>3, code&7
		if exponent == 0 {
			values[code] = math.Ldexp(float64(fraction), -9)
		} else {
			values[code] = math.Ldexp(1+float64(fraction)/8, exponent-7)
		}
	}
	return values
}
func TestSUPFP8HalfConversionExhaustiveNearestEvenDefinition(t *testing.T) {
	values := fp8ReferenceValues()
	for raw := 0; raw < 65536; raw++ {
		half := uint16(raw)
		value := math.Abs(float64(signal.Float32FromHalf(half)))
		best := 127
		if !math.IsNaN(value) && value < 480 {
			best = 0
			distance := value
			for code := 1; code < 128; code++ {
				d := math.Abs(value - values[code])
				if d < distance || d == distance && code&1 == 0 {
					best, distance = code, d
				}
			}
		}
		want := byte(best) | byte(half>>8)&0x80
		if got := HalfToSUPFP8(half); got != want {
			t.Fatalf("half%04x got%02x want%02x", half, got, want)
		}
	}
}

// permuteFP8Reference composes the source's tensor views independently of the
// production packer's direct original-row and tile-offset formulas.
func permuteFP8Reference(src []byte, shape, axes []int) []byte {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	out := make([]byte, len(src))
	for flat := range out {
		remaining, original := flat, 0
		for i := len(axes) - 1; i >= 0; i-- {
			axis := axes[i]
			coord := remaining % shape[axis]
			remaining /= shape[axis]
			original += coord * strides[axis]
		}
		out[flat] = src[original]
	}
	return out
}
func TestSUPFP8FeedForwardPackingMatchesSourceViewSequence(t *testing.T) {
	values := fp8ReferenceValues()
	input := func(n int) []uint16 {
		out := make([]uint16, n)
		for i := range out {
			mixed := uint32(i) * 2654435761
			mixed ^= mixed >> 16
			code := 1 + mixed%126
			out[i] = signal.Float16Bits(float32(values[code]))
		}
		return out
	}
	fc1, fc2 := input(4096*512), input(512*2048)
	before1, before2 := append([]uint16(nil), fc1...), append([]uint16(nil), fc2...)
	cache, err := PackSUPFeedForwardFP8(fc1, fc2)
	if err != nil {
		t.Fatal(err)
	}
	convert := func(in []uint16) []byte {
		out := make([]byte, len(in))
		for i, v := range in {
			out[i] = HalfToSUPFP8(v)
		}
		return out
	}
	interleaved := permuteFP8Reference(convert(fc1), []int{2, 128, 16, 512}, []int{1, 0, 2, 3})
	shuffled := make([]byte, len(interleaved))
	index := []int{0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15}
	for group := 0; group < 256; group++ {
		for lane, sourceLane := range index {
			dst := (group*16 + lane) * 512
			src := (group*16 + sourceLane) * 512
			copy(shuffled[dst:dst+512], interleaved[src:src+512])
		}
	}
	want1 := permuteFP8Reference(shuffled, []int{256, 16, 32, 16}, []int{0, 2, 1, 3})
	want2 := permuteFP8Reference(convert(fc2), []int{32, 16, 128, 16}, []int{0, 2, 1, 3})
	got := cache.Snapshot()
	if !bytes.Equal(got.FC1, want1) || !bytes.Equal(got.FC2, want2) {
		t.Fatal("FP8 cache differs from source view/interleave/shuffle sequence")
	}
	for i, v := range fc1 {
		if v != before1[i] {
			t.Fatal("fc1 source mutated")
		}
	}
	for i, v := range fc2 {
		if v != before2[i] {
			t.Fatal("fc2 source mutated")
		}
	}
	got.FC1[0] ^= 1
	if cache.Snapshot().FC1[0] == got.FC1[0] {
		t.Fatal("snapshot mutates packed cache")
	}
}
func TestSUPFP8FeedForwardRejectsUnqualifiedShapesAndDomain(t *testing.T) {
	if _, err := PackSUPFeedForwardFP8(nil, nil); err == nil {
		t.Fatal("empty parameters accepted")
	}
	fc1, fc2 := make([]uint16, 4096*512), make([]uint16, 512*2048)
	for _, bad := range []uint16{0x7c00, 0x7e00, 0x5fff} {
		fc1[37] = bad
		if _, err := PackSUPFeedForwardFP8(fc1, fc2); err == nil {
			t.Fatalf("unsupported half%04x accepted", bad)
		}
	}
}
