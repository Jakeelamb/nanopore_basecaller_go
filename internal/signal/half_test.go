package signal

import (
	"math"
	"testing"
)

func TestEveryHalfRoundTrip(t *testing.T) {
	for value := 0; value < 65536; value++ {
		bits := uint16(value)
		expected := bits
		if bits&0x7c00 == 0x7c00 && bits&1023 != 0 {
			expected |= 0x0200
		}
		if got := Float16Bits(Float32FromHalf(bits)); got != expected {
			t.Fatalf("half %04x roundtrip %04x want %04x", bits, got, expected)
		}
	}
}

func TestEveryFiniteHalfMidpoint(t *testing.T) {
	for lower := uint16(0); lower < 0x7bff; lower++ {
		midpoint := float32((float64(Float32FromHalf(lower)) + float64(Float32FromHalf(lower+1))) / 2)
		expected := lower
		if lower&1 != 0 {
			expected++
		}
		for _, sign := range []float32{1, -1} {
			signed := midpoint * sign
			want := expected
			if sign < 0 {
				want |= 0x8000
			}
			if got := Float16Bits(signed); got != want {
				t.Fatalf("midpoint %04x sign %v got %04x want %04x", lower, sign, got, want)
			}
		}
		if got := Float16Bits(math.Nextafter32(midpoint, 0)); got != lower {
			t.Fatalf("below midpoint %04x got %04x", lower, got)
		}
		if got := Float16Bits(math.Nextafter32(midpoint, float32(math.Inf(1)))); got != lower+1 {
			t.Fatalf("above midpoint %04x got %04x", lower, got)
		}
	}
}

func TestHalfOverflowAndSpecial(t *testing.T) {
	for _, test := range []struct {
		value float32
		bits  uint16
	}{
		{65504, 0x7bff}, {65519, 0x7bff}, {65520, 0x7c00}, {-65520, 0xfc00},
		{float32(math.Inf(1)), 0x7c00}, {float32(math.Inf(-1)), 0xfc00},
		{math.Float32frombits(0x80000000), 0x8000}, {math.SmallestNonzeroFloat32, 0},
	} {
		if got := Float16Bits(test.value); got != test.bits {
			t.Fatalf("%v got %04x want %04x", test.value, got, test.bits)
		}
	}
}
