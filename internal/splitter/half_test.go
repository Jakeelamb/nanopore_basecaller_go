package splitter

import (
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
)

func TestAllHalfComparisons(t *testing.T) {
	threshold := signal.Float32FromHalf(poreThreshold)
	if poreThreshold != signal.Float16Bits(2.8) {
		t.Fatal("PA threshold binary16 differs")
	}
	for bits := 0; bits < 65536; bits++ {
		if bits&0x7c00 == 0x7c00 {
			continue
		}
		if got, want := halfComparable(uint16(bits)) > halfComparable(poreThreshold), signal.Float32FromHalf(uint16(bits)) > threshold; got != want {
			t.Fatalf("half comparison differs at %04x", bits)
		}
	}
}
