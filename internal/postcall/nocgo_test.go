//go:build !cgo

package postcall

import (
	"errors"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/decode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/subread"
)

func TestWithoutCGOFailsExplicitly(t *testing.T) {
	in := subread.Read{ID: "id", Sequence: "ACGTA", Qstring: "IIIII", Moves: []uint8{1, 1, 1, 1, 1}, Signal: make([]uint16, 30), Stride: 6, SampleRate: 5000, Inherited: subread.Inherited{SequencingKit: "SQK-LSK114"}}
	p := DefaultPolicy()
	p.Split, p.Trim = false, false
	if _, err := Process(in, p); !errors.Is(err, decode.ErrCGORequired) {
		t.Fatalf("missing libm boundary did not reject: %v", err)
	}
}
