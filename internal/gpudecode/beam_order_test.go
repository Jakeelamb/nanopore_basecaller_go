package gpudecode

import (
	"math"
	"testing"
)

func TestBeamCompactionMatchesLiteralPTXLaneOrder(t *testing.T) {
	front := make([]BeamCandidate, 160)
	for i := range front {
		front[i] = BeamCandidate{Key: uint32(i), ScoreBits: math.Float32bits(7), State: uint16(i)}
	}
	got, err := CompactBeam(front, math.Float32bits(7))
	if err != nil {
		t.Fatal(err)
	}
	// Independent original-PTX block interpreter fixture, not ascending indices.
	want := []uint32{0, 32, 64, 96, 128, 1, 33, 65, 97, 129, 2, 34, 66, 98, 130, 3, 35, 67, 99, 131, 4, 36, 68, 100, 132, 5, 37, 69, 101, 133, 6, 38}
	if len(got) != len(want) {
		t.Fatalf("survivors %d", len(got))
	}
	for i, v := range got {
		if v.Key != want[i] {
			t.Fatalf("slot%d got%d want%d", i, v.Key, want[i])
		}
	}
	best, err := TerminalBeam(got)
	if err != nil || best.Key != 0 {
		t.Fatalf("finite tie winner %d err=%v", best.Key, err)
	}
	// A wrong right-on-equality reduction would choose the final retained key38.
	got[31].ScoreBits = math.Float32bits(8)
	best, err = TerminalBeam(got)
	if err != nil || best.Key != 38 {
		t.Fatalf("strict maximum winner %d err=%v", best.Key, err)
	}
}

func TestBeamOrderingUsesFTZComparisonAndPreservesRecordBits(t *testing.T) {
	bits := []uint32{0x80000001, 0, 1, 0x80000000, math.Float32bits(-1)}
	front := make([]BeamCandidate, len(bits))
	for i, v := range bits {
		front[i] = BeamCandidate{Key: uint32(i), ScoreBits: v}
	}
	got, err := CompactBeam(front, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 4 {
		t.Fatalf("FTZ survivors %d", len(got))
	}
	for i, v := range got {
		if v.Key != uint32(i) || v.ScoreBits != bits[i] {
			t.Fatal("comparison altered retained record")
		}
	}
	best, err := TerminalBeam(got)
	if err != nil || best.Key != 0 || best.ScoreBits != bits[0] {
		t.Fatalf("FTZ finite tie changed: %+v %v", best, err)
	}
}

func TestBeamOrderingRejectsUnqualifiedFrontAndNonfiniteScores(t *testing.T) {
	for _, n := range []int{0, 1, 4, 161, 165} {
		if _, err := CompactBeam(make([]BeamCandidate, n), 0); err == nil {
			t.Fatalf("front length%d accepted", n)
		}
	}
	front := make([]BeamCandidate, 5)
	for _, bad := range []uint32{0x7f800000, 0xff800000, 0x7fc00001} {
		front[0].ScoreBits = bad
		if _, err := CompactBeam(front, 0); err == nil {
			t.Fatal("nonfinite candidate accepted")
		}
		if _, err := TerminalBeam(front); err == nil {
			t.Fatal("nonfinite terminal accepted")
		}
		front[0].ScoreBits = 0
		if _, err := CompactBeam(front, bad); err == nil {
			t.Fatal("nonfinite cutoff accepted")
		}
	}
	if _, err := TerminalBeam(nil); err == nil {
		t.Fatal("empty terminal accepted")
	}
	if _, err := TerminalBeam(make([]BeamCandidate, 33)); err == nil {
		t.Fatal("oversize terminal accepted")
	}
}
