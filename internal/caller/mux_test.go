package caller

import (
	"bytes"
	"reflect"
	"strings"
	"testing"
)

func TestMuxEndReasonAndTailAccounting(t *testing.T) {
	r := testRead()
	r.NumSamples = 610
	r.EndReasonName = "signal_positive"
	moves := bytes.Repeat([]byte{1}, 100)
	normalized := make([]uint16, 600)
	out := CalledRead{ID: r.ID.String(), Sequence: strings.Repeat("A", 100), Qstring: strings.Repeat("I", 70) + strings.Repeat("!", 30), Moves: moves, NumSamples: 610, Trimmed: 10}
	before := append([]byte(nil), moves...)
	for _, reason := range []uint8{0, 1, 2, 3, 4, 9} {
		r.EndReason = reason
		got, err := trimMux(r, normalized, 6, out)
		if err != nil {
			t.Fatal(err)
		}
		if reason == 1 || reason == 2 {
			if len(got.Sequence) != 69 || len(got.Qstring) != 69 || len(got.Moves) != 69 || got.TailTrimmedSamples != 186 {
				t.Fatalf("end reason %d: %+v", reason, got)
			}
			if got.ID != out.ID || got.NumSamples != 610 || got.Trimmed != 10 {
				t.Fatal("mux trim changed identity or original/leading sample accounting")
			}
		} else if !reflect.DeepEqual(got, out) {
			t.Fatalf("end reason %d changed a non-mux read", reason)
		}
	}
	if !bytes.Equal(moves, before) || len(out.Sequence) != 100 {
		t.Fatal("mux integration mutated the stitched input")
	}
}

func TestMuxIntegrationRejectsInvalidMapping(t *testing.T) {
	r := testRead()
	r.EndReason = 1
	out := CalledRead{Sequence: "A", Qstring: "I", Moves: []byte{1}}
	if _, err := trimMux(r, make([]uint16, 6), 1<<32+6, out); err == nil {
		t.Fatal("stride narrowed before validation")
	}
	if _, err := trimMux(r, make([]uint16, 6), 0, out); err == nil {
		t.Fatal("zero stride accepted")
	}
	out.Moves = []byte{0}
	if _, err := trimMux(r, make([]uint16, 6), 6, out); err == nil {
		t.Fatal("inconsistent move mapping accepted")
	}
}
