package gpudecode

import (
	"reflect"
	"testing"
)

func TestQualityNeighborOrder(t *testing.T) {
	for _, tc := range []struct {
		states int
		state  int32
		want   []uint16
	}{
		{256, 0, []uint16{64, 1, 128, 2, 192, 3}},
		{256, 1, []uint16{0, 4, 64, 5, 128, 6, 192, 7}},
		{256, 85, []uint16{21, 84, 149, 86, 213, 87}},
		{256, 255, []uint16{63, 252, 127, 253, 191, 254}},
		{1024, 1, []uint16{0, 4, 256, 5, 512, 6, 768, 7}},
	} {
		got, err := PlanQuality(BeamTrace{Path: []int32{tc.state}, Moves: []byte{1}}, tc.states, false)
		if err != nil {
			t.Fatal(err)
		}
		f := got.Frames[0]
		if f.PosteriorRow != 1 || f.State != uint16(tc.state) || !reflect.DeepEqual(f.Neighbors[:f.NeighborCount], tc.want) {
			t.Fatalf("states=%d state=%d frame=%+v want=%v", tc.states, tc.state, f, tc.want)
		}
	}
}

func TestQualityNeighborDomainExhaustive(t *testing.T) {
	for _, states := range []int{256, 1024} {
		for state := 0; state < states; state++ {
			got, err := PlanQuality(BeamTrace{Path: []int32{int32(state)}, Moves: []byte{1}}, states, false)
			if err != nil {
				t.Fatal(err)
			}
			f := got.Frames[0]
			seen := map[uint16]bool{uint16(state): true}
			for _, neighbor := range f.Neighbors[:f.NeighborCount] {
				if int(neighbor) >= states || seen[neighbor] {
					t.Fatalf("invalid neighbor states=%d state=%d frame=%+v", states, state, f)
				}
				seen[neighbor] = true
			}
			// Independently compare the selected set to the two de Bruijn
			// adjacency sets; literal cases above check numerical addition order.
			for candidate := 0; candidate < states; candidate++ {
				adjacent := candidate == state || candidate/4 == state%(states/4) || candidate%(states/4) == state/4
				if seen[uint16(candidate)] != adjacent {
					t.Fatalf("missing/extra adjacency states=%d state=%d candidate=%d", states, state, candidate)
				}
			}
		}
	}
}

func TestQualityMoveGroupsAndOwnedTrace(t *testing.T) {
	for _, pad := range []bool{false, true} {
		trace := BeamTrace{Path: []int32{5, 6, 7, 8, 9, 10}, Moves: []byte{1, 0, 1, 0, 0, 1}}
		got, err := PlanQuality(trace, 256, pad)
		if err != nil {
			t.Fatal(err)
		}
		want := []QualityGroup{{0, 2, 0, 'G'}, {2, 5, 1, 'C'}, {5, 6, 2, 'G'}}
		if pad {
			want[1].Output = 2
			want[2].Output = 5
		}
		if !reflect.DeepEqual(got.Groups, want) || !reflect.DeepEqual(got.Path, []int32{1, 2, 3, 0, 1, 2}) || !reflect.DeepEqual(got.Moves, trace.Moves) {
			t.Fatalf("pad=%v got=%+v", pad, got)
		}
		trace.Path[0], trace.Moves[0] = 255, 0
		if got.Frames[0].State != 5 || got.Path[0] != 1 || got.Moves[0] != 1 {
			t.Fatal("schedule borrows mutable traceback")
		}
	}
}

func TestQualityRejectsInvalidTrace(t *testing.T) {
	for _, tc := range []struct {
		trace  BeamTrace
		states int
	}{
		{BeamTrace{}, 256},
		{BeamTrace{Path: []int32{0}, Moves: []byte{1}}, 64},
		{BeamTrace{Path: []int32{0, 1}, Moves: []byte{1}}, 256},
		{BeamTrace{Path: []int32{0}, Moves: []byte{0}}, 256},
		{BeamTrace{Path: []int32{-1}, Moves: []byte{1}}, 256},
		{BeamTrace{Path: []int32{256}, Moves: []byte{1}}, 256},
		{BeamTrace{Path: []int32{0, 1}, Moves: []byte{1, 2}}, 256},
	} {
		if _, err := PlanQuality(tc.trace, tc.states, false); err == nil {
			t.Fatalf("accepted invalid trace %+v", tc)
		}
	}
}
