package gpudecode

import "fmt"

// QualityFrame selects the ordered neighboring posterior values for one frame.
// The center value is added after the neighbor sum in the selected SM120 code.
// PosteriorRow is one-based: decode_step reads rows 1 through T.
type QualityFrame struct {
	PosteriorRow  int
	State         uint16
	Neighbors     [8]uint16
	NeighborCount uint8
}

// QualityGroup contains [First, End) frames sharing one quality calculation.
// Base is taken from the final frame, as in decode_step. Output is the compact
// sequence index, or the first frame index when move padding is enabled.
type QualityGroup struct {
	First, End, Output int
	Base               byte
}

type QualitySchedule struct {
	Frames  []QualityFrame
	Groups  []QualityGroup
	Path    []int32
	Moves   []byte
	MovePad bool
}

// PlanQuality owns state-neighbor selection and move grouping in Go. It does
// not evaluate GPU approximate probability, average or quality arithmetic.
// Inputs are the complete traceback before state IDs are replaced by base IDs.
func PlanQuality(trace BeamTrace, states int, movePad bool) (QualitySchedule, error) {
	if (states != 256 && states != 1024) || len(trace.Path) == 0 || len(trace.Path) != len(trace.Moves) {
		return QualitySchedule{}, fmt.Errorf("quality requires a nonempty complete HAC/SUP traceback")
	}
	if trace.Moves[0] != 1 {
		return QualitySchedule{}, fmt.Errorf("quality traceback must start with a move")
	}
	// Validate the full external traceback before constructing a partial plan.
	for i, state := range trace.Path {
		if state < 0 || int(state) >= states || trace.Moves[i] > 1 {
			return QualitySchedule{}, fmt.Errorf("quality traceback frame %d exceeds state or move domain", i)
		}
	}
	out := QualitySchedule{Frames: make([]QualityFrame, len(trace.Path)), Path: make([]int32, len(trace.Path)), Moves: append([]byte(nil), trace.Moves...), MovePad: movePad}
	first := 0
	for i, state := range trace.Path {
		f := QualityFrame{PosteriorRow: i + 1, State: uint16(state)}
		// PTX alternates incoming and outgoing neighbors, excluding the center
		// and previously selected states. S is at least 256, so each individual
		// incoming/outgoing four-state list has distinct entries.
		for k := 0; k < 4; k++ {
			for _, candidate := range [2]int{int(state)/4 + k*(states/4), (4*int(state))%states + k} {
				if candidate == int(state) {
					continue
				}
				duplicate := false
				for j := uint8(0); j < f.NeighborCount; j++ {
					duplicate = duplicate || f.Neighbors[j] == uint16(candidate)
				}
				if !duplicate {
					f.Neighbors[f.NeighborCount] = uint16(candidate)
					f.NeighborCount++
				}
			}
		}
		out.Frames[i] = f
		out.Path[i] = state % 4
		if i+1 == len(trace.Path) || trace.Moves[i+1] != 0 {
			output := len(out.Groups)
			if movePad {
				output = first
			}
			out.Groups = append(out.Groups, QualityGroup{First: first, End: i + 1, Output: output, Base: "ACGT"[state%4]})
			first = i + 1
		}
	}
	return out, nil
}
