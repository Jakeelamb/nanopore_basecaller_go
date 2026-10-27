package gpudecode

import (
	"fmt"
	"math"
)

// BeamCandidate is the logical twelve-byte source front record. Wire encoding
// must be explicit; the Go struct's memory representation is not its ABI.
// ScoreBits includes the next-frame guide, before source guide subtraction.
type BeamCandidate struct {
	Key       uint32
	ScoreBits uint32
	State     uint16
	Previous  uint8
	Stay      bool
}

func finiteBeamScore(bits uint32) bool { return bits&0x7f800000 != 0x7f800000 }
func beamComparisonScore(bits uint32) float32 {
	if bits&0x7f800000 == 0 {
		bits &= 0x80000000
	}
	return math.Float32frombits(bits)
}

// CompactBeam implements the source warp's lane-major survivor order for W32.
// It takes the already expanded/merged front: five candidates per prior beam.
// The cutoff's numerical construction belongs to the separately verified math.
func CompactBeam(front []BeamCandidate, cutoffBits uint32) ([]BeamCandidate, error) {
	if len(front) == 0 || len(front) > 160 || len(front)%5 != 0 || !finiteBeamScore(cutoffBits) {
		return nil, fmt.Errorf("W32 finite beam front and cutoff required")
	}
	for _, v := range front {
		if !finiteBeamScore(v.ScoreBits) {
			return nil, fmt.Errorf("nonfinite beam candidate")
		}
	}
	cutoff := beamComparisonScore(cutoffBits)
	survivors := make([]BeamCandidate, 0, 32)
	for lane := 0; lane < 32; lane++ {
		for i := lane; i < len(front); i += 32 {
			if beamComparisonScore(front[i].ScoreBits) >= cutoff && len(survivors) < 32 {
				survivors = append(survivors, front[i])
			}
		}
	}
	return survivors, nil
}

// TerminalBeam returns the earliest finite maximum in compacted order. The
// inspected last-frame partial merge emits one winner per pair, not a full sort.
// Only its terminal winner is established here; other scratch slots are opaque.
func TerminalBeam(compacted []BeamCandidate) (BeamCandidate, error) {
	if len(compacted) == 0 || len(compacted) > 32 {
		return BeamCandidate{}, fmt.Errorf("one to 32 compacted beam candidates required")
	}
	for _, v := range compacted {
		if !finiteBeamScore(v.ScoreBits) {
			return BeamCandidate{}, fmt.Errorf("nonfinite terminal beam")
		}
	}
	best := compacted[0]
	for _, v := range compacted[1:] {
		if beamComparisonScore(v.ScoreBits) > beamComparisonScore(best.ScoreBits) {
			best = v
		}
	}
	return best, nil
}
