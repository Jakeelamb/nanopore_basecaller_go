package caller

import (
	"errors"
	"math"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/pod5"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/readtrim"
)

func trimMux(r pod5.Read, normalized []uint16, stride uint64, out CalledRead) (CalledRead, error) {
	if stride > math.MaxInt32 {
		return CalledRead{}, errors.New("mux trim stride exceeds native int")
	}
	// DataLoader.cpp selects these two POD5 enum values, excluding data-service unblock.
	isMux := r.EndReason == 1 || r.EndReason == 2
	result, err := readtrim.MuxChangeTrim(readtrim.MuxRead{
		Sequence: out.Sequence, Qstring: out.Qstring, Moves: out.Moves,
		Signal: normalized, Stride: uint32(stride), IsMuxChange: isMux,
		NumMinknowEvents: r.NumMinknowEvents, PreTrimSequenceLength: uint64(len(out.Sequence)),
	})
	if err != nil {
		return CalledRead{}, err
	}
	out.Sequence, out.Qstring, out.Moves = result.Read.Sequence, result.Read.Qstring, result.Read.Moves
	out.TailTrimmedSamples = uint64(len(normalized)) - result.NumSamples
	return out, nil
}
