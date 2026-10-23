// Package decode ports Dorado's float32 CPU CRF beam-search algorithm.
// Go owns the algorithm; five scalar float operations use native libm through cgo.
package decode

import "errors"

// ErrCGORequired means this build cannot provide native float libm semantics.
var ErrCGORequired = errors.New("decode requires CGO_ENABLED=1 and native float libm; no pure-Go numerical fallback is provided")

// Input holds contiguous row-major tensors. Scores is [Blocks,4*States];
// BackGuides and Posts are [Blocks+1,States]. Posts contains probabilities.
// Slices are borrowed for the call and never modified.
type Input struct {
	Blocks, States            int
	Scores, BackGuides, Posts []float32
}

// Options controls the pinned CPU beam search. BeamCut is zero (no cut) or >=1.
// QShift and positive QScale apply to per-base Phred scores before clamping to [1,50].
type Options struct {
	BeamWidth                           int
	BeamCut, BlankScore, QShift, QScale float32
}

// Result owns its storage. States contains emitted bases (0=A..3=T), as in
// Dorado after quality generation. QualData is [Blocks,4] before base aggregation.
type Result struct {
	Sequence, Qstring string
	Moves             []uint8
	States            []int32
	QualData          []float32
	FinalScore        float32
}
