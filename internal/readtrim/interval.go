package readtrim

import (
	"fmt"
	"math"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/decode"
)

// Interval retains the half-open base range [Start, End).
type Interval struct{ Start, End int }

// TrimResult borrows input storage. Copy slices before modifying shared data.
// LeadingTrimSamples includes the supplied prior leading trim. NumSamples is
// the current signal length plus LeadingTrimSamples, as in Dorado's ns tag.
type TrimResult struct {
	Read                           MuxRead
	Interval                       Interval
	MoveStart, MoveEnd             int
	SignalStart, SignalEnd         int
	LeadingTrimSamples, NumSamples uint64
	Trimmed                        bool
}

// ApplyInterval applies the plain DNA SimplexRead trim. It leaves pre-trim
// length, event count, and all caller-owned identity and clocks unchanged.
// Invalid intervals are errors here. The pipeline must skip invalid detector
// intersections before calling this helper, as TrimmerNode does.
func ApplyInterval(read MuxRead, interval Interval, leadingTrim uint64) (TrimResult, error) {
	if err := validateIntervalRead(read); err != nil {
		return TrimResult{}, err
	}
	n := len(read.Sequence)
	if interval.Start < 0 || interval.End < interval.Start || interval.End > n || (interval.Start == n && n > 0) {
		return TrimResult{}, fmt.Errorf("invalid base interval [%d,%d) for %d", interval.Start, interval.End, n)
	}
	r := TrimResult{Read: read, Interval: interval, MoveEnd: len(read.Moves), SignalEnd: len(read.Signal), LeadingTrimSamples: leadingTrim}
	if interval.End-interval.Start != n {
		start, end := 0, 0
		if interval.End > interval.Start {
			base := -1
			for i, move := range read.Moves {
				if move == 1 {
					base++
				}
				if base >= interval.End {
					break
				}
				if base >= interval.Start {
					end = i + 1
				} else {
					start++
					end = start
				}
			}
		}
		removed := uint64(start) * uint64(read.Stride)
		if removed > math.MaxInt32 {
			return TrimResult{}, fmt.Errorf("native leading sample multiplication overflow")
		}
		if leadingTrim > math.MaxUint64-removed {
			return TrimResult{}, fmt.Errorf("leading trim overflow")
		}
		r.LeadingTrimSamples += removed
		r.MoveStart, r.MoveEnd = start, end
		r.SignalStart = min(len(read.Signal), int(removed))
		r.SignalEnd = min(len(read.Signal), int(uint64(end)*uint64(read.Stride)))
		r.Read.Sequence = read.Sequence[interval.Start:interval.End]
		r.Read.Qstring = read.Qstring[interval.Start:interval.End]
		r.Read.Moves = read.Moves[start:end:end]
		r.Read.Signal = read.Signal[r.SignalStart:r.SignalEnd:r.SignalEnd]
		r.Trimmed = true
	}
	if r.LeadingTrimSamples > math.MaxUint64-uint64(len(r.Read.Signal)) {
		return TrimResult{}, fmt.Errorf("sample count overflow")
	}
	r.NumSamples = r.LeadingTrimSamples + uint64(len(r.Read.Signal))
	return r, nil
}

func validateIntervalRead(read MuxRead) error {
	// Native trim is alphabet agnostic. N is admitted for end-detector fixtures.
	for i := range read.Sequence {
		if read.Sequence[i] != 'A' && read.Sequence[i] != 'C' && read.Sequence[i] != 'G' && read.Sequence[i] != 'T' && read.Sequence[i] != 'N' {
			return fmt.Errorf("invalid DNA base at %d", i)
		}
	}
	// Reuse the frozen MUX shape validator without narrowing this API's alphabet.
	r := read
	r.Sequence = strings.ReplaceAll(read.Sequence, "N", "A")
	return validate(r)
}

// FilterReason is a bit set because length and quality can both fail.
type FilterReason uint8

const (
	FilterPass       FilterReason = 0
	FilterShort      FilterReason = 1
	FilterLowQuality FilterReason = 2
)

type FilterResult struct {
	Keep   bool
	MeanQ  float32
	Reason FilterReason
}

// Filter applies the five-base minimum and optional positive Q threshold.
// A zero or negative finite threshold disables Q filtering. MeanQ is always
// computed for the returned metadata and requires decode's native libm bridge.
func Filter(seq, q string, meanQStart int, minQ float32) (FilterResult, error) {
	if len(seq) != len(q) || meanQStart < 0 || math.IsNaN(float64(minQ)) || math.IsInf(float64(minQ), 0) {
		return FilterResult{}, fmt.Errorf("invalid filter input")
	}
	for i := range seq {
		if seq[i] != 'A' && seq[i] != 'C' && seq[i] != 'G' && seq[i] != 'T' && seq[i] != 'N' {
			return FilterResult{}, fmt.Errorf("invalid DNA base at %d", i)
		}
		if q[i] < 33 || q[i] > 126 {
			return FilterResult{}, fmt.Errorf("invalid quality at %d", i)
		}
	}
	selected := q
	if len(q) > meanQStart {
		selected = q[meanQStart:]
	}
	mean, err := decode.MeanQScore(selected)
	if err != nil {
		return FilterResult{}, err
	}
	r := FilterResult{Keep: true, MeanQ: mean}
	if len(seq) < 5 {
		r.Reason |= FilterShort
	}
	if minQ > 0 && mean < minQ {
		r.Reason |= FilterLowQuality
	}
	r.Keep = r.Reason == FilterPass
	return r, nil
}
