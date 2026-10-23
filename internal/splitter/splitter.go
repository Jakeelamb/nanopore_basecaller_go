// Package splitter implements the pinned simplex DNA PA pore-adapter decision.
package splitter

import (
	"fmt"
	"math"
	"sort"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/align"
)

const InternalAdapter = "TACTTCGTTCAGTTACGTATTGCT"

const (
	poreThreshold   uint16 = 0x419a // c10::Half(2.8f).
	ignorePrefix           = 5000
	clusterDistance        = 500
	maxPoreBases           = 500
	topCandidates          = 10
)

// Input borrows normalized binary16 signal and binary moves. Callers must not
// change the borrowed slices while FindSimplexPA runs.
type Input struct {
	Signal            []uint16
	Moves             []uint8
	Sequence, Qstring string
	Stride            uint64
}

// Interval has an inclusive start and exclusive end, in bases.
type Interval struct{ Start, End uint64 }

// SampleCluster has half-open sample coordinates and the original peak bits.
type SampleCluster struct {
	Start, End, Argmax uint64
	Peak               uint16
}

type Rejection uint8

const (
	Eligible Rejection = iota
	MoveEndOutside
	NoCalledBase
	PoreTooWide
	QualityTooHigh
)

// PoreCandidate records the mapping before and after argmax reduction. BaseSpan
// and MeanQ are meaningful only after the earlier rejection checks pass.
type PoreCandidate struct {
	MoveStart, MoveEnd, MoveArgmax uint64
	BaseSpan, Range                Interval
	MeanQ                          float32
	Rejection                      Rejection
}

// AdapterCheck records the first strict-equality HW/LOC match within Search.
// Match has half-open coordinates and is meaningful only when Found is true.
type AdapterCheck struct {
	Pore, Search, Match Interval
	Found               bool
}

// Decision owns all returned slices. RawCandidates contains every quality and
// width eligible base interval, sorted by coordinates. Candidates contains the
// top ten by native (peak, start, end) order, again sorted by coordinates.
// Spacers contains the one-base pore intervals with a nearby adapter.
type Decision struct {
	Clusters       []SampleCluster
	MoveSums       []uint64
	PoreCandidates []PoreCandidate
	RawCandidates  []Interval
	Candidates     []Interval
	AdapterChecks  []AdapterCheck
	Spacers        []Interval
}

// FindSimplexPA applies only the SQK-LSK114 nonrapid simplex DNA PA contract.
// It does not create children, mutate inputs, or perform model inference.
func FindSimplexPA(in Input) (Decision, error) {
	if err := validate(in); err != nil {
		return Decision{}, err
	}
	out := Decision{Clusters: poreClusters(in.Signal), MoveSums: make([]uint64, len(in.Moves))}
	var sum uint64
	for i, move := range in.Moves {
		sum += uint64(move)
		out.MoveSums[i] = sum
	}
	type ranked struct {
		peak uint16
		r    Interval
	}
	var raw []ranked
	for _, cluster := range out.Clusters {
		p := PoreCandidate{MoveStart: cluster.Start / in.Stride, MoveEnd: cluster.End / in.Stride, MoveArgmax: cluster.Argmax / in.Stride}
		switch {
		case p.MoveEnd >= uint64(len(out.MoveSums)):
			p.Rejection = MoveEndOutside
		case out.MoveSums[p.MoveStart] == 0:
			p.Rejection = NoCalledBase
		default:
			p.BaseSpan = Interval{out.MoveSums[p.MoveStart] - 1, out.MoveSums[p.MoveEnd]}
			if p.BaseSpan.End > p.BaseSpan.Start+maxPoreBases {
				p.Rejection = PoreTooWide
			} else {
				start := out.MoveSums[p.MoveArgmax] - 1
				p.Range = Interval{start, start + 1}
				end := min(start+5, uint64(len(in.Qstring)))
				var quality uint64
				for i := start; i < end; i++ {
					quality += uint64(in.Qstring[i] - 33)
				}
				p.MeanQ = float32(float64(quality) / float64(end-start))
				// The native subtraction rounds back to 10 in binary32.
				threshold := float32(10) - float32(0x1p-23)
				if p.MeanQ > threshold {
					p.Rejection = QualityTooHigh
				} else {
					raw = append(raw, ranked{cluster.Peak, p.Range})
				}
			}
		}
		out.PoreCandidates = append(out.PoreCandidates, p)
	}
	sort.Slice(raw, func(i, j int) bool {
		if raw[i].peak != raw[j].peak {
			return raw[i].peak < raw[j].peak
		}
		return before(raw[i].r, raw[j].r)
	})
	for _, candidate := range raw {
		out.RawCandidates = append(out.RawCandidates, candidate.r)
	}
	sortIntervals(out.RawCandidates)
	for _, candidate := range raw[max(0, len(raw)-topCandidates):] {
		out.Candidates = append(out.Candidates, candidate.r)
	}
	sortIntervals(out.Candidates)
	for _, candidate := range out.Candidates {
		check := AdapterCheck{Pore: candidate, Search: Interval{candidate.Start, min(candidate.End+50, uint64(len(in.Sequence)))}}
		location, err := align.HW(InternalAdapter, in.Sequence[check.Search.Start:check.Search.End], align.EqualityExact)
		if err != nil {
			return Decision{}, err
		}
		if location.Valid && location.Distance <= 4 {
			check.Found = true
			check.Match = Interval{check.Search.Start + uint64(location.Start), check.Search.Start + uint64(location.End+1)}
			out.Spacers = append(out.Spacers, candidate)
		}
		out.AdapterChecks = append(out.AdapterChecks, check)
	}
	return out, nil
}

func validate(in Input) error {
	if len(in.Sequence) != len(in.Qstring) {
		return fmt.Errorf("sequence/quality length mismatch")
	}
	if len(in.Moves) == 0 || len(in.Moves) > math.MaxInt32 || len(in.Sequence) > math.MaxInt32-50 || uint64(len(in.Signal)) > math.MaxInt64 {
		return fmt.Errorf("counts outside native domain")
	}
	if in.Stride == 0 || in.Stride > math.MaxInt32 {
		return fmt.Errorf("stride outside native positive int domain")
	}
	for i := range in.Sequence {
		switch in.Sequence[i] {
		case 'A', 'C', 'G', 'T', 'N':
		default:
			return fmt.Errorf("invalid DNA base at %d", i)
		}
		if in.Qstring[i] <= 33 || in.Qstring[i] > 126 {
			return fmt.Errorf("quality outside native positive Phred domain at %d", i)
		}
	}
	var bases uint64
	for i, move := range in.Moves {
		if move > 1 {
			return fmt.Errorf("nonbinary move at %d", i)
		}
		bases += uint64(move)
	}
	if bases != uint64(len(in.Sequence)) {
		return fmt.Errorf("move base count differs from sequence length")
	}
	samples, moves := uint64(len(in.Signal)), uint64(len(in.Moves))
	floor := samples / in.Stride
	ceil := floor
	if samples%in.Stride != 0 {
		ceil++
	}
	if moves != floor && moves != ceil {
		return fmt.Errorf("signal/move stride mismatch")
	}
	for i, sample := range in.Signal {
		if sample&0x7c00 == 0x7c00 {
			return fmt.Errorf("nonfinite binary16 signal at %d", i)
		}
	}
	return nil
}

func before(a, b Interval) bool  { return a.Start < b.Start || a.Start == b.Start && a.End < b.End }
func sortIntervals(r []Interval) { sort.Slice(r, func(i, j int) bool { return before(r[i], r[j]) }) }

func halfComparable(bits uint16) int16 {
	em := int16(bits & 0x7fff)
	if bits&0x8000 != 0 {
		return -em
	}
	return em
}

func poreClusters(signal []uint16) []SampleCluster {
	var clusters []SampleCluster
	for i := ignorePrefix; i < len(signal); {
		for i < len(signal) && halfComparable(signal[i]) <= halfComparable(poreThreshold) {
			i++
		}
		if i == len(signal) {
			break
		}
		cluster := SampleCluster{Start: uint64(i), Argmax: uint64(i), Peak: signal[i]}
		i++
		for i < len(signal) && halfComparable(signal[i]) > halfComparable(poreThreshold) {
			if signal[i] >= cluster.Peak {
				cluster.Peak, cluster.Argmax = signal[i], uint64(i)
			}
			i++
		}
		cluster.End = uint64(i)
		i++ // The native scanner skips the first sample below the threshold.
		if len(clusters) == 0 || cluster.Start-clusters[len(clusters)-1].End > clusterDistance {
			clusters = append(clusters, cluster)
		} else {
			last := &clusters[len(clusters)-1]
			last.End = cluster.End
			if cluster.Peak >= last.Peak {
				last.Peak, last.Argmax = cluster.Peak, cluster.Argmax
			}
		}
	}
	return clusters
}
