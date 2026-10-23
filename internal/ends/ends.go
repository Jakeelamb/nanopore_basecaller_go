// Package ends detects the pinned default SQK-LSK114 DNA adapters and cDNA primers.
package ends

import (
	"fmt"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/align"
)

const Unclassified = "unclassified"
const SupportedKit = "SQK-LSK114"
const adapterFront = "CCTGTACTTCGTTCAGTTACGTATTGC"
const adapterRear = "AGCAATACGTAACTGAAC"
const primerFront = "TTTCTGTTGGTGCTGATATTGCTGGG"
const primerRear = "ACTTGCCTGTCGCTCTATCTTCTTT"

type Admission string

const (
	KitSupported Admission = "supported"
	KitMissing   Admission = "missing_no_stock_candidates"
)

type Orientation int8

const (
	Reverse Orientation = -1
	Unknown Orientation = 0
	Forward Orientation = 1
)

// BaseInterval is a half-open interval of sequence bases.
type BaseInterval struct{ Start, End int }

// End uses inclusive alignment coordinates, including native -1 sentinels.
type End struct {
	Score float32
	Name  string
	Start int
	End   int
}

type Pair struct{ Front, Rear End }

type Detection struct {
	Raw      Pair
	Accepted Pair
	Retained BaseInterval
}

type Classification struct {
	Name        string
	Orientation Orientation
}

type Options struct{ Adapters, Primers bool }

func DefaultOptions() Options { return Options{Adapters: true, Primers: true} }

type Result struct {
	Kit                  string
	Admission            Admission
	Adapters             Detection
	Primers              Detection
	PrimerClassification Classification
	Retained             BaseInterval
	Conflict             bool
}

type query struct{ name, front, rear string }

// Detect owns no read metadata and does not mutate sequence. It is safe for concurrent calls.
// Empty kit follows native ANY selection, which has no stock candidates.
// Explicit kits outside the initial qualified target are rejected.
func Detect(sequence, kit string, options Options) (Result, error) {
	kit = strings.ToUpper(kit)
	admission := KitSupported
	if kit == "" || kit == "ANY" {
		kit, admission = "ANY", KitMissing
	} else if kit != SupportedKit {
		return Result{}, fmt.Errorf("unsupported end-detection kit %q", kit)
	}
	for i := range sequence {
		if !strings.ContainsRune("ACGTN", rune(sequence[i])) {
			return Result{}, fmt.Errorf("invalid DNA base at %d", i)
		}
	}
	if uint64(len(sequence)) > uint64(1<<31-1) {
		return Result{}, fmt.Errorf("sequence exceeds native coordinate range")
	}
	full := BaseInterval{0, len(sequence)}
	result := Result{Kit: kit, Admission: admission, Retained: full, PrimerClassification: Classification{Unclassified, Unknown}}
	var adapters, primers []query
	if admission == KitSupported {
		adapters = []query{{"LSK110", adapterFront, adapterRear}}
		primers = []query{{"cDNA_FWD", primerFront, reverseComplement(primerRear)}, {"cDNA_REV", primerRear, reverseComplement(primerFront)}}
	}
	result.Adapters = detection(sequence, nil, 75)
	result.Primers = detection(sequence, nil, 150)
	if options.Adapters {
		result.Adapters = detection(sequence, adapters, 75)
	}
	if options.Primers {
		result.Primers = detection(sequence, primers, 150)
		result.PrimerClassification = classify(result.Primers.Accepted)
	}
	result.Retained = BaseInterval{max(result.Adapters.Retained.Start, result.Primers.Retained.Start), min(result.Adapters.Retained.End, result.Primers.Retained.End)}
	if (options.Adapters || options.Primers) && result.Retained.Start >= result.Retained.End {
		result.Retained = full
		result.Conflict = true
	}
	return result, nil
}

func emptyEnd() End { return End{-1, Unclassified, -1, -1} }

func detection(sequence string, queries []query, window int) Detection {
	fronts, rears := make([]End, 0, len(queries)), make([]End, 0, len(queries))
	rearStart := max(0, len(sequence)-window)
	for _, q := range queries {
		if q.front != "" {
			fronts = append(fronts, match(q.front, sequence[:min(window, len(sequence))], q.name+"_FRONT", -1))
		}
		if q.rear != "" {
			rears = append(rears, match(q.rear, sequence[rearStart:], q.name+"_REAR", rearStart))
		}
	}
	raw := Pair{bestResult(fronts), bestResult(rears)}
	accepted, retained := trimInterval(raw, len(sequence))
	return Detection{raw, accepted, retained}
}

func match(query, target, name string, offset int) End {
	location, _ := align.HW(query, target, align.EqualityNWildcard)
	result := emptyEnd()
	if location.Valid {
		result.Score = 1 - float32(location.Distance)/float32(len(query))
		result.Start, result.End = location.Start, location.End
	}
	if offset >= 0 {
		result.Start += offset
		result.End += offset
	}
	result.Name = name
	return result
}

func bestResult(results []End) End {
	best, bestScore := -1, float32(-1)
	const epsilon float32 = 0.1
	for i, result := range results {
		oldSpan := 0
		if best != -1 {
			oldSpan = results[best].End - results[best].Start
		}
		newSpan := result.End - result.Start
		if result.Score > bestScore+epsilon {
			bestScore, best = result.Score, i
		}
		difference := result.Score - bestScore
		if difference < 0 {
			difference = -difference
		}
		if difference <= epsilon && newSpan > oldSpan {
			bestScore, best = result.Score, i
		}
	}
	if best == -1 {
		return emptyEnd()
	}
	return results[best]
}

func trimInterval(pair Pair, length int) (Pair, BaseInterval) {
	interval := BaseInterval{0, length}
	if pair.Front.Name == Unclassified || pair.Front.Score < float32(0.8) {
		pair.Front.Name = Unclassified
	} else {
		interval.Start = pair.Front.End + 1
	}
	if pair.Rear.Name == Unclassified || pair.Rear.Score < float32(0.8) {
		pair.Rear.Name = Unclassified
	} else {
		interval.End = pair.Rear.Start
	}
	if interval.End <= interval.Start {
		interval = BaseInterval{0, length}
		pair.Front.Name, pair.Rear.Name = Unclassified, Unclassified
	}
	return pair, interval
}

func classify(pair Pair) Classification {
	front := strings.SplitN(pair.Front.Name, "_FRONT", 2)[0]
	rear := strings.SplitN(pair.Rear.Name, "_REAR", 2)[0]
	name := Unclassified
	switch {
	case front != Unclassified && rear != Unclassified:
		if front == rear {
			name = front
		}
	case front != Unclassified:
		name = front
	case rear != Unclassified:
		name = rear
	}
	orientation := Unknown
	if strings.Contains(name, "_FWD") {
		orientation = Forward
	} else if strings.Contains(name, "_REV") {
		orientation = Reverse
	}
	return Classification{name, orientation}
}

func reverseComplement(sequence string) string {
	out := make([]byte, len(sequence))
	for i := range sequence {
		out[len(sequence)-i-1] = "TGCA"[strings.IndexByte("ACGT", sequence[i])]
	}
	return string(out)
}
