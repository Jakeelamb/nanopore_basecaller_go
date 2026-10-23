// Package align implements bounded-window edit-distance alignment without native dependencies.
package align

import "fmt"

type Equality uint8

const (
	EqualityExact Equality = iota
	EqualityNWildcard
)

// Location is the first HW/LOC result. Start and End are inclusive.
// Valid is false when native edlib does not return start locations for an empty target.
type Location struct {
	Distance int
	Start    int
	End      int
	Valid    bool
}

// HW aligns all of query to a substring of target, allowing free target flanks.
// Equal-distance results choose the first end, then the earliest start for that end.
func HW(query, target string, equality Equality) (Location, error) {
	if query == "" {
		return Location{}, fmt.Errorf("alignment query is empty")
	}
	if equality != EqualityExact && equality != EqualityNWildcard {
		return Location{}, fmt.Errorf("unsupported alignment equality %d", equality)
	}
	if target == "" {
		return Location{Distance: len(query), Start: -1, End: -1}, nil
	}
	type cell struct{ distance, start int }
	previous := make([]cell, len(query)+1)
	current := make([]cell, len(query)+1)
	for i := range previous {
		previous[i] = cell{i, 0}
	}
	best := Location{Distance: len(query), Start: 0, End: -1, Valid: true}
	// Original edlib cannot report the empty target prefix when its final
	// 64-bit query block has no padding. Its first candidate then ends at zero.
	if len(query)%64 == 0 {
		best.Distance++
	}
	for j := 0; j < len(target); j++ {
		current[0] = cell{0, j + 1}
		for i := 1; i <= len(query); i++ {
			cost := 1
			if equal(query[i-1], target[j], equality) {
				cost = 0
			}
			v := cell{previous[i-1].distance + cost, previous[i-1].start}
			for _, candidate := range [2]cell{{current[i-1].distance + 1, current[i-1].start}, {previous[i].distance + 1, previous[i].start}} {
				if candidate.distance < v.distance || candidate.distance == v.distance && candidate.start < v.start {
					v = candidate
				}
			}
			current[i] = v
		}
		if current[len(query)].distance < best.Distance {
			best = Location{current[len(query)].distance, current[len(query)].start, j, true}
		}
		previous, current = current, previous
	}
	return best, nil
}

func equal(a, b byte, equality Equality) bool {
	if a == b {
		return true
	}
	if equality != EqualityNWildcard {
		return false
	}
	if a == 'N' {
		return b == 'A' || b == 'C' || b == 'G' || b == 'T'
	}
	if b == 'N' {
		return a == 'A' || a == 'C' || a == 'G' || a == 'T'
	}
	return false
}
