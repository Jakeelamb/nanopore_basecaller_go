package align

import (
	"strings"
	"testing"
)

func TestHWLocations(t *testing.T) {
	for _, c := range []struct {
		q, target string
		equality  Equality
		want      Location
	}{
		{"AC", "GGACAC", EqualityExact, Location{0, 2, 3, true}},
		{"CA", "AA", EqualityExact, Location{1, 0, 0, true}},
		{"CC", "AA", EqualityExact, Location{2, 0, -1, true}},
		{"AC", "", EqualityExact, Location{2, -1, -1, false}},
		{"AC", "NN", EqualityNWildcard, Location{0, 0, 1, true}},
		{"NN", "AC", EqualityNWildcard, Location{0, 0, 1, true}},
		{"NN", "AC", EqualityExact, Location{2, 0, -1, true}},
		{strings.Repeat("A", 64), "C", EqualityExact, Location{64, 0, 0, true}},
		{strings.Repeat("A", 65), "C", EqualityExact, Location{65, 0, -1, true}},
	} {
		got, err := HW(c.q, c.target, c.equality)
		if err != nil || got != c.want {
			t.Errorf("HW(%q,%q,%d)=%+v,%v; want %+v", c.q, c.target, c.equality, got, err, c.want)
		}
	}
}

func TestHWRejectsInvalidOptions(t *testing.T) {
	if _, err := HW("", "A", EqualityExact); err == nil {
		t.Fatal("empty query accepted")
	}
	if _, err := HW("A", "A", Equality(9)); err == nil {
		t.Fatal("unknown equality accepted")
	}
}

func TestHWAgainstAllSubstrings(t *testing.T) {
	var words func(int) []string
	words = func(n int) []string {
		out := []string{""}
		if n == 0 {
			return out
		}
		for _, s := range words(n - 1) {
			if len(s) == n-1 {
				for _, b := range []string{"A", "C", "N"} {
					out = append(out, s+b)
				}
			} else if s != "" {
				out = append(out, s)
			}
		}
		return out
	}
	for _, q := range words(3) {
		if q == "" {
			continue
		}
		for _, target := range words(4) {
			for _, eq := range []Equality{EqualityExact, EqualityNWildcard} {
				want := Location{len(q), 0, -1, true}
				if target == "" {
					want = Location{len(q), -1, -1, false}
				} else {
					for end := 0; end < len(target); end++ {
						for start := 0; start <= end+1; start++ {
							distance := globalDistance(q, target[start:end+1], eq)
							if distance < want.Distance {
								want = Location{distance, start, end, true}
							}
						}
					}
				}
				got, err := HW(q, target, eq)
				if err != nil || got != want {
					t.Fatalf("HW(%q,%q,%d)=%+v,%v want %+v", q, target, eq, got, err, want)
				}
			}
		}
	}
}

func globalDistance(q, target string, eq Equality) int {
	d := make([][]int, len(q)+1)
	for i := range d {
		d[i] = make([]int, len(target)+1)
		d[i][0] = i
	}
	for j := range d[0] {
		d[0][j] = j
	}
	for i := 1; i <= len(q); i++ {
		for j := 1; j <= len(target); j++ {
			cost := 1
			if equal(q[i-1], target[j-1], eq) {
				cost = 0
			}
			d[i][j] = min(d[i-1][j-1]+cost, min(d[i-1][j]+1, d[i][j-1]+1))
		}
	}
	return d[len(q)][len(target)]
}
