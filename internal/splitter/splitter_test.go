package splitter_test

import (
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/splitter"
	oracle "github.com/Jakeelamb/nanopore_basecaller_go/verification/splitter"
)

func TestSourceBoundaries(t *testing.T) {
	counts := map[string][3]int{
		"no-pore": {0, 0, 0}, "adapter-without-pore": {0, 0, 0}, "pore-without-adapter": {1, 1, 0}, "positive-internal-adapter": {1, 1, 1},
		"threshold-below": {0, 0, 0}, "threshold-equal": {0, 0, 0}, "threshold-above": {1, 1, 1},
		"contiguous-equal-peak-later": {1, 1, 1}, "lower-peak-preserves-argmax": {1, 1, 1}, "merged-equal-peak-later": {1, 1, 1},
		"merge-gap-500": {1, 1, 1}, "merge-gap-501": {2, 2, 2}, "mean-exactly-10": {1, 1, 1}, "mean-above-10": {1, 0, 0}, "mean-fractional-below-10": {1, 1, 1},
		"clipped-quality-window": {1, 1, 0}, "exclusive-end-outside-moves": {1, 0, 0}, "uncalled-prefix": {1, 0, 0},
		"four-adapter-edits": {1, 1, 1}, "five-adapter-edits": {1, 1, 0}, "N-is-exact": {1, 1, 0},
		"adapter-search-right-four-deletions": {1, 1, 1}, "adapter-search-right-five-deletions": {1, 1, 0}, "adapter-first-native-tie": {1, 1, 1},
		"prefix-4999": {0, 0, 0}, "prefix-5000": {1, 1, 0}, "width-500-passes": {1, 1, 0}, "width-501-rejected": {1, 0, 0},
		"top-ten-equal-peaks-later-pairs": {12, 10, 10}, "top-ten-higher-early-peak": {12, 10, 10},
		"top-count-nine": {9, 9, 9}, "top-count-ten": {10, 10, 10}, "top-count-eleven": {11, 10, 10},
		"partial-stride-floor-moves": {1, 1, 1}, "partial-stride-ceil-moves": {1, 1, 1}, "partial-tail-floor-rejected": {1, 0, 0}, "partial-tail-ceil-admitted": {1, 1, 0}, "one-sample": {0, 0, 0}, "below-prefix-size": {0, 0, 0}, "at-prefix-size": {0, 0, 0}, "no-called-bases": {1, 0, 0},
	}
	for _, c := range oracle.Cases() {
		t.Run(c.Name, func(t *testing.T) {
			beforeSignal, beforeMoves := append([]uint16(nil), c.Input.Signal...), append([]uint8(nil), c.Input.Moves...)
			out, err := splitter.FindSimplexPA(c.Input)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(beforeSignal, c.Input.Signal) || !reflect.DeepEqual(beforeMoves, c.Input.Moves) {
				t.Fatal("input was mutated")
			}
			if want, ok := counts[c.Name]; ok {
				got := [3]int{len(out.Clusters), len(out.Candidates), len(out.Spacers)}
				if got != want {
					t.Fatalf("cluster/candidate/spacer counts %v != %v", got, want)
				}
			}
			for _, r := range out.Spacers {
				if r.End != r.Start+1 || r.End > uint64(len(c.Input.Sequence)) {
					t.Fatal("spacer not a valid one-base interval")
				}
			}
			switch c.Name {
			case "contiguous-equal-peak-later":
				if out.Clusters[0].Argmax != 6012 || out.Candidates[0].Start != 1002 {
					t.Fatal("contiguous argmax tie did not choose later sample")
				}
			case "lower-peak-preserves-argmax":
				if out.Clusters[0].Argmax != 6000 {
					t.Fatal("lower signal replaced peak")
				}
			case "merged-equal-peak-later":
				if out.Clusters[0].Argmax != 6504 {
					t.Fatal("merged argmax tie did not choose later cluster")
				}
			case "positive-internal-adapter":
				p := out.PoreCandidates[0]
				if p.MoveStart != 1000 || p.MoveEnd != 1000 || p.BaseSpan != (splitter.Interval{Start: 1000, End: 1001}) || p.Range != (splitter.Interval{Start: 1000, End: 1001}) || p.MeanQ != 9 {
					t.Fatalf("inclusive move mapping differs: %+v", p)
				}
				if out.AdapterChecks[0].Match != (splitter.Interval{Start: 1000, End: 1024}) {
					t.Fatal("adapter half-open match differs")
				}
			case "mean-exactly-10":
				if out.PoreCandidates[0].MeanQ != 10 {
					t.Fatal("Q10 trace differs")
				}
			case "clipped-quality-window":
				if out.PoreCandidates[0].MeanQ != 9 {
					t.Fatal("clipped local mean differs")
				}
			case "width-501-rejected":
				if out.PoreCandidates[0].Rejection != splitter.PoreTooWide {
					t.Fatal("width gate must precede argmax reduction")
				}
			case "exclusive-end-outside-moves":
				if out.PoreCandidates[0].Rejection != splitter.MoveEndOutside {
					t.Fatal("exclusive source end was not rejected")
				}
			case "uncalled-prefix", "no-called-bases":
				if out.PoreCandidates[0].Rejection != splitter.NoCalledBase {
					t.Fatal("uncalled move start was admitted")
				}
			case "top-ten-equal-peaks-later-pairs":
				if out.Candidates[0].Start != 6300 || len(out.RawCandidates) != 12 {
					t.Fatal("equal peaks did not keep later native pairs")
				}
			case "top-ten-higher-early-peak":
				if out.Candidates[0].Start != 5100 || out.Candidates[1].Start != 6900 {
					t.Fatal("top ten did not rank by peak before coordinate")
				}
			case "adapter-first-native-tie":
				if out.AdapterChecks[0].Match.Start != 1000 {
					t.Fatal("adapter tie did not use first native match")
				}
			}
		})
	}
}

func TestRejectMalformedWithoutMutation(t *testing.T) {
	base := oracle.Cases()[3].Input
	for _, c := range []struct {
		name   string
		change func(*splitter.Input)
	}{
		{"empty-moves", func(in *splitter.Input) { in.Moves = nil }},
		{"quality-length", func(in *splitter.Input) { in.Qstring = in.Qstring[:len(in.Qstring)-1] }},
		{"quality-zero", func(in *splitter.Input) { in.Qstring = "!" + in.Qstring[1:] }},
		{"quality-high", func(in *splitter.Input) { in.Qstring = "\x7f" + in.Qstring[1:] }},
		{"invalid-DNA", func(in *splitter.Input) { in.Sequence = "X" + in.Sequence[1:] }},
		{"nonbinary", func(in *splitter.Input) { in.Moves[0] = 2 }},
		{"count", func(in *splitter.Input) { in.Moves[0] = 0 }},
		{"zero-stride", func(in *splitter.Input) { in.Stride = 0 }},
		{"stride-overflow", func(in *splitter.Input) { in.Stride = 1 << 32 }},
		{"sample-shape", func(in *splitter.Input) { in.Signal = in.Signal[:len(in.Signal)-12] }},
		{"nan", func(in *splitter.Input) { in.Signal[0] = 0x7e00 }},
		{"positive-infinity", func(in *splitter.Input) { in.Signal[0] = 0x7c00 }},
		{"negative-infinity", func(in *splitter.Input) { in.Signal[0] = 0xfc00 }},
	} {
		t.Run(c.name, func(t *testing.T) {
			in := base
			in.Signal = append([]uint16(nil), base.Signal...)
			in.Moves = append([]uint8(nil), base.Moves...)
			c.change(&in)
			beforeSignal, beforeMoves := append([]uint16(nil), in.Signal...), append([]uint8(nil), in.Moves...)
			out, err := splitter.FindSimplexPA(in)
			if err == nil || !reflect.DeepEqual(out, splitter.Decision{}) {
				t.Fatal("malformed input returned a decision")
			}
			if !reflect.DeepEqual(beforeSignal, in.Signal) || !reflect.DeepEqual(beforeMoves, in.Moves) {
				t.Fatal("invalid input was mutated")
			}
		})
	}
}

func TestReentrant(t *testing.T) {
	in := oracle.Cases()[3].Input
	want, err := splitter.FindSimplexPA(in)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				got, err := splitter.FindSimplexPA(in)
				if err != nil || !reflect.DeepEqual(got, want) {
					t.Error("concurrent decision differs")
				}
			}
		}()
	}
	wg.Wait()
	if strings.Contains(in.Sequence, "X") {
		t.Fatal("input mutated")
	}
}
