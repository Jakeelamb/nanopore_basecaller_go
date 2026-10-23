package subreadoracle

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/subread"
)

func Fixture(bases int) subread.Read {
	signal := make([]uint16, bases*6)
	for i := range signal {
		signal[i] = uint16(i * 1009)
	}
	moves := make([]uint8, bases)
	for i := range moves {
		moves[i] = 1
	}
	return subread.Read{
		Signal: signal, Sequence: strings.Repeat("ACGTN", (bases+4)/5)[:bases], Qstring: strings.Repeat("I", bases), Moves: moves, Stride: 6,
		ID: "550e8400-e29b-41d4-a716-446655440000", PrevID: "outer-prev", NextID: "outer-next",
		StartSample: 100, EndSample: uint64(100 + len(signal) + 10), AcquisitionStartMS: 1700000000123, SampleRate: 5000,
		StartTimeMS: 20, StartTime: "original-clock", LeadingTrim: 10, SplitPoint: 17, SubreadID: 71, SplitCount: 72,
		ReadNumber: 42, NumSamples: uint64(len(signal) + 10), NumEvents: 999, PreTrimSeqLen: 123, MeanQStart: 17, EndReason: "unblock_mux_change", IsMuxChange: true,
		Inherited: subread.Inherited{
			DigitisationBits: math.Float32bits(8192), RangeBits: math.Float32bits(1467.6), OffsetBits: math.Float32bits(-12.25), OpenPoreLevelBits: 0x7fc00123,
			ShiftBits: math.Float32bits(1.25), ScaleBits: math.Float32bits(2.5), ScalingBits: math.Float32bits(0.175), ScalingMethod: "pa",
			RunID: "run-id", FlowCellProductCode: "FLO-PRO114M", SequencingKit: "sqk-lsk114", FlowcellID: "PAQ12345", PositionID: "P1", AcquisitionID: "acq-id", ExperimentID: "experiment-id", ModelName: "dna-model", SampleID: "sample-id", ProtocolStartMS: 1699990000000,
			ReadTag: 456, ClientToken: 789, Mux: 3, Channel: 42, Filename: "input.pod5", PoreType: "R10",
		},
	}
}

func Cases() []Case {
	var cases []Case
	add := func(name string, spacers []subread.Interval, change func(*subread.Read)) {
		read := Fixture(8)
		if change != nil {
			change(&read)
		}
		cases = append(cases, Case{name, read, spacers})
	}
	add("root-no-spacers", nil, nil)
	add("two-children", []subread.Interval{{Start: 2, End: 3}}, nil)
	add("multiple-children", []subread.Interval{{Start: 1, End: 2}, {Start: 3, End: 4}, {Start: 6, End: 7}}, nil)
	add("full-spacer-no-children", []subread.Interval{{Start: 0, End: 8}}, nil)
	add("prefix-one-child-outer-neighbors", []subread.Interval{{Start: 0, End: 3}}, nil)
	add("suffix-one-child-end-reason", []subread.Interval{{Start: 3, End: 8}}, nil)
	add("adjacent-spacers-zero-base-omitted", []subread.Interval{{Start: 1, End: 3}, {Start: 3, End: 5}}, nil)
	add("zero-width-front", []subread.Interval{{Start: 0, End: 0}}, nil)
	add("zero-width-middle", []subread.Interval{{Start: 3, End: 3}}, nil)
	add("zero-width-end", []subread.Interval{{Start: 8, End: 8}}, nil)
	add("duplicate-zero-width", []subread.Interval{{Start: 3, End: 3}, {Start: 3, End: 3}}, nil)
	add("leading-zero-moves-drop-prefix", []subread.Interval{{Start: 2, End: 3}}, func(r *subread.Read) {
		r.Moves[0] = 0
		r.Moves[1] = 0
		r.Sequence = r.Sequence[:6]
		r.Qstring = r.Qstring[:6]
	})
	add("leading-zero-moves-no-spacers-identity", nil, func(r *subread.Read) {
		r.Moves[0] = 0
		r.Moves[1] = 0
		r.Sequence = r.Sequence[:6]
		r.Qstring = r.Qstring[:6]
	})
	add("partial-tail-floor-moves", []subread.Interval{{Start: 2, End: 3}}, func(r *subread.Read) { r.Signal = append(r.Signal, 0xabcd) })
	add("partial-tail-ceil-moves", []subread.Interval{{Start: 2, End: 3}}, func(r *subread.Read) {
		r.Signal = append(r.Signal, 0xabcd)
		r.Moves = append(r.Moves, 1)
		r.Sequence += "A"
		r.Qstring += "I"
	})
	add("partial-tail-zero-whole-stride-omitted", []subread.Interval{{Start: 0, End: 8}}, func(r *subread.Read) {
		r.Signal = append(r.Signal, 0xabcd)
		r.Moves = append(r.Moves, 1)
		r.Sequence += "A"
		r.Qstring += "I"
	})
	add("existing-root-parent-no-spacers-finalized", nil, func(r *subread.Read) { r.ParentID = "root-parent" })
	add("existing-root-parent-split", []subread.Interval{{Start: 2, End: 3}}, func(r *subread.Read) { r.ParentID = "root-parent" })
	add("uint32-split-point-wrap", []subread.Interval{{Start: 2, End: 3}}, func(r *subread.Read) { r.SplitPoint = math.MaxUint32 - 5 })
	add("large-leading-trim-wrap", []subread.Interval{{Start: 2, End: 3}}, func(r *subread.Read) { r.LeadingTrim = uint64(math.MaxUint32) + 11 })
	add("non-mux-last-reason", []subread.Interval{{Start: 2, End: 3}}, func(r *subread.Read) { r.EndReason = "signal_positive"; r.IsMuxChange = false })
	for _, sample := range []uint64{0, 1, 2, 3, 4, 5, 999, 1000, 1001} {
		add(fmt.Sprintf("clock-2000hz-start-%d", sample), []subread.Interval{{Start: 0, End: 0}}, func(r *subread.Read) { r.StartSample = sample; r.LeadingTrim = 0; r.SampleRate = 2000 })
	}
	for _, clock := range []struct {
		name                         string
		acquisition, sample, leading uint64
		spacer                       subread.Interval
	}{
		{"last-acquisition", 9223372036854, 0, 0, subread.Interval{}},
		{"last-summed-rounded", 9223372036853, 1, 0, subread.Interval{}},
		{"last-trim-and-spacer", 9223372036849, 1, 2, subread.Interval{Start: 0, End: 1}},
	} {
		add("clock-boundary-"+clock.name, []subread.Interval{clock.spacer}, func(r *subread.Read) {
			r.AcquisitionStartMS, r.StartSample, r.LeadingTrim, r.SampleRate = clock.acquisition, clock.sample, clock.leading, 2000
		})
	}
	empty := Fixture(0)
	empty.ParentID = "already-parented"
	cases = append(cases, Case{"empty-sequence-bypass", empty, nil})
	noMoves := Fixture(8)
	noMoves.Moves = nil
	noMoves.ParentID = "already-parented"
	cases = append(cases, Case{"empty-moves-bypass", noMoves, []subread.Interval{{Start: 2, End: 3}}})
	allZero := Fixture(8)
	for i := range allZero.Moves {
		allZero.Moves[i] = 0
	}
	allZero.Sequence = ""
	allZero.Qstring = ""
	cases = append(cases, Case{"zero-base-bypass", allZero, nil})
	rng := rand.New(rand.NewSource(7301022))
	for k := 0; k < 200; k++ {
		r := Fixture(1 + rng.Intn(40))
		for i := range r.Moves {
			if rng.Intn(4) == 0 {
				r.Moves[i] = 0
			}
		}
		bases := 0
		for _, m := range r.Moves {
			bases += int(m)
		}
		r.Sequence = r.Sequence[:bases]
		r.Qstring = r.Qstring[:bases]
		if rng.Intn(2) == 0 {
			r.Signal = append(r.Signal, 1, 2, 3)
			if rng.Intn(2) == 0 {
				r.Moves = append(r.Moves, 0)
			}
		}
		var spacers []subread.Interval
		for start := 0; start < bases; {
			start += rng.Intn(5)
			if start > bases {
				break
			}
			end := min(bases, start+rng.Intn(4))
			spacers = append(spacers, subread.Interval{Start: uint64(start), End: uint64(end)})
			start = max(end, start+1)
		}
		r.StartSample = uint64(rng.Intn(50000))
		r.LeadingTrim = uint64(rng.Intn(200))
		r.SplitPoint = rng.Uint32()
		if rng.Intn(2) == 0 {
			r.ParentID = "retained-root"
		}
		if rng.Intn(2) == 0 {
			r.IsMuxChange = false
			r.EndReason = "signal_positive"
		}
		cases = append(cases, Case{fmt.Sprintf("deterministic-sparse-%03d", k), r, spacers})
	}
	return cases
}
func UUIDCases() []UUIDCase {
	var out []UUIDCase
	for _, parent := range []string{"550e8400-e29b-41d4-a716-446655440000", "ROOT-Verbatim", ""} {
		for _, index := range []uint64{0, 1, 9, 10, 1 << 32, 1 << 63, math.MaxUint64} {
			out = append(out, UUIDCase{parent, index})
		}
	}
	return out
}
