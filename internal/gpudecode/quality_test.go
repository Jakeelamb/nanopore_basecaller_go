package gpudecode

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"testing"
)

// This fake returns host test values. It does not model GPU approximations.
type hostQualityNumbers struct {
	hook      func(string) error
	values    []QualityValues
	groups    [][]uint32
	summaries []QualitySummaryRow
	bad       string
}

func qbits(v float32) uint32 { return math.Float32bits(v) }
func (n *hostQualityNumbers) call(op string) error {
	if n.hook != nil {
		return n.hook(op)
	}
	return nil
}
func (n *hostQualityNumbers) Probabilities(_ context.Context, v []QualityValues) ([]uint32, error) {
	n.values = append([]QualityValues(nil), v...)
	if e := n.call("probability"); e != nil {
		return nil, e
	}
	r := make([]uint32, len(v))
	for i := range r {
		r[i] = qbits(float32(i%6+1) / 16)
	}
	if n.bad == "probability" {
		r[0] = 0x7f800000
	}
	return r, nil
}
func (n *hostQualityNumbers) Groups(_ context.Context, rows [][]uint32, scale, shift uint32) ([]QualityGroupNumber, error) {
	n.groups = rows
	if e := n.call("group"); e != nil {
		return nil, e
	}
	r := make([]QualityGroupNumber, len(rows))
	for i, row := range rows {
		r[i] = QualityGroupNumber{qbits(float32(len(row)) / 16), byte(34 + len(row))}
	}
	if n.bad == "group" {
		r[0].Character = 33
	}
	return r, nil
}
func (n *hostQualityNumbers) Summaries(_ context.Context, rows []QualitySummaryRow) ([]QualitySummaryNumber, error) {
	n.summaries = rows
	if e := n.call("summary"); e != nil {
		return nil, e
	}
	r := make([]QualitySummaryNumber, len(rows))
	for i, row := range rows {
		sum := float32(0)
		for _, b := range row.Errors {
			sum += math.Float32frombits(b)
		}
		mean := float32(math.Inf(1))
		if row.Emitted > 0 {
			mean = sum / float32(row.Emitted)
		}
		r[i] = QualitySummaryNumber{qbits(float32(row.Frames-row.Emitted) / float32(row.Frames)), qbits(float32(row.Emitted) / float32(row.Frames)), qbits(mean)}
	}
	if n.bad == "summary" {
		r[0].StayBits = 0x7fc00000
	}
	return r, nil
}

func qualityTestInput(t *testing.T, pad bool) (DecodePlan, QualityInput) {
	t.Helper()
	g, e := PlanGuides(64, 6, 256, 0, 2)
	if e != nil {
		t.Fatal(e)
	}
	p, e := PlanDecode(g, 1.02, -.3, pad)
	if e != nil {
		t.Fatal(e)
	}
	in := QualityInput{WorkingScoresNTC: make([]uint16, 64*6*1024), Trace: make([]BeamTrace, 64), FinalScoreBits: make([]uint32, 64)}
	for i := range in.WorkingScoresNTC {
		in.WorkingScoresNTC[i] = uint16(0x1000 + i%0x2000)
	}
	for i := range in.Trace {
		in.Trace[i] = BeamTrace{Path: []int32{5, 6, 7, 8, 9, 10}, Moves: []byte{1, 0, 1, 0, 0, 1}}
		in.FinalScoreBits[i] = qbits(12)
	}
	return p, in
}

func TestDecodeQualityGatherGroupsOutputAndPaddedCounter(t *testing.T) {
	for _, pad := range []bool{false, true} {
		p, in := qualityTestInput(t, pad)
		saved := append([]uint16(nil), in.WorkingScoresNTC...)
		n := &hostQualityNumbers{}
		got, e := DecodeQuality(context.Background(), p, in, n)
		if e != nil {
			t.Fatal(e)
		}
		if len(got.Chunks) != 64 || len(n.values) != 64*6 || len(n.groups) != 64*3 || len(n.summaries) != 64 {
			t.Fatal("incomplete batch")
		}
		schedule, _ := PlanQuality(in.Trace[0], 256, pad)
		for i, f := range schedule.Frames {
			v := n.values[i]
			offset := f.PosteriorRow * 256
			if v.Center != in.WorkingScoresNTC[offset+int(f.State)] || v.Count != f.NeighborCount {
				t.Fatal("center/row gather differs")
			}
			for j := uint8(0); j < v.Count; j++ {
				if v.Neighbors[j] != in.WorkingScoresNTC[offset+int(f.Neighbors[j])] {
					t.Fatal("neighbor order differs")
				}
			}
		}
		for chunk, out := range got.Chunks {
			wantSeq, wantQ := make([]byte, 6), make([]byte, 6)
			for i, g := range schedule.Groups {
				wantSeq[g.Output], wantQ[g.Output] = g.Base, byte(34+[]int{2, 3, 1}[i])
			}
			if !reflect.DeepEqual(out.Sequence, wantSeq) || !reflect.DeepEqual(out.Q, wantQ) || !reflect.DeepEqual(out.Path, []int32{1, 2, 3, 0, 1, 2}) || !reflect.DeepEqual(out.Moves, in.Trace[chunk].Moves) {
				t.Fatal("output placement/path differs")
			}
			wantCount := uint32(3)
			if pad {
				wantCount = 0
			}
			if out.ResultWords[0] != uint32(chunk*6) || out.ResultWords[1] != 6 || out.ResultWords[2] != wantCount || out.ResultWords[3] != qbits(12) || out.ResultWords[6] != 0 {
				t.Fatal("raw result words differ")
			}
			if pad && finiteBeamScore(out.ResultWords[7]) {
				t.Fatal("padded zero denominator was corrected")
			}
		}
		got.Chunks[0].Path[0] = 255
		got.Chunks[0].Moves[0] = 0
		if in.Trace[0].Path[0] != 5 || in.Trace[0].Moves[0] != 1 || got.Chunks[1].Path[0] != 1 || !reflect.DeepEqual(in.WorkingScoresNTC, saved) {
			t.Fatal("borrowed input/output alias")
		}
	}
}

func TestDecodeQualitySourceNegativeScoreFallbackIgnoresHistoryAndPosteriors(t *testing.T) {
	p, in := qualityTestInput(t, true)
	for i := range in.Trace {
		in.FinalScoreBits[i] = qbits(-1)
		in.Trace[i] = BeamTrace{}
	}
	for i := range in.WorkingScoresNTC {
		in.WorkingScoresNTC[i] = 0x7e00
	}
	n := &hostQualityNumbers{hook: func(string) error { t.Fatal("fallback called numeric provider"); return nil }}
	got, e := DecodeQuality(context.Background(), p, in, n)
	if e != nil {
		t.Fatal(e)
	}
	for chunk, v := range got.Chunks {
		if !reflect.DeepEqual(v.Path, make([]int32, 6)) || !reflect.DeepEqual(v.Moves, []byte{1, 0, 0, 0, 0, 0}) || !reflect.DeepEqual(v.Sequence, []byte{'A', 0, 0, 0, 0, 0}) || !reflect.DeepEqual(v.Q, []byte{'!', 0, 0, 0, 0, 0}) || v.ResultWords != [8]uint32{uint32(chunk * 6), 6, 1, qbits(-1), 0, qbits(1), 0, qbits(1)} {
			t.Fatal("source fallback changed")
		}
	}
}

func TestDecodeQualityFailuresAndCancellationPublishNothing(t *testing.T) {
	stop := errors.New("injected quality failure")
	for _, op := range []string{"probability", "group", "summary"} {
		for _, mode := range []string{"error", "cancel", "poison"} {
			p, in := qualityTestInput(t, false)
			ctx, cancel := context.WithCancel(context.Background())
			n := &hostQualityNumbers{}
			if mode == "poison" {
				n.bad = op
			} else {
				n.hook = func(name string) error {
					if name == op {
						if mode == "error" {
							return stop
						}
						cancel()
					}
					return nil
				}
			}
			out, e := DecodeQuality(ctx, p, in, n)
			cancel()
			if e == nil || out.Chunks != nil {
				t.Fatal("partial batch escaped", op, mode)
			}
		}
	}
	for _, change := range []func(*QualityInput){func(in *QualityInput) { in.Trace[0].Path[0] = 256 }, func(in *QualityInput) { in.WorkingScoresNTC[256+5] = 0x7e00 }, func(in *QualityInput) { in.FinalScoreBits[0] = 0x7f800000 }, func(in *QualityInput) { in.WorkingScoresNTC = in.WorkingScoresNTC[1:] }} {
		p, in := qualityTestInput(t, false)
		change(&in)
		n := &hostQualityNumbers{hook: func(string) error { t.Fatal("invalid input reached numerics"); return nil }}
		if out, e := DecodeQuality(context.Background(), p, in, n); e == nil || out.Chunks != nil {
			t.Fatal("invalid quality input admitted")
		}
	}
}

func TestQualityFromHistoryAndExplicitResultWire(t *testing.T) {
	p, in := qualityTestInput(t, false)
	history := make([][][]BeamHistoryRecord, 64)
	for c := range history {
		history[c] = make([][]BeamHistoryRecord, 7)
		for row := range history[c] {
			history[c][row] = []BeamHistoryRecord{{State: 0, StayByte: 1}}
		}
	}
	got, e := QualityFromHistory(context.Background(), p, in.WorkingScoresNTC, history, in.FinalScoreBits, &hostQualityNumbers{})
	if e != nil || got.Chunks[0].Moves[0] != 1 || got.Chunks[0].ResultWords[2] != 1 {
		t.Fatal("Go traceback handoff differs", e)
	}
	words := [8]uint32{1, 2, 3, 0x80000000, 5, 6, 7, 0x7fc12345}
	wire := PackQualityResult(words)
	for i, w := range words {
		if binary.LittleEndian.Uint32(wire[4*i:]) != w {
			t.Fatal("result float bits or integer words changed")
		}
	}
}

func TestQualityNumericAdmissionAndRaggedOrder(t *testing.T) {
	if _, e := PlanDecode(GuidePlan{}, 1, 0, false); e == nil {
		t.Fatal("unchecked plan")
	}
	g, _ := PlanGuides(32, 1, 1024, 0, 2)
	if _, e := PlanDecode(g, float32(math.NaN()), 0, false); e == nil {
		t.Fatal("NaN quality scale")
	}
	keys, buckets, e := qualityBuckets([]int{3, 1, 3, 2})
	if e != nil || !reflect.DeepEqual(keys, []int{1, 2, 3}) || !reflect.DeepEqual(buckets[3], []int{0, 2}) {
		t.Fatal("numeric bucketing reordered a row")
	}
	if _, _, e := qualityBuckets([]int{0}); e == nil {
		t.Fatal("zero group row")
	}
	var n *GPUQualityNumbers
	if _, e := n.Groups(context.Background(), nil, 0, 0); e == nil {
		t.Fatal("nil numeric owner")
	}
}
