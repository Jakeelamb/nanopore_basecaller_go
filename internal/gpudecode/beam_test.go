package gpudecode

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

// fakeBeamNumbers uses host libm only to test Go policy. It is not a numerical
// oracle for PTX approximations and cannot be selected by production code.
type fakeBeamNumbers struct {
	hook      func(string, []F32Pair) error
	expand    func([]ExpandRow) ([]ExpandResult, error)
	midpoint  func([]F32Pair) ([]uint32, error)
	logPairs  [][]F32Pair
	maxPairs  [][]F32Pair
	callNames []string
}

func fb(v float32) uint32 { return math.Float32bits(v) }
func ff(v uint32) float32 { return beamComparisonScore(v) }
func ftzBits(v float32) uint32 {
	b := fb(v)
	if b&0x7f800000 == 0 {
		b &= 0x80000000
	}
	return b
}
func (n *fakeBeamNumbers) call(name string, p []F32Pair) error {
	n.callNames = append(n.callNames, name)
	if n.hook != nil {
		return n.hook(name, p)
	}
	return nil
}
func (n *fakeBeamNumbers) LogCuts(_ context.Context, cuts []uint32) ([]uint32, error) {
	if err := n.call("log", nil); err != nil {
		return nil, err
	}
	out := make([]uint32, len(cuts))
	for i, c := range cuts {
		out[i] = fb(float32(math.Log(float64(ff(c)))))
	}
	return out, nil
}
func (n *fakeBeamNumbers) Expand5(_ context.Context, rows []ExpandRow, clamp, blank uint32) ([]ExpandResult, error) {
	if err := n.call("expand", nil); err != nil {
		return nil, err
	}
	if n.expand != nil {
		return n.expand(rows)
	}
	out := make([]ExpandResult, len(rows))
	limit := ff(clamp)
	if limit == 0 {
		limit = math.MaxFloat32
	}
	for i, row := range rows {
		maximum := -float32(math.MaxFloat32)
		for a := 0; a < 4; a++ {
			v := signal.Float32FromHalf(row.Transition[a])
			v = min(limit, max(-limit, v))
			out[i].Scores[a] = ftzBits(ff(row.Guide[a]) + ff(ftzBits(ff(row.Previous)+v)))
			maximum = max(maximum, ff(out[i].Scores[a]))
		}
		out[i].Scores[4] = ftzBits(ff(ftzBits(ff(blank)+ff(row.Previous))) + ff(row.Guide[4]))
		out[i].OrderedMax = ftzBits(max(maximum, ff(out[i].Scores[4])))
	}
	return out, nil
}
func (n *fakeBeamNumbers) pair(name string, pairs []F32Pair, f func(float32, float32) float32) ([]uint32, error) {
	if err := n.call(name, pairs); err != nil {
		return nil, err
	}
	out := make([]uint32, len(pairs))
	for i, p := range pairs {
		out[i] = ftzBits(f(ff(p.First), ff(p.Second)))
	}
	return out, nil
}
func (n *fakeBeamNumbers) LogAddPairs(_ context.Context, pairs []F32Pair) ([]uint32, error) {
	n.logPairs = append(n.logPairs, append([]F32Pair(nil), pairs...))
	return n.pair("logadd", pairs, func(a, b float32) float32 {
		d := math.Abs(float64(a) - float64(b))
		if d >= 17 {
			return max(a, b)
		}
		return max(a, b) + float32(math.Log1p(math.Exp(-d)))
	})
}
func (n *fakeBeamNumbers) SubPairs(_ context.Context, pairs []F32Pair) ([]uint32, error) {
	return n.pair("sub", pairs, func(a, b float32) float32 { return a - b })
}
func (n *fakeBeamNumbers) Midpoints(_ context.Context, pairs []F32Pair) ([]uint32, error) {
	if n.midpoint != nil {
		if err := n.call("midpoint", pairs); err != nil {
			return nil, err
		}
		return n.midpoint(pairs)
	}
	return n.pair("midpoint", pairs, func(a, b float32) float32 { return ff(ftzBits(a+b)) / 2 })
}
func (n *fakeBeamNumbers) MaxPairs(_ context.Context, pairs []F32Pair) ([]uint32, error) {
	n.maxPairs = append(n.maxPairs, append([]F32Pair(nil), pairs...))
	return n.pair("max", pairs, func(a, b float32) float32 { return max(a, b) })
}

func beamTestInput(t *testing.T, frames int64) (BeamPlan, BeamInput) {
	t.Helper()
	g, err := PlanGuides(64, frames, 256, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	p, err := PlanBeam(g, 100)
	if err != nil {
		t.Fatal(err)
	}
	in := BeamInput{ScoresNTC: make([]uint16, 64*frames*1024), GuidesTNS: make([]uint32, 64*(frames+1)*256)}
	for i := range in.ScoresNTC {
		in.ScoresNTC[i] = signal.Float16Bits(-1000)
	}
	return p, in
}

func TestBeamPlanRequiresCheckedFixedGuideAndCut100(t *testing.T) {
	g, _ := PlanGuides(32, 7, 1024, 0, 2)
	if _, err := PlanBeam(g, 100); err != nil {
		t.Fatal(err)
	}
	for _, cut := range []float32{0, -1, .5, 1, 99, 101, float32(math.NaN()), float32(math.Inf(1))} {
		if _, err := PlanBeam(g, cut); err == nil {
			t.Fatalf("unqualified cut %v admitted", cut)
		}
	}
	if _, err := PlanBeam(GuidePlan{}, 100); err == nil {
		t.Fatal("zero plan admitted")
	}
}

func TestBeamSeedMatchesRetainedInitialThresholdPTXCases(t *testing.T) {
	// Actual PTX-block fixture source 7e46f596...63b01a, original 613..783.
	// Retained initial_threshold.result.json has S256/S1024 in all four modes.
	for _, states := range []int{256, 1024} {
		for _, mode := range []string{"ascending", "descending", "ties", "ftz_ties"} {
			t.Run(fmt.Sprintf("S%d/%s", states, mode), func(t *testing.T) {
				guides := make([]uint32, states)
				for i := range guides {
					switch mode {
					case "ascending":
						guides[i] = fb(float32(i))
					case "descending":
						guides[i] = fb(float32(states - i))
					case "ties":
						guides[i] = fb(7)
					case "ftz_ties":
						guides[i] = []uint32{0, 0x80000000, 1, 0x80000001}[i%4]
					}
				}
				got := seedBeam(guides)
				if len(got) != 32 {
					t.Fatal("initial threshold used terminal top-one")
				}
				for i, v := range got {
					state := i
					if mode == "ascending" {
						state += states - 32
					}
					if int(v.State) != state || v.ScoreBits != 0 || v.Key != InitialBeamKey(uint16(state)) || v.Stay || v.Previous != 0 {
						t.Fatalf("seed %d = %+v; expected ascending state %d with score +0", i, v, state)
					}
				}
			})
		}
	}
}

func TestBeamTerminalPreservesFullRetainedPTXScratchFixture(t *testing.T) {
	// Source interpreter original 1966..2113, all_tie_final_top_one.output.
	// These labels retain untouched expanded B slots, including all odd slots.
	retained := []int{0, 32, 64, 96, 128, 1, 33, 65, 97, 129, 2, 34, 66, 98, 130, 3, 35, 67, 99, 131, 4, 36, 68, 100, 132, 5, 37, 69, 101, 133, 6, 38}
	expected := []int{0, 1, 64, 3, 128, 5, 33, 7, 97, 9, 2, 11, 66, 13, 130, 15, 35, 17, 99, 19, 4, 21, 68, 23, 132, 25, 37, 27, 101, 29, 6, 31}
	record := func(label int) BeamCandidate {
		return BeamCandidate{uint32(label), fb(1), uint16(label), uint8(3 * label), label%2 != 0}
	}
	front := make([]BeamCandidate, 160)
	for i := range front {
		front[i] = record(i)
	}
	active := make([]BeamCandidate, 32)
	for i, label := range retained {
		active[i] = record(label)
	}
	terminalBeamScratch(active, front)
	for i, label := range expected {
		if active[i] != record(label) {
			t.Fatalf("initialized final slot %d = %+v, want label %d", i, active[i], label)
		}
	}
	if reflect.DeepEqual(expected, retained) {
		t.Fatal("full sort negative control did not differ")
	}
}

func TestBeamTerminalHeadAndScratchAcrossOddCounts(t *testing.T) {
	for n := 1; n <= 32; n++ {
		front := make([]BeamCandidate, 5*n)
		active := make([]BeamCandidate, n)
		for i := range front {
			front[i] = BeamCandidate{Key: uint32(1000 + i), ScoreBits: fb(-100), State: uint16(i)}
		}
		for i := range active {
			active[i] = BeamCandidate{Key: uint32(i), ScoreBits: fb(float32((i * 17) % 11)), State: uint16(i)}
		}
		want, err := TerminalBeam(active)
		if err != nil {
			t.Fatal(err)
		}
		terminalBeamScratch(active, front)
		if active[0] != want {
			t.Fatalf("n%d terminal head %+v want %+v", n, active[0], want)
		}
	}
}

func mergeTestFrame() beamFrame {
	f := beamFrame{active: make([]BeamCandidate, 2), front: make([]BeamCandidate, 10)}
	for i := range f.front {
		f.front[i] = BeamCandidate{Key: uint32(i + 100), ScoreBits: invalidBeamScore, State: uint16(i), Previous: uint8(i)}
	}
	f.front[8] = BeamCandidate{Key: 7, ScoreBits: fb(2), State: 4, Previous: 0, Stay: true}
	f.front[9].Stay = true
	f.front[0] = BeamCandidate{Key: 7, ScoreBits: fb(2), State: 12, Previous: 1}
	f.front[4] = BeamCandidate{Key: 7, ScoreBits: fb(1), State: 99, Previous: 0}
	f.laneMax[0] = fb(2)
	return f
}

func TestBeamMergeCRCOnlyTieToEmissionNoBreakAndInvalidatedKeys(t *testing.T) {
	for _, invalidSecond := range []bool{false, true} {
		frames := []beamFrame{mergeTestFrame()}
		if invalidSecond {
			frames[0].front[4].ScoreBits = invalidBeamScore
		}
		n := &fakeBeamNumbers{}
		before := append([]BeamCandidate(nil), frames[0].front...)
		if err := beamMerge(context.Background(), frames, 3, n); err != nil {
			t.Fatal(err)
		}
		if len(n.logPairs) != 2 || n.logPairs[1][0].First != invalidBeamScore {
			t.Fatalf("merge skipped later key after invalidation: %+v", n.logPairs)
		}
		if frames[0].front[8].ScoreBits != invalidBeamScore || frames[0].front[0].ScoreBits == invalidBeamScore {
			t.Fatal("finite equality did not keep emission representative")
		}
		for i, v := range frames[0].front {
			v.ScoreBits = before[i].ScoreBits
			if v != before[i] {
				t.Fatal("merge changed metadata instead of score alone")
			}
		}
	}
}

func TestBeamMergeRejectsOverlappingStayWritersBeforeArithmetic(t *testing.T) {
	f := mergeTestFrame()
	f.front[9].Key, f.front[9].State = 7, 0
	n := &fakeBeamNumbers{}
	frames := []beamFrame{f}
	err := beamMerge(context.Background(), frames, 18, n)
	var collision *BeamCollisionScheduleError
	if !errors.As(err, &collision) || collision.Frame != 18 || collision.Emission != 0 || collision.FirstLane != 0 || collision.SecondLane != 1 || len(n.callNames) != 0 {
		t.Fatalf("collision qualification %v calls=%v", err, n.callNames)
	}
}

func TestBeamSearchReachesExactBeamStageCollisionWitnessAndPublishesNothing(t *testing.T) {
	// Retained trace_controls.py beam-stage witness. Zero guides are supplied
	// directly, and +256/-1024 scores are outside the HAC/SUP production range.
	// This does not establish production collision reachability or GPU numerics.
	p, in := beamTestInput(t, 20)
	for i := range in.ScoresNTC {
		in.ScoresNTC[i] = signal.Float16Bits(-1024)
	}
	sequence := [2][17]uint8{{}, {1, 0, 3, 3, 2, 1, 3, 1, 0, 3, 2, 3, 1, 1, 0, 0, 1}}
	channel := func(state int, base uint8) int { return 4*((4*state)&255) + 4*state/256 + 4*int(base) }
	for chunk := 0; chunk < 64; chunk++ {
		states := [2]int{}
		for frame := 0; frame < 20; frame++ {
			scores := in.ScoresNTC[(chunk*20+frame)*1024:]
			if frame < 17 {
				for path := range states {
					base := sequence[path][frame]
					scores[channel(states[path], base)] = signal.Float16Bits(256)
					states[path] = (4 * states[path] & 255) + int(base)
				}
			} else {
				for state := 0; state < 256; state++ {
					scores[channel(state, 0)] = signal.Float16Bits(2)
				}
			}
		}
	}
	rows := 0
	out, err := beamSearch(context.Background(), p, in, &fakeBeamNumbers{}, func(frame int, frames []beamFrame) {
		rows++
		want := 2
		if frame == 17 {
			want = 4
		}
		if len(frames[0].active) != want {
			t.Fatalf("exact beam witness row%d count%d want%d", frame+1, len(frames[0].active), want)
		}
	})
	var collision *BeamCollisionScheduleError
	if !errors.As(err, &collision) || collision.Frame != 18 || rows != 18 || out.History != nil || out.FinalScoreBits != nil {
		t.Fatalf("beam-stage witness qualification rows%d collision%+v err%v", rows, collision, err)
	}
}

func TestBeamCutoffNinthConvergenceFallsBackToSavedUpper(t *testing.T) {
	f := beamFrame{front: make([]BeamCandidate, 160)}
	for i := range f.front {
		score := float32(0)
		if i == 0 {
			score = 10
		} else if i < 28 {
			score = 9
		} else if i < 33 {
			score = 5
		}
		f.front[i].ScoreBits = fb(score)
	}
	f.laneMax[17] = fb(10)
	frames := []beamFrame{f}
	calls := 0
	n := &fakeBeamNumbers{midpoint: func(p []F32Pair) ([]uint32, error) {
		calls++
		if calls == 9 {
			return []uint32{fb(8)}, nil
		}
		return []uint32{fb(4)}, nil
	}}
	if err := beamCutoffs(context.Background(), frames, fb(10), n); err != nil {
		t.Fatal(err)
	}
	if calls != 9 || frames[0].cutoff != fb(10) || countBeam(frames[0].front, fb(8)) != 28 {
		t.Fatalf("ninth convergence incorrectly retained midpoint: calls=%d cutoff=%g", calls, ff(frames[0].cutoff))
	}
}

func TestBeamCutoffUsesSnapshotButterflyAndPreservesExpansionMaximum(t *testing.T) {
	f := beamFrame{front: make([]BeamCandidate, 10)}
	for lane := range f.laneMax {
		f.laneMax[lane] = fb(float32(lane))
	}
	frames := []beamFrame{f}
	n := &fakeBeamNumbers{}
	if err := beamCutoffs(context.Background(), frames, fb(5), n); err != nil {
		t.Fatal(err)
	}
	if frames[0].cutoff != fb(26) || len(n.maxPairs) != 5 {
		t.Fatal("cutoff recomputed final-front maximum or refined count below lower target")
	}
	for lane, pair := range n.maxPairs[0] {
		if pair.First != fb(float32(lane)) || pair.Second != fb(float32(lane^16)) {
			t.Fatal("butterfly did not use pre-stage snapshots")
		}
	}
}

func TestBeamSearchHostFakeHistoryImmutableInputsAndNoChunkAliases(t *testing.T) {
	p, in := beamTestInput(t, 2)
	beforeScores, beforeGuides := append([]uint16(nil), in.ScoresNTC...), append([]uint32(nil), in.GuidesTNS...)
	out, err := BeamSearch(context.Background(), p, in, &fakeBeamNumbers{})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.History) != 64 || len(out.FinalScoreBits) != 64 || !reflect.DeepEqual(in.ScoresNTC, beforeScores) || !reflect.DeepEqual(in.GuidesTNS, beforeGuides) {
		t.Fatal("batch extent or borrowed input changed")
	}
	for chunk := range out.History {
		rows := out.History[chunk].Rows
		if len(rows) != 3 || len(rows[0]) != 32 || len(rows[1]) != 32 || len(rows[2]) != 32 || out.FinalScoreBits[chunk] != fb(4) {
			t.Fatalf("chunk %d incomplete history or unguided terminal score", chunk)
		}
		trace, err := TraceBeamHistory(rows[1:], 256)
		if err != nil || !reflect.DeepEqual(trace.Path, []int32{0, 0}) || !reflect.DeepEqual(trace.Moves, []byte{1, 0}) {
			t.Fatalf("host-fake traceback %+v %v", trace, err)
		}
	}
	out.History[0].Rows[0][0].State = 255
	out.History[0].Rows[1][0].Previous = 31
	if out.History[1].Rows[0][0].State != 0 || out.History[1].Rows[1][0].Previous != 0 {
		t.Fatal("history aliases another chunk")
	}
}

func TestBeamSearchGathersSourceChannelsAndExpansionOrdering(t *testing.T) {
	p, in := beamTestInput(t, 2)
	for chunk := 0; chunk < 64; chunk++ {
		for state := 0; state < 256; state++ {
			in.GuidesTNS[chunk*256+state] = fb(float32(state))
		}
	}
	for i := range in.ScoresNTC {
		in.ScoresNTC[i] = uint16(i % 0x7000)
	}
	checked := false
	n := &fakeBeamNumbers{expand: func(rows []ExpandRow) ([]ExpandResult, error) {
		if !checked {
			checked = true
			for chunk := 0; chunk < 64; chunk++ {
				for i := 0; i < 32; i++ {
					state := 224 + i
					u, q := (4*state)&255, 4*state/256
					row := rows[chunk*32+i]
					for a := 0; a < 4; a++ {
						if row.Transition[a] != in.ScoresNTC[chunk*2048+4*u+q+4*a] || row.Previous != 0 {
							t.Fatal("source score channel or initial +0 differs")
						}
					}
				}
			}
		}
		out := make([]ExpandResult, len(rows))
		for i := range out {
			out[i] = ExpandResult{[5]uint32{fb(-1000), fb(-1000), fb(-1000), fb(-1000), fb(2)}, fb(2)}
		}
		return out, nil
	}}
	observed := 0
	_, err := beamSearch(context.Background(), p, in, n, func(frame int, frames []beamFrame) {
		if frame != 0 {
			return
		}
		observed++
		front := frames[0].front
		for i := 0; i < 32; i++ {
			state := 224 + i
			for a := 0; a < 4; a++ {
				v := front[4*i+a]
				if v.State != uint16((4*state&255)+a) || v.Previous != uint8(i) || v.Stay {
					t.Fatal("emission order or metadata differs")
				}
			}
			if v := front[128+i]; v.State != uint16(state) || v.Previous != uint8(i) || !v.Stay {
				t.Fatal("stay expansion order differs")
			}
		}
		frames[0].active[0].State = 1023
	})
	if err != nil || !checked || observed != 1 {
		t.Fatalf("gather/expand %v", err)
	}
}

func TestBeamSearchLateFailureCancellationAndPoisonReturnNoPartialBatch(t *testing.T) {
	for _, mode := range []string{"failure", "cancel", "extent", "nonfinite", "late-cancel"} {
		t.Run(mode, func(t *testing.T) {
			p, in := beamTestInput(t, 2)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			sentinel := errors.New("injected numeric failure")
			n := &fakeBeamNumbers{}
			subtractions := 0
			n.hook = func(name string, _ []F32Pair) error {
				if name == "sub" {
					subtractions++
					if subtractions == 4 {
						if mode == "failure" {
							return sentinel
						}
						if mode == "cancel" {
							cancel()
						}
					}
				}
				return nil
			}
			if mode == "extent" || mode == "nonfinite" {
				n.expand = func(rows []ExpandRow) ([]ExpandResult, error) {
					if mode == "extent" {
						return nil, nil
					}
					out := make([]ExpandResult, len(rows))
					out[len(out)-1].OrderedMax = 0x7f800000
					return out, nil
				}
			}
			out, err := beamSearch(ctx, p, in, n, func(frame int, _ []beamFrame) {
				if mode == "late-cancel" && frame == 1 {
					cancel()
				}
			})
			if err == nil || out.History != nil || out.FinalScoreBits != nil {
				t.Fatalf("partial success escaped %v %+v", err, out)
			}
			if mode == "failure" && !errors.Is(err, sentinel) || (mode == "cancel" || mode == "late-cancel") && !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
		})
	}
}

func TestBeamSearchRejectsInputsBeforeNumericProvider(t *testing.T) {
	for _, mode := range []string{"plan", "nil-context", "extent", "half-nan", "guide-inf", "canceled"} {
		t.Run(mode, func(t *testing.T) {
			p, in := beamTestInput(t, 1)
			ctx := context.Background()
			switch mode {
			case "plan":
				p = BeamPlan{}
			case "nil-context":
				ctx = nil
			case "extent":
				in.ScoresNTC = in.ScoresNTC[1:]
			case "half-nan":
				in.ScoresNTC[0] = 0x7e00
			case "guide-inf":
				in.GuidesTNS[len(in.GuidesTNS)-1] = 0x7f800000
			case "canceled":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			}
			n := &fakeBeamNumbers{}
			out, err := BeamSearch(ctx, p, in, n)
			if err == nil || len(n.callNames) != 0 || out.History != nil {
				t.Fatal("unsafe input reached arithmetic")
			}
		})
	}
}

func TestBeamWirePackingUsesExplicitLittleEndianFields(t *testing.T) {
	front := []BeamCandidate{{0x11223344, 0x55667788, 0x99aa, 0xbb, true}}
	want := []byte{0x44, 0x33, 0x22, 0x11, 0x88, 0x77, 0x66, 0x55, 0xaa, 0x99, 0xbb, 1}
	if !reflect.DeepEqual(PackBeamFront(front), want) || !reflect.DeepEqual(PackBeamHistory([]BeamHistoryRecord{{0x99aa, 0xbb, 0xcc}}), []byte{0xaa, 0x99, 0xbb, 0xcc}) {
		t.Fatal("logical struct was mistaken for wire ABI")
	}
	operand := f32BeamOperand(2, []uint32{0x11223344, 0x55667788})
	if binary.LittleEndian.Uint32(operand.bytes[4:]) != 0x55667788 {
		t.Fatal("numeric operands are not explicitly packed")
	}
}

func TestBeamOptionalDiagnosticsOwnAllInitializedFrameEvidence(t *testing.T) {
	p, in := beamTestInput(t, 2)
	batch, trace, err := BeamSearchWithTrace(context.Background(), p, in, &fakeBeamNumbers{})
	if err != nil || len(trace) != 2 || len(trace[0].Chunks) != 64 {
		t.Fatalf("diagnostic result %v", err)
	}
	for _, frame := range trace {
		for _, chunk := range frame.Chunks {
			if len(chunk.Expanded) != 160 || len(chunk.Merged) != 160 || len(chunk.Active) != 32 || len(chunk.RetainedIndices) != 32 || len(chunk.History) != 32 || len(chunk.Cutoffs) != 1 {
				t.Fatal("diagnostic omitted initialized frame decisions")
			}
		}
	}
	trace[0].Chunks[0].History[0].State = 255
	trace[0].Chunks[0].Active[0].State = 254
	trace[0].Chunks[0].Expanded[0].ScoreBits = 0
	if batch.History[0].Rows[1][0].State != 0 || trace[1].Chunks[0].History[0].State != 0 || trace[0].Chunks[0].Merged[0].ScoreBits == 0 {
		t.Fatal("diagnostic aliases history or another phase/frame")
	}
	ctx, cancel := context.WithCancel(context.Background())
	n := &fakeBeamNumbers{hook: func(name string, _ []F32Pair) error {
		if name == "expand" {
			cancel()
		}
		return nil
	}}
	batch, trace, err = BeamSearchWithTrace(ctx, p, in, n)
	if err == nil || trace != nil || batch.History != nil {
		t.Fatal("partial diagnostics published on cancellation")
	}
}

func TestBeamGPUAdapterRejectsUncheckedOwnersWorkspaceAndDomainBeforeDispatch(t *testing.T) {
	// Empty typed handles exercise only admission. No backend exists here.
	n := &GPUBeamNumbers{device: &tensor.Device{}, stream: &tensor.Stream{}, maxRows: 1}
	ctx := context.Background()
	if _, err := n.LogCuts(ctx, []uint32{fb(100), fb(100)}); err == nil {
		t.Fatal("workspace overflow admitted")
	}
	if _, err := n.LogCuts(ctx, []uint32{1}); err == nil {
		t.Fatal("FTZ-zero positive log operand admitted")
	}
	if _, err := n.Expand5(ctx, []ExpandRow{{}}, 0x80000001, fb(2)); err == nil {
		t.Fatal("negative subnormal clamp admitted")
	}
	if _, err := n.SubPairs(ctx, []F32Pair{{0x7f800000, 0}}); err == nil {
		t.Fatal("nonfinite pair admitted")
	}
	if _, err := n.LogCuts(ctx, nil); err != nil {
		t.Fatal("zero rows dispatched", err)
	}
	if _, err := n.Expand5(ctx, nil, 0, fb(2)); err != nil {
		t.Fatal("zero expand rows dispatched", err)
	}
	var absent *GPUBeamNumbers
	if _, err := absent.SubPairs(ctx, nil); err == nil {
		t.Fatal("nil provider admitted")
	}
	if _, err := NewGPUBeamNumbers(nil, nil, [32]byte{}, 1); err == nil {
		t.Fatal("unauthenticated program admitted")
	}
	p, _ := beamTestInput(t, 1)
	if _, err := DownloadBeamInput(ctx, p, nil, nil, nil); err == nil {
		t.Fatal("absent immutable storage admitted")
	}
}
