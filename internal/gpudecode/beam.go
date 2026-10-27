package gpudecode

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
)

const beamWidth = 32
const invalidBeamScore uint32 = 0xff7fffff

type BeamInput struct {
	ScoresNTC []uint16
	GuidesTNS []uint32
}

type ChunkBeamHistory struct{ Rows [][]BeamHistoryRecord }

type BeamBatch struct {
	History        []ChunkBeamHistory
	FinalScoreBits []uint32
}

type checkedBeamPlan struct {
	guide GuidePlan
	cut   uint32
}

type BeamPlan struct{ checked checkedBeamPlan }

func PlanBeam(guide GuidePlan, cut float32) (BeamPlan, error) {
	want, err := PlanGuides(guide.n, guide.t, guide.states, guide.clamp, guide.blank)
	if err != nil || want != guide || math.Float32bits(cut) != math.Float32bits(100) {
		return BeamPlan{}, fmt.Errorf("initialized fixed guide plan and cut100 required")
	}
	return BeamPlan{checkedBeamPlan{guide, math.Float32bits(cut)}}, nil
}

type F32Pair struct{ First, Second uint32 }
type ExpandRow struct {
	Previous   uint32
	Transition [4]uint16
	Guide      [5]uint32
}
type ExpandResult struct {
	Scores     [5]uint32
	OrderedMax uint32
}

// BeamNumbers performs arithmetic only. Go selects every operand and decision.
type BeamNumbers interface {
	LogCuts(context.Context, []uint32) ([]uint32, error)
	Expand5(context.Context, []ExpandRow, uint32, uint32) ([]ExpandResult, error)
	LogAddPairs(context.Context, []F32Pair) ([]uint32, error)
	SubPairs(context.Context, []F32Pair) ([]uint32, error)
	Midpoints(context.Context, []F32Pair) ([]uint32, error)
	MaxPairs(context.Context, []F32Pair) ([]uint32, error)
}

// BeamCollisionScheduleError marks an unqualified source warp schedule. Such
// inputs are legal source inputs. No serialized lane schedule is assumed here.
type BeamCollisionScheduleError struct {
	Chunk, Frame, Emission, FirstLane, SecondLane int
}

func (e *BeamCollisionScheduleError) Error() string {
	return fmt.Sprintf("decoder beam collision schedule unqualified at chunk %d frame %d emission %d, stay lanes %d and %d", e.Chunk, e.Frame, e.Emission, e.FirstLane, e.SecondLane)
}

type BeamNumericError struct{ Operation, Reason string }

func (e *BeamNumericError) Error() string { return "decoder beam " + e.Operation + ": " + e.Reason }

type BeamMergeTrace struct {
	Lane, Emission, Winner int
	Operands               F32Pair
	Combined               uint32
}

type BeamCutoffTrace struct {
	Step, Count int
	CutoffBits  uint32
	Fallback    bool
}

type BeamChunkTrace struct {
	Expanded, Merged, Active       []BeamCandidate
	ExpansionLaneMax, MergeLaneMax [beamWidth]uint32
	Pairs                          []BeamMergeTrace
	Cutoffs                        []BeamCutoffTrace
	RetainedIndices                []int
	History                        []BeamHistoryRecord
}

type BeamFrameTrace struct {
	Frame  int
	Chunks []BeamChunkTrace
}

type beamFrame struct {
	active     []BeamCandidate
	front      []BeamCandidate
	laneMax    [beamWidth]uint32
	cutoff     uint32
	diagnostic *BeamChunkTrace
}

func beamBitsCall(ctx context.Context, op string, count int, call func() ([]uint32, error)) ([]uint32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, nil
	}
	out, err := call()
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(out) != count {
		return nil, &BeamNumericError{op, "result extent differs"}
	}
	for _, b := range out {
		if !finiteBeamScore(b) {
			return nil, &BeamNumericError{op, "nonfinite result"}
		}
	}
	return out, nil
}

func seedBeam(guides []uint32) []BeamCandidate {
	ordered := append([]uint32(nil), guides...)
	sort.SliceStable(ordered, func(i, j int) bool {
		return beamComparisonScore(ordered[i]) > beamComparisonScore(ordered[j])
	})
	threshold := beamComparisonScore(ordered[beamWidth-1])
	active := make([]BeamCandidate, 0, beamWidth)
	for state, bits := range guides {
		if beamComparisonScore(bits) >= threshold {
			active = append(active, BeamCandidate{Key: InitialBeamKey(uint16(state)), State: uint16(state)})
			if len(active) == beamWidth {
				break
			}
		}
	}
	return active
}

func beamHistory(active []BeamCandidate) []BeamHistoryRecord {
	row := make([]BeamHistoryRecord, len(active))
	for i, v := range active {
		var stay uint8
		if v.Stay {
			stay = 1
		}
		row[i] = BeamHistoryRecord{v.State, v.Previous, stay}
	}
	return row
}

// PackBeamHistory serializes initialized records, independent of Go ABI layout.
func PackBeamHistory(row []BeamHistoryRecord) []byte {
	out := make([]byte, 4*len(row))
	for i, v := range row {
		binary.LittleEndian.PutUint16(out[4*i:], v.State)
		out[4*i+2], out[4*i+3] = v.Previous, v.StayByte
	}
	return out
}

// PackBeamFront serializes logical records, independent of Go ABI layout.
func PackBeamFront(row []BeamCandidate) []byte {
	out := make([]byte, 12*len(row))
	for i, v := range row {
		binary.LittleEndian.PutUint32(out[12*i:], v.Key)
		binary.LittleEndian.PutUint32(out[12*i+4:], v.ScoreBits)
		binary.LittleEndian.PutUint16(out[12*i+8:], v.State)
		out[12*i+10] = v.Previous
		if v.Stay {
			out[12*i+11] = 1
		}
	}
	return out
}

// terminalBeamScratch preserves the partial writes to both source buffers.
// Unwritten expanded slots are observable in initialized final history rows.
func terminalBeamScratch(active, expanded []BeamCandidate) {
	n := len(active)
	if n < 2 {
		return
	}
	src, dst := active, expanded[:n]
	for width := 1; width < n; width *= 2 {
		for start := 0; start < n; start += 2 * width {
			winner := src[start]
			if right := start + width; right < n && beamComparisonScore(src[right].ScoreBits) > beamComparisonScore(winner.ScoreBits) {
				winner = src[right]
			}
			dst[start] = winner
		}
		src, dst = dst, src
	}
	copy(active, src)
}

func countBeam(front []BeamCandidate, cutoff uint32) int {
	count := 0
	for _, v := range front {
		if beamComparisonScore(v.ScoreBits) >= beamComparisonScore(cutoff) {
			count++
		}
	}
	return count
}

func beamMerge(ctx context.Context, frames []beamFrame, frame int, numbers BeamNumbers) error {
	// Validate the complete pair graph before the first mutable score operation.
	for chunk := range frames {
		f := &frames[chunk]
		b := len(f.active)
		owners := make([]int, 4*b)
		for i := range owners {
			owners[i] = -1
		}
		for lane := 0; lane < b; lane++ {
			stay := f.front[4*b+lane]
			for j := 0; j < b; j++ {
				k := 4*j + int(stay.State&3)
				if stay.Key != f.front[k].Key {
					continue
				}
				if owners[k] >= 0 {
					return &BeamCollisionScheduleError{chunk, frame, k, owners[k], lane}
				}
				owners[k] = lane
			}
		}
	}
	// Ascending j order is preserved independently in every stay lane. Different
	// lanes are batched only after proving their mutable emission sets disjoint.
	type binding struct{ chunk, lane, emission int }
	for j := 0; j < beamWidth; j++ {
		var pairs []F32Pair
		var bindings []binding
		for chunk := range frames {
			f := &frames[chunk]
			b := len(f.active)
			if j >= b {
				continue
			}
			for lane := 0; lane < b; lane++ {
				stay := f.front[4*b+lane]
				k := 4*j + int(stay.State&3)
				if stay.Key == f.front[k].Key {
					pairs = append(pairs, F32Pair{stay.ScoreBits, f.front[k].ScoreBits})
					bindings = append(bindings, binding{chunk, lane, k})
				}
			}
		}
		combined, err := beamBitsCall(ctx, "log_add", len(pairs), func() ([]uint32, error) { return numbers.LogAddPairs(ctx, pairs) })
		if err != nil {
			return err
		}
		maxPairs := make([]F32Pair, len(pairs))
		for k, v := range bindings {
			f := &frames[v.chunk]
			stay := 4*len(f.active) + v.lane
			winner, loser := stay, v.emission
			if beamComparisonScore(pairs[k].First) <= beamComparisonScore(pairs[k].Second) {
				winner, loser = loser, winner
			}
			f.front[winner].ScoreBits = combined[k]
			f.front[loser].ScoreBits = invalidBeamScore
			if f.diagnostic != nil {
				f.diagnostic.Pairs = append(f.diagnostic.Pairs, BeamMergeTrace{v.lane, v.emission, winner, pairs[k], combined[k]})
			}
			maxPairs[k] = F32Pair{f.laneMax[v.lane], combined[k]}
		}
		maxima, err := beamBitsCall(ctx, "merge_max", len(maxPairs), func() ([]uint32, error) { return numbers.MaxPairs(ctx, maxPairs) })
		if err != nil {
			return err
		}
		for k, v := range bindings {
			frames[v.chunk].laneMax[v.lane] = maxima[k]
		}
	}
	return nil
}

func beamCutoffs(ctx context.Context, frames []beamFrame, logCut uint32, numbers BeamNumbers) error {
	for i := range frames {
		if frames[i].diagnostic != nil {
			frames[i].diagnostic.MergeLaneMax = frames[i].laneMax
		}
	}
	for _, distance := range []int{16, 8, 4, 2, 1} {
		pairs := make([]F32Pair, len(frames)*beamWidth)
		for chunk := range frames {
			for lane := 0; lane < beamWidth; lane++ {
				pairs[chunk*beamWidth+lane] = F32Pair{frames[chunk].laneMax[lane], frames[chunk].laneMax[lane^distance]}
			}
		}
		values, err := beamBitsCall(ctx, "butterfly_max", len(pairs), func() ([]uint32, error) { return numbers.MaxPairs(ctx, pairs) })
		if err != nil {
			return err
		}
		for chunk := range frames {
			copy(frames[chunk].laneMax[:], values[chunk*beamWidth:(chunk+1)*beamWidth])
		}
	}
	pairs := make([]F32Pair, len(frames))
	for i := range frames {
		pairs[i] = F32Pair{frames[i].laneMax[0], logCut}
	}
	values, err := beamBitsCall(ctx, "initial_cutoff", len(pairs), func() ([]uint32, error) { return numbers.SubPairs(ctx, pairs) })
	if err != nil {
		return err
	}
	type search struct {
		lower, upper uint32
		tooMany, run bool
	}
	searches := make([]search, len(frames))
	for i := range frames {
		frames[i].cutoff = values[i]
		count := countBeam(frames[i].front, values[i])
		searches[i] = search{values[i], frames[i].laneMax[0], true, count > beamWidth}
		if frames[i].diagnostic != nil {
			frames[i].diagnostic.Cutoffs = append(frames[i].diagnostic.Cutoffs, BeamCutoffTrace{0, count, values[i], false})
		}
	}
	for step := 1; step <= 9; step++ {
		var pairs []F32Pair
		var chunks []int
		for i := range frames {
			s := &searches[i]
			if !s.run {
				continue
			}
			partner := s.lower
			if s.tooMany {
				s.lower = frames[i].cutoff
				partner = s.upper
			} else {
				s.upper = frames[i].cutoff
			}
			pairs = append(pairs, F32Pair{frames[i].cutoff, partner})
			chunks = append(chunks, i)
		}
		values, err := beamBitsCall(ctx, "midpoint", len(pairs), func() ([]uint32, error) { return numbers.Midpoints(ctx, pairs) })
		if err != nil {
			return err
		}
		for k, i := range chunks {
			s := &searches[i]
			frames[i].cutoff = values[k]
			count := countBeam(frames[i].front, values[k])
			if frames[i].diagnostic != nil {
				frames[i].diagnostic.Cutoffs = append(frames[i].diagnostic.Cutoffs, BeamCutoffTrace{step, count, values[k], false})
			}
			s.tooMany = count > beamWidth
			s.run = (s.tooMany || count < 25) && step < 9
			if step == 9 {
				frames[i].cutoff = s.upper
				if frames[i].diagnostic != nil {
					frames[i].diagnostic.Cutoffs = append(frames[i].diagnostic.Cutoffs, BeamCutoffTrace{step, countBeam(frames[i].front, s.upper), s.upper, true})
				}
			}
		}
		if len(chunks) == 0 {
			break
		}
	}
	return nil
}

// BeamSearch owns all beam policy and publishes only complete batches.
// Scores are F16 [N,T,4*S]; immutable guides are F32 [T+1,N,S].
func BeamSearch(ctx context.Context, p BeamPlan, in BeamInput, numbers BeamNumbers) (BeamBatch, error) {
	return beamSearch(ctx, p, in, numbers, nil)
}

// BeamSearchWithTrace retains optional source-level differential diagnostics.
// On failure it publishes neither partial history nor partial frame evidence.
func BeamSearchWithTrace(ctx context.Context, p BeamPlan, in BeamInput, numbers BeamNumbers) (BeamBatch, []BeamFrameTrace, error) {
	var trace []BeamFrameTrace
	batch, err := beamSearch(ctx, p, in, numbers, func(frame int, frames []beamFrame) {
		row := BeamFrameTrace{Frame: frame, Chunks: make([]BeamChunkTrace, len(frames))}
		for i := range frames {
			row.Chunks[i] = *frames[i].diagnostic
		}
		trace = append(trace, row)
	})
	if err != nil {
		return BeamBatch{}, nil, err
	}
	return batch, trace, nil
}

// observe is an internal, synchronous diagnostic hook. Its records are copies.
func beamSearch(ctx context.Context, p BeamPlan, in BeamInput, numbers BeamNumbers, observe func(int, []beamFrame)) (BeamBatch, error) {
	var empty BeamBatch
	g := p.checked.guide
	want, err := PlanBeam(g, math.Float32frombits(p.checked.cut))
	if err != nil || want != p || ctx == nil || numbers == nil {
		return empty, fmt.Errorf("initialized beam plan, context and numeric provider required")
	}
	if err := ctx.Err(); err != nil {
		return empty, err
	}
	if int64(len(in.ScoresNTC)) != g.n*g.t*4*g.states || int64(len(in.GuidesTNS)) != (g.t+1)*g.n*g.states {
		return empty, fmt.Errorf("beam input extents differ from checked geometry")
	}
	for _, bits := range in.ScoresNTC {
		if bits&0x7c00 == 0x7c00 {
			return empty, fmt.Errorf("beam scores must be finite half values")
		}
	}
	for _, bits := range in.GuidesTNS {
		if !finiteBeamScore(bits) {
			return empty, fmt.Errorf("beam guides must be finite float values")
		}
	}
	logCuts, err := beamBitsCall(ctx, "log_cut", 1, func() ([]uint32, error) { return numbers.LogCuts(ctx, []uint32{p.checked.cut}) })
	if err != nil {
		return empty, err
	}
	batch := BeamBatch{History: make([]ChunkBeamHistory, g.n), FinalScoreBits: make([]uint32, g.n)}
	frames := make([]beamFrame, g.n)
	for chunk := range frames {
		frames[chunk].active = seedBeam(in.GuidesTNS[int64(chunk)*g.states : int64(chunk+1)*g.states])
		batch.History[chunk].Rows = make([][]BeamHistoryRecord, g.t+1)
		batch.History[chunk].Rows[0] = beamHistory(frames[chunk].active)
	}
	for t := int64(0); t < g.t; t++ {
		if err := ctx.Err(); err != nil {
			return empty, err
		}
		var rows []ExpandRow
		for chunk := range frames {
			f := &frames[chunk]
			if observe != nil {
				f.diagnostic = &BeamChunkTrace{}
			}
			f.front = make([]BeamCandidate, 5*len(f.active))
			for lane := range f.laneMax {
				f.laneMax[lane] = invalidBeamScore
			}
			guide := in.GuidesTNS[((t+1)*g.n+int64(chunk))*g.states:]
			scores := in.ScoresNTC[(int64(chunk)*g.t+t)*4*g.states:]
			for i, v := range f.active {
				u := (4 * int64(v.State)) & (g.states - 1)
				q := 4 * int64(v.State) / g.states
				row := ExpandRow{Previous: v.ScoreBits}
				for a := int64(0); a < 4; a++ {
					row.Transition[a] = scores[4*u+q+4*a]
					row.Guide[a] = guide[u+a]
					f.front[4*i+int(a)] = BeamCandidate{Key: beamCRC(v.Key, uint32(a), 2), State: uint16(u + a), Previous: uint8(i)}
				}
				row.Guide[4] = guide[v.State]
				f.front[4*len(f.active)+i] = BeamCandidate{Key: v.Key, State: v.State, Previous: uint8(i), Stay: true}
				rows = append(rows, row)
			}
		}
		if err := ctx.Err(); err != nil {
			return empty, err
		}
		expanded, err := numbers.Expand5(ctx, rows, math.Float32bits(g.clamp), math.Float32bits(g.blank))
		if err != nil {
			return empty, err
		}
		if err := ctx.Err(); err != nil {
			return empty, err
		}
		if len(expanded) != len(rows) {
			return empty, &BeamNumericError{"expand", "result extent differs"}
		}
		k := 0
		for chunk := range frames {
			f := &frames[chunk]
			for i := range f.active {
				v := expanded[k]
				k++
				for a, score := range v.Scores {
					if !finiteBeamScore(score) {
						return empty, &BeamNumericError{"expand", "nonfinite score"}
					}
					index := 4*i + a
					if a == 4 {
						index = 4*len(f.active) + i
					}
					f.front[index].ScoreBits = score
				}
				if !finiteBeamScore(v.OrderedMax) {
					return empty, &BeamNumericError{"expand", "nonfinite lane maximum"}
				}
				f.laneMax[i] = v.OrderedMax
			}
			if f.diagnostic != nil {
				f.diagnostic.Expanded = append([]BeamCandidate(nil), f.front...)
				f.diagnostic.ExpansionLaneMax = f.laneMax
			}
		}
		if err := beamMerge(ctx, frames, int(t), numbers); err != nil {
			return empty, err
		}
		if err := beamCutoffs(ctx, frames, logCuts[0], numbers); err != nil {
			return empty, err
		}
		var pairs []F32Pair
		for chunk := range frames {
			f := &frames[chunk]
			if f.diagnostic != nil {
				f.diagnostic.Merged = append([]BeamCandidate(nil), f.front...)
				for lane := 0; lane < beamWidth; lane++ {
					for i := lane; i < len(f.front); i += beamWidth {
						if beamComparisonScore(f.front[i].ScoreBits) >= beamComparisonScore(f.cutoff) && len(f.diagnostic.RetainedIndices) < beamWidth {
							f.diagnostic.RetainedIndices = append(f.diagnostic.RetainedIndices, i)
						}
					}
				}
			}
			f.active, err = CompactBeam(f.front, f.cutoff)
			if err != nil {
				return empty, err
			}
			if len(f.active) == 0 {
				return empty, &BeamNumericError{"compact", "beam extinction is not qualified"}
			}
			if t == g.t-1 {
				terminalBeamScratch(f.active, f.front)
			}
			guide := in.GuidesTNS[((t+1)*g.n+int64(chunk))*g.states:]
			for _, v := range f.active {
				pairs = append(pairs, F32Pair{v.ScoreBits, guide[v.State]})
			}
		}
		values, err := beamBitsCall(ctx, "subtract_guide", len(pairs), func() ([]uint32, error) { return numbers.SubPairs(ctx, pairs) })
		if err != nil {
			return empty, err
		}
		k = 0
		for chunk := range frames {
			f := &frames[chunk]
			for i := range f.active {
				f.active[i].ScoreBits = values[k]
				k++
			}
			batch.History[chunk].Rows[t+1] = beamHistory(f.active)
			if f.diagnostic != nil {
				f.diagnostic.Active = append([]BeamCandidate(nil), f.active...)
				f.diagnostic.History = append([]BeamHistoryRecord(nil), batch.History[chunk].Rows[t+1]...)
			}
		}
		if observe != nil {
			copyFrames := append([]beamFrame(nil), frames...)
			for i := range copyFrames {
				copyFrames[i].active = append([]BeamCandidate(nil), frames[i].active...)
				copyFrames[i].front = append([]BeamCandidate(nil), frames[i].front...)
			}
			observe(int(t), copyFrames)
		}
	}
	if err := ctx.Err(); err != nil {
		return empty, err
	}
	for chunk := range frames {
		batch.FinalScoreBits[chunk] = frames[chunk].active[0].ScoreBits
	}
	if err := ctx.Err(); err != nil {
		return empty, err
	}
	return batch, nil
}
