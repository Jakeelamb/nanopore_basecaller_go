package gpudecode

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
)

type DecodePlan struct {
	guide          GuidePlan
	qscale, qshift uint32
	movePad        bool
}

func PlanDecode(g GuidePlan, qscale, qshift float32, movePad bool) (DecodePlan, error) {
	want, err := PlanGuides(g.n, g.t, g.states, g.clamp, g.blank)
	if err != nil || want != g || !finiteBeamScore(math.Float32bits(qscale)) || !finiteBeamScore(math.Float32bits(qshift)) {
		return DecodePlan{}, fmt.Errorf("checked fixed guide geometry and finite quality scalars required")
	}
	return DecodePlan{g, math.Float32bits(qscale), math.Float32bits(qshift), movePad}, nil
}

type QualityValues struct {
	Neighbors [8]uint16
	Count     uint8
	Center    uint16
}

type QualityGroupNumber struct {
	ErrorBits uint32
	Character byte
}

type QualitySummaryRow struct {
	Errors          []uint32
	Frames, Emitted uint32
}

type QualitySummaryNumber struct {
	StayBits, MoveBits, MeanErrorBits uint32
}

type QualityNumbers interface {
	Probabilities(context.Context, []QualityValues) ([]uint32, error)
	Groups(context.Context, [][]uint32, uint32, uint32) ([]QualityGroupNumber, error)
	Summaries(context.Context, []QualitySummaryRow) ([]QualitySummaryNumber, error)
}

type QualityInput struct {
	WorkingScoresNTC []uint16
	Trace            []BeamTrace
	FinalScoreBits   []uint32
}

type QualityChunk struct {
	Path               []int32
	Moves, Sequence, Q []byte
	// These are the eight initialized source words, with float values kept as
	// bits. Padded mode retains its zero count and nonfinite mean-error word.
	ResultWords [8]uint32
}

type QualityBatch struct{ Chunks []QualityChunk }

func posteriorProbability(b uint16) bool {
	return b&0x7fff <= 0x3c00 && (b&0x8000 == 0 || b&0x7fff == 0)
}

func qualityContext(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("quality context required")
	}
	return ctx.Err()
}

// DecodeQuality selects every posterior state, group and output slot in Go.
// Trace contains the Go traceback before full states are replaced by base IDs.
func DecodeQuality(ctx context.Context, p DecodePlan, in QualityInput, numbers QualityNumbers) (QualityBatch, error) {
	var empty QualityBatch
	g := p.guide
	want, err := PlanDecode(g, math.Float32frombits(p.qscale), math.Float32frombits(p.qshift), p.movePad)
	if err != nil || want != p || numbers == nil {
		return empty, fmt.Errorf("initialized quality plan and numeric provider required")
	}
	if err := qualityContext(ctx); err != nil {
		return empty, err
	}
	if int64(len(in.WorkingScoresNTC)) != g.n*g.t*4*g.states || int64(len(in.Trace)) != g.n || int64(len(in.FinalScoreBits)) != g.n {
		return empty, fmt.Errorf("quality input extents differ from checked geometry")
	}
	batch := QualityBatch{Chunks: make([]QualityChunk, g.n)}
	schedules := make([]QualitySchedule, g.n)
	var values []QualityValues
	var normal []int
	for chunk := range batch.Chunks {
		score := in.FinalScoreBits[chunk]
		if !finiteBeamScore(score) {
			return empty, fmt.Errorf("finite terminal beam score required")
		}
		out := &batch.Chunks[chunk]
		out.ResultWords[0], out.ResultWords[1], out.ResultWords[3] = uint32(int64(chunk)*g.t), uint32(g.t), score
		out.Path, out.Moves = make([]int32, g.t), make([]byte, g.t)
		out.Sequence, out.Q = make([]byte, g.t), make([]byte, g.t)
		if beamComparisonScore(score) < 0 {
			out.Moves[0], out.Sequence[0], out.Q[0] = 1, 'A', '!'
			out.ResultWords[2], out.ResultWords[5], out.ResultWords[7] = 1, 0x3f800000, 0x3f800000
			continue
		}
		if int64(len(in.Trace[chunk].Path)) != g.t || int64(len(in.Trace[chunk].Moves)) != g.t {
			return empty, fmt.Errorf("complete fixed-length traceback required")
		}
		schedule, err := PlanQuality(in.Trace[chunk], int(g.states), p.movePad)
		if err != nil {
			return empty, err
		}
		schedules[chunk] = schedule
		normal = append(normal, chunk)
		for _, frame := range schedule.Frames {
			row := in.WorkingScoresNTC[(int64(chunk)*g.t*4+int64(frame.PosteriorRow))*g.states:]
			v := QualityValues{Count: frame.NeighborCount, Center: row[frame.State]}
			if !posteriorProbability(v.Center) {
				return empty, fmt.Errorf("selected posterior center exceeds half probability domain")
			}
			for i := uint8(0); i < v.Count; i++ {
				v.Neighbors[i] = row[frame.Neighbors[i]]
				if !posteriorProbability(v.Neighbors[i]) {
					return empty, fmt.Errorf("selected posterior neighbor exceeds half probability domain")
				}
			}
			values = append(values, v)
		}
		out.Path, out.Moves = schedule.Path, schedule.Moves
	}
	if err := ctx.Err(); err != nil {
		return empty, err
	}
	if len(normal) == 0 {
		return batch, nil
	}
	probabilities, err := numbers.Probabilities(ctx, values)
	if err != nil {
		return empty, err
	}
	if err := ctx.Err(); err != nil {
		return empty, err
	}
	if len(probabilities) != len(values) {
		return empty, fmt.Errorf("quality probability result extent differs")
	}
	for _, bits := range probabilities {
		if !finiteBeamScore(bits) || beamComparisonScore(bits) < 0 || beamComparisonScore(bits) > 1 {
			return empty, fmt.Errorf("quality probability result exceeds finite domain")
		}
	}
	var groupRows [][]uint32
	for index, chunk := range normal {
		row := probabilities[int64(index)*g.t : int64(index+1)*g.t]
		for _, group := range schedules[chunk].Groups {
			groupRows = append(groupRows, append([]uint32(nil), row[group.First:group.End]...))
		}
	}
	groups, err := numbers.Groups(ctx, groupRows, p.qscale, p.qshift)
	if err != nil {
		return empty, err
	}
	if err := ctx.Err(); err != nil {
		return empty, err
	}
	if len(groups) != len(groupRows) {
		return empty, fmt.Errorf("quality group result extent differs")
	}
	for _, v := range groups {
		if !finiteBeamScore(v.ErrorBits) || v.Character < 34 || v.Character > 83 {
			return empty, fmt.Errorf("quality group result exceeds source output domain")
		}
	}
	summaries := make([]QualitySummaryRow, len(normal))
	index := 0
	for i, chunk := range normal {
		schedule, out := schedules[chunk], &batch.Chunks[chunk]
		summaries[i].Frames = uint32(g.t)
		if !p.movePad {
			summaries[i].Emitted = uint32(len(schedule.Groups))
		}
		out.ResultWords[2] = summaries[i].Emitted
		for _, group := range schedule.Groups {
			numeric := groups[index]
			index++
			out.Sequence[group.Output], out.Q[group.Output] = group.Base, numeric.Character
			summaries[i].Errors = append(summaries[i].Errors, numeric.ErrorBits)
		}
	}
	result, err := numbers.Summaries(ctx, summaries)
	if err != nil {
		return empty, err
	}
	if err := ctx.Err(); err != nil {
		return empty, err
	}
	if len(result) != len(normal) {
		return empty, fmt.Errorf("quality summary result extent differs")
	}
	for i, chunk := range normal {
		v := result[i]
		if !finiteBeamScore(v.StayBits) || !finiteBeamScore(v.MoveBits) || (summaries[i].Emitted > 0) != finiteBeamScore(v.MeanErrorBits) {
			return empty, fmt.Errorf("quality summary finite/nonfinite result contract differs")
		}
		batch.Chunks[chunk].ResultWords[4], batch.Chunks[chunk].ResultWords[5], batch.Chunks[chunk].ResultWords[7] = v.StayBits, v.MoveBits, v.MeanErrorBits
	}
	if err := ctx.Err(); err != nil {
		return empty, err
	}
	return batch, nil
}

// QualityFromHistory reuses the Go traceback over rows 1..T. Negative-score
// chunks follow the source fallback without reading their opaque history.
func QualityFromHistory(ctx context.Context, p DecodePlan, working []uint16, history [][][]BeamHistoryRecord, scores []uint32, numbers QualityNumbers) (QualityBatch, error) {
	if err := qualityContext(ctx); err != nil {
		return QualityBatch{}, err
	}
	if int64(len(history)) != p.guide.n || len(scores) != len(history) {
		return QualityBatch{}, fmt.Errorf("fixed per-chunk history and score extents required")
	}
	traces := make([]BeamTrace, len(history))
	for chunk, rows := range history {
		if beamComparisonScore(scores[chunk]) < 0 {
			continue
		}
		if int64(len(rows)) != p.guide.t+1 {
			return QualityBatch{}, fmt.Errorf("history must include rows zero through T")
		}
		trace, err := TraceBeamHistory(rows[1:], int(p.guide.states))
		if err != nil {
			return QualityBatch{}, err
		}
		traces[chunk] = trace
	}
	return DecodeQuality(ctx, p, QualityInput{working, traces, scores}, numbers)
}

func PackQualityResult(words [8]uint32) []byte {
	out := make([]byte, 32)
	for i, b := range words {
		binary.LittleEndian.PutUint32(out[4*i:], b)
	}
	return out
}
