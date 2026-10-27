package gpudecode

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

// GPUBeamNumbers dispatches six arithmetic operations to an authenticated program.
// It has no state IDs, keys, cut decisions, time loop or host libm fallback.
type GPUBeamNumbers struct {
	device  *tensor.Device
	stream  *tensor.Stream
	maxRows int
}

func NewGPUBeamNumbers(d *tensor.Device, s *tensor.Stream, programSHA256 [32]byte, maxRows int) (*GPUBeamNumbers, error) {
	if d == nil || s == nil || maxRows <= 0 || maxRows > 1<<20 || programSHA256 == [32]byte{} {
		return nil, fmt.Errorf("live numeric device/stream, authenticated program and bounded workspace required")
	}
	program, err := d.DecoderBeamProgram(s)
	if err != nil {
		return nil, err
	}
	if sha256.Sum256(program) != programSHA256 {
		return nil, fmt.Errorf("decoder beam linked program authentication differs")
	}
	return &GPUBeamNumbers{d, s, maxRows}, nil
}

type beamOperand struct {
	dtype tensor.DType
	width int
	bytes []byte
}

func (n *GPUBeamNumbers) checkRows(ctx context.Context, rows int) error {
	if ctx == nil || n == nil || n.device == nil || n.stream == nil || n.maxRows <= 0 || rows < 0 || rows > n.maxRows {
		return fmt.Errorf("checked beam numeric context and workspace required")
	}
	return ctx.Err()
}

func f32BeamOperand(width int, bits []uint32) beamOperand {
	data := make([]byte, 4*len(bits))
	for i, b := range bits {
		binary.LittleEndian.PutUint32(data[4*i:], b)
	}
	return beamOperand{tensor.DTypeF32, width, data}
}

func (n *GPUBeamNumbers) dispatch(ctx context.Context, rows, outputWidth int, inputs []beamOperand, call func([]*tensor.Storage, *tensor.Storage) error) (result []uint32, err error) {
	if rows == 0 {
		return nil, nil
	}
	var owned []*tensor.Storage
	defer func() {
		for i := len(owned) - 1; i >= 0; i-- {
			err = errors.Join(err, owned[i].Close())
		}
		if err != nil {
			result = nil
		}
	}()
	create := func(v beamOperand) (*tensor.Storage, error) {
		h, err := n.device.NewStorage(n.stream, v.dtype, []int64{int64(rows), int64(v.width)}, []tensor.KoiTag{tensor.TagN, tensor.TagC}, v.bytes)
		if err == nil {
			owned = append(owned, h)
		}
		return h, err
	}
	operands := make([]*tensor.Storage, len(inputs))
	for i, v := range inputs {
		if len(v.bytes) != rows*v.width*v.dtype.Size() {
			return nil, fmt.Errorf("beam numeric operand extent differs")
		}
		operands[i], err = create(v)
		if err != nil {
			return nil, err
		}
	}
	poison := make([]byte, rows*outputWidth*4)
	for i := 0; i < len(poison); i += 4 {
		binary.LittleEndian.PutUint32(poison[i:], 0x7fc00000)
	}
	out, err := create(beamOperand{tensor.DTypeF32, outputWidth, poison})
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err = call(operands, out); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	data, err := out.Download(n.stream)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(data) != rows*outputWidth*4 {
		return nil, &BeamNumericError{"download", "result byte extent differs"}
	}
	result = make([]uint32, rows*outputWidth)
	for i := range result {
		result[i] = binary.LittleEndian.Uint32(data[4*i:])
		if !finiteBeamScore(result[i]) {
			return nil, &BeamNumericError{"download", "nonfinite result"}
		}
	}
	return result, nil
}

func (n *GPUBeamNumbers) LogCuts(ctx context.Context, cuts []uint32) ([]uint32, error) {
	if err := n.checkRows(ctx, len(cuts)); err != nil {
		return nil, err
	}
	for _, b := range cuts {
		if !finiteBeamScore(b) || beamComparisonScore(b) <= 0 {
			return nil, fmt.Errorf("beam numeric log cuts must be finite positive FTZ values")
		}
	}
	return n.dispatch(ctx, len(cuts), 1, []beamOperand{f32BeamOperand(1, cuts)}, func(in []*tensor.Storage, out *tensor.Storage) error {
		return n.device.DecoderBeamLogCuts(n.stream, in[0], out)
	})
}

func (n *GPUBeamNumbers) Expand5(ctx context.Context, rows []ExpandRow, clamp, blank uint32) ([]ExpandResult, error) {
	if err := n.checkRows(ctx, len(rows)); err != nil {
		return nil, err
	}
	if !finiteBeamScore(clamp) || math.Float32frombits(clamp) < 0 || !finiteBeamScore(blank) {
		return nil, fmt.Errorf("finite nonnegative clamp and finite blank required")
	}
	previous, guides := make([]uint32, len(rows)), make([]uint32, len(rows)*5)
	transitions := make([]byte, len(rows)*8)
	for i, row := range rows {
		if !finiteBeamScore(row.Previous) {
			return nil, fmt.Errorf("finite previous beam score required")
		}
		previous[i] = row.Previous
		for a, b := range row.Transition {
			if b&0x7c00 == 0x7c00 {
				return nil, fmt.Errorf("finite half transition required")
			}
			binary.LittleEndian.PutUint16(transitions[i*8+a*2:], b)
		}
		for a, b := range row.Guide {
			if !finiteBeamScore(b) {
				return nil, fmt.Errorf("finite beam guide required")
			}
			guides[i*5+a] = b
		}
	}
	bits, err := n.dispatch(ctx, len(rows), 6, []beamOperand{f32BeamOperand(1, previous), {tensor.DTypeF16, 4, transitions}, f32BeamOperand(5, guides)}, func(in []*tensor.Storage, out *tensor.Storage) error {
		return n.device.DecoderBeamExpand5(n.stream, in[0], in[1], in[2], out, math.Float32frombits(clamp), math.Float32frombits(blank))
	})
	if err != nil {
		return nil, err
	}
	result := make([]ExpandResult, len(rows))
	for i := range result {
		copy(result[i].Scores[:], bits[6*i:6*i+5])
		result[i].OrderedMax = bits[6*i+5]
	}
	return result, nil
}

func (n *GPUBeamNumbers) pairs(ctx context.Context, pairs []F32Pair, call func(*tensor.Storage, *tensor.Storage) error) ([]uint32, error) {
	if err := n.checkRows(ctx, len(pairs)); err != nil {
		return nil, err
	}
	bits := make([]uint32, 2*len(pairs))
	for i, p := range pairs {
		if !finiteBeamScore(p.First) || !finiteBeamScore(p.Second) {
			return nil, fmt.Errorf("finite beam numeric pair required")
		}
		bits[2*i], bits[2*i+1] = p.First, p.Second
	}
	return n.dispatch(ctx, len(pairs), 1, []beamOperand{f32BeamOperand(2, bits)}, func(in []*tensor.Storage, out *tensor.Storage) error { return call(in[0], out) })
}

func (n *GPUBeamNumbers) LogAddPairs(ctx context.Context, pairs []F32Pair) ([]uint32, error) {
	return n.pairs(ctx, pairs, func(in, out *tensor.Storage) error { return n.device.DecoderBeamLogAddPairs(n.stream, in, out) })
}
func (n *GPUBeamNumbers) SubPairs(ctx context.Context, pairs []F32Pair) ([]uint32, error) {
	return n.pairs(ctx, pairs, func(in, out *tensor.Storage) error { return n.device.DecoderBeamSubPairs(n.stream, in, out) })
}
func (n *GPUBeamNumbers) Midpoints(ctx context.Context, pairs []F32Pair) ([]uint32, error) {
	return n.pairs(ctx, pairs, func(in, out *tensor.Storage) error { return n.device.DecoderBeamMidpoints(n.stream, in, out) })
}
func (n *GPUBeamNumbers) MaxPairs(ctx context.Context, pairs []F32Pair) ([]uint32, error) {
	return n.pairs(ctx, pairs, func(in, out *tensor.Storage) error { return n.device.DecoderBeamMaxPairs(n.stream, in, out) })
}

// DownloadBeamInput copies each immutable input once. It accepts only the
// checked Go score and guide layouts, without altering their device owners.
func DownloadBeamInput(ctx context.Context, p BeamPlan, s *tensor.Stream, scores, guides *tensor.Storage) (BeamInput, error) {
	var empty BeamInput
	g := p.checked.guide
	want, err := PlanBeam(g, math.Float32frombits(p.checked.cut))
	if err != nil || want != p || ctx == nil || s == nil || scores == nil || guides == nil {
		return empty, fmt.Errorf("checked beam plan, context, stream and inputs required")
	}
	if err := ctx.Err(); err != nil {
		return empty, err
	}
	for i, h := range []*tensor.Storage{scores, guides} {
		m, err := h.Meta()
		if err != nil {
			return empty, err
		}
		dtype := tensor.DTypeF16
		shape, strides := []int64{g.n, g.t, 4 * g.states}, []int64{g.t * 4 * g.states, 4 * g.states, 1}
		tags := []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC}
		if i == 1 {
			dtype = tensor.DTypeF32
			shape, strides = []int64{g.t + 1, g.n, g.states}, []int64{g.n * g.states, g.states, 1}
			tags = []tensor.KoiTag{tensor.TagT, tensor.TagN, tensor.TagC}
		}
		if m.DType != dtype || m.ScalePresent || len(m.Shape) != 3 || len(m.Strides) != 3 || len(m.Tags) != 3 {
			return empty, fmt.Errorf("beam device input type, rank or scale differs")
		}
		for j := 0; j < 3; j++ {
			if m.Shape[j] != shape[j] || m.Strides[j] != strides[j] || m.Tags[j] != tags[j] {
				return empty, fmt.Errorf("beam device input geometry or layout differs")
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return empty, err
	}
	scoreBytes, err := scores.Download(s)
	if err != nil {
		return empty, err
	}
	if err := ctx.Err(); err != nil {
		return empty, err
	}
	guideBytes, err := guides.Download(s)
	if err != nil {
		return empty, err
	}
	if err := ctx.Err(); err != nil {
		return empty, err
	}
	if int64(len(scoreBytes)) != g.n*g.t*4*g.states*2 || int64(len(guideBytes)) != (g.t+1)*g.n*g.states*4 {
		return empty, fmt.Errorf("downloaded beam input byte extent differs")
	}
	in := BeamInput{make([]uint16, len(scoreBytes)/2), make([]uint32, len(guideBytes)/4)}
	for i := range in.ScoresNTC {
		in.ScoresNTC[i] = binary.LittleEndian.Uint16(scoreBytes[2*i:])
	}
	for i := range in.GuidesTNS {
		in.GuidesTNS[i] = binary.LittleEndian.Uint32(guideBytes[4*i:])
	}
	if err := ctx.Err(); err != nil {
		return empty, err
	}
	return in, nil
}
