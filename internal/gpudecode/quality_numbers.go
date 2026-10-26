package gpudecode

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

type GPUQualityNumbers struct {
	device  *tensor.Device
	stream  *tensor.Stream
	maxRows int
}

func NewGPUQualityNumbers(d *tensor.Device, s *tensor.Stream, programSHA256 [32]byte, maxRows int) (*GPUQualityNumbers, error) {
	if d == nil || s == nil || programSHA256 == [32]byte{} || maxRows <= 0 || maxRows > 1<<20 {
		return nil, fmt.Errorf("live quality owners, authenticated program and checked workspace required")
	}
	program, err := d.DecoderPosteriorProgram(s)
	if err != nil {
		return nil, err
	}
	if sha256.Sum256(program) != programSHA256 {
		return nil, fmt.Errorf("posterior/quality linked program authentication differs")
	}
	return &GPUQualityNumbers{d, s, maxRows}, nil
}

func (n *GPUQualityNumbers) check(ctx context.Context, rows int) error {
	if n == nil || n.device == nil || n.stream == nil || n.maxRows <= 0 || rows < 0 || rows > n.maxRows {
		return fmt.Errorf("checked quality numeric provider and workspace required")
	}
	return qualityContext(ctx)
}

type qualityOperand struct {
	dtype tensor.DType
	width int
	data  []byte
}
type qualityNumericOutput struct {
	bits  []uint32
	extra []byte
}

func qualityWords(dtype tensor.DType, width int, bits []uint32) qualityOperand {
	b := make([]byte, 4*len(bits))
	for i, x := range bits {
		binary.LittleEndian.PutUint32(b[4*i:], x)
	}
	return qualityOperand{dtype, width, b}
}

func (n *GPUQualityNumbers) dispatch(ctx context.Context, rows, outWidth int, inputs [2]qualityOperand, aux bool, call func(*tensor.Storage, *tensor.Storage, *tensor.Storage, *tensor.Storage) error) (result qualityNumericOutput, err error) {
	if rows == 0 {
		return result, nil
	}
	var owned []*tensor.Storage
	defer func() {
		for i := len(owned) - 1; i >= 0; i-- {
			err = errors.Join(err, owned[i].Close())
		}
		if err != nil {
			result = qualityNumericOutput{}
		}
	}()
	create := func(v qualityOperand) (*tensor.Storage, error) {
		if v.width <= 0 || rows > (1<<27)/v.width || len(v.data) != rows*v.width*v.dtype.Size() {
			return nil, fmt.Errorf("numeric row extent exceeds checked workspace")
		}
		h, e := n.device.NewStorage(n.stream, v.dtype, []int64{int64(rows), int64(v.width)}, []tensor.KoiTag{tensor.TagN, tensor.TagC}, v.data)
		if e == nil {
			owned = append(owned, h)
		}
		return h, e
	}
	a, err := create(inputs[0])
	if err != nil {
		return result, err
	}
	b, err := create(inputs[1])
	if err != nil {
		return result, err
	}
	poison := make([]byte, rows*outWidth*4)
	for i := 0; i < len(poison); i += 4 {
		binary.LittleEndian.PutUint32(poison[i:], 0x7fc00000)
	}
	out, err := create(qualityOperand{tensor.DTypeF32, outWidth, poison})
	if err != nil {
		return result, err
	}
	var extra *tensor.Storage
	if aux {
		extra, err = create(qualityOperand{tensor.DTypeU8, 1, make([]byte, rows)})
		if err != nil {
			return result, err
		}
	}
	if err = ctx.Err(); err != nil {
		return result, err
	}
	if err = call(a, b, out, extra); err != nil {
		return result, err
	}
	if err = ctx.Err(); err != nil {
		return result, err
	}
	data, err := out.Download(n.stream)
	if err != nil {
		return result, err
	}
	if len(data) != rows*outWidth*4 {
		return result, fmt.Errorf("numeric result extent differs")
	}
	result.bits = make([]uint32, rows*outWidth)
	for i := range result.bits {
		result.bits[i] = binary.LittleEndian.Uint32(data[4*i:])
	}
	if aux {
		result.extra, err = extra.Download(n.stream)
		if err != nil {
			return result, err
		}
		if len(result.extra) != rows {
			return result, fmt.Errorf("numeric auxiliary result extent differs")
		}
	}
	if err = ctx.Err(); err != nil {
		return result, err
	}
	return result, nil
}

func (n *GPUQualityNumbers) Probabilities(ctx context.Context, values []QualityValues) ([]uint32, error) {
	if err := n.check(ctx, len(values)); err != nil {
		return nil, err
	}
	data := make([]byte, 18*len(values))
	counts := make([]uint32, len(values))
	for i, v := range values {
		if v.Count > 8 || !posteriorProbability(v.Center) {
			return nil, fmt.Errorf("quality half values/count exceed source domain")
		}
		counts[i] = uint32(v.Count)
		for j := uint8(0); j < v.Count; j++ {
			if !posteriorProbability(v.Neighbors[j]) {
				return nil, fmt.Errorf("quality neighbor exceeds half probability domain")
			}
			binary.LittleEndian.PutUint16(data[18*i+2*int(j):], v.Neighbors[j])
		}
		binary.LittleEndian.PutUint16(data[18*i+16:], v.Center)
	}
	out, err := n.dispatch(ctx, len(values), 1, [2]qualityOperand{{tensor.DTypeF16, 9, data}, qualityWords(tensor.DTypeI32, 1, counts)}, false, func(a, b, out, _ *tensor.Storage) error {
		return n.device.DecoderQualityProbabilities(n.stream, a, b, out)
	})
	if err != nil {
		return nil, err
	}
	for _, x := range out.bits {
		if !finiteBeamScore(x) || beamComparisonScore(x) < 0 || beamComparisonScore(x) > 1 {
			return nil, fmt.Errorf("quality probability output exceeds finite domain")
		}
	}
	return out.bits, nil
}

// Numeric rows are already selected by Go policy. Bucketing equal lengths only
// avoids rectangular padding and does not choose or reorder values in a row.
func qualityBuckets(lengths []int) ([]int, map[int][]int, error) {
	buckets := map[int][]int{}
	total := 0
	for i, width := range lengths {
		if width <= 0 || width > 1<<20 || total > (1<<27)-width {
			return nil, nil, fmt.Errorf("positive numeric rows exceed workspace")
		}
		total += width
		buckets[width] = append(buckets[width], i)
	}
	keys := make([]int, 0, len(buckets))
	for width := range buckets {
		keys = append(keys, width)
	}
	sort.Ints(keys)
	return keys, buckets, nil
}

func (n *GPUQualityNumbers) Groups(ctx context.Context, rows [][]uint32, scale, shift uint32) ([]QualityGroupNumber, error) {
	if err := n.check(ctx, len(rows)); err != nil {
		return nil, err
	}
	if !finiteBeamScore(scale) || !finiteBeamScore(shift) {
		return nil, fmt.Errorf("finite quality scale and shift required")
	}
	lengths := make([]int, len(rows))
	for i, row := range rows {
		lengths[i] = len(row)
		for _, x := range row {
			if !finiteBeamScore(x) || beamComparisonScore(x) < 0 || beamComparisonScore(x) > 1 {
				return nil, fmt.Errorf("group probability exceeds finite domain")
			}
		}
	}
	keys, buckets, err := qualityBuckets(lengths)
	if err != nil {
		return nil, err
	}
	result := make([]QualityGroupNumber, len(rows))
	for _, width := range keys {
		indices := buckets[width]
		data := make([]uint32, 0, len(indices)*width)
		counts := make([]uint32, len(indices))
		for i, index := range indices {
			data = append(data, rows[index]...)
			counts[i] = uint32(width)
		}
		out, err := n.dispatch(ctx, len(indices), 1, [2]qualityOperand{qualityWords(tensor.DTypeF32, width, data), qualityWords(tensor.DTypeI32, 1, counts)}, true, func(a, b, out, q *tensor.Storage) error {
			return n.device.DecoderQualityGroups(n.stream, a, b, out, q, math.Float32frombits(scale), math.Float32frombits(shift))
		})
		if err != nil {
			return nil, err
		}
		for i, index := range indices {
			if !finiteBeamScore(out.bits[i]) || out.extra[i] < 34 || out.extra[i] > 83 {
				return nil, fmt.Errorf("quality group output contains poison or exceeds source domain")
			}
			result[index] = QualityGroupNumber{out.bits[i], out.extra[i]}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

func (n *GPUQualityNumbers) Summaries(ctx context.Context, rows []QualitySummaryRow) ([]QualitySummaryNumber, error) {
	if err := n.check(ctx, len(rows)); err != nil {
		return nil, err
	}
	lengths := make([]int, len(rows))
	for i, row := range rows {
		lengths[i] = len(row.Errors)
		if row.Frames == 0 || row.Frames > math.MaxInt32 || uint64(len(row.Errors)) > uint64(row.Frames) || (row.Emitted != 0 && uint64(row.Emitted) != uint64(len(row.Errors))) {
			return nil, fmt.Errorf("summary counters differ from selected rows")
		}
		for _, x := range row.Errors {
			if !finiteBeamScore(x) {
				return nil, fmt.Errorf("finite selected group errors required")
			}
		}
	}
	keys, buckets, err := qualityBuckets(lengths)
	if err != nil {
		return nil, err
	}
	result := make([]QualitySummaryNumber, len(rows))
	for _, width := range keys {
		indices := buckets[width]
		data := make([]uint32, 0, len(indices)*width)
		counts := make([]uint32, len(indices)*2)
		for i, index := range indices {
			data = append(data, rows[index].Errors...)
			counts[2*i], counts[2*i+1] = rows[index].Frames, rows[index].Emitted
		}
		out, err := n.dispatch(ctx, len(indices), 3, [2]qualityOperand{qualityWords(tensor.DTypeF32, width, data), qualityWords(tensor.DTypeI32, 2, counts)}, true, func(a, b, out, written *tensor.Storage) error {
			return n.device.DecoderQualitySummaries(n.stream, a, b, out, written)
		})
		if err != nil {
			return nil, err
		}
		for i, index := range indices {
			x := out.bits[3*i : 3*i+3]
			if out.extra[i] != 1 || !finiteBeamScore(x[0]) || !finiteBeamScore(x[1]) || (rows[index].Emitted > 0) != finiteBeamScore(x[2]) {
				return nil, fmt.Errorf("quality summary output contains poison or has wrong finite/nonfinite class")
			}
			result[index] = QualitySummaryNumber{x[0], x[1], x[2]}
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}
