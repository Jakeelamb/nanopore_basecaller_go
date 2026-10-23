// Package caller composes the pinned Go-owned CPU HAC basecalling path.
package caller

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/chunk"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/crf"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/decode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/pod5"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/stitch"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

var ErrClosed = errors.New("caller is closed")
var ErrUnsupported = errors.New("caller supports only pinned DNA 5000Hz CPU HAC with no adapter trimming or read splitting")

type Config struct {
	TensorLibrary, ModelDir, CaptureDir string
	ChunkSize, Overlap                  uint64
}
type CalledRead struct {
	ID         string  `json:"id"`
	Sequence   string  `json:"sequence"`
	Qstring    string  `json:"qstring"`
	Moves      []uint8 `json:"moves"`
	NumSamples uint64  `json:"num_samples"`
	Trimmed    int     `json:"trimmed"`
}
type Caller struct {
	mu      sync.Mutex
	engine  *tensor.Engine
	model   *model.Model
	chunks  chunk.Config
	options decode.Options
	capture string
	runFile payload
}

func configuration(s model.Spec, c Config) (chunk.Config, decode.Options, error) {
	if s.Architecture != model.HAC || s.SampleRate != 5000 || s.Stride != 6 {
		return chunk.Config{}, decode.Options{}, ErrUnsupported
	}
	size, overlap := c.ChunkSize, c.Overlap
	if size == 0 && overlap != 0 {
		return chunk.Config{}, decode.Options{}, errors.New("ChunkSize zero selects all model defaults and requires Overlap zero")
	}
	if size == 0 {
		size = uint64(s.ChunkSize)
	}
	if c.ChunkSize == 0 && c.Overlap == 0 {
		overlap = uint64(s.Overlap)
	}
	cc := chunk.Config{ChunkSize: size, Overlap: overlap, Stride: uint64(s.Stride), Granularity: size}
	if size < uint64(s.Stride) || size > uint64(s.ChunkSize) {
		return cc, decode.Options{}, errors.New("caller chunk size must be between stride and the model default")
	}
	if _, err := chunk.Fixed(1, cc); err != nil {
		return cc, decode.Options{}, err
	}
	return cc, decode.Options{BeamWidth: 32, BeamCut: 100, BlankScore: s.BlankScore, QShift: s.QBias, QScale: s.QScale}, nil
}
func Open(ctx context.Context, c Config) (*Caller, error) {
	if ctx == nil {
		return nil, errors.New("nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	spec, err := model.Inspect(c.ModelDir)
	if err != nil {
		return nil, err
	}
	cc, options, err := configuration(spec, c)
	if err != nil {
		return nil, err
	}
	e, err := tensor.Open(c.TensorLibrary)
	if err != nil {
		return nil, err
	}
	m, err := model.Open(e, c.ModelDir)
	if err != nil {
		return nil, errors.Join(err, e.Close())
	}
	result := &Caller{engine: e, model: m, chunks: cc, options: options, capture: c.CaptureDir}
	if err := ctx.Err(); err != nil {
		return nil, errors.Join(err, result.Close())
	}
	if c.CaptureDir != "" {
		result.runFile, err = startCapture(c, spec, m.Weights(), cc, options)
		if err != nil {
			return nil, errors.Join(err, result.Close())
		}
	}
	return result, nil
}
func (c *Caller) Close() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.engine == nil {
		return nil
	}
	err := errors.Join(c.model.Close(), c.engine.Close())
	c.engine = nil
	c.model = nil
	return err
}

// adjustment matches ScalerNode's float32 ADC-domain pore-level correction.
func adjustment(flow string, level, scale float32) float32 {
	if math.IsNaN(float64(level)) {
		return 0
	}
	var expected float32
	switch strings.ToUpper(flow) {
	case "FLO-PRO114", "FLO-PRO114M", "FLO-PRO114HD":
		expected = 199.21
	case "FLO-MIN114", "FLO-MIN114HD":
		expected = 197.61
	case "FLO-FLG114", "FLO-FLG114HD":
		expected = 200
	default:
		return 0
	}
	return float32(float32(level-expected) / scale)
}
func validateRead(r pod5.Read) error {
	if r.NumSamples != uint64(len(r.Signal)) || len(r.Signal) == 0 {
		return errors.New("read signal length differs from num_samples or is empty")
	}
	if r.SampleRate != 5000 {
		return fmt.Errorf("sample rate %d: %w", r.SampleRate, ErrUnsupported)
	}
	flow, kit := strings.ToUpper(r.FlowCellProductCode), strings.ToUpper(r.SequencingKit)
	hd := false
	switch flow {
	case "FLO-PRO114", "FLO-PRO114M", "FLO-MIN114", "FLO-FLG114":
	case "FLO-PRO114HD", "FLO-MIN114HD", "FLO-FLG114HD":
		hd = true
	default:
		return fmt.Errorf("flowcell %q: %w", flow, ErrUnsupported)
	}
	if hd {
		if kit != "SQK-LSK114" && kit != "SQK-LSK114-XL" {
			return fmt.Errorf("HD kit %q: %w", kit, ErrUnsupported)
		}
	} else {
		switch kit {
		case "SQK-LSK114", "SQK-LSK114-XL", "SQK-ULK114", "SQK-RAD114", "SQK-PCS114", "SQK-HTB114-96", "SQK-NBD114-24", "SQK-NBD114-96", "SQK-RBK114-24", "SQK-RBK114-96", "SQK-RPB114-24", "SQK-MLK114-96-XL", "SQK-16S114-24", "SQK-MAB114-24", "SQK-PCB114-24":
		default:
			return fmt.Errorf("kit %q: %w", kit, ErrUnsupported)
		}
	}
	return nil
}
func (c *Caller) Call(ctx context.Context, r pod5.Read) (CalledRead, error) {
	if c == nil {
		return CalledRead{}, ErrClosed
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.engine == nil {
		return CalledRead{}, ErrClosed
	}
	if ctx == nil {
		return CalledRead{}, errors.New("nil context")
	}
	if err := ctx.Err(); err != nil {
		return CalledRead{}, err
	}
	if err := validateRead(r); err != nil {
		return CalledRead{}, err
	}
	spec := c.model.Spec()
	normalized, err := signal.Normalize(r.Signal, signal.Calibration{Scale: r.CalibrationScale, Offset: r.CalibrationOffset}, signal.Standardisation{Mean: spec.Mean, Stdev: spec.Stdev}, adjustment(r.FlowCellProductCode, r.OpenPoreLevel, r.CalibrationScale))
	if err != nil {
		return CalledRead{}, err
	}
	offsets, err := chunk.Fixed(uint64(len(normalized.Samples)), c.chunks)
	if err != nil {
		return CalledRead{}, err
	}
	var capture *captureRead
	if c.capture != "" {
		capture, err = beginRead(c.capture, c.runFile, r, normalized)
		if err != nil {
			return CalledRead{}, err
		}
	}
	decoded := make([]stitch.Chunk, 0, len(offsets))
	for i, offset := range offsets {
		if err := ctx.Err(); err != nil {
			return CalledRead{}, err
		}
		end := min(offset+c.chunks.ChunkSize, uint64(len(normalized.Samples)))
		padded, err := chunk.RepeatPad(normalized.Samples[offset:end], c.chunks.ChunkSize)
		if err != nil {
			return CalledRead{}, err
		}
		values := make([]float32, len(padded))
		for i, v := range padded {
			values[i] = signal.Float32FromHalf(v)
		}
		result, err := c.infer(ctx, values)
		if err != nil {
			return CalledRead{}, fmt.Errorf("chunk %d: %w", i, err)
		}
		if err := ctx.Err(); err != nil {
			return CalledRead{}, err
		}
		d, err := decode.Decode(result.DecodeInput(), c.options)
		if err != nil {
			return CalledRead{}, fmt.Errorf("chunk %d decode: %w", i, err)
		}
		dc := stitch.Chunk{Offset: offset, RawSize: end - offset, Granularity: c.chunks.ChunkSize, Sequence: d.Sequence, Qstring: d.Qstring, Moves: d.Moves}
		decoded = append(decoded, dc)
		if capture != nil {
			if err := capture.addChunk(i, dc, result); err != nil {
				return CalledRead{}, err
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return CalledRead{}, err
	}
	assembled, err := stitch.Stitch(stitch.Config{RawSamples: uint64(len(normalized.Samples)), Stride: c.chunks.Stride}, decoded)
	if err != nil {
		return CalledRead{}, err
	}
	out := CalledRead{ID: r.ID.String(), Sequence: assembled.Sequence, Qstring: assembled.Qstring, Moves: assembled.Moves, NumSamples: r.NumSamples, Trimmed: normalized.Trimmed}
	if capture != nil {
		if err := capture.finish(out); err != nil {
			return CalledRead{}, err
		}
	}
	return out, nil
}
func (c *Caller) infer(ctx context.Context, values []float32) (out crf.Result, err error) {
	var handles []*tensor.Tensor
	defer func() {
		for i := len(handles) - 1; i >= 0; i-- {
			err = errors.Join(err, handles[i].Close())
		}
	}()
	x, err := c.engine.NewFloat32([]int64{1, 1, int64(len(values))}, values)
	if err != nil {
		return out, err
	}
	handles = append(handles, x)
	scores, err := c.model.Forward(ctx, x)
	if err != nil {
		return out, err
	}
	handles = append(handles, scores)
	tnc, err := c.engine.Transpose(scores, 0, 1)
	if err != nil {
		return out, err
	}
	handles = append(handles, tnc)
	contiguous, err := c.engine.Contiguous(tnc)
	if err != nil {
		return out, err
	}
	handles = append(handles, contiguous)
	return crf.Compute(ctx, c.engine, contiguous, c.options.BlankScore)
}
