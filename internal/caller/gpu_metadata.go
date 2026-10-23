package caller

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/pod5"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/postcall"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/stitch"
)

// GPURead is one decoded GPU read with the host metadata the accepted postcall
// pipeline needs. Assembled must come from the pinned GPU decoder; this type
// carries no CPU inference result.
type GPURead struct {
	Read       pod5.Read
	Normalized signal.Normalized
	Stride     uint64
	Assembled  stitch.Result
}

// gpuStage rebuilds the core seam the CPU caller reaches after inference. It
// keeps MUX trimming, clock derivation and FASTQ metadata in one place so the
// GPU path cannot fork postcall behaviour.
func gpuStage(spec model.Spec, g GPURead) (coreStage, error) {
	if err := validateAdmission(g.Read); err != nil {
		return coreStage{}, err
	}
	if _, err := pod5.ParseID(g.Read.ID.String()); err != nil {
		return coreStage{}, err
	}
	if g.Read.NumSamples == 0 {
		return coreStage{}, errors.New("GPU read has no samples")
	}
	if g.Read.AcquisitionStartTimeMS < 0 {
		return coreStage{}, errors.New("GPU read has a negative acquisition clock")
	}
	if g.Stride == 0 || g.Stride > math.MaxInt32 {
		return coreStage{}, fmt.Errorf("GPU read stride %d is outside the native int domain", g.Stride)
	}
	if spec.Stride <= 0 || uint64(spec.Stride) != g.Stride {
		return coreStage{}, fmt.Errorf("GPU read stride %d differs from model stride %d", g.Stride, spec.Stride)
	}
	if len(g.Normalized.Samples) == 0 {
		return coreStage{}, errors.New("GPU read has no normalized signal")
	}
	if len(g.Assembled.Sequence) != len(g.Assembled.Qstring) {
		return coreStage{}, errors.New("GPU read sequence and quality lengths differ")
	}
	bases := 0
	for _, move := range g.Assembled.Moves {
		if move > 1 {
			return coreStage{}, errors.New("GPU read has a non-binary move")
		}
		bases += int(move)
	}
	if bases != len(g.Assembled.Sequence) {
		return coreStage{}, errors.New("GPU read move count differs from sequence length")
	}
	metadata, err := fastqMetadata(g.Read, spec)
	if err != nil {
		return coreStage{}, err
	}
	called := CalledRead{
		ID: g.Read.ID.String(), Sequence: g.Assembled.Sequence, Qstring: g.Assembled.Qstring,
		Moves: g.Assembled.Moves, NumSamples: g.Read.NumSamples, Trimmed: g.Normalized.Trimmed, FASTQ: metadata,
	}
	called, err = trimMux(g.Read, g.Normalized.Samples, g.Stride, called)
	if err != nil {
		return coreStage{}, err
	}
	return coreStage{CalledRead: called, Input: g.Read, Normalized: g.Normalized, Stride: g.Stride}, nil
}

// ProcessGPURead applies the accepted default postcall pipeline to a decoded GPU
// read without loading a model or running inference. It is the producer-free
// seam used by capture replay.
func ProcessGPURead(spec model.Spec, g GPURead, policy postcall.Policy) (Processed, error) {
	if err := policy.Validate(g.Read.SequencingKit); err != nil {
		return Processed{}, err
	}
	stage, err := gpuStage(spec, g)
	if err != nil {
		return Processed{}, err
	}
	return stage.process(policy)
}

// ProcessGPU runs postcall for a decoded GPU read while the caller holds the
// pinned model identity. Inference is not repeated.
func (c *Caller) ProcessGPU(ctx context.Context, g GPURead, policy postcall.Policy) (Processed, error) {
	if c == nil {
		return Processed{}, ErrClosed
	}
	if ctx == nil {
		return Processed{}, errors.New("nil context")
	}
	if err := ctx.Err(); err != nil {
		return Processed{}, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.engine == nil || c.model == nil {
		return Processed{}, ErrClosed
	}
	out, err := ProcessGPURead(c.model.Spec(), g, policy)
	if err != nil {
		return Processed{}, err
	}
	if err := ctx.Err(); err != nil {
		return Processed{}, err
	}
	return out, nil
}
