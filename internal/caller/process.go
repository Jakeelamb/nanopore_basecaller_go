package caller

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/pod5"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/postcall"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/subread"
)

// Processed retains all postcall children, including filtered reads. Reads holds
// final writer records in deterministic native child order.
type Processed struct {
	Postcall postcall.Result
	Reads    []CalledRead
}

func (p Processed) FASTQBytes() ([]byte, error) {
	var b bytes.Buffer
	for _, r := range p.Reads {
		if err := r.WriteFASTQ(&b); err != nil {
			return nil, err
		}
	}
	return b.Bytes(), nil
}

// Process runs inference once, then consumes the same normalized stage buffer.
// Call remains the baseline producer API.
func (c *Caller) Process(ctx context.Context, r pod5.Read, policy postcall.Policy) (Processed, error) {
	if err := policy.Validate(r.SequencingKit); err != nil {
		return Processed{}, err
	}
	s, err := c.callStage(ctx, r)
	if err != nil {
		return Processed{}, err
	}
	out, err := s.process(policy)
	if err != nil {
		return Processed{}, err
	}
	if err := ctx.Err(); err != nil {
		return Processed{}, err
	}
	if c.capture != "" {
		dir := filepath.Join(c.capture, r.ID.String())
		if _, err := captureProcessed(dir, filepath.Join(dir, "manifest.json"), out); err != nil {
			return Processed{}, err
		}
	}
	return out, nil
}

func (s coreStage) process(policy postcall.Policy) (Processed, error) {
	r, called := s.Input, s.CalledRead
	if called.FASTQ == nil || called.Trimmed < 0 || r.AcquisitionStartTimeMS < 0 || called.TailTrimmedSamples > uint64(len(s.Normalized.Samples)) {
		return Processed{}, errors.New("invalid core metadata for postprocessing")
	}
	signal := s.Normalized.Samples[:uint64(len(s.Normalized.Samples))-called.TailTrimmedSamples]
	events, pretrim, samples := r.NumMinknowEvents, uint64(len(called.Sequence)), r.NumSamples
	if called.TailTrimmedSamples > 0 {
		events, samples = 0, uint64(len(signal))
	}
	start, err := timestamp(called.FASTQ.StartTimeMS)
	if err != nil {
		return Processed{}, err
	}
	if r.StartSample > math.MaxUint64/1000 {
		return Processed{}, errors.New("sample clock overflow")
	}
	read := subread.Read{
		ID: called.ID, Signal: signal, Sequence: called.Sequence, Qstring: called.Qstring, Moves: called.Moves, Stride: s.Stride,
		StartSample: r.StartSample, EndSample: r.EndSample, AcquisitionStartMS: uint64(r.AcquisitionStartTimeMS), SampleRate: r.SampleRate,
		StartTimeMS: r.StartSample * 1000 / r.SampleRate, StartTime: start, LeadingTrim: uint64(called.Trimmed), ReadNumber: int64(r.ReadNumber),
		NumSamples: samples, NumEvents: events, PreTrimSeqLen: pretrim, MeanQStart: uint32(called.FASTQ.MeanQStart), EndReason: r.EndReasonName, IsMuxChange: r.EndReason == 1 || r.EndReason == 2,
		Inherited: subread.Inherited{OffsetBits: math.Float32bits(r.CalibrationOffset), OpenPoreLevelBits: math.Float32bits(r.OpenPoreLevel), ShiftBits: math.Float32bits(s.Normalized.Shift), ScaleBits: math.Float32bits(s.Normalized.Scale), ScalingBits: math.Float32bits(r.CalibrationScale), ScalingMethod: "pa", RunID: r.RunID, FlowCellProductCode: r.FlowCellProductCode, SequencingKit: r.SequencingKit, FlowcellID: r.FlowCellID, PositionID: r.PositionID, AcquisitionID: r.AcquisitionID, ExperimentID: r.ExperimentID, ModelName: called.FASTQ.ModelName, SampleID: r.SampleID, ProtocolStartMS: r.ProtocolStartTimeMS, Mux: uint32(r.Well), Channel: int32(r.Channel), Filename: r.Filename, PoreType: r.PoreType},
	}
	result, err := postcall.Process(read, policy)
	if err != nil {
		return Processed{}, err
	}
	out := Processed{Postcall: result, Reads: []CalledRead{}}
	for _, child := range result.Children {
		if !child.Filter.Keep {
			continue
		}
		r := child.Read
		clock, err := time.Parse("2006-01-02T15:04:05.000000+00:00", r.StartTime)
		if err != nil {
			return Processed{}, fmt.Errorf("child clock: %w", err)
		}
		metadata := FASTQMetadata{
			Channel: uint16(r.Inherited.Channel), StartTimeMS: clock.UnixMilli(),
			ProtocolStartTimeMS: r.Inherited.ProtocolStartMS, RunID: r.Inherited.RunID,
			ModelName: r.Inherited.ModelName, FlowCellID: r.Inherited.FlowcellID,
			MeanQStart: int(r.MeanQStart),
		}
		if policy.Trim {
			metadata.TrimFlags = TrimAdapter | TrimPrimer
		}
		if r.LeadingTrim > uint64(math.MaxInt) || r.LeadingTrim > math.MaxUint64-uint64(len(r.Signal)) {
			return Processed{}, errors.New("child trim count overflow")
		}
		out.Reads = append(out.Reads, CalledRead{ID: r.ID, Sequence: r.Sequence, Qstring: r.Qstring, Moves: r.Moves, NumSamples: r.NumSamples, Trimmed: int(r.LeadingTrim), FASTQ: &metadata})
	}
	return out, nil
}
