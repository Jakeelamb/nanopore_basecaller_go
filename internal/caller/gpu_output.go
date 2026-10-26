package caller

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"math"
	"runtime/debug"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/bam"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/pod5"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/postcall"
)

type gpuRecordOutput interface {
	write(Processed) error
	finish() error
}

type gpuDestination struct {
	writer io.Writer
	err    error
}

func (w *gpuDestination) Write(data []byte) (int, error) {
	if w.err != nil {
		return 0, w.err
	}
	n, err := w.writer.Write(data)
	if n < 0 || n > len(data) {
		n, err = 0, errors.New("GPU destination returned invalid write count")
	}
	if err == nil && n != len(data) {
		err = io.ErrShortWrite
	}
	w.err = err
	return n, err
}

type gpuOutput struct {
	destination *gpuDestination
	fastq       *bufio.Writer
	bam         *bam.Writer
	readGroup   string
	emitMoves   bool
	err         error
	closed      bool
}

func gpuProgram(commandLine string) bam.Program {
	version := "devel"
	if info, ok := debug.ReadBuildInfo(); ok {
		if info.Main.Version != "" && info.Main.Version != "(devel)" {
			version = info.Main.Version
		}
		for _, setting := range info.Settings {
			if setting.Key == "vcs.revision" {
				version += "+" + setting.Value
			}
			if setting.Key == "vcs.modified" && setting.Value == "true" {
				version += ".modified"
			}
		}
	}
	return bam.Program{ID: "nanopore_basecaller_go", Name: "nanopore_basecaller_go", Version: version, CommandLine: commandLine}
}

func newGPUOutput(dst io.Writer, cfg GPUConfig, spec model.Spec, metadata pod5.FileMetadata) (*gpuOutput, error) {
	w := &gpuOutput{destination: &gpuDestination{writer: dst}, emitMoves: cfg.EmitMoves}
	if cfg.Format == FASTQ {
		w.fastq = bufio.NewWriter(w.destination)
		return w, nil
	}
	if len(metadata.Runs) != 1 {
		return nil, fmt.Errorf("%w: BAM requires one file run until multi-read-group headers are ported", ErrGPUUnqualified)
	}
	run := metadata.Runs[0]
	if run.RunID == "" {
		return nil, errors.New("GPU BAM input has no protocol run ID")
	}
	flags := TrimFlags(0)
	if cfg.Policy.Trim {
		flags = TrimAdapter | TrimPrimer
	}
	trimFlags, err := flags.text()
	if err != nil {
		return nil, err
	}
	start, err := timestamp(run.AcquisitionStartTimeMS)
	if err != nil {
		return nil, err
	}
	experimentStart, err := timestamp(run.ProtocolStartTimeMS)
	if err != nil {
		return nil, err
	}
	w.readGroup = bam.ReadGroupID(run.RunID, spec.Name, "")
	experiment := run.ExperimentID
	if experiment == "" {
		experiment = run.TrackingID["protocol_group_id"]
	}
	header := bam.Header{Version: "1.6", SortOrder: "unknown", Program: gpuProgram(cfg.CommandLine), ReadGroup: &bam.ReadGroup{
		ID: w.readGroup, FlowCellID: run.FlowCellID, DeviceID: run.SystemName,
		ExperimentStartTime: experimentStart, RunID: run.RunID, BasecallModel: spec.Name,
		ExperimentID: experiment, AcquisitionStartTime: start, SampleID: run.SampleID,
		ModelStride: int(spec.Stride), TrimFlags: trimFlags,
	}}
	w.bam = bam.NewWriter(w.destination)
	if err = w.bam.WriteHeader(header); err != nil {
		w.err = err
		return w, err
	}
	return w, nil
}

func (w *gpuOutput) write(p Processed) error {
	if w.err != nil {
		return w.err
	}
	if w.closed {
		return errors.New("GPU output is closed")
	}
	if w.fastq != nil {
		for _, read := range p.Reads {
			if err := read.WriteFASTQ(w.fastq); err != nil {
				w.err = err
				return err
			}
		}
		return nil
	}
	for _, child := range p.Postcall.Children {
		if !child.Filter.Keep {
			continue
		}
		record, err := gpuBAMRecord(child, w.readGroup, w.emitMoves)
		if err == nil {
			err = w.bam.WriteRecord(record)
		}
		if err != nil {
			w.err = err
			return err
		}
	}
	return nil
}

func gpuBAMRecord(child postcall.Record, readGroup string, emitMoves bool) (bam.Record, error) {
	r := child.Read
	if r.NumSamples > math.MaxInt32 || r.LeadingTrim > math.MaxInt32 || r.ReadNumber < math.MinInt32 || r.ReadNumber > math.MaxInt32 || r.NumEvents > math.MaxUint32 || r.Stride > math.MaxUint8 || r.Stride == 0 || r.SampleRate == 0 || r.Inherited.Mux > math.MaxInt32 {
		return bam.Record{}, errors.New("GPU BAM metadata exceeds source integer domain")
	}
	if r.ParentID != "" && r.SplitPoint > math.MaxInt32 {
		return bam.Record{}, errors.New("GPU BAM split point exceeds source integer domain")
	}
	if bam.ReadGroupID(r.Inherited.RunID, r.Inherited.ModelName, "") != readGroup {
		return bam.Record{}, errors.New("GPU BAM child read group differs from header")
	}
	if _, err := time.Parse("2006-01-02T15:04:05.000000+00:00", r.StartTime); err != nil {
		return bam.Record{}, err
	}
	quality := make([]byte, len(r.Qstring))
	for i := range quality {
		if r.Qstring[i] < 33 || r.Qstring[i] > 126 {
			return bam.Record{}, errors.New("GPU BAM quality is outside printable FASTQ domain")
		}
		quality[i] = r.Qstring[i] - 33
	}
	var moves []uint8
	if emitMoves {
		moves = r.Moves
	}
	tags, err := bam.Tags(bam.Metadata{
		MeanQ: child.Filter.MeanQ, Duration: float32(r.NumSamples) / float32(r.SampleRate),
		NumSamples: int32(r.NumSamples), Trimmed: int32(r.LeadingTrim),
		Mux: int32(r.Inherited.Mux), Channel: r.Inherited.Channel, StartTime: r.StartTime,
		ReadNumber: int32(r.ReadNumber), Filename: r.Inherited.Filename,
		Shift: math.Float32frombits(r.Inherited.ShiftBits), Scale: math.Float32frombits(r.Inherited.ScaleBits),
		ScalingMethod: r.Inherited.ScalingMethod, ReadGroup: readGroup, ParentID: r.ParentID,
		SplitPoint: int32(r.SplitPoint), PoreType: r.Inherited.PoreType, EndReason: r.EndReason,
		MinknowEvents: uint32(r.NumEvents), Moves: moves, ModelStride: uint8(r.Stride), SampleRate: r.SampleRate,
	})
	return bam.Record{Name: r.ID, Flag: 4, Sequence: r.Sequence, Quality: quality, Aux: tags}, err
}

func (w *gpuOutput) finish() error {
	if w.closed {
		return w.err
	}
	w.closed = true
	var err error
	if w.fastq != nil {
		err = w.fastq.Flush()
	}
	if w.bam != nil {
		err = w.bam.Close()
	}
	w.err = errors.Join(w.err, err, w.destination.err)
	return w.err
}
