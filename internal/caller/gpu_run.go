package caller

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/chunk"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpubatch"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/pod5"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/postcall"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/scheduler"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

type Chunking uint8

const (
	VariableChunks Chunking = iota + 1
	FixedChunks
)

type OutputFormat uint8

const (
	FASTQ OutputFormat = iota + 1
	BAM
)

type GPULimits struct {
	LiveParents, PendingChunks  int
	LiveSamples, MaxReadSamples uint64
}
type GPUConfig struct {
	NativeBundleDir, ModelDir, POD5Path string
	ReadIDs                             []pod5.ID
	BatchSize                           int
	ChunkSize, Overlap                  uint64
	Chunking                            Chunking
	Limits                              GPULimits
	Policy                              postcall.Policy
	Format                              OutputFormat
	EmitMoves                           bool
	CommandLine                         string
}

// Counts on error describe an incomplete prefix. Only a nil RunGPU error
// establishes complete input, finalized output and successful resource drain.
type GPUReport struct {
	ExpectedParents, InputParents, CompletedParents, SerializedParents uint64
	OutputRecords, FilteredChildren, OriginalSamples                   uint64
}

type gpuInput interface {
	IDs(context.Context) ([]pod5.ID, error)
	Metadata(context.Context) (pod5.FileMetadata, error)
	Iterate(context.Context, []pod5.ID, func(pod5.Read) error) error
	Close() error
}
type gpuRecipe struct {
	chunkSize, overlap uint64
	variable           bool
}
type gpuSourceChunk struct{ offset, rawLength, granularity uint64 }
type gpuParent struct {
	ordinal       uint64
	read          pod5.Read
	normalized    signal.Normalized
	chunks        []gpuSourceChunk
	profile, next int
	collector     *Collector
}
type gpuChunkRef struct {
	parent *gpuParent
	index  int
	order  uint64
}
type gpuProfile struct {
	scheduler  *scheduler.Scheduler
	invocation scheduler.Invocation
}
type gpuRun struct {
	cfg                    GPUConfig
	spec                   model.Spec
	bundle                 gpuBundle
	ctx                    context.Context
	cancel                 context.CancelFunc
	input                  gpuInput
	output                 gpuRecordOutput
	engine                 *tensor.Engine
	device                 *tensor.Device
	hac                    *model.HACGPU
	sup                    *model.SUPGPU
	recipes                []gpuRecipe
	profiles               []*gpuProfile
	parents                []*gpuParent
	pending                map[string]gpuChunkRef
	expected               []pod5.ID
	liveSamples, nextChunk uint64
	report                 GPUReport
	eof                    bool
	newInvocation          func(*gpuRun, int) (scheduler.Invocation, error)
	closeShared            func() error
}

func validateGPUConfig(c GPUConfig) error {
	if c.Chunking != FixedChunks && c.Chunking != VariableChunks {
		return errors.New("GPU chunking must be fixed or variable")
	}
	if c.Format != FASTQ && c.Format != BAM {
		return errors.New("GPU output must be fastq or bam")
	}
	if c.BatchSize <= 0 || c.BatchSize > gpubatch.MaxCoordinate {
		return fmt.Errorf("GPU batch size requires explicit positive geometry")
	}
	l := c.Limits
	if l.LiveParents <= 0 || l.LiveParents > 1_000_000 || l.PendingChunks <= 0 || l.PendingChunks > gpubatch.MaxCoordinate || l.MaxReadSamples == 0 || l.MaxReadSamples > math.MaxInt32 || l.LiveSamples < l.MaxReadSamples || l.LiveSamples > math.MaxInt64/2 {
		return errors.New("GPU parent, signal and pending-chunk limits must be finite and admit one maximum read")
	}
	if c.Policy.Version != postcall.PolicyVersion || c.Policy.MinQ != 0 {
		return errors.New("unsupported GPU postcall policy")
	}
	return nil
}

// RunGPU owns one bounded, synchronous POD5-to-output operation on CUDA ordinal
// zero. The installation must have independently approved native qualification.
// dst is borrowed and never closed.
func RunGPU(ctx context.Context, cfg GPUConfig, dst io.Writer) (report GPUReport, err error) {
	if ctx == nil || dst == nil {
		return report, errors.New("GPU context and destination are required")
	}
	if err = ctx.Err(); err != nil {
		return report, err
	}
	if err = validateGPUConfig(cfg); err != nil {
		return report, err
	}
	if cfg.ModelDir == "" || cfg.POD5Path == "" {
		return report, errors.New("GPU model and POD5 paths are required")
	}
	if _, set := os.LookupEnv("DORADO_EXTRA_CHUNK_SIZES"); set {
		return report, errors.New("unset DORADO_EXTRA_CHUNK_SIZES for the pinned default GPU profile policy")
	}
	bundle, err := loadGPUBundle(cfg.NativeBundleDir)
	if err != nil {
		return report, err
	}
	spec, err := model.Inspect(cfg.ModelDir)
	if err != nil {
		return report, err
	}
	recipes, err := gpuSourceProfiles(spec, cfg)
	if err != nil {
		return report, err
	}
	cfg.ReadIDs = appendPreservingNil(cfg.ReadIDs)
	workCtx, cancel := context.WithCancel(ctx)
	r := &gpuRun{cfg: cfg, spec: spec, bundle: bundle, ctx: workCtx, cancel: cancel, recipes: recipes, profiles: make([]*gpuProfile, len(recipes)), pending: make(map[string]gpuChunkRef), newInvocation: newGPUInvocation}
	defer func() { err = r.finish(err); report = r.report }()
	reader, err := pod5.OpenContext(workCtx, bundle.pod5Library, cfg.POD5Path, pod5.Options{MaxSamples: cfg.Limits.MaxReadSamples})
	if reader != nil {
		r.input = reader
	}
	if err != nil {
		return report, err
	}
	if err = r.selectInput(); err != nil {
		return report, err
	}
	metadata, err := r.input.Metadata(workCtx)
	if err != nil {
		return report, err
	}
	out, err := newGPUOutput(dst, cfg, spec, metadata)
	if out != nil {
		r.output = out
	}
	if err != nil {
		return report, err
	}
	r.engine, err = tensor.Open(bundle.tensorLibrary)
	if err != nil {
		return report, err
	}
	r.device, err = r.engine.NewCUDADevice(0)
	if err != nil {
		return report, err
	}
	return report, r.execute()
}

func appendPreservingNil(ids []pod5.ID) []pod5.ID {
	if ids == nil {
		return nil
	}
	return append([]pod5.ID{}, ids...)
}

// CudaCaller.cpp calculate_batch_sizes uses descending unique rounded sizes.
// Its default simplex policy adds half the request except for variable SUP.
func gpuSourceProfiles(spec model.Spec, cfg GPUConfig) ([]gpuRecipe, error) {
	pinned, err := model.Pinned(spec.Architecture)
	if err != nil || spec != pinned {
		return nil, ErrUnsupported
	}
	requested, overlap := cfg.ChunkSize, cfg.Overlap
	if requested == 0 {
		if overlap != 0 {
			return nil, errors.New("zero GPU chunk size requires zero overlap and selects model defaults")
		}
		requested, overlap = uint64(spec.ChunkSize), uint64(spec.Overlap)
	}
	if requested > math.MaxInt32 || overlap >= math.MaxInt32 || overlap%uint64(spec.InnerStride) != 0 {
		return nil, errors.New("GPU chunk geometry is outside the pinned source domain")
	}
	granularity := uint64(spec.Granularity)
	minimum := ((overlap + 1 + granularity - 1) / granularity) * granularity
	adjust := func(size uint64) uint64 { return max(minimum, size/granularity*granularity) }
	sizes := []uint64{adjust(requested)}
	variable := cfg.Chunking == VariableChunks
	if !(variable && spec.Architecture == model.SUP) {
		half := adjust(uint64(float32(requested) * float32(.5)))
		if half != sizes[0] {
			sizes = append(sizes, half)
		}
	}
	sort.Slice(sizes, func(i, j int) bool { return sizes[i] > sizes[j] })
	profiles := make([]gpuRecipe, len(sizes))
	for i, size := range sizes {
		if size > math.MaxInt32 || size*uint64(cfg.BatchSize) > math.MaxInt32 {
			return nil, errors.New("GPU profile capacity exceeds source integer domain")
		}
		profiles[i] = gpuRecipe{size, overlap, variable}
	}
	return profiles, nil
}

// Preserve the strict source comparisons, including the equality boundary.
func gpuSelectProfile(recipes []gpuRecipe, samples uint64) int {
	best := 0
	for i := 1; i < len(recipes); i++ {
		a, b := recipes[best].chunkSize, recipes[i].chunkSize
		if (a < samples && a < b) || (samples < b && b < a) {
			best = i
		}
	}
	return best
}

func (r *gpuRun) selectInput() error {
	ids, err := r.input.IDs(r.ctx)
	if err != nil {
		return err
	}
	known := make(map[pod5.ID]bool, len(ids))
	for _, id := range ids {
		if known[id] {
			return errors.New("duplicate input UUID")
		}
		known[id] = true
	}
	r.expected = appendPreservingNil(ids)
	if r.cfg.ReadIDs != nil {
		seen := make(map[pod5.ID]bool, len(r.cfg.ReadIDs))
		for _, id := range r.cfg.ReadIDs {
			if !known[id] {
				return fmt.Errorf("%w: %s", pod5.ErrNotFound, id)
			}
			if seen[id] {
				return errors.New("duplicate selected UUID")
			}
			seen[id] = true
		}
		r.expected = appendPreservingNil(r.cfg.ReadIDs)
	}
	r.report.ExpectedParents = uint64(len(r.expected))
	return nil
}

func (r *gpuRun) execute() error {
	if err := r.input.Iterate(r.ctx, r.cfg.ReadIDs, r.acceptParent); err != nil {
		return err
	}
	r.eof = true
	for len(r.pending) != 0 {
		if err := r.flushOldest(); err != nil {
			return err
		}
	}
	if err := r.ctx.Err(); err != nil {
		return err
	}
	return r.emitReady()
}

func (r *gpuRun) acceptParent(read pod5.Read) error {
	if err := r.ctx.Err(); err != nil {
		return err
	}
	ordinal := r.report.InputParents
	if ordinal >= uint64(len(r.expected)) || read.ID != r.expected[ordinal] {
		return errors.New("GPU input changed selected parent identity or order")
	}
	r.report.InputParents++
	if err := validateRead(read); err != nil {
		return err
	}
	if read.NumSamples > r.cfg.Limits.MaxReadSamples || read.NumSamples > math.MaxUint64-r.report.OriginalSamples {
		return errors.New("GPU read exceeds sample limit or total overflows")
	}
	r.report.OriginalSamples += read.NumSamples
	if err := r.cfg.Policy.Validate(read.SequencingKit); err != nil {
		return err
	}
	if read.AcquisitionStartTimeMS < 0 {
		return errors.New("GPU read has negative acquisition clock")
	}
	if _, err := fastqMetadata(read, r.spec); err != nil {
		return err
	}
	normalizedSize := read.NumSamples
	if normalizedSize > 10 {
		normalizedSize -= 10
	}
	for len(r.parents) >= r.cfg.Limits.LiveParents || normalizedSize > r.cfg.Limits.LiveSamples-r.liveSamples {
		if len(r.pending) == 0 {
			return errors.New("GPU parent admission cannot make progress")
		}
		if err := r.flushOldest(); err != nil {
			return err
		}
	}
	normalized, err := signal.Normalize(read.Signal, signal.Calibration{Scale: read.CalibrationScale, Offset: read.CalibrationOffset}, signal.Standardisation{Mean: r.spec.Mean, Stdev: r.spec.Stdev}, adjustment(read.FlowCellProductCode, read.OpenPoreLevel, read.CalibrationScale))
	if err != nil {
		return err
	}
	profile := gpuSelectProfile(r.recipes, uint64(len(normalized.Samples)))
	plan, err := r.planParent(profile, uint64(len(normalized.Samples)))
	if err != nil {
		return err
	}
	if err = r.ensureProfile(profile); err != nil {
		return err
	}
	c, err := NewCollector(map[string]ReadShape{read.ID.String(): {RawSamples: uint64(len(normalized.Samples)), Stride: uint64(r.spec.Stride), Variable: r.recipes[profile].variable, Chunks: len(plan)}})
	if err != nil {
		return err
	}
	read.Signal = nil
	p := &gpuParent{ordinal: ordinal, read: read, normalized: normalized, chunks: plan, profile: profile, collector: c}
	r.parents = append(r.parents, p)
	r.liveSamples += uint64(len(normalized.Samples))
	for p.next < len(p.chunks) {
		if err = r.ctx.Err(); err != nil {
			return err
		}
		if len(r.pending) == r.cfg.Limits.PendingChunks {
			if err = r.flushOldest(); err != nil {
				return err
			}
		}
		id := strconv.FormatUint(p.ordinal, 10) + "/" + strconv.Itoa(p.next)
		if err = r.profiles[profile].scheduler.Submit(r.ctx, gpubatch.Chunk{ID: id, Length: int(p.chunks[p.next].rawLength)}); err != nil {
			return err
		}
		r.pending[id] = gpuChunkRef{p, p.next, r.nextChunk}
		r.nextChunk++
		p.next++
	}
	return nil
}

func (r *gpuRun) planParent(profile int, samples uint64) ([]gpuSourceChunk, error) {
	p := r.recipes[profile]
	c := chunk.Config{ChunkSize: p.chunkSize, Overlap: p.overlap, Stride: uint64(r.spec.Stride), Granularity: uint64(r.spec.Granularity)}
	var intervals []chunk.Interval
	granularity := p.chunkSize
	if p.variable {
		var err error
		if r.spec.Architecture == model.SUP {
			intervals, err = chunk.Transformer(samples, c)
			granularity = uint64(r.spec.Granularity)
		} else {
			intervals, err = chunk.Variable(samples, c)
			granularity = uint64(r.spec.Stride)
		}
		if err != nil {
			return nil, err
		}
	} else {
		offsets, err := chunk.Fixed(samples, c)
		if err != nil {
			return nil, err
		}
		intervals = make([]chunk.Interval, len(offsets))
		for i, offset := range offsets {
			intervals[i] = chunk.Interval{Start: offset, End: min(offset+p.chunkSize, samples)}
		}
	}
	plan := make([]gpuSourceChunk, len(intervals))
	for i, interval := range intervals {
		plan[i] = gpuSourceChunk{interval.Start, interval.End - interval.Start, granularity}
	}
	return plan, nil
}

func (r *gpuRun) geometry(profile int) (gpubatch.Config, error) {
	p := r.recipes[profile]
	lane := gpubatch.HAC
	if r.spec.Architecture == model.SUP {
		lane = gpubatch.SUP
	}
	if p.variable && lane == gpubatch.SUP {
		return gpubatch.Config{}, fmt.Errorf("%w: packed SUP convolution layout and GPU graph are unfinished", ErrGPUUnqualified)
	}
	g := gpubatch.Config{Lane: lane, ChunkSize: int(p.chunkSize), BatchSize: r.cfg.BatchSize, Stride: int(r.spec.Stride), InnerStride: int(r.spec.InnerStride), Granularity: int(r.spec.Granularity), VariableChunks: p.variable}
	return g, g.Validate()
}

func (r *gpuRun) ensureProfile(index int) error {
	if r.profiles[index] != nil {
		return nil
	}
	p := &gpuProfile{}
	r.profiles[index] = p
	var err error
	p.invocation, err = r.newInvocation(r, index)
	if err != nil {
		return err
	}
	geometry, err := r.geometry(index)
	if err != nil {
		return err
	}
	p.scheduler, err = scheduler.New(scheduler.Config{Geometry: geometry, MaxPending: r.cfg.Limits.PendingChunks, Selection: scheduler.SelectionExplicit}, p.invocation)
	return err
}

func (r *gpuRun) flushOldest() error {
	before := len(r.pending)
	if before == 0 {
		return errors.New("GPU flush has no pending work")
	}
	oldest := uint64(math.MaxUint64)
	profile := -1
	for _, ref := range r.pending {
		if ref.order < oldest {
			oldest, profile = ref.order, ref.parent.profile
		}
	}
	if err := r.profiles[profile].scheduler.Flush(r.ctx); err != nil {
		return err
	}
	if len(r.pending) >= before {
		return errors.New("GPU invocation did not publish its complete batch")
	}
	if err := r.ctx.Err(); err != nil {
		return err
	}
	return r.emitReady()
}

func (r *gpuRun) publish(b scheduler.Batch, work []DecodedWork) error {
	if len(work) != b.Real || len(b.Placements) != b.Real {
		return errors.New("GPU decoded membership differs from real batch")
	}
	for j, placement := range b.Placements {
		ref, ok := r.pending[placement.Chunk.ID]
		if !ok {
			return errors.New("GPU completion has unknown chunk")
		}
		c, w := ref.parent.chunks[ref.index], work[j]
		if w.ReadID != ref.parent.read.ID.String() || w.ChunkIndex != ref.index || w.Offset != c.offset || w.RawLength != c.rawLength || w.Granularity != c.granularity || placement.Chunk.Length != int(c.rawLength) || len(w.Moves) != placement.Length/int(r.spec.Stride) || len(w.Sequence) != len(w.Qstring) {
			return errors.New("GPU completion changed source coordinates or output extent")
		}
		bases := 0
		for _, move := range w.Moves {
			if move > 1 {
				return errors.New("GPU completion has nonbinary move")
			}
			bases += int(move)
		}
		if bases != len(w.Sequence) {
			return errors.New("GPU completion move count differs from bases")
		}
	}
	for j, placement := range b.Placements {
		ref := r.pending[placement.Chunk.ID]
		if err := ref.parent.collector.Add(work[j]); err != nil {
			return err
		}
	}
	for _, placement := range b.Placements {
		delete(r.pending, placement.Chunk.ID)
	}
	return nil
}

func (r *gpuRun) emitReady() error {
	for len(r.parents) > 0 {
		p := r.parents[0]
		if p.next != len(p.chunks) || !p.collector.Ready(p.read.ID.String()) {
			break
		}
		if err := r.ctx.Err(); err != nil {
			return err
		}
		assembled, err := p.collector.Assemble(p.read.ID.String())
		if err != nil {
			return err
		}
		processed, err := ProcessGPURead(r.spec, GPURead{Read: p.read, Normalized: p.normalized, Stride: uint64(r.spec.Stride), Assembled: assembled}, r.cfg.Policy)
		if err != nil {
			return err
		}
		r.report.CompletedParents++
		kept := 0
		for _, child := range processed.Postcall.Children {
			if child.Filter.Keep {
				kept++
			} else {
				r.report.FilteredChildren++
			}
		}
		if kept != len(processed.Reads) {
			return errors.New("GPU postcall children differ from writer records")
		}
		if err = r.output.write(processed); err != nil {
			return err
		}
		r.report.SerializedParents++
		r.report.OutputRecords += uint64(kept)
		r.liveSamples -= uint64(len(p.normalized.Samples))
		r.parents[0] = nil
		r.parents = r.parents[1:]
	}
	return nil
}

func (r *gpuRun) finish(workErr error) error {
	workErr = errors.Join(workErr, r.ctx.Err())
	if workErr != nil {
		r.cancel()
	}
	var cleanup error
	profilesDrained := true
	for _, p := range r.profiles {
		if p == nil {
			continue
		}
		if p.scheduler != nil {
			cleanup = errors.Join(cleanup, p.scheduler.Shutdown(r.ctx))
			profilesDrained = profilesDrained && p.scheduler.Stats().Drained
		} else if p.invocation != nil {
			e := p.invocation.Drain(context.WithoutCancel(r.ctx))
			cleanup = errors.Join(cleanup, e)
			profilesDrained = profilesDrained && e == nil
		}
	}
	sharedClosed := false
	if profilesDrained || r.device == nil {
		cleanup = errors.Join(cleanup, r.closeModels())
		if r.closeShared != nil {
			cleanup = errors.Join(cleanup, r.closeShared())
			sharedClosed = true
		}
	}
	if r.device != nil {
		e := r.device.Close()
		cleanup = errors.Join(cleanup, e)
		if e == nil {
			cleanup = errors.Join(cleanup, r.recoverModelOwners())
			profilesDrained = true
		}
	}
	if r.engine != nil {
		e := r.engine.Close()
		cleanup = errors.Join(cleanup, e)
		if e == nil {
			cleanup = errors.Join(cleanup, r.recoverModelOwners())
			profilesDrained = true
		}
	}
	if !sharedClosed && profilesDrained && r.closeShared != nil {
		cleanup = errors.Join(cleanup, r.closeShared())
	}
	if r.output != nil {
		cleanup = errors.Join(cleanup, r.output.finish())
	}
	if r.input != nil {
		cleanup = errors.Join(cleanup, r.input.Close())
	}
	result := errors.Join(workErr, cleanup, r.ctx.Err())
	if result == nil {
		if !r.eof || r.report.InputParents != r.report.ExpectedParents || r.report.CompletedParents != r.report.ExpectedParents || r.report.SerializedParents != r.report.ExpectedParents || len(r.pending) != 0 || len(r.parents) != 0 || r.liveSamples != 0 {
			result = errors.New("GPU caller did not account for complete input")
		}
		for _, p := range r.profiles {
			if p != nil && p.scheduler != nil {
				st := p.scheduler.Stats()
				if st.FailedChunks != 0 || st.Abandoned != 0 || st.Pending != 0 || !st.Drained {
					result = errors.Join(result, errors.New("GPU scheduler has failed, abandoned or undrained work"))
				}
			}
		}
	}
	r.cancel()
	clear(r.pending)
	r.parents = nil
	r.liveSamples = 0
	return result
}

func (r *gpuRun) closeModels() error {
	var err error
	if r.hac != nil {
		e := r.hac.Close()
		err = errors.Join(err, e)
		if e == nil {
			r.hac = nil
		}
	}
	if r.sup != nil {
		e := r.sup.Close()
		err = errors.Join(err, e)
		if e == nil {
			r.sup = nil
		}
	}
	return err
}

func (r *gpuRun) recoverModelOwners() error {
	var err error
	for _, p := range r.profiles {
		if p != nil {
			if i, ok := p.invocation.(*gpuInvocation); ok {
				err = errors.Join(err, i.release())
			}
		}
	}
	return errors.Join(err, r.closeModels())
}
