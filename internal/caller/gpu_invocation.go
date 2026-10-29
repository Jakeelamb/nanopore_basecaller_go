package caller

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/chunk"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpubatch"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpudecode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/scheduler"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

type gpuInvocation struct {
	run      *gpuRun
	profile  int
	stream   *tensor.Stream
	hacPlan  model.HACGPUPlan
	supPlan  model.SUPGPUPlan
	input    *tensor.Storage
	beam     *gpudecode.GPUBeamNumbers
	quality  *gpudecode.GPUQualityNumbers
	retained []*tensor.Storage
	drained  bool
	drainErr error
}

func newGPUInvocation(r *gpuRun, profile int) (owner scheduler.Invocation, err error) {
	p := r.recipes[profile]
	if err := r.bundle.record.validatePartialProfiles(); err != nil {
		return nil, err
	}
	key := gpuProfileKey{string(r.spec.Architecture), int(p.chunkSize), r.cfg.BatchSize, p.variable}
	if !r.bundle.record.profiles[key] {
		return nil, fmt.Errorf("%w: no native qualification for %s chunk%d N%d variable=%t", ErrGPUUnqualified, key.lane, key.chunk, key.batch, key.variable)
	}
	if p.variable || (p.chunkSize != 12288 && p.chunkSize != 6144) {
		return nil, fmt.Errorf("%w: fixed GPU graphs require chunk6144 or chunk12288; packed graphs/decoder indexing remain unfinished", ErrGPUUnqualified)
	}
	batch := 64
	if r.spec.Architecture == model.SUP {
		batch = 32
	}
	if r.cfg.BatchSize != batch {
		return nil, fmt.Errorf("%w: existing fixed graph requires N%d", ErrGPUUnqualified, batch)
	}
	i := &gpuInvocation{run: r, profile: profile}
	owner = i
	i.stream, err = r.device.NewStream()
	if err != nil {
		return owner, err
	}
	program, err := r.device.DecoderGuideProgram()
	if err != nil {
		return owner, err
	}
	if sha256.Sum256(program) != r.bundle.record.guide {
		return owner, errors.New("GPU guide program differs from independent bundle pin")
	}
	program, err = r.device.DecoderPosteriorProgram(i.stream)
	if err != nil {
		return owner, err
	}
	if sha256.Sum256(program) != r.bundle.record.posterior {
		return owner, errors.New("GPU posterior program differs from independent bundle pin")
	}
	i.beam, err = gpudecode.NewGPUBeamNumbers(r.device, i.stream, r.bundle.record.beam, 1<<20)
	if err != nil {
		return owner, err
	}
	i.quality, err = gpudecode.NewGPUQualityNumbers(r.device, i.stream, r.bundle.record.posterior, 1<<20)
	if err != nil {
		return owner, err
	}
	if r.spec.Architecture == model.HAC {
		profile := model.SourceDefaultHACGPUProfile()
		if p.chunkSize == 6144 {
			profile = model.SourceShortHACGPUProfile()
		}
		i.hacPlan, err = model.PlanHACGPU(profile)
		if err != nil {
			return owner, err
		}
		if r.hac == nil {
			weights, e := model.LoadHACGPUWeights(r.ctx, r.engine, r.cfg.ModelDir)
			if e != nil {
				return owner, e
			}
			r.hac, err = model.NewHACGPU(r.device, i.stream, weights)
		}
	} else {
		profile := model.FullFixedSUPGPUProfile()
		if p.chunkSize == 6144 {
			profile = model.ShortFixedSUPGPUProfile()
		}
		i.supPlan, err = model.PlanSUPGPU(profile)
		if err != nil {
			return owner, err
		}
		if r.sup == nil {
			weights, e := model.LoadSUPGPUWeights(r.ctx, r.engine, r.cfg.ModelDir)
			if e != nil {
				return owner, e
			}
			r.sup, err = model.NewSUPGPU(r.ctx, r.device, i.stream, weights)
		}
	}
	return owner, err
}

func (i *gpuInvocation) Call(ctx context.Context, b scheduler.Batch) (err error) {
	if i.drained {
		return errors.New("GPU invocation is drained")
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	p, r := i.run.recipes[i.profile], i.run
	data, err := i.prepareInput(b)
	if err != nil {
		return err
	}
	if i.input == nil {
		i.input, err = r.device.NewStorage(i.stream, tensor.DTypeF16, []int64{int64(b.PaddedSlots), 1, int64(p.chunkSize)}, []tensor.KoiTag{tensor.TagN, tensor.TagC, tensor.TagT}, data)
	} else {
		err = i.input.Upload(i.stream, data)
	}
	if err != nil {
		return err
	}
	var owners []*tensor.Storage
	defer func() {
		if err != nil {
			i.retained = append(i.retained, owners...)
		}
	}()
	var scores *tensor.Storage
	var qscale, qbias, blank, clamp float32
	var stateLen int64
	if r.hac != nil {
		result, e := r.hac.Forward(ctx, i.stream, i.input, i.hacPlan, model.HACGPUBatch{ActiveSlots: b.PaddedSlots})
		if result != nil {
			scores = result.Scores
			owners = append(owners, scores)
		}
		if e != nil {
			return e
		}
		qscale, qbias, blank, clamp, stateLen = result.Metadata.QScale, result.Metadata.QBias, result.Metadata.BlankScore, result.Metadata.DecoderClampValue, result.Metadata.StateLen
	} else if r.sup != nil {
		result, e := r.sup.Forward(ctx, i.stream, i.input, i.supPlan)
		if result != nil {
			scores = result.Scores
			owners = append(owners, scores)
		}
		if e != nil {
			return e
		}
		qscale, qbias, blank, clamp, stateLen = result.Metadata.QScale, result.Metadata.QBias, result.Metadata.BlankScore, result.Metadata.DecoderClampValue, result.Metadata.StateLen
	} else {
		return errors.New("GPU invocation has no concrete graph")
	}
	if scores == nil || stateLen != r.spec.StateLen {
		return errors.New("GPU graph returned invalid decoder metadata")
	}
	guidePlan, e := gpudecode.PlanGuides(int64(b.PaddedSlots), int64(p.chunkSize)/r.spec.Stride, 1<<(2*stateLen), clamp, blank)
	if e != nil {
		return e
	}
	guides, e := gpudecode.BackGuides(ctx, r.device, i.stream, scores, guidePlan)
	if guides != nil {
		owners = append(owners, guides)
	}
	if e != nil {
		return e
	}
	beamPlan, e := gpudecode.PlanBeam(guidePlan, 100)
	if e != nil {
		return e
	}
	beamInput, e := gpudecode.DownloadBeamInput(ctx, beamPlan, i.stream, scores, guides)
	if e != nil {
		return e
	}
	beam, e := gpudecode.BeamSearch(ctx, beamPlan, beamInput, i.beam)
	if e != nil {
		return e
	}
	working, e := gpudecode.Posteriors(ctx, r.device, i.stream, scores, guides, guidePlan)
	if working != nil {
		owners = append(owners, working)
	}
	if e != nil {
		return e
	}
	workingBytes, e := working.Download(i.stream)
	if e != nil {
		return e
	}
	bits := make([]uint16, len(workingBytes)/2)
	for j := range bits {
		bits[j] = binary.LittleEndian.Uint16(workingBytes[j*2:])
	}
	decodePlan, e := gpudecode.PlanDecode(guidePlan, qscale, qbias, false)
	if e != nil {
		return e
	}
	history := make([][][]gpudecode.BeamHistoryRecord, len(beam.History))
	for j := range history {
		history[j] = beam.History[j].Rows
	}
	quality, e := gpudecode.QualityFromHistory(ctx, decodePlan, bits, history, beam.FinalScoreBits, i.quality)
	if e != nil {
		return e
	}
	work, e := i.decodedWork(b, quality)
	if e != nil {
		return e
	}
	if err = ctx.Err(); err != nil {
		return err
	}
	var failed []*tensor.Storage
	for j := len(owners) - 1; j >= 0; j-- {
		if owners[j] != nil {
			if e = owners[j].Close(); e != nil {
				err = errors.Join(err, e)
				failed = append(failed, owners[j])
			}
		}
	}
	owners = failed
	if err != nil {
		return err
	}
	return r.publish(b, work)
}

func (i *gpuInvocation) validateBatch(b scheduler.Batch) error {
	p, r := i.run.recipes[i.profile], i.run
	wantLane := gpubatch.HAC
	if r.spec.Architecture == model.SUP {
		wantLane = gpubatch.SUP
	}
	if b.Variable || b.Lane != wantLane || b.PaddedSlots != r.cfg.BatchSize || b.PaddedSamples != r.cfg.BatchSize*int(p.chunkSize) || b.Real < 1 || b.Real > b.PaddedSlots || len(b.Placements) != b.Real {
		return errors.New("GPU invocation geometry differs from its profile")
	}
	if err := r.bundle.record.validatePartialProfiles(); err != nil {
		return err
	}
	key := gpuProfileKey{string(r.spec.Architecture), int(p.chunkSize), r.cfg.BatchSize, p.variable}
	if b.Real < b.PaddedSlots && !r.bundle.record.partialProfiles[key] {
		return fmt.Errorf("%w: partial fixed profile requires independent publication qualification", ErrGPUUnqualified)
	}
	seen := make(map[string]bool, b.Real)
	for slot, placement := range b.Placements {
		ref, ok := r.pending[placement.Chunk.ID]
		if !ok || seen[placement.Chunk.ID] || ref.parent == nil || ref.parent.profile != i.profile || ref.index < 0 || ref.index >= len(ref.parent.chunks) || placement.Slot != slot || placement.Offset != slot*int(p.chunkSize) || placement.Length != int(p.chunkSize) {
			return errors.New("GPU input placement changed profile, identity or fixed slot")
		}
		seen[placement.Chunk.ID] = true
		c := ref.parent.chunks[ref.index]
		samples := uint64(len(ref.parent.normalized.Samples))
		if c.rawLength == 0 || c.rawLength > p.chunkSize || c.offset > samples || c.rawLength > samples-c.offset || placement.Chunk.Length != int(c.rawLength) {
			return errors.New("GPU input changed unpadded chunk length")
		}
	}
	return nil
}

// Every upload replaces the complete physical input, including absent rows.
func (i *gpuInvocation) prepareInput(b scheduler.Batch) ([]byte, error) {
	if err := i.validateBatch(b); err != nil {
		return nil, err
	}
	data := make([]byte, b.PaddedSamples*2)
	for _, placement := range b.Placements {
		ref := i.run.pending[placement.Chunk.ID]
		c := ref.parent.chunks[ref.index]
		padded, e := chunk.RepeatPad(ref.parent.normalized.Samples[c.offset:c.offset+c.rawLength], uint64(placement.Length))
		if e != nil {
			return nil, e
		}
		for j, value := range padded {
			binary.LittleEndian.PutUint16(data[(placement.Offset+j)*2:], value)
		}
	}
	return data, nil
}

// Decode all physical rows before selecting real membership. Errors from graph,
// beam and quality computation remain fatal, including errors in absent lanes.
func (i *gpuInvocation) decodedWork(b scheduler.Batch, physical gpudecode.QualityBatch) ([]DecodedWork, error) {
	if err := i.validateBatch(b); err != nil {
		return nil, err
	}
	if len(physical.Chunks) != b.PaddedSlots {
		return nil, errors.New("GPU decoder result count differs from physical batch")
	}
	work := make([]DecodedWork, b.Real)
	for j := range work {
		decoded := physical.Chunks[j]
		ref := i.run.pending[b.Placements[j].Chunk.ID]
		c := ref.parent.chunks[ref.index]
		if len(decoded.Moves) != b.Placements[j].Length/int(i.run.spec.Stride) {
			return nil, errors.New("GPU decoder move extent differs from physical profile")
		}
		sequence, qstring, err := gpuExtractChunk(decoded)
		if err != nil {
			return nil, err
		}
		work[j] = DecodedWork{ReadID: ref.parent.read.ID.String(), ChunkIndex: ref.index, Offset: c.offset, RawLength: c.rawLength, Granularity: c.granularity, Sequence: sequence, Qstring: qstring, Moves: decoded.Moves}
	}
	return work, nil
}

// The source allocates T sequence/Q bytes but only sum(moves) bytes are live.
func gpuExtractChunk(decoded gpudecode.QualityChunk) (string, string, error) {
	bases := 0
	for _, move := range decoded.Moves {
		if move > 1 {
			return "", "", errors.New("GPU decoder returned nonbinary moves")
		}
		bases += int(move)
	}
	if bases > len(decoded.Sequence) || bases > len(decoded.Q) {
		return "", "", errors.New("GPU decoder sequence/Q storage is shorter than move count")
	}
	return string(decoded.Sequence[:bases]), string(decoded.Q[:bases]), nil
}

// Stream.Close synchronizes before destruction. Keep every graph/storage owner
// live through that call; an unsuccessful drain leaves them for device recovery.
func (i *gpuInvocation) Drain(_ context.Context) error {
	if i.drained {
		return i.drainErr
	}
	i.drained = true
	i.drainErr = i.release()
	return i.drainErr
}

func (i *gpuInvocation) release() error {
	var cleanup error
	if i.stream != nil {
		if err := i.stream.Close(); err != nil {
			return err
		}
		i.stream = nil
	}
	var failed []*tensor.Storage
	for j := len(i.retained) - 1; j >= 0; j-- {
		if h := i.retained[j]; h != nil {
			if err := h.Close(); err != nil {
				cleanup = errors.Join(cleanup, err)
				failed = append(failed, h)
			}
		}
	}
	i.retained = failed
	if i.input != nil {
		err := i.input.Close()
		cleanup = errors.Join(cleanup, err)
		if err == nil {
			i.input = nil
		}
	}
	return cleanup
}
