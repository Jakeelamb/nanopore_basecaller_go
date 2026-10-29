//go:build cgo

package caller

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpudecode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/pod5"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/postcall"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/scheduler"
)

type gpuHostInput struct {
	reads      []pod5.Read
	closed     int
	events     *[]string
	earlyEOF   bool
	failAfter  int
	iterateErr error
}

func (in *gpuHostInput) IDs(ctx context.Context) ([]pod5.ID, error) {
	ids := make([]pod5.ID, len(in.reads))
	for j, r := range in.reads {
		ids[j] = r.ID
	}
	return ids, ctx.Err()
}
func (in *gpuHostInput) Metadata(context.Context) (pod5.FileMetadata, error) {
	return pod5.FileMetadata{}, nil
}
func (in *gpuHostInput) Iterate(ctx context.Context, ids []pod5.ID, visit func(pod5.Read) error) error {
	for j, r := range in.reads {
		if err := ctx.Err(); err != nil {
			return err
		}
		if in.iterateErr != nil && j == in.failAfter {
			return in.iterateErr
		}
		if in.earlyEOF && j == len(in.reads)-1 {
			return nil
		}
		if err := visit(r); err != nil {
			return err
		}
	}
	return ctx.Err()
}
func (in *gpuHostInput) Close() error {
	in.closed++
	*in.events = append(*in.events, "input-close")
	return nil
}

type gpuHostOutput struct {
	ids              []string
	writes, finishes int
	failWrite        int
	finalErr         error
	events           *[]string
}

func (w *gpuHostOutput) write(p Processed) error {
	w.writes++
	if w.writes == w.failWrite {
		return errors.New("writer failed")
	}
	for _, r := range p.Reads {
		w.ids = append(w.ids, r.ID)
	}
	return nil
}
func (w *gpuHostOutput) finish() error {
	w.finishes++
	*w.events = append(*w.events, "output-finish")
	return w.finalErr
}

type gpuHostInvocation struct {
	run     *gpuRun
	harness *gpuHostHarness
	profile int
	drains  int
}

func (i *gpuHostInvocation) Call(ctx context.Context, b scheduler.Batch) error {
	h := i.harness
	h.calls++
	h.real = append(h.real, b.Real)
	h.profileCalls = append(h.profileCalls, i.profile)
	h.eofAtCall = append(h.eofAtCall, i.run.eof)
	if len(i.run.pending) > i.run.cfg.Limits.PendingChunks || len(i.run.parents) > i.run.cfg.Limits.LiveParents || i.run.liveSamples > i.run.cfg.Limits.LiveSamples {
		h.t.Fatal("host buffers exceeded limits")
	}
	if h.calls == h.failCall {
		return errors.New("middle batch failed")
	}
	if h.calls == h.cancelCall {
		h.run.cancel()
		return ctx.Err()
	}
	work := make([]DecodedWork, len(b.Placements))
	for j, p := range b.Placements {
		ref := i.run.pending[p.Chunk.ID]
		c := ref.parent.chunks[ref.index]
		moves := make([]uint8, p.Length/int(i.run.spec.Stride))
		for k := range moves {
			moves[k] = 1
		}
		work[j] = DecodedWork{ReadID: ref.parent.read.ID.String(), ChunkIndex: ref.index, Offset: c.offset, RawLength: c.rawLength, Granularity: c.granularity, Sequence: strings.Repeat("A", len(moves)), Qstring: strings.Repeat("I", len(moves)), Moves: moves}
	}
	if h.physicalDecode {
		quality := gpudecode.QualityBatch{Chunks: make([]gpudecode.QualityChunk, b.PaddedSlots)}
		for j := range quality.Chunks {
			quality.Chunks[j] = gpudecode.QualityChunk{Moves: []byte{2}, Sequence: []byte("ABSENT"), Q: []byte("!!!!!!")}
		}
		for j, w := range work {
			quality.Chunks[j] = gpudecode.QualityChunk{Moves: w.Moves, Sequence: []byte(w.Sequence), Q: []byte(w.Qstring)}
		}
		var err error
		work, err = (&gpuInvocation{run: i.run, profile: i.profile}).decodedWork(b, quality)
		if err != nil {
			return err
		}
	}
	if h.omitResult {
		work = work[:len(work)-1]
	}
	if h.badCoordinates {
		work[0].Offset++
	}
	if err := i.run.publish(b, work); err != nil {
		return err
	}
	if h.calls == h.cancelAfterCall {
		h.run.cancel()
	}
	return nil
}
func (i *gpuHostInvocation) Drain(context.Context) error {
	i.drains++
	*i.harness.events = append(*i.harness.events, fmt.Sprintf("drain-%d", i.profile))
	return i.harness.drainErr
}

type gpuHostHarness struct {
	t                                            *testing.T
	run                                          *gpuRun
	input                                        *gpuHostInput
	output                                       *gpuHostOutput
	invocations                                  []*gpuHostInvocation
	events                                       *[]string
	calls, failCall, cancelCall, cancelAfterCall int
	real, profileCalls                           []int
	omitResult                                   bool
	badCoordinates                               bool
	physicalDecode                               bool
	eofAtCall                                    []bool
	drainErr, sharedErr                          error
}

func gpuHostRead(id byte, normalized int) pod5.Read {
	r := testRead()
	r.ID[0] = id
	r.Signal = make([]int16, normalized+10)
	r.NumSamples = uint64(len(r.Signal))
	r.EndSample = r.StartSample + r.NumSamples
	r.RunID = "run"
	r.FlowCellID = "flow"
	r.Channel = 42
	r.AcquisitionStartTimeMS = 1000
	r.ProtocolStartTimeMS = 900
	r.EndReasonName = "unknown"
	return r
}

func newGPUHostHarness(t *testing.T, reads []pod5.Read) *gpuHostHarness {
	t.Helper()
	events := []string{}
	h := &gpuHostHarness{t: t, events: &events}
	h.input = &gpuHostInput{reads: reads, events: &events}
	h.output = &gpuHostOutput{events: &events}
	policy := postcall.DefaultPolicy()
	policy.Trim, policy.Split = false, false
	cfg := GPUConfig{BatchSize: 64, ChunkSize: 12288, Overlap: 600, Chunking: FixedChunks, Format: FASTQ, Policy: policy, Limits: GPULimits{LiveParents: 2, PendingChunks: 2, LiveSamples: 500000, MaxReadSamples: 250000}}
	recipes, err := gpuSourceProfiles(testSpec(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	r := &gpuRun{cfg: cfg, spec: testSpec(), ctx: ctx, cancel: cancel, input: h.input, output: h.output, recipes: recipes, profiles: make([]*gpuProfile, len(recipes)), pending: make(map[string]gpuChunkRef)}
	h.run = r
	r.newInvocation = func(r *gpuRun, profile int) (scheduler.Invocation, error) {
		i := &gpuHostInvocation{run: r, harness: h, profile: profile}
		h.invocations = append(h.invocations, i)
		return i, nil
	}
	r.closeShared = func() error {
		for _, i := range h.invocations {
			if i.drains != 1 {
				t.Fatal("shared device closed before every invocation drained")
			}
		}
		events = append(events, "shared-close")
		return h.sharedErr
	}
	if err = r.selectInput(); err != nil {
		t.Fatal(err)
	}
	return h
}

func (h *gpuHostHarness) execute() error { return h.run.finish(h.run.execute()) }
func (h *gpuHostHarness) assertRetired() {
	h.t.Helper()
	if len(h.run.parents) != 0 || len(h.run.pending) != 0 || h.run.liveSamples != 0 || h.input.closed != 1 || h.output.finishes != 1 {
		h.t.Fatal("run retained parent state or missed finalization")
	}
	for _, i := range h.invocations {
		if i.drains != 1 {
			h.t.Fatal("invocation drain count differs")
		}
	}
}

func TestGPUHostLongReadProgress(t *testing.T) {
	n := 12288 + 7*(12288-600)
	h := newGPUHostHarness(t, []pod5.Read{gpuHostRead(1, n)})
	h.run.cfg.Limits.LiveParents = 1
	if err := h.execute(); err != nil {
		t.Fatal(err)
	}
	if h.calls != 4 || h.run.report.ExpectedParents != 1 || h.run.report.InputParents != 1 || h.run.report.CompletedParents != 1 || h.run.report.SerializedParents != 1 || h.run.report.OutputRecords != 1 || h.run.report.OriginalSamples != uint64(n+10) {
		t.Fatalf("progress/accounting %+v calls=%d", h.run.report, h.calls)
	}
	h.assertRetired()
}

func TestGPUHostCrossParentAndRetirement(t *testing.T) {
	reads := []pod5.Read{gpuHostRead(1, 7002), gpuHostRead(2, 7002), gpuHostRead(3, 7002)}
	h := newGPUHostHarness(t, reads)
	if err := h.execute(); err != nil {
		t.Fatal(err)
	}
	if h.calls != 2 || h.real[0] != 2 || h.real[1] != 1 || len(h.output.ids) != 3 || h.run.report.OriginalSamples != 3*7012 {
		t.Fatalf("cross-read batches=%v report=%+v", h.real, h.run.report)
	}
	for j, r := range reads {
		if h.output.ids[j] != r.ID.String() {
			t.Fatal("output parent order changed")
		}
	}
	h.assertRetired()
}

func TestGPUHostSampleAdmissionAndFilteredParent(t *testing.T) {
	h := newGPUHostHarness(t, []pod5.Read{gpuHostRead(1, 7002), gpuHostRead(2, 7002), gpuHostRead(3, 7002)})
	h.run.cfg.Limits.LiveParents = 4
	h.run.cfg.Limits.MaxReadSamples = 10000
	h.run.cfg.Limits.LiveSamples = 14004
	h.run.cfg.Limits.PendingChunks = 64
	if err := h.execute(); err != nil {
		t.Fatal(err)
	}
	if h.calls != 2 || h.real[0] != 2 {
		t.Fatal("sample barrier did not flush before admitting a third normalized parent")
	}
	h.assertRetired()
	h = newGPUHostHarness(t, []pod5.Read{gpuHostRead(4, 12)})
	if err := h.execute(); err != nil {
		t.Fatal(err)
	}
	if h.run.report.SerializedParents != 1 || h.run.report.OutputRecords != 0 || h.run.report.FilteredChildren != 1 {
		t.Fatalf("filtered parent disappeared: %+v", h.run.report)
	}
	h.assertRetired()
}

func TestGPUHostStrictProfileBoundary(t *testing.T) {
	h := newGPUHostHarness(t, []pod5.Read{gpuHostRead(1, 6143), gpuHostRead(2, 6144), gpuHostRead(3, 6145)})
	if err := h.execute(); err != nil {
		t.Fatal(err)
	}
	if len(h.invocations) != 2 || h.real[0] != 1 || h.profileCalls[0] != 1 {
		t.Fatalf("source profile boundary differs: %v", h.profileCalls)
	}
	if h.run.report.SerializedParents != 3 {
		t.Fatal("profile transition lost a parent")
	}
	h.assertRetired()
}

func TestGPUHostFailuresNeverComplete(t *testing.T) {
	for _, name := range []string{"middle-batch", "cancel-before", "cancel-during", "cancel-after", "writer", "final-flush", "drain", "missing-result", "coordinates", "early-eof"} {
		t.Run(name, func(t *testing.T) {
			h := newGPUHostHarness(t, []pod5.Read{gpuHostRead(1, 12288+7*(12288-600))})
			switch name {
			case "middle-batch":
				h.failCall = 2
			case "cancel-before":
				h.run.cancel()
			case "cancel-during":
				h.cancelCall = 1
			case "cancel-after":
				h.cancelAfterCall = 1
			case "writer":
				h.output.failWrite = 1
			case "final-flush":
				h.output.finalErr = errors.New("final flush failed")
			case "drain":
				h.drainErr = errors.New("drain failed")
			case "missing-result":
				h.omitResult = true
			case "coordinates":
				h.badCoordinates = true
			case "early-eof":
				h.input.earlyEOF = true
			}
			if err := h.execute(); err == nil {
				t.Fatal("incomplete run returned success")
			}
			if name == "middle-batch" && h.calls != 2 {
				t.Fatal("failed scheduler resumed")
			}
			if strings.HasPrefix(name, "cancel-") && h.calls > 1 {
				t.Fatal("batch started after cancellation")
			}
			if name == "writer" && h.run.report.SerializedParents != 0 {
				t.Fatal("failed parent counted as serialized")
			}
			h.assertRetired()
		})
	}
}

func TestGPUHostRejectsSelectionBeforeCallbacks(t *testing.T) {
	for _, duplicate := range []bool{false, true} {
		h := newGPUHostHarness(t, []pod5.Read{gpuHostRead(1, 7002)})
		h.run.cfg.ReadIDs = []pod5.ID{h.input.reads[0].ID, h.input.reads[0].ID}
		if !duplicate {
			h.run.cfg.ReadIDs[1][0] = 99
		}
		err := h.run.selectInput()
		if err == nil || h.calls != 0 || len(h.invocations) != 0 {
			t.Fatal("invalid selection reached invocation")
		}
		if err = h.run.finish(err); err == nil {
			t.Fatal("invalid selection became complete input")
		}
		h.assertRetired()
	}
}

func TestGPUHostJoinsEveryDrainAndSharedError(t *testing.T) {
	h := newGPUHostHarness(t, []pod5.Read{gpuHostRead(1, 7002), gpuHostRead(2, 6000)})
	drain, shared, flush := errors.New("profile drain failed"), errors.New("shared close failed"), errors.New("output flush failed")
	h.drainErr, h.sharedErr, h.output.finalErr = drain, shared, flush
	err := h.execute()
	if !errors.Is(err, drain) || !errors.Is(err, shared) || !errors.Is(err, flush) || len(h.invocations) != 2 {
		t.Fatalf("cleanup errors lost: %v", err)
	}
	h.assertRetired()
}

func TestGPUHostPartialInitializationDrains(t *testing.T) {
	h := newGPUHostHarness(t, []pod5.Read{gpuHostRead(1, 7002)})
	original := h.run.newInvocation
	h.run.newInvocation = func(r *gpuRun, profile int) (scheduler.Invocation, error) {
		i, _ := original(r, profile)
		return i, errors.New("initialization failed after stream ownership")
	}
	if err := h.execute(); err == nil {
		t.Fatal("partial initialization succeeded")
	}
	if h.calls != 0 {
		t.Fatal("partially initialized invocation ran")
	}
	h.assertRetired()
}

func TestGPUHostRejectsMissingNativeProfile(t *testing.T) {
	h := newGPUHostHarness(t, []pod5.Read{gpuHostRead(1, 6143)})
	h.run.newInvocation = newGPUInvocation
	err := h.execute()
	if !errors.Is(err, ErrGPUUnqualified) || h.calls != 0 || h.run.report.CompletedParents != 0 {
		t.Fatalf("unavailable 6144 profile was promoted: %v", err)
	}
	h.assertRetired()
}

func TestGPUHostSUPRecipe(t *testing.T) {
	spec, _ := model.Pinned(model.SUP)
	h := newGPUHostHarness(t, nil)
	h.run.cfg.BatchSize = 32
	profiles, err := gpuSourceProfiles(spec, h.run.cfg)
	if err != nil || len(profiles) != 2 || profiles[1].chunkSize != 6144 {
		t.Fatalf("SUP fixed profiles: %+v %v", profiles, err)
	}
	h.run.cfg.Chunking = VariableChunks
	profiles, err = gpuSourceProfiles(spec, h.run.cfg)
	if err != nil || len(profiles) != 1 || profiles[0].chunkSize != 12288 {
		t.Fatalf("SUP packed profiles: %+v %v", profiles, err)
	}
}

func TestGPUOutputFASTQFinalFlushFailure(t *testing.T) {
	policy := postcall.DefaultPolicy()
	policy.Trim, policy.Split = false, false
	processed, err := ProcessGPURead(testSpec(), gpuFixture("AAAAA"), policy)
	if err != nil {
		t.Fatal(err)
	}
	w, err := newGPUOutput(gpuAlwaysShortWriter{}, GPUConfig{Format: FASTQ}, testSpec(), pod5.FileMetadata{})
	if err != nil {
		t.Fatal(err)
	}
	if err = w.write(processed); err != nil {
		t.Fatalf("small FASTQ should remain buffered until finalization: %v", err)
	}
	if err = w.finish(); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("final FASTQ flush failure lost: %v", err)
	}
}

func TestGPUHostSharedModelClosesAfterProfileDrains(t *testing.T) {
	for _, drainFailure := range []bool{false, true} {
		t.Run(fmt.Sprintf("drainFailure=%t", drainFailure), func(t *testing.T) {
			h := newGPUHostHarness(t, []pod5.Read{gpuHostRead(1, 6143), gpuHostRead(2, 6144)})
			shared := &model.HACGPU{}
			h.run.hac = shared
			if drainFailure {
				h.drainErr = errors.New("synthetic profile drain failed")
			}
			closeShared := h.run.closeShared
			closes := 0
			h.run.closeShared = func() error {
				closes++
				if h.run.hac != nil {
					t.Fatal("run retained successfully closed model")
				}
				if _, err := shared.Forward(context.Background(), nil, nil, model.HACGPUPlan{}, model.HACGPUBatch{ActiveSlots: 64}); !errors.Is(err, model.ErrClosed) {
					t.Fatal("shared model was not closed", err)
				}
				return closeShared()
			}
			err := h.execute()
			if drainFailure && !errors.Is(err, h.drainErr) || !drainFailure && err != nil {
				t.Fatal("drain status changed", err)
			}
			if closes != 1 || len(h.invocations) != 2 {
				t.Fatal("shared lifetime did not span both profiles", closes, len(h.invocations))
			}
			h.assertRetired()
		})
	}
}

// Private host records exercise publication only; they never enter the registry.
func enableGPUHostPhysicalDecode(h *gpuHostHarness) {
	h.physicalDecode = true
	h.run.bundle.record.profiles = make(map[gpuProfileKey]bool)
	h.run.bundle.record.partialProfiles = make(map[gpuProfileKey]bool)
	for _, p := range h.run.recipes {
		key := gpuProfileKey{string(h.run.spec.Architecture), int(p.chunkSize), h.run.cfg.BatchSize, false}
		h.run.bundle.record.profiles[key] = true
		h.run.bundle.record.partialProfiles[key] = true
	}
}

func TestGPUHostPartialPhysicalAccounting(t *testing.T) {
	for _, architecture := range []model.Architecture{model.HAC, model.SUP} {
		t.Run(string(architecture), func(t *testing.T) {
			reads := []pod5.Read{gpuHostRead(1, 7002), gpuHostRead(2, 7002), gpuHostRead(3, 7002)}
			h := newGPUHostHarness(t, reads)
			h.run.spec, _ = model.Pinned(architecture)
			if architecture == model.SUP {
				h.run.cfg.BatchSize = 32
			}
			enableGPUHostPhysicalDecode(h)
			if err := h.execute(); err != nil {
				t.Fatal(err)
			}
			if h.calls != 2 || h.real[0] != 2 || h.real[1] != 1 || h.eofAtCall[0] || !h.eofAtCall[1] {
				t.Fatalf("pre-EOF and EOF partial membership: real=%v eof=%v", h.real, h.eofAtCall)
			}
			st := h.run.profiles[0].scheduler.Stats()
			if st.Submitted != 3 || st.RealChunks != 3 || st.Dispatched != 2 || st.PaddedSlots != 2*h.run.cfg.BatchSize || st.PaddedSamples != 2*h.run.cfg.BatchSize*12288 || st.FailedChunks != 0 || st.Abandoned != 0 || st.Pending != 0 || !st.Drained {
				t.Fatalf("physical vs real accounting: %+v", st)
			}
			if h.run.report.ExpectedParents != 3 || h.run.report.InputParents != 3 || h.run.report.CompletedParents != 3 || h.run.report.SerializedParents != 3 || h.run.report.OutputRecords != 3 || len(h.output.ids) != 3 {
				t.Fatalf("absent decoded rows affected publication: %+v ids=%v", h.run.report, h.output.ids)
			}
			for j, read := range reads {
				if h.output.ids[j] != read.ID.String() {
					t.Fatal("real parent identity or order changed")
				}
			}
			h.assertRetired()
		})
	}
}

func TestGPUHostLoaderErrorAfterAcceptedParents(t *testing.T) {
	for _, accepted := range []int{1, 3} {
		t.Run(fmt.Sprintf("accepted%d", accepted), func(t *testing.T) {
			h := newGPUHostHarness(t, []pod5.Read{gpuHostRead(1, 7002), gpuHostRead(2, 7002), gpuHostRead(3, 7002), gpuHostRead(4, 7002)})
			enableGPUHostPhysicalDecode(h)
			loaderErr := errors.New("POD5 loader failed after accepted parent")
			h.input.failAfter, h.input.iterateErr = accepted, loaderErr
			if err := h.execute(); !errors.Is(err, loaderErr) {
				t.Fatalf("loader failure lost: %v", err)
			}
			published := accepted - 1
			if h.run.eof || h.run.report.InputParents != uint64(accepted) || h.run.report.CompletedParents != uint64(published) || h.run.report.SerializedParents != uint64(published) || h.run.report.OutputRecords != uint64(published) || len(h.output.ids) != published || h.calls != published/2 {
				t.Fatalf("loader failure flushed pending work or lost earlier output: %+v calls=%d", h.run.report, h.calls)
			}
			st := h.run.profiles[0].scheduler.Stats()
			if st.Submitted != accepted || st.RealChunks != published || st.Abandoned != 1 || st.FailedChunks != 0 || st.Pending != 0 || !st.Drained || st.PaddedSlots != h.calls*64 || st.PaddedSamples != h.calls*64*12288 {
				t.Fatalf("loader failure accounting: %+v", st)
			}
			h.assertRetired()
		})
	}
}

func TestGPUHostFinalPartialOutcomes(t *testing.T) {
	for _, name := range []string{"call-fails", "cancel-during", "cancel-after-publication", "writer", "drain", "finalize"} {
		t.Run(name, func(t *testing.T) {
			h := newGPUHostHarness(t, []pod5.Read{gpuHostRead(1, 7002), gpuHostRead(2, 7002), gpuHostRead(3, 7002)})
			enableGPUHostPhysicalDecode(h)
			switch name {
			case "call-fails":
				h.failCall = 2
			case "cancel-during":
				h.cancelCall = 2
			case "cancel-after-publication":
				h.cancelAfterCall = 2
			case "writer":
				h.output.failWrite = 3
			case "drain":
				h.drainErr = errors.New("final partial drain failed")
			case "finalize":
				h.output.finalErr = errors.New("final partial writer finalization failed")
			}
			err := h.execute()
			if err == nil || h.calls != 2 || h.real[1] != 1 || !h.eofAtCall[1] {
				t.Fatalf("final partial outcome lost: %v real=%v eof=%v", err, h.real, h.eofAtCall)
			}
			if strings.HasPrefix(name, "cancel-") && !errors.Is(err, context.Canceled) {
				t.Fatal("cancellation cause lost", err)
			}
			st := h.run.profiles[0].scheduler.Stats()
			failed := name == "call-fails" || name == "cancel-during"
			wantReal, wantBatches, wantFailed := 3, 2, 0
			if failed {
				wantReal, wantBatches, wantFailed = 2, 1, 1
			}
			if st.Submitted != 3 || st.RealChunks != wantReal || st.Dispatched != wantBatches || st.FailedChunks != wantFailed || st.FailedBatches != wantFailed || st.PaddedSlots != wantBatches*64 || st.PaddedSamples != wantBatches*64*12288 || st.Abandoned != 0 || st.Pending != 0 || st.Drained != (name != "drain") {
				t.Fatalf("final partial scheduler accounting: %+v", st)
			}
			wantSerialized := uint64(3)
			if failed || name == "cancel-after-publication" || name == "writer" {
				wantSerialized = 2
			}
			wantCompleted := wantSerialized
			if name == "writer" {
				wantCompleted = 3
			}
			if h.run.report.InputParents != 3 || h.run.report.CompletedParents != wantCompleted || h.run.report.SerializedParents != wantSerialized || h.run.report.OutputRecords != wantSerialized || len(h.output.ids) != int(wantSerialized) {
				t.Fatalf("final partial publication accounting: %+v", h.run.report)
			}
			h.assertRetired()
		})
	}
}

func TestGPUHostZeroSelectedReads(t *testing.T) {
	h := newGPUHostHarness(t, nil)
	if err := h.execute(); err != nil {
		t.Fatal(err)
	}
	if !h.run.eof || h.calls != 0 || len(h.invocations) != 0 || len(h.output.ids) != 0 || h.run.report != (GPUReport{}) {
		t.Fatalf("zero reads created work: %+v", h.run.report)
	}
	h.assertRetired()
}
