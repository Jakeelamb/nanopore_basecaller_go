package scheduler_test

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpubatch"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/scheduler"
)

func hacFixed() gpubatch.Config {
	return gpubatch.Config{
		Lane: gpubatch.HAC, ChunkSize: 768, BatchSize: 64,
		Stride: 6, InnerStride: 6, Granularity: 6,
	}
}

func hacVariable() gpubatch.Config {
	cfg := hacFixed()
	cfg.VariableChunks = true
	return cfg
}

func supVariable() gpubatch.Config {
	return gpubatch.Config{
		Lane: gpubatch.SUP, ChunkSize: 768, BatchSize: 32,
		Stride: 6, InnerStride: 12, Granularity: 768,
		FirstConvPadding: 2, MaxConvPadding: 4, VariableChunks: true,
	}
}

func chunks(from, to, length int) []gpubatch.Chunk {
	out := make([]gpubatch.Chunk, 0, to-from)
	for i := from; i < to; i++ {
		out = append(out, gpubatch.Chunk{ID: fmt.Sprintf("c%03d", i), Length: length})
	}
	return out
}

func ids(from, to int) []string {
	out := make([]string, 0, to-from)
	for i := from; i < to; i++ {
		out = append(out, fmt.Sprintf("c%03d", i))
	}
	return out
}

type fakeCall struct {
	Sequence      int
	IDs           []string
	Placements    []gpubatch.Placement
	Real          int
	PaddedSlots   int
	PaddedSamples int
	Failed        bool
	Cancelled     bool
	Err           error
}

type fakeInvocation struct {
	mu       sync.Mutex
	calls    []fakeCall
	drains   int
	drainErr error
	failAt   int
	panicAt  int
	cancelAt int
	cancel   context.CancelFunc
	onCall   func(scheduler.Batch)
}

func (f *fakeInvocation) Call(ctx context.Context, b scheduler.Batch) error {
	rec := fakeCall{
		Sequence: b.Sequence, Real: b.Real,
		PaddedSlots: b.PaddedSlots, PaddedSamples: b.PaddedSamples,
		Placements: append([]gpubatch.Placement(nil), b.Placements...),
	}
	for _, p := range b.Placements {
		rec.IDs = append(rec.IDs, p.Chunk.ID)
	}
	if f.cancel != nil && b.Sequence == f.cancelAt {
		f.cancel()
		rec.Cancelled = true
	}
	if f.onCall != nil {
		f.onCall(b)
	}
	var err error
	if b.Sequence == f.failAt {
		err = fmt.Errorf("injected invocation failure at batch %d", b.Sequence)
		rec.Failed = true
	}
	rec.Err = err
	f.mu.Lock()
	f.calls = append(f.calls, rec)
	f.mu.Unlock()
	if b.Sequence == f.panicAt {
		panic(fmt.Sprintf("injected invocation panic at batch %d", b.Sequence))
	}
	return err
}

func (f *fakeInvocation) Drain(ctx context.Context) error {
	f.mu.Lock()
	f.drains++
	f.mu.Unlock()
	return f.drainErr
}

func (f *fakeInvocation) snapshot() ([]fakeCall, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]fakeCall(nil), f.calls...), f.drains
}

func newScheduler(t *testing.T, cfg gpubatch.Config, maxPending int, inv scheduler.Invocation) *scheduler.Scheduler {
	t.Helper()
	s, err := scheduler.New(scheduler.Config{
		Geometry: cfg, MaxPending: maxPending, Selection: scheduler.SelectionExplicit,
	}, inv)
	if err != nil {
		t.Fatalf("new scheduler: %v", err)
	}
	return s
}

func TestExplicitGeometrySeparatesRealWorkFromPaddedCapacity(t *testing.T) {
	inv := &fakeInvocation{}
	s := newScheduler(t, hacFixed(), 128, inv)
	if err := s.Submit(context.Background(), chunks(0, 70, 768)...); err != nil {
		t.Fatal(err)
	}
	if err := s.Flush(context.Background()); err != nil {
		t.Fatal(err)
	}
	calls, drains := inv.snapshot()
	if drains != 0 {
		t.Fatalf("drain ran before shutdown: %d", drains)
	}
	if len(calls) != 2 {
		t.Fatalf("got %d batches, want 2", len(calls))
	}
	if calls[0].Real != 64 || calls[0].PaddedSlots != 64 {
		t.Fatalf("full batch real=%d padded=%d", calls[0].Real, calls[0].PaddedSlots)
	}
	if calls[1].Real != 6 || calls[1].PaddedSlots != 64 {
		t.Fatalf("partial batch real=%d padded=%d", calls[1].Real, calls[1].PaddedSlots)
	}
	for i, call := range calls {
		if call.PaddedSamples != 64*768 {
			t.Fatalf("batch %d padded samples %d", i, call.PaddedSamples)
		}
		if !reflect.DeepEqual(call.IDs, ids(i*64, i*64+call.Real)) {
			t.Fatalf("batch %d membership %v", i, call.IDs)
		}
	}
	st := s.Stats()
	if st.Dispatched != 2 || st.RealChunks != 70 {
		t.Fatalf("stats %+v", st)
	}
	if st.PaddedSlots != 128 || st.PaddedSamples != 128*768 {
		t.Fatalf("padded capacity counted as real work: %+v", st)
	}
	if st.PaddedSlots <= st.RealChunks {
		t.Fatalf("partial batch capacity was not reported separately: %+v", st)
	}
}

func TestDeterministicMembershipAndOrder(t *testing.T) {
	run := func(split bool) []fakeCall {
		inv := &fakeInvocation{}
		s := newScheduler(t, hacFixed(), 256, inv)
		ctx := context.Background()
		if split {
			if err := s.Submit(ctx, chunks(0, 30, 768)...); err != nil {
				t.Fatal(err)
			}
			if err := s.Submit(ctx, chunks(30, 150, 768)...); err != nil {
				t.Fatal(err)
			}
		} else if err := s.Submit(ctx, chunks(0, 150, 768)...); err != nil {
			t.Fatal(err)
		}
		if err := s.Flush(ctx); err != nil {
			t.Fatal(err)
		}
		calls, _ := inv.snapshot()
		return calls
	}
	one, two := run(false), run(false)
	if !reflect.DeepEqual(one, two) {
		t.Fatalf("repeat run differs:\n%v\n%v", one, two)
	}
	split := run(true)
	if !reflect.DeepEqual(one, split) {
		t.Fatalf("split submission changed membership:\n%v\n%v", one, split)
	}
	seen := 0
	for i, call := range callsIDs(one) {
		if !reflect.DeepEqual(call, ids(seen, seen+len(call))) {
			t.Fatalf("batch %d order %v", i, call)
		}
		seen += len(call)
	}
	if seen != 150 {
		t.Fatalf("dispatched %d chunks, want 150", seen)
	}
}

func callsIDs(calls []fakeCall) [][]string {
	out := make([][]string, len(calls))
	for i, call := range calls {
		out[i] = call.IDs
	}
	return out
}

func TestVariableLaneUsesSourceRounding(t *testing.T) {
	inv := &fakeInvocation{}
	s := newScheduler(t, hacVariable(), 256, inv)
	lengths := []int{100, 101, 102, 767, 768}
	var batch []gpubatch.Chunk
	for i := 0; i < 70; i++ {
		batch = append(batch, gpubatch.Chunk{ID: fmt.Sprintf("v%03d", i), Length: lengths[i%len(lengths)]})
	}
	if err := s.Submit(context.Background(), batch...); err != nil {
		t.Fatal(err)
	}
	if err := s.Flush(context.Background()); err != nil {
		t.Fatal(err)
	}
	calls, _ := inv.snapshot()
	if len(calls) != 1 || calls[0].Real != 70 {
		t.Fatalf("variable batch %+v", calls)
	}
	// A variable batch is bounded by packed samples, not by the N dimension, so
	// more real chunks than PaddedSlots are legal while offsets stay inside it.
	if calls[0].PaddedSamples != 64*768 || calls[0].PaddedSlots != 64 {
		t.Fatalf("variable capacity %+v", calls[0])
	}
	packed := 0
	prev := -1
	for _, p := range calls[0].Placements {
		if p.Length%6 != 0 || p.Length < p.Chunk.Length {
			t.Fatalf("chunk %q padded to %d from %d", p.Chunk.ID, p.Length, p.Chunk.Length)
		}
		if p.Offset <= prev {
			t.Fatalf("packed offset %d did not advance past %d", p.Offset, prev)
		}
		prev = p.Offset
		packed = p.Offset + p.Length
	}
	if packed > calls[0].PaddedSamples {
		t.Fatalf("packed samples %d exceed capacity %d", packed, calls[0].PaddedSamples)
	}
}

func TestVariableFlushUsesSourcePartAccumulator(t *testing.T) {
	inv := &fakeInvocation{}
	s := newScheduler(t, hacVariable(), 256, inv)
	ctx := context.Background()
	if err := s.Submit(ctx, chunks(0, 70, 768)...); err != nil {
		t.Fatal(err)
	}
	if err := s.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	calls, _ := inv.snapshot()
	if len(calls) != 2 || calls[0].Real != 64 || calls[1].Real != 6 {
		t.Fatalf("variable flush batches %+v", calls)
	}
	if !reflect.DeepEqual(calls[1].IDs, ids(64, 70)) {
		t.Fatalf("variable flush order %v", calls[1].IDs)
	}
	if calls[1].Placements[0].Offset != 0 {
		t.Fatalf("flush did not restart packed offsets: %+v", calls[1].Placements[0])
	}
}

func TestRejectsAutomaticSelection(t *testing.T) {
	for _, sel := range []scheduler.Selection{scheduler.SelectionAutomatic, scheduler.SelectionMemory} {
		_, err := scheduler.New(scheduler.Config{Geometry: hacFixed(), MaxPending: 8, Selection: sel}, &fakeInvocation{})
		var automatic *scheduler.AutomaticSelectionError
		if !errors.As(err, &automatic) {
			t.Fatalf("%q: got %v, want AutomaticSelectionError", sel, err)
		}
		if !errors.Is(err, scheduler.ErrAutomaticSelection) {
			t.Fatalf("%q: sentinel not matched: %v", sel, err)
		}
	}
	cfg := hacFixed()
	cfg.BatchSize = 0
	_, err := scheduler.New(scheduler.Config{Geometry: cfg, MaxPending: 8, Selection: scheduler.SelectionExplicit}, &fakeInvocation{})
	var automatic *scheduler.AutomaticSelectionError
	if !errors.As(err, &automatic) {
		t.Fatalf("zero batch size: got %v, want AutomaticSelectionError", err)
	}
}

func TestRejectsInvalidGeometryAndQueue(t *testing.T) {
	negative := hacFixed()
	negative.BatchSize = -64
	unaligned := hacFixed()
	unaligned.BatchSize = 63
	badLane := hacFixed()
	badLane.Lane = 9
	for name, cfg := range map[string]gpubatch.Config{
		"negative-batch": negative, "unaligned-batch": unaligned, "lane": badLane,
	} {
		if _, err := scheduler.New(scheduler.Config{Geometry: cfg, MaxPending: 8, Selection: scheduler.SelectionExplicit}, &fakeInvocation{}); err == nil {
			t.Fatalf("%s geometry accepted", name)
		}
	}
	if _, err := scheduler.New(scheduler.Config{Geometry: hacFixed(), MaxPending: 8}, &fakeInvocation{}); err == nil {
		t.Fatal("undeclared selection accepted")
	}
	if _, err := scheduler.New(scheduler.Config{Geometry: hacFixed(), MaxPending: 0, Selection: scheduler.SelectionExplicit}, &fakeInvocation{}); err == nil {
		t.Fatal("unbounded queue accepted")
	}
	if _, err := scheduler.New(scheduler.Config{Geometry: hacFixed(), MaxPending: 8, Selection: scheduler.SelectionExplicit}, nil); err == nil {
		t.Fatal("nil invocation accepted")
	}
}

func TestBoundedQueueRejectsOverCapacity(t *testing.T) {
	s := newScheduler(t, hacFixed(), 4, &fakeInvocation{})
	ctx := context.Background()
	if err := s.Submit(ctx, chunks(0, 4, 768)...); err != nil {
		t.Fatal(err)
	}
	err := s.Submit(ctx, chunks(4, 5, 768)...)
	if !errors.Is(err, scheduler.ErrQueueFull) {
		t.Fatalf("got %v, want ErrQueueFull", err)
	}
	if st := s.Stats(); st.Pending != 4 || st.Submitted != 4 {
		t.Fatalf("over-capacity submission changed the queue: %+v", st)
	}
	err = s.Submit(ctx, chunks(4, 7, 768)...)
	if !errors.Is(err, scheduler.ErrQueueFull) {
		t.Fatalf("got %v, want ErrQueueFull", err)
	}
	if st := s.Stats(); st.Pending != 4 || st.Submitted != 4 {
		t.Fatalf("partial submission was enqueued: %+v", st)
	}
}

func TestDuplicateAndInvalidMembership(t *testing.T) {
	inv := &fakeInvocation{}
	s := newScheduler(t, hacFixed(), 16, inv)
	ctx := context.Background()
	dup := []gpubatch.Chunk{{ID: "a", Length: 768}, {ID: "a", Length: 768}}
	if err := s.Submit(ctx, dup...); err == nil {
		t.Fatal("duplicate in one submission accepted")
	}
	if err := s.Submit(ctx, gpubatch.Chunk{ID: "a", Length: 768}); err != nil {
		t.Fatal(err)
	}
	if err := s.Submit(ctx, gpubatch.Chunk{ID: "a", Length: 768}); err == nil {
		t.Fatal("duplicate pending chunk accepted")
	}
	for name, chunk := range map[string]gpubatch.Chunk{
		"empty-id": {ID: "", Length: 768},
		"zero":     {ID: "b", Length: 0},
		"negative": {ID: "c", Length: -1},
		"too-long": {ID: "d", Length: 769},
	} {
		if err := s.Submit(ctx, chunk); err == nil {
			t.Fatalf("%s chunk accepted", name)
		}
	}
	if st := s.Stats(); st.Pending != 1 || st.Submitted != 1 {
		t.Fatalf("rejected chunks changed the queue: %+v", st)
	}

	// A chunk inside the running batch is duplicate in-flight work, and a chunk
	// in a later batch of the same snapshot is still claimed.
	var inFlightErr, laterErr error
	var sched *scheduler.Scheduler
	inv2 := &fakeInvocation{}
	inv2.onCall = func(b scheduler.Batch) {
		if b.Sequence != 1 || inFlightErr != nil {
			return
		}
		inFlightErr = sched.Submit(ctx, gpubatch.Chunk{ID: b.Placements[0].Chunk.ID, Length: 768})
		laterErr = sched.Submit(ctx, gpubatch.Chunk{ID: "c064", Length: 768})
	}
	sched = newScheduler(t, hacFixed(), 128, inv2)
	if err := sched.Submit(ctx, chunks(0, 70, 768)...); err != nil {
		t.Fatal(err)
	}
	if err := sched.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	if inFlightErr == nil {
		t.Fatal("duplicate in-flight chunk accepted")
	}
	if laterErr == nil {
		t.Fatal("chunk claimed by a later batch of the snapshot was accepted")
	}
}

func TestCancelMidBatchDrainsInFlightWork(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	inv := &fakeInvocation{cancel: cancel, cancelAt: 1}
	s := newScheduler(t, hacFixed(), 128, inv)
	if err := s.Submit(ctx, chunks(0, 70, 768)...); err != nil {
		t.Fatal(err)
	}
	err := s.Flush(ctx)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	calls, _ := inv.snapshot()
	if len(calls) != 1 || !calls[0].Cancelled || calls[0].Failed {
		t.Fatalf("in-flight batch was not completed exactly once: %+v", calls)
	}
	st := s.Stats()
	if st.Dispatched != 1 || st.RealChunks != 64 || st.FailedBatches != 0 {
		t.Fatalf("cancelled drain reported partial work: %+v", st)
	}
	if st.Pending != 6 {
		t.Fatalf("undispatched work was not returned to the queue: %+v", st)
	}
	if err := s.Flush(context.Background()); err != nil {
		t.Fatal(err)
	}
	calls, _ = inv.snapshot()
	if len(calls) != 2 || calls[1].Sequence != 2 {
		t.Fatalf("drained batch sequence %+v", calls)
	}
	if !reflect.DeepEqual(calls[1].IDs, ids(64, 70)) {
		t.Fatalf("requeued order %v", calls[1].IDs)
	}
	if st := s.Stats(); st.RealChunks != 70 || st.Pending != 0 {
		t.Fatalf("second drain stats %+v", st)
	}
}

func TestInvocationErrorSticksAndRecordsFailedChunks(t *testing.T) {
	inv := &fakeInvocation{failAt: 2}
	s := newScheduler(t, hacFixed(), 256, inv)
	ctx := context.Background()
	if err := s.Submit(ctx, chunks(0, 130, 768)...); err != nil {
		t.Fatal(err)
	}
	err := s.Flush(ctx)
	if err == nil || err.Error() != "injected invocation failure at batch 2" {
		t.Fatalf("got %v, want the injected error", err)
	}
	calls, _ := inv.snapshot()
	if len(calls) != 2 || !calls[1].Failed {
		t.Fatalf("calls %+v", calls)
	}
	st := s.Stats()
	if st.Dispatched != 1 || st.FailedBatches != 1 {
		t.Fatalf("failed batch reported as success: %+v", st)
	}
	if st.RealChunks != 64 || st.PaddedSlots != 64 {
		t.Fatalf("failed batch work counted: %+v", st)
	}
	if st.FailedChunks != 64 || !reflect.DeepEqual(st.FailedChunkIDs, ids(64, 128)) {
		t.Fatalf("failed chunk accounting %+v", st)
	}
	if st.Pending != 2 {
		t.Fatalf("later batches were not returned to the queue: %+v", st)
	}

	// The failure is sticky: a later drain must not report the hole as success.
	err = s.Flush(ctx)
	if !errors.Is(err, scheduler.ErrInvocationFailed) {
		t.Fatalf("second flush: %v, want ErrInvocationFailed", err)
	}
	if !strings.Contains(err.Error(), "injected invocation failure at batch 2") {
		t.Fatalf("second flush lost the original error: %v", err)
	}
	if calls, _ := inv.snapshot(); len(calls) != 2 {
		t.Fatalf("sticky failure dispatched %d batches", len(calls))
	}
	if st := s.Stats(); st.Pending != 2 || st.Dispatched != 1 {
		t.Fatalf("sticky flush changed the queue: %+v", st)
	}

	// The record is a copy, so a caller cannot erase the failed chunk list.
	st = s.Stats()
	st.FailedChunkIDs[0] = "tampered"
	if got := s.Stats().FailedChunkIDs[0]; got != "c064" {
		t.Fatalf("stats snapshot aliases the scheduler record: %q", got)
	}

	if err := s.Resume(); err != nil {
		t.Fatalf("resume: %v", err)
	}
	if err := s.Resume(); err != nil {
		t.Fatalf("second resume: %v", err)
	}
	if err := s.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	calls, _ = inv.snapshot()
	if len(calls) != 3 || calls[2].Sequence != 3 || !reflect.DeepEqual(calls[2].IDs, ids(128, 130)) {
		t.Fatalf("recovered drain %+v", calls)
	}
	st = s.Stats()
	if st.Dispatched != 2 || st.RealChunks != 66 || st.Pending != 0 {
		t.Fatalf("recovered stats %+v", st)
	}
	if st.FailedBatches != 1 || st.FailedChunks != 64 || !reflect.DeepEqual(st.FailedChunkIDs, ids(64, 128)) {
		t.Fatalf("resume cleared the failure record: %+v", st)
	}
}

func TestShutdownRefusesDrainAfterFailure(t *testing.T) {
	inv := &fakeInvocation{failAt: 2}
	s := newScheduler(t, hacFixed(), 256, inv)
	ctx := context.Background()
	if err := s.Submit(ctx, chunks(0, 130, 768)...); err != nil {
		t.Fatal(err)
	}
	if err := s.Flush(ctx); err == nil {
		t.Fatal("injected failure was not reported")
	}
	err := s.Shutdown(ctx)
	if !errors.Is(err, scheduler.ErrInvocationFailed) || !strings.Contains(err.Error(), "injected invocation failure at batch 2") {
		t.Fatalf("shutdown after failure: %v", err)
	}
	calls, drains := inv.snapshot()
	if len(calls) != 2 {
		t.Fatalf("shutdown drained past the failure: %d calls", len(calls))
	}
	if drains != 1 {
		t.Fatalf("shutdown did not release the invocation: %d drains", drains)
	}
	st := s.Stats()
	if st.Abandoned != 2 || st.Pending != 0 || !st.Closed || !st.Drained {
		t.Fatalf("shutdown after failure stats %+v", st)
	}
	if st.Dispatched != 1 || st.FailedChunks != 64 {
		t.Fatalf("shutdown after failure work %+v", st)
	}
	if err := s.Resume(); !errors.Is(err, scheduler.ErrClosed) {
		t.Fatalf("resume after shutdown: %v", err)
	}
	if err := s.Shutdown(ctx); !errors.Is(err, scheduler.ErrInvocationFailed) {
		t.Fatalf("repeated shutdown lost the failure: %v", err)
	}
}

func TestPanickingInvocationReleasesClaims(t *testing.T) {
	inv := &fakeInvocation{panicAt: 2}
	s := newScheduler(t, hacFixed(), 256, inv)
	ctx := context.Background()
	if err := s.Submit(ctx, chunks(0, 130, 768)...); err != nil {
		t.Fatal(err)
	}
	err := s.Flush(ctx)
	if !errors.Is(err, scheduler.ErrInvocationPanic) || !strings.Contains(err.Error(), "injected invocation panic at batch 2") {
		t.Fatalf("got %v, want ErrInvocationPanic", err)
	}
	calls, _ := inv.snapshot()
	if len(calls) != 2 || !reflect.DeepEqual(calls[1].IDs, ids(64, 128)) {
		t.Fatalf("panic record %+v", calls)
	}
	st := s.Stats()
	if st.Dispatched != 1 || st.FailedBatches != 1 || st.FailedChunks != 64 || st.Pending != 2 {
		t.Fatalf("panic accounting %+v", st)
	}
	if !reflect.DeepEqual(st.FailedChunkIDs, ids(64, 128)) {
		t.Fatalf("panic failed chunks %+v", st.FailedChunkIDs)
	}

	// The panicked batch's claims are released, so its chunks can be resubmitted
	// once the caller has failed those reads.
	if err := s.Resume(); err != nil {
		t.Fatal(err)
	}
	if err := s.Submit(ctx, chunks(64, 65, 768)...); err != nil {
		t.Fatalf("chunk from the panicked batch is still claimed: %v", err)
	}
	if err := s.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	calls, _ = inv.snapshot()
	if len(calls) != 3 {
		t.Fatalf("recovered calls %+v", calls)
	}
	if !reflect.DeepEqual(calls[2].IDs, []string{"c128", "c129", "c064"}) {
		t.Fatalf("recovered order %v", calls[2].IDs)
	}
	if st := s.Stats(); st.RealChunks != 67 || st.Pending != 0 {
		t.Fatalf("recovered stats %+v", st)
	}
}

func TestShutdownDrainsAndIsIdempotent(t *testing.T) {
	inv := &fakeInvocation{}
	s := newScheduler(t, hacFixed(), 128, inv)
	ctx := context.Background()
	if err := s.Submit(ctx, chunks(0, 70, 768)...); err != nil {
		t.Fatal(err)
	}
	if err := s.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	calls, drains := inv.snapshot()
	if len(calls) != 2 || drains != 1 {
		t.Fatalf("shutdown calls=%d drains=%d", len(calls), drains)
	}
	if st := s.Stats(); !st.Closed || !st.Drained || st.RealChunks != 70 || st.Pending != 0 {
		t.Fatalf("shutdown stats %+v", st)
	}
	if err := s.Shutdown(ctx); err != nil {
		t.Fatalf("second shutdown: %v", err)
	}
	if _, drains := inv.snapshot(); drains != 1 {
		t.Fatalf("second shutdown drained again: %d", drains)
	}
	if err := s.Submit(ctx, chunks(70, 71, 768)...); !errors.Is(err, scheduler.ErrClosed) {
		t.Fatalf("submit after shutdown: %v", err)
	}
	if err := s.Flush(ctx); !errors.Is(err, scheduler.ErrClosed) {
		t.Fatalf("flush after shutdown: %v", err)
	}
}

func TestShutdownAbandonsUndispatchedWork(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	inv := &fakeInvocation{cancel: cancel, cancelAt: 1}
	s := newScheduler(t, hacFixed(), 256, inv)
	if err := s.Submit(ctx, chunks(0, 130, 768)...); err != nil {
		t.Fatal(err)
	}
	if err := s.Flush(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("flush: %v", err)
	}
	if err := s.Shutdown(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("shutdown: %v", err)
	}
	calls, drains := inv.snapshot()
	if len(calls) != 1 || drains != 1 {
		t.Fatalf("calls=%d drains=%d", len(calls), drains)
	}
	st := s.Stats()
	if st.Abandoned != 66 || st.RealChunks != 64 || st.Dispatched != 1 || st.FailedBatches != 0 {
		t.Fatalf("abandoned work reported as success: %+v", st)
	}
	if err := s.Shutdown(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("repeated shutdown lost the first error: %v", err)
	}
	if _, drains := inv.snapshot(); drains != 1 {
		t.Fatalf("repeated shutdown drained again: %d", drains)
	}
}

func TestShutdownReportsDrainFailure(t *testing.T) {
	drainErr := errors.New("stream release failed")
	inv := &fakeInvocation{drainErr: drainErr}
	s := newScheduler(t, hacFixed(), 8, inv)
	if err := s.Shutdown(context.Background()); !errors.Is(err, drainErr) {
		t.Fatalf("got %v, want the drain error", err)
	}
	if st := s.Stats(); st.Drained {
		t.Fatalf("failed drain reported as drained: %+v", st)
	}
}

func TestEmptyInputAndEmptyFlush(t *testing.T) {
	inv := &fakeInvocation{}
	s := newScheduler(t, hacFixed(), 8, inv)
	if err := s.Submit(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := s.Flush(context.Background()); err != nil {
		t.Fatal(err)
	}
	if calls, _ := inv.snapshot(); len(calls) != 0 {
		t.Fatalf("empty flush dispatched %d batches", len(calls))
	}
	if err := s.Shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestSubmitDuringFlushWaitsForNextDrain(t *testing.T) {
	var sched *scheduler.Scheduler
	inv := &fakeInvocation{}
	inv.onCall = func(b scheduler.Batch) {
		if b.Sequence == 1 {
			if err := sched.Submit(context.Background(), gpubatch.Chunk{ID: "late", Length: 768}); err != nil {
				t.Errorf("late submit: %v", err)
			}
		}
	}
	sched = newScheduler(t, hacFixed(), 128, inv)
	if err := sched.Submit(context.Background(), chunks(0, 64, 768)...); err != nil {
		t.Fatal(err)
	}
	if err := sched.Flush(context.Background()); err != nil {
		t.Fatal(err)
	}
	if st := sched.Stats(); st.Pending != 1 || st.RealChunks != 64 {
		t.Fatalf("late work joined the running drain: %+v", st)
	}
	if err := sched.Flush(context.Background()); err != nil {
		t.Fatal(err)
	}
	calls, _ := inv.snapshot()
	if len(calls) != 2 || !reflect.DeepEqual(calls[1].IDs, []string{"late"}) {
		t.Fatalf("late work not drained in order: %+v", calls)
	}
}

func TestReentrantDrainRejected(t *testing.T) {
	var sched *scheduler.Scheduler
	var flushErr, shutdownErr error
	inv := &fakeInvocation{}
	inv.onCall = func(b scheduler.Batch) {
		if b.Sequence != 1 {
			return
		}
		flushErr = sched.Flush(context.Background())
		shutdownErr = sched.Shutdown(context.Background())
	}
	sched = newScheduler(t, hacFixed(), 128, inv)
	if err := sched.Submit(context.Background(), chunks(0, 64, 768)...); err != nil {
		t.Fatal(err)
	}
	if err := sched.Flush(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !errors.Is(flushErr, scheduler.ErrDrainInProgress) {
		t.Fatalf("reentrant flush: %v", flushErr)
	}
	if !errors.Is(shutdownErr, scheduler.ErrDrainInProgress) {
		t.Fatalf("reentrant shutdown: %v", shutdownErr)
	}
	if st := sched.Stats(); st.Closed || st.RealChunks != 64 {
		t.Fatalf("reentrant drain changed state: %+v", st)
	}
}

func TestNoGoroutineLeak(t *testing.T) {
	baseline := runtime.NumGoroutine()
	for i := 0; i < 4; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		inv := &fakeInvocation{cancel: cancel, cancelAt: 1, failAt: 2}
		s := newScheduler(t, hacFixed(), 256, inv)
		if err := s.Submit(ctx, chunks(0, 130, 768)...); err != nil {
			t.Fatal(err)
		}
		_ = s.Flush(ctx)
		_ = s.Shutdown(ctx)
		cancel()
	}
	deadline := time.Now().Add(2 * time.Second)
	for runtime.NumGoroutine() > baseline && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	// A leak shows as growth. The baseline can include a transient runtime
	// goroutine that exits during the loop, so a lower count is not a leak.
	if got := runtime.NumGoroutine(); got > baseline {
		t.Fatalf("goroutines %d, baseline %d", got, baseline)
	}
}

func TestVariableSUPBatchGeometry(t *testing.T) {
	inv := &fakeInvocation{}
	s := newScheduler(t, supVariable(), 128, inv)
	ctx := context.Background()
	batch := make([]gpubatch.Chunk, 0, 70)
	for i := 0; i < 70; i++ {
		batch = append(batch, gpubatch.Chunk{ID: fmt.Sprintf("s%03d", i), Length: 500 + i%3})
	}
	if err := s.Submit(ctx, batch...); err != nil {
		t.Fatal(err)
	}
	if err := s.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	calls, _ := inv.snapshot()
	if len(calls) < 2 {
		t.Fatalf("SUP variable batch did not flush: %+v", calls)
	}
	total := 0
	for _, call := range calls {
		if call.Real > 32 {
			t.Fatalf("batch exceeds padded slots: %+v", call)
		}
		total += call.Real
	}
	if total != 70 {
		t.Fatalf("dispatched %d chunks, want 70", total)
	}
}
