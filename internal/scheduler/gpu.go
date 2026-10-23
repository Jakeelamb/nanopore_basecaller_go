// Package scheduler owns the host-side dispatch boundary for the pinned GPU
// caller.
//
// It forms explicit-geometry batches with internal/gpubatch, keeps real work
// separate from padded capacity, and hands each batch to one invocation owner
// that retains its device views, caches, scratch and stream until drain. The
// package imports no CUDA, tensor, cgo or model code. It creates no goroutines
// and uses synchronous primitives only: the pinned source's per-device CUDA
// thread and cross-caller task queue are not ported here.
package scheduler

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpubatch"
)

// Selection names how the batch size was chosen. Only an explicit size is
// ported. CudaCaller.cpp:410-505 derives a size from device memory and
// CudaCaller.cpp:516-664 benchmarks one; P01/P02 have not pinned that policy,
// so both requests are rejected instead of guessed.
type Selection string

const (
	// SelectionExplicit supplies the batch size and geometry directly.
	SelectionExplicit Selection = "explicit"
	// SelectionAutomatic is the source's requested_batch_size == 0 path, which
	// benchmarks batch sizes on the device.
	SelectionAutomatic Selection = "automatic"
	// SelectionMemory is the source's device-memory-derived batch limit.
	SelectionMemory Selection = "memory"
)

var (
	// ErrAutomaticSelection is the sentinel matched by AutomaticSelectionError.
	ErrAutomaticSelection = errors.New("scheduler: automatic batch or memory selection rejected")
	// ErrQueueFull reports a submission that would exceed the bounded queue.
	ErrQueueFull = errors.New("scheduler: bounded queue is full")
	// ErrClosed reports work offered after shutdown.
	ErrClosed = errors.New("scheduler: scheduler is closed")
	// ErrDrainInProgress reports a second concurrent flush or shutdown.
	ErrDrainInProgress = errors.New("scheduler: a drain is already in progress")
	// ErrInvocationFailed marks the sticky failure a drain leaves behind. Until
	// Resume is called, Flush and Shutdown refuse to drain again and return the
	// original invocation error wrapped with this sentinel.
	ErrInvocationFailed = errors.New("scheduler: an invocation failed")
	// ErrInvocationPanic reports an Invocation.Call that panicked. The batch is
	// failed as a whole, the panic value is reported and the stack is not kept.
	ErrInvocationPanic = errors.New("scheduler: invocation panicked")
)

// AutomaticSelectionError is returned for any request that asks the scheduler
// to derive a batch size or memory budget. Match it with errors.Is(err,
// ErrAutomaticSelection) or errors.As(err, *AutomaticSelectionError).
type AutomaticSelectionError struct {
	// Field names the automatic input that was requested.
	Field string
}

func (e *AutomaticSelectionError) Error() string {
	field := e.Field
	if field == "" {
		field = "batch geometry"
	}
	return fmt.Sprintf("%v: %s is not pinned; supply explicit geometry", ErrAutomaticSelection, field)
}

func (e *AutomaticSelectionError) Is(target error) bool { return target == ErrAutomaticSelection }

// Config fixes one bounded queue over one explicit batch geometry.
type Config struct {
	Geometry   gpubatch.Config
	MaxPending int
	Selection  Selection
}

// Validate rejects automatic selection, invalid geometry and an unbounded
// queue. A zero batch size is the source's automatic sentinel
// (CudaCaller.cpp:516-524), so it is rejected as automatic rather than invalid.
func (c Config) Validate() error {
	switch c.Selection {
	case SelectionExplicit:
	case SelectionAutomatic:
		return &AutomaticSelectionError{Field: "batch size"}
	case SelectionMemory:
		return &AutomaticSelectionError{Field: "device memory budget"}
	default:
		return fmt.Errorf("scheduler: batch selection %q is not declared; explicit geometry is required", string(c.Selection))
	}
	if c.Geometry.BatchSize == 0 {
		return &AutomaticSelectionError{Field: "batch size"}
	}
	if err := c.Geometry.Validate(); err != nil {
		return fmt.Errorf("scheduler: invalid batch geometry: %w", err)
	}
	if int64(c.Geometry.BatchSize)*int64(c.Geometry.ChunkSize) > gpubatch.MaxCoordinate {
		return fmt.Errorf("scheduler: padded batch capacity exceeds the native int domain")
	}
	if c.MaxPending <= 0 || c.MaxPending > gpubatch.MaxCoordinate {
		return fmt.Errorf("scheduler: invalid bounded queue capacity %d", c.MaxPending)
	}
	return nil
}

// Batch is one immutable batch handed to the invocation owner. Real counts the
// real chunks the source places; PaddedSlots and PaddedSamples are the capacity
// the explicit geometry reserves and are never counted as real work.
//
// PaddedSlots is the source N dimension. On a fixed lane it is also the real
// slot count, so a partial batch has Real below PaddedSlots. On a variable lane
// the packed sample bound is PaddedSamples, so Real may exceed PaddedSlots; the
// packed offsets stay inside PaddedSamples.
//
// Sequence is the dispatch ordinal for this scheduler, starting at 1. It is
// assigned immediately before Call, so batches skipped by cancellation or an
// earlier failure leave no gap.
type Batch struct {
	Sequence      int
	Lane          gpubatch.Lane
	Variable      bool
	Real          int
	PaddedSlots   int
	PaddedSamples int
	Placements    []gpubatch.Placement
}

// Invocation is the narrow owner boundary. One implementation owns the device
// views, caches, scratch and stream for its lane.
//
// Call runs exactly the batch it is given, returns only when that batch is
// complete, and returns an error instead of a partial result. Implementations
// retain their resources across calls until Drain returns. Call must not call
// Flush or Shutdown; those return ErrDrainInProgress while a drain is running.
// Drain is called once with the shutdown context; an owner that must release
// resources unconditionally should ignore cancellation or wrap it with
// context.WithoutCancel.
type Invocation interface {
	Call(ctx context.Context, b Batch) error
	Drain(ctx context.Context) error
}

// Stats is a snapshot. Dispatched counts only batches whose invocation returned
// nil; a failed batch contributes to FailedBatches and to none of the work
// counters, and the chunks it held are counted in FailedChunks and listed in
// FailedChunkIDs. Resume does not clear that accounting: it is the record of
// work the caller must fail. PaddedSlots and PaddedSamples are reserved capacity,
// not real chunks.
type Stats struct {
	Submitted      int
	Dispatched     int
	FailedBatches  int
	FailedChunks   int
	FailedChunkIDs []string
	RealChunks     int
	PaddedSlots    int
	PaddedSamples  int
	Abandoned      int
	Pending        int
	Closed         bool
	Drained        bool
}

// Scheduler forms and dispatches batches. It is safe for concurrent Submit
// while one drain runs; only one drain may run at a time.
type Scheduler struct {
	cfg Config
	inv Invocation

	mu          sync.Mutex
	pending     []gpubatch.Chunk
	pendingIDs  map[string]struct{}
	inflight    map[string]struct{}
	stats       Stats
	sequence    int
	closed      bool
	draining    bool
	shutdownErr error
	stickyErr   error

	// cur and formErr are the batch former. Both are guarded by mu: only the
	// drain path writes them, but plan reads them while Submit and Stats may run.
	cur     *gpubatch.Batch
	formErr error
}

// New validates the explicit configuration and prepares one batch former. The
// invocation owner must be non-nil.
func New(cfg Config, inv Invocation) (*Scheduler, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if inv == nil {
		return nil, errors.New("scheduler: invocation owner required")
	}
	cur, err := gpubatch.NewBatch(cfg.Geometry)
	if err != nil {
		return nil, fmt.Errorf("scheduler: invalid batch geometry: %w", err)
	}
	return &Scheduler{
		cfg: cfg, inv: inv, cur: cur,
		pendingIDs: make(map[string]struct{}),
	}, nil
}

// Submit appends real chunks in FIFO order. A submission is all-or-nothing: an
// invalid, duplicate or over-capacity chunk leaves the queue unchanged.
func (s *Scheduler) Submit(ctx context.Context, chunks ...gpubatch.Chunk) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return fmt.Errorf("%w: cannot submit", ErrClosed)
	}
	if len(chunks) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(chunks))
	for _, c := range chunks {
		if c.ID == "" {
			return fmt.Errorf("scheduler: chunk with empty ID")
		}
		if c.Length <= 0 || c.Length > s.cfg.Geometry.ChunkSize {
			return fmt.Errorf("scheduler: chunk %q length %d outside 1..%d", c.ID, c.Length, s.cfg.Geometry.ChunkSize)
		}
		if _, dup := seen[c.ID]; dup {
			return fmt.Errorf("scheduler: duplicate chunk %q in submission", c.ID)
		}
		if _, dup := s.pendingIDs[c.ID]; dup {
			return fmt.Errorf("scheduler: duplicate pending chunk %q", c.ID)
		}
		if _, dup := s.inflight[c.ID]; dup {
			return fmt.Errorf("scheduler: duplicate in-flight chunk %q", c.ID)
		}
		seen[c.ID] = struct{}{}
	}
	if len(s.pending)+len(chunks) > s.cfg.MaxPending {
		return fmt.Errorf("%w: %d pending, %d offered, limit %d", ErrQueueFull, len(s.pending), len(chunks), s.cfg.MaxPending)
	}
	s.pending = append(s.pending, chunks...)
	for _, c := range chunks {
		s.pendingIDs[c.ID] = struct{}{}
		s.stats.Submitted++
	}
	return nil
}

// Flush drains the real work that is pending when it is called. Work submitted
// during the drain waits for the next Flush. The trailing partially filled batch
// is dispatched explicitly and reported with Real below PaddedSlots; the source
// dispatches partial batches on its batch timeouts (CudaCaller.cpp:94-107).
//
// On cancellation the in-flight batch is allowed to finish. Batches not yet
// dispatched are returned to the front of the queue in order and ctx.Err() is
// returned. On an invocation error that batch is failed as a whole, later
// batches are returned to the queue and the first error is returned.
//
// The invocation error is sticky: it is recorded on the scheduler and every
// later Flush returns it, wrapped with ErrInvocationFailed, until Resume is
// called. The chunks the failed batch held are dropped from the queue and are
// listed in Stats().FailedChunkIDs; the caller must fail those reads.
func (s *Scheduler) Flush(ctx context.Context) error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return fmt.Errorf("%w: cannot flush", ErrClosed)
	}
	if s.draining {
		s.mu.Unlock()
		return ErrDrainInProgress
	}
	if s.stickyErr != nil {
		err := s.stickyLocked()
		s.mu.Unlock()
		return err
	}
	s.draining = true
	s.mu.Unlock()
	defer s.endDrain()
	return s.drainPending(ctx)
}

// Resume clears the sticky invocation failure so the next Flush or Shutdown may
// drain again. It does not repair the failed batch: the caller must still fail
// the reads whose chunks are in Stats().FailedChunkIDs. Chunks that were
// requeued behind the failure stay pending and are dispatched by the next drain.
// Resume is a no-op when no failure is recorded and reports ErrClosed once
// Shutdown has closed the scheduler, which never drains again.
func (s *Scheduler) Resume() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return fmt.Errorf("%w: cannot resume", ErrClosed)
	}
	s.stickyErr = nil
	return nil
}

// Shutdown stops new submissions, drains pending real work and then calls
// Invocation.Drain exactly once. A cancelled or failed drain abandons the work
// that never ran and reports it as Abandoned. A recorded invocation failure is
// sticky: Shutdown refuses to drain the remaining work, counts it as Abandoned,
// releases the invocation resources and returns the failure instead of reporting
// the hole as success. Repeated calls are idempotent and return the first
// shutdown result without draining again.
func (s *Scheduler) Shutdown(ctx context.Context) error {
	s.mu.Lock()
	if s.draining {
		s.mu.Unlock()
		return ErrDrainInProgress
	}
	if s.closed {
		err := s.shutdownErr
		s.mu.Unlock()
		return err
	}
	s.closed = true
	s.draining = true
	var stickyErr error
	if s.stickyErr != nil {
		stickyErr = s.stickyLocked()
	}
	s.mu.Unlock()
	defer s.endDrain()

	var flushErr error
	if stickyErr != nil {
		// Refuse the drain. The failed batch was already dropped and recorded;
		// whatever is still queued is abandoned below rather than dispatched
		// behind the caller's back.
		flushErr = stickyErr
	} else {
		flushErr = s.drainPending(ctx)
	}

	s.mu.Lock()
	s.stats.Abandoned = len(s.pending)
	s.pending = nil
	clear(s.pendingIDs)
	s.mu.Unlock()

	drainErr := s.inv.Drain(ctx)

	s.mu.Lock()
	s.stats.Closed = true
	s.stats.Drained = drainErr == nil
	s.shutdownErr = errors.Join(flushErr, drainErr)
	err := s.shutdownErr
	s.mu.Unlock()
	return err
}

// Stats returns a snapshot with the current pending count. The failed chunk list
// is copied, so the caller cannot mutate the scheduler's record.
func (s *Scheduler) Stats() Stats {
	s.mu.Lock()
	defer s.mu.Unlock()
	st := s.stats
	st.FailedChunkIDs = append([]string(nil), s.stats.FailedChunkIDs...)
	st.Pending = len(s.pending)
	return st
}

// stickyLocked wraps the recorded invocation failure. Callers must hold s.mu.
func (s *Scheduler) stickyLocked() error {
	return fmt.Errorf("%w: %w (call Resume to drain again)", ErrInvocationFailed, s.stickyErr)
}

func (s *Scheduler) endDrain() {
	s.mu.Lock()
	s.draining = false
	s.mu.Unlock()
}

// plannedBatch is one batch plus its chunk range in the flush snapshot, so a
// cancelled or failed drain can return the undispatched suffix in order.
type plannedBatch struct {
	batch    Batch
	from, to int
}

func (s *Scheduler) drainPending(ctx context.Context) error {
	s.mu.Lock()
	if len(s.pending) == 0 {
		s.mu.Unlock()
		return nil
	}
	// The snapshot keeps its IDs claimed until each batch is dispatched, so a
	// concurrent duplicate submission cannot slip into the gap.
	snapshot := s.pending
	s.pending = nil
	s.mu.Unlock()

	planned, err := s.plan(snapshot)
	if err != nil {
		s.requeue(snapshot)
		return err
	}
	for i := range planned {
		pb := &planned[i]
		if err := ctx.Err(); err != nil {
			s.requeue(snapshot[pb.from:])
			return err
		}
		b := pb.batch
		s.mu.Lock()
		s.sequence++
		b.Sequence = s.sequence
		s.inflight = make(map[string]struct{}, len(b.Placements))
		for _, p := range b.Placements {
			s.inflight[p.Chunk.ID] = struct{}{}
			delete(s.pendingIDs, p.Chunk.ID)
		}
		s.mu.Unlock()

		if err := s.dispatch(ctx, b); err != nil {
			s.requeue(snapshot[pb.to:])
			return err
		}
	}
	return nil
}

// dispatch runs one batch and always releases its in-flight claims, so a failed
// or panicking invocation cannot leave chunks permanently unsubmittable. A
// non-nil result fails the batch as a whole: its chunks are counted in
// FailedChunks and listed in FailedChunkIDs and the failure is recorded as
// sticky, so no later drain can report the missing work as success.
func (s *Scheduler) dispatch(ctx context.Context, b Batch) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%w: %v", ErrInvocationPanic, r)
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		s.inflight = nil
		if err == nil {
			s.stats.Dispatched++
			s.stats.RealChunks += b.Real
			s.stats.PaddedSlots += b.PaddedSlots
			s.stats.PaddedSamples += b.PaddedSamples
			return
		}
		s.stats.FailedBatches++
		s.stats.FailedChunks += len(b.Placements)
		for _, p := range b.Placements {
			s.stats.FailedChunkIDs = append(s.stats.FailedChunkIDs, p.Chunk.ID)
		}
		s.stickyErr = err
	}()
	return s.inv.Call(ctx, b)
}

// plan applies the source batch formation to the snapshot in order. It keeps
// one former across normal flushes so the HAC part accumulator survives, as the
// pinned worker thread does. cur and formErr are read and written under s.mu; the
// drain path is their only writer.
func (s *Scheduler) plan(chunks []gpubatch.Chunk) ([]plannedBatch, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.formErr != nil {
		return nil, s.formErr
	}
	cur := s.cur
	var planned []plannedBatch
	start := 0
	emit := func(end int, placements []gpubatch.Placement) {
		if len(placements) > 0 {
			planned = append(planned, plannedBatch{batch: s.newBatch(placements), from: start, to: end})
		}
		start = end
	}
	for i, c := range chunks {
		if cur.Full() {
			emit(i, cur.Placements())
			cur.Reset()
		}
		prev := cur.Placements()
		flushed, err := cur.Add(c)
		if err != nil {
			return nil, fmt.Errorf("scheduler: chunk %q: %w", c.ID, err)
		}
		if flushed {
			emit(i, prev)
		}
	}
	emit(len(chunks), cur.Placements())
	cur.Reset()
	s.cur = cur
	return planned, nil
}

func (s *Scheduler) newBatch(placements []gpubatch.Placement) Batch {
	g := s.cfg.Geometry
	return Batch{
		Lane:          g.Lane,
		Variable:      g.VariableChunks,
		Real:          len(placements),
		PaddedSlots:   g.BatchSize,
		PaddedSamples: g.BatchSize * g.ChunkSize,
		Placements:    placements,
	}
}

// requeue returns undispatched chunks to the front of the queue and restarts
// batch formation. The HAC part accumulator cannot be rewound, so only normal
// flush boundaries preserve it.
func (s *Scheduler) requeue(chunks []gpubatch.Chunk) {
	if len(chunks) == 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	rest := make([]gpubatch.Chunk, 0, len(chunks)+len(s.pending))
	rest = append(rest, chunks...)
	rest = append(rest, s.pending...)
	s.pending = rest
	for _, c := range chunks {
		s.pendingIDs[c.ID] = struct{}{}
	}
	next, err := gpubatch.NewBatch(s.cfg.Geometry)
	if err != nil {
		s.formErr = fmt.Errorf("scheduler: rebuild batch formation: %w", err)
		return
	}
	s.cur = next
}
