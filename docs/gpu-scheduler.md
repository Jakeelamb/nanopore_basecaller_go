# GPU batch scheduling and drain

Status is host-verified source implementation. No genuine GPU stream, device
buffer or model launch exists yet: the invocation owner is a Go interface and the
only implementation is a host-only recorder. This is not native parity, not GPU
qualification and not performance evidence.

`internal/scheduler` forms explicit-geometry batches with the accepted
`internal/gpubatch` unit, keeps real work separate from padded capacity, and hands
each batch to one invocation owner. `verification/gpucaller/schedule.go` replays a
host-only plan through that boundary and writes a deterministic receipt. Neither
package imports CUDA, tensor, cgo or model code.

## Source pins

- `basecall/CudaCaller.cpp:111-117,191-200` keeps one task queue per device, shared
  by callers, with the input lock held only around queue mutation.
- `basecall/CudaCaller.cpp:94-107` fixes the first/last chunk and low-latency batch
  timeouts, which are how the source dispatches a partially filled batch.
- `basecall/CudaCaller.cpp:211-214` fixes the batch multiple at 64 for LSTM and 32
  for transformer lanes.
- `basecall/CudaCaller.cpp:246-289` pushes one task, waits for its `done` flag,
  returns empty for `num_chunks == 0`, and rejects auxiliary data that disagrees
  with the variable-chunk mode.
- `basecall/CudaCaller.cpp:292-300` sets the terminate flag, notifies the queue and
  joins the CUDA thread. `restart` at `301-306` reuses the same boundary.
- `basecall/CudaCaller.cpp:308-342` keeps one pinned input/output/auxiliary backing
  allocation per runner, so an owner retains its buffers across batches.
- `basecall/CudaCaller.cpp:410-505` pads the requested batch size to the lane
  multiple and caps it by a device-memory estimate.
- `basecall/CudaCaller.cpp:516-664` benchmarks and selects a batch size when the
  request is zero. That automatic policy is not ported.
- `basecall/CudaCaller.cpp:668-748` is the CUDA thread loop: wait, run the module,
  retry once on `c10::Error`, signal `done`, pop.
- `read_pipeline/nodes/BasecallerNode.cpp:320-405,420-491` owns the source flush
  conditions already ported in [batch geometry](gpu-batch-geometry.md).

## Contract

`Config` carries one `gpubatch.Config`, a `MaxPending` bound on real chunks and a
`Selection`. Only `SelectionExplicit` is accepted. `SelectionAutomatic`,
`SelectionMemory` and a zero batch size (the source's "determine for us" sentinel)
return `*scheduler.AutomaticSelectionError`, which matches
`scheduler.ErrAutomaticSelection`. No automatic memory or batch policy is guessed.

`Batch` separates the work the source places from the capacity it reserves:

- `Real` is the number of real chunks in the batch.
- `PaddedSlots` is the source `N` dimension. On a fixed lane it is also the slot
  count, so a partial batch reports `Real < PaddedSlots`.
- `PaddedSamples` is `N * ChunkSize`, the packed sample bound. A variable lane is
  bounded by samples, so `Real` can exceed `PaddedSlots` while every packed offset
  stays inside `PaddedSamples`.

`Invocation` is the narrow owner boundary:

```go
Call(ctx context.Context, b Batch) error
Drain(ctx context.Context) error
```

`Call` runs exactly the batch it is given and returns only when that batch is
complete; it returns an error instead of a partial result. The implementation
retains its views, caches, scratch and stream across calls until `Drain`. The
scheduler never allocates, frees or inspects device state, and it creates no
goroutines: queue mutation, dispatch and drain are synchronous.

## Queue and drain rules

- `Submit` appends real chunks in FIFO order. A submission is all-or-nothing: an
  invalid length, an empty ID, a duplicate pending or in-flight ID, or an
  over-capacity submission leaves the queue unchanged. `ErrQueueFull` and
  `ErrClosed` are exported.
- `Flush` drains the work pending when it is called; work submitted during the
  drain waits for the next `Flush`. The trailing partial batch is dispatched
  explicitly and recorded as such. The source's time-based partial flush is not
  ported.
- Batch membership and ordering are deterministic. One `gpubatch.Batch` former is
  kept across normal flushes so the HAC part accumulator survives, as the pinned
  worker thread's does. `Batch.Sequence` is assigned immediately before `Call`.
- Cancellation allows the in-flight batch to finish. Batches not yet dispatched go
  back to the front of the queue in order and `ctx.Err()` is returned. A
  cancellation path restarts batch formation because the part accumulator cannot
  be rewound.
- An invocation error fails that batch as a whole, contributes nothing to
  `Dispatched`, `RealChunks`, `PaddedSlots` or `PaddedSamples`, stops the drain and
  is returned unchanged. Later batches are requeued.
- `Shutdown` refuses new work, drains, then calls `Invocation.Drain` exactly once.
  Work that never ran is counted in `Abandoned`. Repeated calls are idempotent and
  return the first shutdown result without draining again. A second concurrent
  drain returns `ErrDrainInProgress`.

## Verification unit

`ScheduleHandlers()` exposes host `Import` and `Replay` for the existing
`ScheduleCheck` unit; `Native` is nil. A plan pins the lane, chunk geometry, batch
size, selection, queue bound and an ordered step script (`submit`, `flush`,
`shutdown`) plus optional `FailAtBatch` and `CancelAtBatch` ordinals. The plan must
end with `shutdown`. The receipt records every step error, every dispatched batch
with its real and padded sizes, the drain count and the scheduler stats, and pins
the canonical plan digest and result digest. `Import` executes the plan twice and
refuses to write a receipt if the results differ; `Replay` re-executes the
embedded plan and rejects a changed plan, digest or result.

## Commands

```sh
go test -json -count=1 ./internal/scheduler
go test -json -count=1 -run TestSchedule ./verification/gpucaller
go test -race -count=1 ./internal/scheduler ./verification/gpucaller
go test -count=1 ./...
go vet ./...
```

The handler is registered by the P05 owner. Until then it runs through
`Dispatch` in tests and through the existing command as
`-unit schedule-check -mode import -input PLAN -out NEW_ATTEMPT` once wired. The
card's `schedule-check -plan PLAN -runtime-admitted` form is not implemented:
native scheduling stays unwired and host replay needs no admission.

## Host evidence

Receipts are under `artifacts/gpu-plan/gpu-scheduler-host-20261023T193547Z/`:
`scheduler-test.json`, `gpucaller-test.json`, `schedule-test.json` and a
`demo/` import/replay pair. 18 scheduler tests and 9 schedule-handler tests pass
with no failures or skips; the whole `verification/gpucaller` package reports 85
passing tests.

The tests cover explicit geometry with a full and a partial batch, real versus
padded accounting, deterministic repeat and split submissions, HAC stride rounding,
the HAC part accumulator across a flush, SUP variable capacity, and the negative
controls: automatic and memory selection, zero and invalid batch, over-capacity
queue, duplicate pending and in-flight membership, empty and invalid chunks,
cancel mid-batch, injected invocation error, repeated shutdown, drain failure,
re-entrant drain, empty input, late submission and goroutine stability. The
receipt tests cover rejection of automatic selection, invalid plans, unknown and
trailing JSON, tampered result, digest and plan, existing output and non-regular
input, and confirm a failed batch is never counted as successful work.

The demo receipt pins HAC `chunk=768 batch=64 stride=6` and reports two batches
(64 + 6 real chunks), `PaddedSlots=128`, `PaddedSamples=98304`, `Drains=1` and
`Abandoned=0`.

## Open

- No genuine stream case exists. The interface still needs an owner backed by the
  real views, scratch, stream and buffers, and a device run showing bounded
  memory, no deadlock or use-after-free and complete error propagation.
- The per-device cross-caller queue and dedicated CUDA thread
  (`CudaCaller.cpp:668-748`) are not ported; this unit has one owner and no device
  arbitration. The single `c10::Error` retry is also not ported.
- Automatic batch and memory selection (`CudaCaller.cpp:410-664`) remains
  rejected until P01/P02 pin the policy.
- Batch timeouts are not modelled: a partial batch is dispatched when the caller
  flushes, not after the source's first/last-chunk timers.
- The queue bound is a real-chunk count, not an allocator byte budget; native
  allocation accounting and OOM handling are unverified.
- Source-only evidence cannot close native acceptance. The card still needs the
  genuine stream case and independent review. No GPU, host lock, compiler, model
  file or native producer was touched for this checkpoint.
