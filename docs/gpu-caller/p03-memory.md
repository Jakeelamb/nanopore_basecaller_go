# P03: checked HAC GPU working-memory plan

Status is **source-only, host-verified**. This unit plans and checks the one
backing extent the HAC GPU caller reserves. It allocates no device memory,
imports no CUDA, compiles no native code, executes no model and takes no host
lock. It is not native parity, GPU qualification, end-to-end basecalling or
biological evidence, and it cannot close native acceptance.

## Source pins

Pinned Dorado revision `8b8fc5d36a9c0baab262a743cb175b5878e38ca6`.

| File | SHA-256 | Lines used |
| --- | --- | --- |
| `dorado/nn/WorkingMemory.cpp` | `4b512195a11e418ce35f8a60cf050dc499cec1b3c5504fa6ce8e21d6986e2b0d` | `30-33` `tensor_bytes`; `35-59` `next`; `61-79` `get_current_NTC_view`; `83-102` `next_TC`; `104-106` `temp`; `108-114` `allocate_backing_tensor` |
| `dorado/nn/include/nn/WorkingMemory.h` | `271f1c1e4451832eacfb8553a1f72902fa8795040c1e36aa282f08d214f1aa6e` | `55-82` class state, `ALIGNMENT`, `reservation_bytes`, `layout`, `is_input_to_rev_lstm`, `N/T/C` |
| `dorado/nn/ConvStack.cpp` | `d71cbdb1100010991753e4b3b80b8866265d97a08855f0b8ff1e41c00c91f6b2` | `191-211` `ConvLayer::reserve_working_memory` |
| `dorado/nn/FLSTMStack.cpp` | `97c2df7d81b7e8bb7c1d3c807ff4e80f39973df0bbc63cd2e2aa38d203a20160` | `97-109` `FLSTMStackImpl::reserve_working_memory` |
| `dorado/basecall/model/CRFModel.cpp` | `37f1f9712c3e61957ff17099d4cffb606c6cc784d2973c040bcc77129b5f9964` | `88-106` one `WorkingMemory` per invocation: reserve, then `allocate_backing_tensor` |

`verification/hacgpu.VerifySources` checks all five hashes before the oracle is
trusted. It was run against the pinned checkout with the maintained verification
run below; it reads files only.

## Rules the plan implements

- **256-byte alignment.** `tensor_bytes` pads every extent to
  `WorkingMemory::ALIGNMENT = 256` (`WorkingMemory.h:56-58`,
  `WorkingMemory.cpp:30-33`). `GPUMemoryReservation.Bytes` is therefore always a
  multiple of 256 and every view offset is one too.
- **Alternating ends.** The use phase starts at the front of the backing extent
  (`current` is undefined, so `current_is_front` is false). Each `next_TC` binds
  the end opposite the live current; a front-anchored current puts the new view
  at `reservation - new_bytes`, otherwise the new view starts at 0
  (`WorkingMemory.cpp:46-52`). The first view is at offset 0, then the sequence
  alternates front/back.
- **Reservation is a maximum, use is what is touched.** The reservation phase
  records `max(current_bytes + new_bytes)` (`WorkingMemory.cpp:38-40`), not a
  sum. `GPUMemoryReservation.PeakLiveBytes` is the same maximum without
  alignment padding, so a capture can separate reserved capacity from bytes the
  model actually touches. The HAC anchor case reserves 203008 bytes and peaks at
  202756 exact bytes.
- **Overlap is the source budget guard.** `WorkingMemory.cpp:42-44` throws
  `WorkingMemory: overlap detected.` when the live pair exceeds the reservation.
  With alternating-end placement that condition is exactly the condition that
  the new extent does not intersect the live current extent, so the plan keeps
  the source guard rather than inventing a second one.
- **CUTLASS is T+5, CUBLAS is T+3.** `next_TC` reserves `[T+5,N,C]` for
  `CUTLASS_TNC_F16` and `CUTLASS_TNC_I8`, `[T+3,N,C]` for `CUBLAS_TNC` and
  `[T+1,N,2,C]` for `CUBLAS_TN2C` (`WorkingMemory.cpp:91-98`). The comments at
  `ConvStack.cpp:60-63` and `LSTMStack.cpp:128` still describe CUTLASS as `T+3`;
  the code is `T+5`, so the plan pins storage from `next_TC`, not from those
  comments. Substituting `CUBLAS_TNC` for a reserved CUTLASS step is rejected.
- **Orientation and native binding.** `is_input_to_rev_lstm` selects the narrow
  start: reverse reads rows `[1,1+T)`, forward reads `[2,2+T)` for CUTLASS and
  `CUBLAS_TNC`, and rows `[1,1+T)` plane 1 / `[0,T)` plane 0 for `CUBLAS_TN2C`
  (`WorkingMemory.cpp:61-79`). The plan requires the direction to be stated per
  step, stores it on the view, and rejects a view bound with the other
  direction. `GPUNativeAllocation` admits a native extent only when it is at
  least the reservation and aligned to a positive multiple of 256.
- **Initialised sentinels and valid ranges.** `LSTMStack.cpp:135-140` and
  `FLSTMStack.cpp:236-237` zero rows `0` and `T+2`; `LSTMStack.cpp:74-78` zeroes
  row 0 plane 1 and row `T` plane 0 for `CUBLAS_TN2C`. `GPUView.RowLayout`
  declares the valid row interval, the zeroed sentinel rows and, for CUTLASS,
  rows `T+3` and `T+4` as source-uninitialised slack. These are declared before
  execution, not inferred from observed bytes.
- **Scratch lifetime.** `temp()` is never made current, and the next `next_TC`
  hands the same end of the backing store to a new tensor
  (`WorkingMemory.h:22-27`). A bound scratch view is invalidated as soon as a
  later view overlaps its extent; reading it afterwards is rejected.
- **One plan per invocation.** `CRFModel.cpp:88-106` constructs a `WorkingMemory`,
  runs the reservation traversal, allocates once and then replays. A
  `GPUMemoryReservation` is therefore per invocation: every `Use()` starts with
  an undefined current and binds the first view at the front.

## Code

- `internal/model/hac_gpu_memory.go`: `GPUMemoryPlan` (reservation traversal),
  `GPUMemoryReservation` (maximum extent, peak live bytes, native admission),
  `GPUMemoryUse` (checked replay, extents, scratch lifetime), `GPUView`
  (`NTCView`, `RowLayout`) and the six source layouts plus the flat `temp`
  extent. All arithmetic is overflow-checked.
- `verification/hacgpu/memory.go`: source-pinned pins, `VerifySources`, and a
  literal transliteration of `tensor_bytes`, `next`, `next_TC` and `temp` as an
  independent planner oracle. It imports no model code.
- `internal/model/hac_gpu_memory_test.go`: the differential and every control.

## Required test

```sh
go test -json -count=1 ./internal/model -run '^TestHACGPUMemory$'
```

Receipt at the committed head: exit 0, 25 `pass` events, 0 `fail`, 0 `skip`,
package elapsed 0.003s. `TestHACGPUMemory` runs all of the following:

| Subtest | What it pins |
| --- | --- |
| `SourcePlannerDifferential/hac-conv-cutlass-lstm` | HAC conv + CUTLASS int8 FLSTM scratch; 203008 reserved / 202756 live; offsets `0,178432,0,66816,0,0` |
| `SourcePlannerDifferential/hac-cublas-tnc-flstm` | `CUBLAS_TNC` half FLSTM scratch |
| `SourcePlannerDifferential/hac-cublas-tn2c` | interleaved `[T+1,N,2,C]`, odd `T`, reverse orientation |
| `SourcePlannerDifferential/conv-window-cutlass-i8` | `ConvStack.cpp:330,344-348` window TNC + int8 convert + CUTLASS input |
| `SourcePlannerDifferential/minimum-shapes` | every layout at `T=C=N=1`, one 256-byte block each |
| `ReservationVersusUse` | reservation is aligned, strictly above the exact live peak, and bounds every view |
| `Overflow` | shape, `next_TC` and `temp` arithmetic |
| `UnderReservation` | a later use step larger than the reserved pair |
| `Overlap` | an aligned pair that exceeds the reservation |
| `UndersizedPlan` | empty plan, short native extent, under-aligned native extent, short use, extra use step |
| `CUTLASSStorage` | CUTLASS `T+5` = 133 rows / 544768 bytes vs `CUBLAS_TNC` `T+3` = 131 rows / 536576 bytes at `N=2,T=128,C=1024` |
| `CUTLASSSubstitution` | CUTLASS reservation replayed as `CUBLAS_TNC` |
| `WrongReverseView` | reverse `NTCView` of forward storage; forward offset `+2*N*C*2` |
| `ReverseViewOffset` | `CUBLAS_TN2C` reverse row/plane offset and interleaved strides |
| `ScratchReuse` | scratch alive, then overwritten by the next current |
| `SentinelRows` | valid start, zeroed sentinels and CUTLASS unused rows per layout |
| `Alignment` | every layout's extent is 256-aligned, `LiveBytes <= Bytes`, padding `< 256` |

## Negative controls

Each control was proved effective by removing the guard, rerunning the required
command and observing a failure. The guard was then restored; the committed tree
is the restored one.

| Control | Guard removed | Observed failure |
| --- | --- | --- |
| `Overflow` | `gpuAdd`/`gpuMul`/`gpuPadTo` checks | `shape overflow: <nil>` |
| `UnderReservation` | `pair > reservation` in `GPUMemoryUse.bind` | `under-reservation accepted: <nil>` |
| `Overlap` | same budget guard | `overlapping pair accepted: <nil>` |
| `UndersizedPlan` | `a.Bytes < r.Bytes` in `AdmitNative` | `undersized native extent accepted: <nil>` |
| `UndersizedPlan` | alignment check in `AdmitNative` | `under-aligned native extent accepted: <nil>` |
| `UndersizedPlan` | `len(views) != reservation.Steps` in `Finish` | `short use accepted: <nil>` |
| `UndersizedPlan` | `u.step >= reservation.Steps` in `bind` | panic on the unreserved step index |
| `CUTLASSSubstitution` | layout comparison in `bind` | `CUTLASS reservation replayed as CUBLAS_TNC: <nil>` |
| `WrongReverseView` | `orientation != v.Orientation` in `NTCView` | `reverse view of forward storage accepted: <nil>` |
| `ScratchReuse` | overlap invalidation in `bind` | `reused scratch accepted: <nil>` |

The differential is also effective against a wrong source rule: changing CUTLASS
to `T+3`, `CUBLAS_TN2C` to `T+2`, or the alignment to 128 makes the
differential fail with a reservation mismatch against the oracle.

## Limits

- No device allocation, no native build, no CUDA compilation, no GPU run, no
  model execution, no host lock.
- The oracle is a host transliteration of the pinned planner. It is not the
  native `WorkingMemory` executing, and a Go test is not native parity.
- `VerifySources` was executed once against the pinned checkout; the pinned
  checkout is outside Git, so the committed tests do not read it. The pins are
  constants in `verification/hacgpu/memory.go`.
- Exact live-byte and sentinel accounting is prospective for a later capture. No
  capture has been compared yet.
- Automatic batch/memory selection, device qualification and every kernel that
  consumes these views remain with their own cards.
