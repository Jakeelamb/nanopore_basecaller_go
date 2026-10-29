# Current ABI5 CPU acceptance

The current Go CPU caller reproduces both frozen HG002 development controls
with the ABI5 backend after the native thread-initialization fix. The full port
remains unfinished. This is one parent read, not a cohort, biological-accuracy,
or performance result. HG005 was not used for selection.

## Current contract and results

The builder emits ABI5, the loader requires ABI5, and the maintained CPU and
CUDA verifier metadata guards require ABI5. CUDA replay explicitly rejects
historical ABI4 receipts. Historical ABI3 and ABI4 evidence remains valid only
at its original pins.

The pre-fix ABI5 CPU-only backend passed all ten maintained checker commands,
including ordinary, no-cgo, ABI3 rejection, native, direct-C ABI, race, and vet.
Its HAC CLI then aborted at foreign-thread Tanh with OpenMP Error 34, EAGAIN,
and SIGABRT. Those contracts did not establish caller acceptance.

The corrected common CPU boundary establishes process settings once and calls
public `at::init_num_threads()` once per entering OS thread. It changes no ABI,
exports, operation, or arithmetic and requires no permanent Go thread locking.
The corrected 1,024-element probe exposed the old backend's foreign OpenMP
maximum of 24 and task growth from 2 to 25. Explicit initialization gave maximum
1 and two tasks. The old four-foreign-thread Go regression also aborted.
The fixed probe and Go overlay pass with four distinct foreign TIDs, Tanh output
checks, and exact identity MatMul checks. MKL query symbols were unavailable;
no measured MKL maximum is claimed.

The earlier million-element probe initialized threads during its operation and
passed on the old backend. `thread-probe-001` is a failed reproduction, not fix
evidence. All attempts remain retained.

The fixed library is
`/home/jake/Projects/nanopore_basecaller_go_worktrees/cpu-thread-native-20261028/artifacts/tensor/thread-init-cpu-001/libnanopore_tensor.so`.
Its SHA256 is `b11d22c1f840a51b6d9aea83284cbf6af856f5edbe8d6693522bce86f1096430`.
The current production `backend.cpp` and `runtime.h` match that build's inputs.
The final regression source SHA256 is
`16fd3b18177d04e5b0eb86a2e1e3ac2d8cbecdedead7efe42d91486cd692b5ee`.
It uses the existing `NANOPORE_TENSOR_BACKEND` opt-in; no extra flag is needed.
The final maintained native checker passes all ten commands after the thread fix.
Its native log records this final test passing on four distinct foreign OS threads.
The ordinary suite passes 767 root tests with 76 skips; its no-cgo run passes
677 roots with 57 skips. The tensor native run passes 66 roots with seven skips
for separate domains and controls. The foreign-thread test does not skip in
that native run. Race and vet also pass.

Fresh HAC and SUP CLI attempts pass with `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
and `OMP_THREAD_LIMIT` explicitly unset. All retained controller runs use the
original 2-CPU, 3-GiB, zero-swap, Tasks64 limits. No caps were increased.

| Model | Core bases | Raw FASTQ SHA256 |
| --- | --- | --- |
| HAC | 1653 | `53ec2ca57acff8ce0041576a6bade2724a623110421867827032990d8df860cd` |
| SUP | 1650 | `dbb28c99d9c29d9808bc27c56f96a1057cfdf0b5fe197ac940e898d380b994c0` |

Both comparisons report exact UUID, header, sequence, qualities, full raw bytes,
and capture replay against unchanged stock controls. Stock FASTQ has no move
table, so replay exactness does not qualify native moves. Cached postprocessing
reuses these fresh captures without repeated inference. It passes all four flag
modes for both models, manifest write and sync failures, 12 rejected CLI
captures, the execution trace, and immutable-input checks. Default outputs remain
1,615 HAC bases and 1,619 SUP bases, with the historical default hashes unchanged.

## Commands and raw receipts

Build and check a new CPU-only ABI5 output with the
[tensor procedure](tensor-backend.md#rebuild-and-verify), then use that same
library in the [selected-read CLI](caller.md#run-one-selected-read). These
procedures require the pinned local Torch archive and extracted tree, Dorado
source, model configs and weights, POD5 file, and POD5 library. They perform no
implicit dependency acquisition. The maintained `verification/caller/run.sh`
defaults to the current CPU-only ABI5 build output,
`artifacts/tensor/abi5-cpu/libnanopore_tensor.so`. Set `TENSOR_LIBRARY` explicitly
to use another qualified library.

The script also passes with `TENSOR_LIBRARY` unset and no existing caller output
directory. That fresh HAC run uses the documented default library path and matches
the same 1,653-base stock FASTQ and capture replay. Its outer receipt is canonical
`cpu-script-default-001/`; the attempt is retained under sibling worktree
`cpu-caller-script-polish-20261028/artifacts/caller/attempt-20261028T222651Z-qU3vGD/`.

The following are reproducible command forms, not executions by this
documentation assignment. Use fresh output directories and controller admission
for native work.

```sh
TENSOR_LIBRARY="$CPU_TENSOR_LIBRARY" verification/caller/run.sh

# Select the matching pinned SUP model and retained stock control for SUP.
MODEL_DIR=/home/jake/Projects/nanopore_basecaller/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0 \
STOCK_CONTROL=/home/jake/Projects/nanopore_basecaller_go/artifacts/native-cpu-control/stock-sup-dev1-001/stdout \
TENSOR_LIBRARY="$CPU_TENSOR_LIBRARY" verification/caller/run.sh

go run ./verification/caller/defaultcmd \
  --hac "$PWD/artifacts/caller/attempt-20261028T182310Z-zi6O9n/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json" \
  --sup "$PWD/artifacts/caller/attempt-20261028T182316Z-BECAlF/capture/bb76c31a-36f0-48a0-abd2-107a10e1ade0/manifest.json" \
  --stock /home/jake/Projects/nanopore_basecaller_go/artifacts/native-cpu-control \
  --output artifacts/caller/default-abi5-new
```

Primary-worktree paths below resolve from this repository root. Canonical paths
resolve from
`/home/jake/Projects/nanopore_basecaller_go/artifacts/pstack-herdr-polish-20261028/`.

| Evidence | Retained location |
| --- | --- |
| Old metadata red overlays and seven focused green roots | Canonical `cpu-check-001/` |
| Ten-command pre-fix native pass | `artifacts/tensor/abi5-cpu-native-check-001/manifest.json` and canonical `cpu-check-001/native.*` |
| Ten-command fixed native pass and final foreign-thread regression | `artifacts/tensor/abi5-cpu-native-final-002/manifest.json` and canonical `cpu-check-final-002/` |
| Initial final-check lock deferral with no child commands | `artifacts/tensor/abi5-cpu-native-final-001/manifest.json` and canonical `cpu-check-final-001/` |
| Original HAC abort | `artifacts/caller/attempt-20261028T175801Z-xlVpaS/` and canonical `cpu-caller-001/` |
| Failed broad probe, corrected old probe, fixed green | Canonical `thread-probe-001/`, `thread-probe-1024-001/`, and `thread-fixed-001/` |
| Current HAC raw call, comparator, capture | `artifacts/caller/attempt-20261028T182310Z-zi6O9n/` |
| Current SUP raw call, comparator, capture | `artifacts/caller/attempt-20261028T182316Z-BECAlF/` |
| Cached default comparisons, controls, trace, input pins | `artifacts/caller/default-current-abi5-001/summary.json` and sibling receipts |
| Actual outer logs, exits, clocks, effective limits | Canonical `cpu-caller-002/` |

Each CLI attempt retains `call.command`, `call.started`, `call.finished`,
`call.exit`, `call.stdout`, `call.stderr`, `compare.command`, `compare.stdout`,
and immutable capture manifests. The default run retains per-step JSON receipts,
raw logs, source snapshots, and before/after input pins. Generated evidence,
signals, weights, and native distributions stay ignored. Reuse requires matching
all declared input pins.

## Bounded HAC native GPU reference

A separate fixed HAC native decoder reference is imported with status
`native_oracle_qualified`. The original capture remains
`native_captured_pending_supervisor`; the returned import authenticates it through
`SourceCapture`. The reference has score shape `[64,32,1024]`, 256 HAC states,
32 buffers across four stages, and five numeric identities.

The retained qualified import is
`/home/jake/Projects/nanopore_basecaller_go_worktrees/decoder-native-completion-20261028/artifacts/d01-current-20261028-001/4-check/check-capture.stdout`,
SHA256 `352657c6d661f10dd6d6b79f96e8a196c79c21384dfcf438951f715fd98300bb`.
Its `SourceCapture` SHA256 is
`9026a505ba4915c77071460a58d30824eb8e8d7b3fa8efe35d9c17b2b2023619`.
The bounded retained-evidence audit is
`/home/jake/.local/state/herdr-orchestrator/tasks/t-d28b39/report.md`, with its
`evidence-index.json` recording actual files, hashes, comparison slices, and
reliance on maintained imports for historical ancestry.

The rejected external-anchor digest corruption proves authentication rejection.
It is not a numerical-payload or mask control. No Go GPU numerical comparison,
SUP reference, full graph, partial or packed batch, whole caller, biological,
or performance qualification follows. The D01 card remains incomplete and
`approvedGPUBundles` remains empty. No production GPU bundle is approved.
