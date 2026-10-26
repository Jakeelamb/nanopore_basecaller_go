# H06 half recurrent fallback candidate

Status: the independently reviewed source candidate is integrated into the
canonical feature worktree. The shared native backend compiled successfully
in `artifacts/tensor/attempt-h06-h07-shared-build-20261025-001/`.
No H06 recurrent native differential, GPU qualification or caller acceptance
has run. The original isolated development worktree remains preserved.

The source contract is pinned Dorado
`8b8fc5d36a9c0baab262a743cb175b5878e38ca6`,
`dorado/nn/FLSTMStack.cpp:239-295`. The fixed candidate uses seven C1024/K128
layers, first reverse, half `[T+3,N,1024]` input and the CUBLAS half operation.
Variable chunks are rejected by the source fallback. The pinned factorised
output projection also does not accept this half layout, so this standalone
fallback does not establish a supported end-to-end fallback model graph.

Go packs transposed half down matrices, concatenated/transposed up matrices,
and the input bias. The hidden bias is never used. Each timestep runs the
input-down GEMM, hidden-down GEMM, joint-up GEMM, then one hard-activation
`host_lstm_step_f16` cell update. The cell is reset for each layer. Go chooses
every layer, direction and row; no native function runs a recurrent layer.

| Direction | Input row | Hidden row | Output row |
| --- | --- | --- | --- |
| Reverse | `T-tick` | `T-tick+2` | `T-tick+1` |
| Forward | `2+tick` | `tick` | `1+tick` |

Rows 0 and T+2 are zeroed once. The seven directions alternate reverse,
forward, reverse, forward, reverse, forward, reverse. Final output occupies
rows `[2,T+2)`. Down scratch is half `[N,2,128]`: its two matrix views have
row stride 256, rather than two separate contiguous matrices. Up scratch is
half `[N,4096]`. The caller supplies the combined half temp owner with
`N*(256+4096)` elements. Input and temp must be contiguous, correctly shaped
and disjoint. Caches and cell have separate retained owners. Run and Close
are serialized; failure and cancellation cannot return a partial output as
success.

The new storage half GEMM and existing HalfBuffer GEMM share the identical
cuBLAS enqueue implementation. The cell boundary fixes C1024, out stride
1024, hard activation and the input bias. Both keep common CUDAWork drain
and retention behavior. Both new native operations are compiled into the
shared backend; neither has an H06 numerical runtime observation yet.

`artifacts/h06-host-20261025-001/` retained four passing model checks with
CGO off and those four plus two fake-ABI checks with CGO on. These check
cache roles, source row order, error/cancellation behavior and actual loader
argument marshalling without a GPU. The subsequent
`artifacts/h06-host-20261025-002/` checks the runtime object's Go compilation
and complete tensor host suite: four model roots and 25 tensor roots pass;
21 unconfigured native tensor roots skip. Current-source
`artifacts/h06-host-20261025-003/` includes the lifecycle correction: five
model roots pass with CGO off; those five and two fake-ABI roots pass with
CGO on, without required skips. Vet passes for both affected packages.
None of these counts are native parity.

The source/ABI review is retained in
`artifacts/parity-resume-review-20261024/h06-source-review.md`, and guarded
integration in `artifacts/h06-integration-20261025-001/receipt.json`.
The source observer preparation tests in
`artifacts/h06-oracle-host-20261025-001/` pass five host tests. They establish
reversible observation insertions and the bounded fixture transform, not
source execution.

The current host pass in `artifacts/h06-oracle-host-20261025-004/` has eight
passing roots, including a complete 936-record synthetic manifest and an
effective changed-payload rejection. The prepared fixture and immutable C++
source copies are in `artifacts/tensor/h06-prepared-002/`. The Go replay
compiles and compares 28 caches, 896 completed timestep buffers, seven layer
owners and three final outputs from one production recurrence call. It has
not executed on a GPU. The two native pre-cell records are retained separately
for cell/activation controls.

Remaining acceptance requires a pinned forced-half stock FLSTM oracle,
all declared cache/timestep/cell/output comparisons and effective F32,
activation, row-offset, initial-cell and recurrent-bias rejection controls.
The producer must preserve an untouched reference and any instrumented
observer equivalence. No source-model producer should be rerun merely to
format a report. All compiler/runtime/source/weight/input pins and raw
receipts must be retained prospectively under the canonical host lock.

## Closed compiler harness

The H06 build harness has separately reviewed compiler-only evidence. It
cannot execute either native target. Runtime capture remains held for the
shared S00 load guard rather than a second H06 guard.

`-unit controller-build -host-admitted -out NEW` creates a CGO0 controller with
actual Go dependency selection, the finite Go tool family, assembler include pins,
immutable source snapshots, a fresh private cache, and a forced rebuild.
It disables automatic PGO and VCS injection. Its `controller.json` seals the
new executable. Native phases require that exact executable to be running.
The tool family includes helpers such as pack and buildid even when a
particular build does not use them. It is conservative coverage, with actual
commands retained by `-x`. GOROOT configuration and version files are included
when present. A second dependency listing checks membership after the build.

`-unit build-plan` requires independent path and SHA256 pairs for `-prepared`,
`-compiler-contract`, `-link-contract`, and `-controller-build`, together with
`-build-dir` and a fresh `-out`. The compiler contract is the accepted D01
closed build plan. The link contract is the S00 host010 closed Torch build
plan. H06 reads those immutable contracts instead of recreating their
compiler selection or linker discovery. It rederives every selected compiler
ELF domain through `verification/nativeproof`, verifies the exact tools,
plugins and loader cache, and checks the current CMake selection.
Only queries for declared selected tools and link inputs are retained. The
S00 contract's unused static/shared alternatives are not promoted into H06
inputs by hashing them later. Resolved absolute link arguments remain intact.

The plan compiles reference main with `H06_OBSERVED=0`. It links untouched
stock FLSTMStack and WorkingMemory from the original archive. The observed
target compiles main with `H06_OBSERVED=1` plus the immutable observed FLSTM
copy. Its link must use stock WorkingMemory but reject a second archived
FLSTMStack implementation. The original stock library list and independently
closed bindings remain the source of link inputs.

Only `-unit oracle-build -plan PLAN -plan-sha SHA -compiler-admitted -out NEW`
can run native compiler commands. The gate requires root review and a fresh
grant. It takes the existing canonical nonblocking host lock and returns exit
75 when that lock is unavailable. The current cgroup must limit CPU to at
most two cores, memory to at most 3GiB, and swap to zero. Each child has a
180-second deadline and process-group cancellation.

The build verifies selected compiler and input bytes before and after each
query, preprocessing, compile and link command. All three `-M` dependency
sets are frozen before the first `-c`. Every actual `-MD` set must match its
own frozen set. Both link traces must use only declared inputs and the three
new objects. Link maps enforce the reference and observed implementation
choices. The receipt retains argv, exact child environment, clocks, exits,
raw logs, dependency files, object hashes, maps and the two executable hashes.
Compiler-only success leaves `RuntimeHeld=true`. It establishes no source
forward, Go/native numerical parity, GPU qualification or biological result.

The exercised host handoff is `artifacts/h06-builder-launch-20261025-003/`.
Its 19 Go test roots and vet pass. The fresh CGO0 controller in
`artifacts/tensor/h06-controller-build-003/` seals 942 inputs and immutable
snapshots. Its actual plan in `artifacts/tensor/h06-build-plan-003/` binds
2,012 inputs, three compilation units, two targets and 27 selected queries.
The same controller rejects a self-rehashed observed-source copy with one
arithmetic byte changed. The original prepared source remains unchanged.
Two earlier plan-adapter failures remain retained. No H06 C++ query,
preprocessing, compilation, link, target load or GPU computation ran in this
host handoff.

The separately admitted compiler attempt is
`artifacts/tensor/h06-native-build-001/build.json`, SHA256
`e7f8f9ffc8f44c160bceca21dcc3d7134dad5f6355c54cd54f73b180efcc5211`.
Its 27 selected queries, three header preflights, three compiles and two links
all exit zero. The three frozen `-M` sets equal their actual `-MD` sets:
5,232 headers/dependencies for each main and 5,155 for the observed FLSTM TU.
The retained link maps and static ELF symbols confirm stock FLSTM in the
reference, the direct observed FLSTM copy in the observed target, and stock
WorkingMemory in both. Neither executable has run.

The metadata audit in
`artifacts/h06-compiler-launch-20261025-001/audit-002.json`, SHA256
`4da61b149ce11c7507c4942c52a1cb95cb1767d7368260ee60481a8b764c631c`,
rehashes all 7,295 declared inputs after the build, totaling 2,891,181,609 bytes,
with zero drift. Raw cgroup files verify two CPUs, 3GiB, zero swap and 64
tasks. The outer service has a 600-second limit and control-group cleanup;
the builder retains its 180-second child limit. Its outer wrapper returned
one because it counted the preexisting supervising `uv` process as a build
descendant. That failed observation remains preserved. Actual controller and
all compiler exits are zero; retained PID identity and the absent final
control group resolve the wrapper error without rerunning the build.

The compiler handoff is
`artifacts/h06-compiler-handoff-20261025-001/handoff.json`. Runtime still needs
the shared audited-load guard, effective source/ATen failure-lifetime controls,
the untouched-reference/observer equivalence check, and all native/Go numeric
comparisons and negatives. The local stock cell owner may unwind before the
observer retains it when an early emit fails. This is a coverage gap, not a
demonstrated CUDA use-after-free. A future runtime consumer must authenticate
the retained producer binary and its original source-snapshot mapping while
declaring its own current consumer closure. The old build manifest remains
unchanged when unrelated Go code is integrated.
