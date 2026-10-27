# D01 fixed HAC staged decoder oracle preparation

This unit prepares a verification-only producer over retained H01 scores. The
first standalone producer and the guarded CUPTI v2 producer passed separately
admitted compiler queries, header freeze, and compile/link. Both executables
remain unexecuted. The current Go guard wiring is a source/host candidate;
native loading and GPU execution remain held. The
four original Koi functions are opaque reference calls. They do not count as a
Go decoder port, and no native fixture or decoder parity is claimed here.

## Input and exact fixed contract

The input is H01 `hacgpu-capture-008/stages/stage-scores.bin`, with the existing
H01 capture and receipt pins checked by the preparation command. Its logical
layout is contiguous half NTC `[64,32,1024]`, element strides `[32768,1024,1]`,
4,194,304 bytes, SHA256
`56b26176b4272906612eb75690087182a72527f94a9171d9937e5dd80b0d12db`.
The source backing offset 524,288 is capture provenance. The oracle copies the
logical payload to a distinct working allocation at offset zero.

This first experiment accepts finite original half values only and rejects any
change to the source payload pin. That is an explicit experimental restriction,
not a claim about Koi's general NaN/Inf behavior. Packed chunks, SUP, other N/T/C,
nondefault beam width, changed strides and move padding are rejected by the
closed preparation contract. Separate experiments need separate contracts.

The source descriptor for chunk i is exactly four little-endian int32 values
`[i*32,32,i*32,0]`. There are 64 descriptors, 1,024 bytes total. The producer
checks every word before its first CUDA call.

The actual HAC model config supplies qscore scale 1.02 and bias -0.3.
`BasecallModelConfig.cpp:64-68` reads these as float; `CudaCaller.cpp:138-139`
assigns q_scale/q_shift. `DecodedChunk.h:15-22` supplies W=32, cut=100, blank=2,
move_pad=false. `Decoder.cpp:20` selects score clamp zero because the pinned HAC
config has no explicit clamp sublayer. The complete config is checked against
H01's retained config pin rather than selected by a model-name heuristic.

| Operand | Exact float32/int32 bits |
| --- | --- |
| score clamp | `00000000` |
| q_scale | `3f828f5c` |
| q_shift | `be99999a` |
| beam cut | `42c80000` |
| fixed stay score | `40000000` |
| beam width / move_pad | `32 / 0` |

`params.bin` is the eight-byte ASCII magic `D01FIX01`, followed by eleven
little-endian uint32 words. They are version 1, N, T, C, W, the five float bit
patterns above in table order, then move_pad. Its size is exactly 52 bytes.

## Capacity and prospective comparison masks

S=C/4=256 and U=N*(T+1)=2,112. The following capacities are source allocations,
not inferred from observed output sizes.

| Buffer | Bytes | Source initialization |
| --- | ---: | --- |
| chunks | 1,024 | exact descriptors |
| results | 2,048 | uninitialized |
| working scores | 4,194,304 | complete immutable-input copy |
| aux guides | 2,162,688 | all 33 guide rows per chunk written by back guide |
| aux history | 270,336 | uninitialized capacity; population not established |
| path | 8,448 | zero |
| moves, sequence, qstring | 2,048 each | zero |

Aux is one 2,433,024-byte allocation. Chunk i guide rows begin at
`4*(i*32+i)*256`; there are 33 float32 rows. History begins at byte 2,162,688.
These address equations and guide terminal/reverse-loop stores come from the
pinned SM120 PTX documented in `d01-decoder-investigation.md`.

After back guide, compare only the complete guide interval inside aux and no
result bytes. After beam and later stages, compare result words 0, 1 and 3 only,
using per-chunk spans `[32*i,8]` and `[32*i+12,4]`. The original PTX beam epilogue
stores those words. Other result fields remain raw-only here even though the
final kernel has additional stores, because their conditional/error paths and
semantics need further review. No history slot mask is inferred from data.

Chunks, scores, path and byte outputs have complete initialized storage from
the source's descriptors/copy/zero operations. Preserve and compare stage-local
bytes without interpreting zero tails as emitted sequence. Retain all eight
result words as raw bits; do not cast provisional fields into an accepted public
result struct. All raw aux/results bytes are retained, including excluded bytes.
The contract records exclusions before any producer execution.

## Producer and failure custody

The current `verification/decodergpu/native/oracle.cpp` adds guarded phase
checkpoints and passive image observations to the earlier compiled producer.
This revision passed the genuine CUDA/Koi/CUPTI build retained under
`artifacts/tensor/d01-v2-build-001/`. Its executable has not run.
It validates fixed parameters, descriptor words, score byte count, finite half
domain and the exact input digest before any CUDA operation. It retains
`scores-original.f16` separately from the device working allocation.

It calls back guide, beam search, posterior and decode exactly once each, in
that order, with null optional weights and move_pad zero. Posterior overwrites
working scores in place. After every completed call it checks returned status,
launch status and completion, then retains every complete buffer. `buffers.tsv`
records stage, role, byte count, SHA256 and raw file. Stage events are flushed
before calls and after completed captures. An immutable host input is checked
again after all stages. Input and snapshot host buffers live outside the throwing
scope and remain owned through the final stream drain. A failed final drain
flushes the failure log and exits without unwinding host or device owners.
Device records are reserved before allocation so recording an allocation cannot
throw and lose its owner. Each raw file is explicitly closed and its final stream
state checked before its manifest row or successful capture event is written.

The SHA helper is copied verbatim from the existing H01 oracle helper, avoiding
an extra OpenSSL dependency. This new translation unit still requires separate
source review and an admitted compile; host Go tests cannot prove its syntax or
native behavior. Device capability checks do not establish which embedded
SM120/SM120a cubin or PTX image the CUDA loader selected. That remains an explicit
runtime qualification gate.

## Closed build and controller plan

The preparation command reads CMake metadata through the existing read-only
`hacgpu.OracleToolchainForSource` parser, selecting CUDADecoder.cpp flags and the
retained stock link line. It pins direct files, the selected libkoi archive,
matching koi.h, source and metadata without executing a native tool. Every
native command in its output plan is marked held.

Before any genuine compilation, a separately admitted build phase must close:

1. Compiler, cc1plus, assembler, collect2 and linker executable bytes, loaders and
   runtime DSOs; specs, multilib/search paths and all CRT/start/end objects.
2. Exact `-M` dependency output including system headers, frozen before compile.
   Actual `-MD` dependencies must match; added or changed files fail the attempt.
3. Every direct archive/DSO and `-l` reference, including references wrapped by
   `-Wl`. Expand transitive GNU GROUP/INPUT/AS_NEEDED/INCLUDE scripts before
   compile. Unknown linker tokens or trace inputs are failures.
4. `-Wl,-t` and link-map output checked against already frozen inputs and newly
   compiled objects. Do not retroactively add an observed library to make a
   failed closure pass. Reuse the H07 depfile/link-trace parsers read-only.
5. The prospective ELF interpreter and complete transitive runtime mapping,
   including CUDA runtime and driver. Retain actual maps and hashes, environment,
   GPU UUID/architecture and selected-image proof before stage admission.

The Go preparation controller is built independently with CGO disabled. It
records `go list -deps -json` selected Go/Cgo/assembly/header/embedded files,
freezes those bytes before `go build`, checks them afterward, records the Go
executable/environment and build command, and retains the resulting executable
with its digest. The bootstrap executable is retained separately. Native build
and capture require the frozen controller, source and binary pins to match.
Preparation executes no native tool. The maintained controller below requires
a coordinator release for each native phase.

## Maintained compiler controller

`-unit plan-native` reads the prepared finite case and the executing retained Go
controller, then copies the oracle and SHA helper into immutable source files.
Its single TU uses only the selected Koi and CUDA headers. Explicit link inputs
are the original pinned `libkoi.a` and dynamic `libcudart.so` already present in
the stock link line. The TU calls no Torch API and owns its buffers directly, so
stock Torch/model archives would add unused dependencies. Archive/kernel bytes
and the four original Koi calls stay unchanged. Implicit C++/libc/libm/libgcc and
startup inputs are still frozen explicitly.

A direct read of the retained `decoder_lib.cu.o` symbol table supports this
choice: its undefined symbols are CUDA registration/launch functions, C++ guard
and iostream initialization, unwind/personality, exit/atexit and linker globals.
There are no Torch or cuBLAS imports in that member. Its fixed member hash is
`930b35afa9e352b412b58e07a43d693630280ed9f422dc324106b2b8e700758f`;
the complete archive remains pinned separately. This is a static link rationale,
not proof that the genuine link or CUDA runtime has succeeded.

Before any compiler query, the plan reads the qualified GCC15 distribution
manifest and closes the selected driver, cc1plus, collect2, assembler, linker,
nm, C driver/cc1, LTO tools, explicit LTO plugin and finite BFD plugin family.
Shared `verification/nativeproof` reads ELF metadata without executing code.
D01 adds loader-cache and tool/plugin policy pins. Prospective explicit search
does not emulate glibc cache/hwcaps or interposition; unsupported or ambiguous
choices must reject.

The compiler family spans separate executables and plugin loader contexts.
Each selected tool/plugin root retains its own ELF nodes and dependency bindings;
their canonical path/SHA pins form a checked union. This preserves both the
qualified GCC15 and system BFD plugins even when their SONAME matches. Ambiguous
dependency choices inside a root's closure still reject. The decoder runtime is
a separate single-process closure.

Three separately admitted phases consume externally pinned immutable receipts:

1. `query` confirms prospective program and implicit file selections.
2. `headers` requires the query receipt, runs complete system-header `-M`, and
   freezes the before-compile input map. It never runs `-c`.
3. `build` requires that header receipt, compiles once with `-MD`, and requires
   the same dependency set. Link `-t` and map inputs must already be frozen; the
   map must identify original `libkoi.a(decoder_lib.cu.o)`. Unknown inputs fail
   rather than being added afterward.

The selected tool/plugin/ELF proof is rebuilt before and after each child,
including failure. Launches use a clean fixed environment, the existing
nonblocking canonical lock, fresh attempts and raw argv/clocks/exits/logs. A
180-second process-group cap bounds each child. Before child logs are created,
the launcher checks cgroup v2 CPU quota at most two cores, memory at most 3 GiB
and zero swap. Admission, resource limits and the lock remain independent.

Go preparation now snapshots every selected source, assembly, header, embed and
tool before building its controller. Native entry checks original current bytes,
immutable copies, actual metadata/build command roles and the running binary.
Historical source custody can authenticate snapshots independently of later
consumer source edits.

Maintained command forms follow. Genuine phases remain held:

```sh
FROZEN_CONTROLLER -unit plan-native -preparation PREPARED_JSON \
  -preparation-sha256 PREPARED_SHA -out artifacts/tensor/d01-native-plan-NEW
FROZEN_CONTROLLER -unit query -plan PLAN_JSON -plan-sha256 PLAN_SHA \
  -admitted -out artifacts/tensor/d01-query-NEW
FROZEN_CONTROLLER -unit headers -plan PLAN_JSON -plan-sha256 PLAN_SHA \
  -prior QUERY_JSON -prior-sha256 QUERY_SHA -admitted \
  -out artifacts/tensor/d01-headers-NEW
FROZEN_CONTROLLER -unit build -plan PLAN_JSON -plan-sha256 PLAN_SHA \
  -prior HEADERS_JSON -prior-sha256 HEADERS_SHA -admitted \
  -out artifacts/tensor/d01-build-NEW
FROZEN_CONTROLLER -unit plan-runtime -build BUILD_JSON -build-sha256 BUILD_SHA \
  -out artifacts/tensor/d01-runtime-NEW
```

`plan-runtime` is post-link host-only metadata. It freezes the executable,
dynamic CUDA runtime and finite driver/compute/JIT dlopen families, recursively
closes ELF dependencies and binds the cache. Actual file-backed mappings must
agree before the first decoder stage and after each stage. A pure host checker
rejects unknown or deleted mappings.

Capture remains held for source/host review and a separate root native grant.
The candidate `-unit capture` requires the exact v3 runtime, independently
pinned guard plan/phase contract, accepted immutable native producer and current
consumer. `-admitted` alone cannot bypass those checks. No new model inference
is needed.

The installed `/opt/cuda/include/cupti_callbacks.h` declares
`CUPTI_CBID_RESOURCE_MODULE_LOADED` and `CUpti_ModuleResourceData` with module ID,
cubin size and cubin pointer. Its documentation limits those bytes to the
callback invocation, so the observer must copy them before returning. Reading
this header is not a CUPTI load or proof that a callback observed a kernel.

## Guarded observation candidate

The producer's phase pipe uses the reviewed S00 wire protocol, with audit
requests on FDs 3/4 and main-thread phases on FDs 5/6. D01 does not implement
another loader auditor or Go process-guard framework. It uses the accepted
`supgpu.ExecuteGuarded` and `ReplayGuarded` API with this exact phase list:

1. `pre-model`, after fixed input validation and before CUPTI or any CUDA API.
2. `pre-forward`, after observer subscription and input upload completion.
3. `post-back-guide`, after the first original call, synchronization and snapshots.
4. `post-beam-search`, after the second original call and observations.
5. `post-posterior`, after the third original call and observations.
6. `post-staged`, after the fourth original call and observations.

The existing auditor rendezvous must approve each selected object before its
initializers. Every phase must retain actual executable mappings and enforce
the separately frozen D01 provider policy. The D01 root, auditor, CUPTI,
dynamic CUDA runtime/driver and CPU providers form its own process closure.
The S00 qualified provider evidence is provenance for provider choices, not an
allowlist permitting unused Torch or cuDNN code in this standalone process.
Unknown code, a rejected provider tier, a missing auditor, or out-of-order
phase denies the run. This source phase does not authorize a loader execution.

`native/image_observer.inc` subscribes before the first CUDA API. It enables
only `CUPTI_ACTIVITY_KIND_FUNCTION`, which has no timestamp fields, and the
four driver launch callback forms declared by the installed CUPTI headers.
The resource callback copies each module's bytes while the callback owns them.
Launch callbacks copy context pointer/UID, correlation ID, callback site and
kernel name. Activity completion copies `CUpti_ActivityFunction` IDs, context,
module ID, symbol index and name using CUPTI's documented buffer iterator.
No CUDA API runs inside any callback. Every callback catches exceptions and
records a failure instead of throwing across the C callback boundary.

Storage is bounded before CUDA initialization: sixteen modules of at most
1 MiB each, 512 function records, 512 launch records and eight 64-KiB activity
buffers. Exhaustion, truncated module bytes, invalid records or dropped records
fail qualification. A truncated failure payload is explicitly narrower evidence;
it is never accepted as a complete image. `cuptiActivityFlushAll(0)` runs outside
callbacks after each original stage synchronization. No timing activity, warmup,
extra kernel or extra decoder call is added.

Each stage retains complete module bytes plus module/function/launch TSV files
and callback status. The Go importer requires one successful enter/exit launch
pair for each original kernel, in the original four stage positions. It joins
the launch context UID/name to exactly one FUNCTION record, then joins that
record's module ID and the launch's context pointer to the copied module. The
function index must equal the prospective symbol index in that pinned cubin.
Missing, unsupported or ambiguous FUNCTION collection leaves a failed report
with any narrower module-only observation retained. No symbol-uniqueness guess,
device-SM assertion or mismatch-derived exclusion releases the gate.

Both permitted direct-cubin candidates come from the pinned original selected
decoder member. The SM120 hash is above. A selected-object-only extraction of
SM120a is retained in `artifacts/decoder-static-20261025-sm120a-001/receipt.json`.
Its image SHA256 is
`8c5d3abd2702a48d99a4c6f69e47de6235f5e280c468c38d74a910fda5acea40`.
The exact command was `cuobjdump --extract-elf decoder_lib.sm_120a.cubin
decoder_lib.cu.o`, with absolute input/tool paths in the receipt. No other
archive member was inspected. A module digest matching neither original cubin
is retained as failure, including a different JIT-produced image; it is not
relabeled as a proven PTX selection.

The new native plan has schema v2 and explicitly pins/links dynamic CUPTI along
with its transitive ELF inputs before queries. Its immutable source set contains
the oracle, SHA helper, phase gate and image observer. The earlier v1 build
retains its original two-source set and original recipe.

## Immutable producer and current consumer

The accepted compiler handoff is
`artifacts/decodergpu-host-20261025/closed-controller/compiler-handoff-001.json`,
SHA256 `3404c922788c2bad01dccbb632d3262b79917d2a571b38f5d90faec83ff2e645`.
Its original oracle SHA256 is
`cdfa90f321ee3ad520c0cc6c8f84a56ce60720fa0e62f335f7f7304422f7eb94`.
It passed 21 resource selections, a 295-file prospective/actual dependency
match, 1,790 frozen inputs, and a closed link trace/map identifying the original
decoder member. Its runtime metadata plan is
`artifacts/tensor/d01-runtime-plan-001/runtime.json`, SHA256
`dccfc1355b0a2207d521f586973e832ecda85498278d6dc9e1c2c935cb095bbe`.
These artifacts remain immutable and unexecuted.

Later Go ABI and observer-source edits cannot be checked against those old
live-source pins. `CheckProducerConsumer` authenticates the old build through
its original selected Go snapshots, native source copies, receipts, dependency
files, objects, trace/map and executable bytes. Only the exact native origins
listed by that build schema can resolve to their pinned immutable copies.
Other parameters, scores, configuration, archive, tools and header inputs must
still match. It separately authenticates a freshly prepared current consumer
and the actual executing consumer binary. Changed old snapshots, missing
bindings or changed non-source inputs reject. This does not permit the old
ungated executable to serve as the new observed producer; the v2 candidate
requires a fresh admitted native build after review.

## Host checks and remaining admission

Use bounded host resources for preparation and tests.

```sh
CGO_ENABLED=0 GOMAXPROCS=2 GOMEMLIMIT=3GiB \
  go test ./verification/decodergpu/... -count=1
CGO_ENABLED=1 GOMAXPROCS=2 GOMEMLIMIT=3GiB \
  go test ./verification/decodergpu/... -count=1
CGO_ENABLED=0 GOMAXPROCS=2 GOMEMLIMIT=3GiB \
  go run ./verification/decodergpu/cmd -out artifacts/tensor/d01-preparation-001
```

Host rejection controls cover changed shape/stride/scalar/descriptor/mask bytes,
short/corrupt/nonfinite half scores, hidden linker references, transitive scripts
and post hoc link inputs. They establish preparation behavior only. They do not
execute or qualify the genuine C++ producer. An additional opt-in host fixture
includes the actual producer with fake CUDA/Koi headers and implementations. It
tracks borrowed host allocations and fails on release before a successful drain.
Planned controls inject H2D/D2H enqueue and completion errors, a failed final
drain, and a buffered raw capture close failure through `/dev/full`; the latter
must leave no manifest row or successful capture event. A deliberate early-owner
release checks that the detector is effective. These controls require a separate
small host compiler release and do not link CUDA/Koi or access a device:

```sh
D01_FAKE_HOST_TEST=1 D01_FAKE_CXX=/usr/bin/c++ CGO_ENABLED=0 \
  GOMAXPROCS=2 GOMEMLIMIT=3GiB go test ./verification/decodergpu \
  -run '^TestNativeProducerFakeHostFailures$' -count=1 -v
```

This fixture passed all eight controls in the bounded host test receipt
`artifacts/decodergpu-host-20261025/review-correction/host-controls-001/receipt.json`.
H2D and D2H enqueue/completion errors each reached a successful final drain
before all eight device frees. The failed final drain retained its borrowed
snapshot and performed no device frees; the intentional early-release control
exited 91. Both direct and full-capture `/dev/full` close failures were observed,
with no successful manifest row/capture event. The whole command exited zero
under CPUQuota 200%, MemoryMax 3G and MemorySwapMax 0, using 205.4 MiB peak memory.
Its source pins remained unchanged during execution. A final repeat against the
fully frozen source/doc set is retained under `host-controls-002` in that same
correction directory. Ordinary tests skip this fixture without
explicit opt-in. Genuine compiler queries, CUDA preprocessing/building, native
runtime loading and GPU admission remain separate held phases.

Before a later native launch, root must separately release tool queries and
preprocessing, inspect the resulting full before-compile closure, release actual
compile/link, and then release a bounded device experiment under the canonical
host lock. Retain immutable argv/environment/clocks/exits/raw logs and executable
custody for each phase. No model inference rerun is needed.

Native algorithm controls still required after a genuine positive capture
include all-stay/no-emission, literal ties/near-ties/collisions and cut equality,
beam boundaries, short tails, probability/Q rounding, both move_pad settings,
consecutive batches and poison/error handling. Their fixtures and masks are not
silently admitted by this first fixed preparation.

## Historical host preparation receipts

The last reviewed preparation is `artifacts/tensor/d01-preparation-005/preparation.json`,
status `host_prepared_native_held`, with the corrected producer and then-current
shared dependencies. Its outer receipt is
`artifacts/decodergpu-host-20261025/preparation-wrapper-005/receipt.json`.
It exited zero under CPUQuota 200%, MemoryMax 3G and MemorySwapMax 0 and retained
source pins unchanged. Only Go dependency metadata, Go environment metadata and a
CGO-disabled Go controller build executed; all native proposals remain held.
Exact final source, receipt and retained controller digests are recorded in
`artifacts/decodergpu-host-20261025/review-correction/source-handoff-005.json`.

Historical host preparation is `artifacts/tensor/d01-preparation-003/preparation.json`,
with status `host_prepared_native_held`. The outer command exited zero in
`artifacts/decodergpu-host-20261025/preparation-wrapper-003/receipt.json`.
It records 84 direct input pins and 926 selected Go source/tool/header pins.
Only Go dependency metadata, Go environment metadata and a CGO-disabled Go build
executed. All 25 listed native command proposals remain held.

The retained Go controller is `artifacts/tensor/d01-preparation-003/decoder-controller`,
SHA256 `5a452adea0962a19b20e33f7f8786d3c1ce6f5b22748919ccf882fddd8bb3a1d`.
Go tools compile, asm and link, plus the Go assembler include directory, are
pinned. Dynamically linked Go tools are rejected until their runtime closure is
explicit. PGO is disabled; the controller build retains `-x` tool output.

Preparation001 predates the extra Go tool/header pins. Preparation002 failed
before controller build because it required a nonexistent separate buildid tool.
The actual installed tool inventory was then used. Both attempts remain retained.
The maintained compiler controller is source-only candidate work. Historical Go
closures predate it and cannot authorize its execution. A new preparation and
exact source/host review are required before genuine compiler execution.

Independent source review found preparation003 stale against subsequently changed
declared pins in `internal/tensor/abi.h`, `internal/tensor/loader.h` and
`verification/hacgpu/h07/consumer.go`. It cannot be reused against the current
tree. Preserve it unchanged. After shared dependencies freeze and the host
resource hold is released, a fresh preparation must pin the corrected producer,
actual selected Go inputs and retained controller. Host controls and a new
Go-only preparation are now admitted; every genuine CUDA phase remains held.
Controls001 and preparation004 both succeeded, but subsequent documentation
updates changed a pin additionally declared by their external wrapper. They
remain historical receipts, with the document-only mismatch recorded in
`review-correction/document-only-drift.json`. Final controls002/preparation005
use documentation frozen before execution; do not waive any declared input.

Documentation supplies context, not decoder parameters. It is absent from the
preparation's computational `Inputs` and actual selected Go `Pins`. The source
handoff preserves the exact document bytes seen by controls002/preparation005
under `review-correction/context/d01-decoder-oracle-at-005.md`, with the original
path/hash recorded as context metadata. A final document snapshot is separate.
The external wrapper now snapshots documentation into each fresh attempt and
pins that immutable context file; it excludes the mutable source document from
computational `source_pins`. Future report edits therefore require neither a Go
controller rebuild nor an oracle recapture. All actual code, parameter, scalar,
model, tool and runtime pins remain strict.

## Compiler-only review checkpoint

Preparation008 is a retained source/host checkpoint after the first shared ELF repair
and model metadata freeze. Its JSON is
`artifacts/tensor/d01-preparation-008/preparation.json`, SHA256
`b22c18b0421e9bdc791d74202bb02591a722bbe552fe0be00b341b7b22cdac6d`.
The retained controller SHA256 is
`8c4245b1ab51a5ef908038c31f652f254805f054a53c899c36ca3d572740d181`.
It binds 40 preparation inputs and 936 actual selected Go inputs to 936 snapshot
bindings (932 unique immutable files), including the actual GOMOD selection.

The host-only native plan is `artifacts/tensor/d01-native-plan-002/plan.json`,
SHA256 `9f5d83e5fc5009917e579bf28e47ebb95b9e05b7788d8618512a9280deb7f0e4`.
It has 1,671 native input pins, ten selected tools, four plugins and fourteen
separate loader contexts with a 37-file compiler runtime union. The twenty-one
tool/resource queries have not executed.

`closed-controller/host-tests-006` retained twelve top-level host test passes and
eight fake-runtime fault subtests, with no skips. The actual retained controller
then accepted the genuine plan and rejected independently rehashed missing-DSO,
joint DSO/node/input omission, compiler search override and extra-input plans.
Per-control argv, clocks, exits and raw logs are in
`artifacts/decodergpu-host-20261025/closed-controller/plan-checks-001/`.
The final source/receipt/context pins and compiler-only pickup command are in
that directory's sibling `source-handoff-008.json`.

Earlier failures are retained: host-tests001 found a private helper name
collision; preparation006 exposed missing static fallback for a linker-script
`-lgcc`; plan002 exposed the cross-process plugin SONAME collision. The resolver
and per-root grouping now pass the stated host controls. Outer plan001 and
preparation-wrapper008 deferred before child execution when the canonical lock
was busy. No genuine D01 compiler query, preprocess, compile/link, native target
load, device query, GPU/model execution or decoder numerical parity occurred.

A subsequent shared-helper review found competing partial PT_LOAD/BSS overlap
handling incomplete (NP1b). That holds compiler admission and any claim that
008 is the final corrected helper/controller closure. Preserve these receipts;
after the helper repair freezes, a fresh Go-only preparation and plan must bind
the corrected helper. This does not justify a model or decoder producer rerun.

The accepted NP1b repair is now bound by preparation009 and native-plan003.
Their exact hashes, immutable controller, all current source/context pins and
the compiler-only pickup command are in
`artifacts/decodergpu-host-20261025/closed-controller/source-handoff-009.json`.
Host-tests007 passed all twelve top-level tests and eight fake fault subtests.
Plan-checks002 passed ten actual retained-controller checks: the positive plan,
four rehashed plan mutations, four native modes without admission and a changed
controller executable. Negative JSON results report failure, and canonical lock
contention maps to exit 75. The capture observer/handshake remains held; this
checkpoint supplies no genuine compiler or decoder numerical result.

The guarded-observation source checkpoint uses preparation010, SHA256
`64e6c0159cfb6aa82d33a1207728a71b323db9778e59ce7fe27b5688ab8c8122`,
and current controller SHA256
`7313e980eebd4ba33049a4cfe67807728a7fce87b17e2d1850c38a02d85bcc13`.
Native-plan004 is the unexecuted v2 plan, SHA256
`6fc5209c0435ff88f5300d41766d0a46e25d37b664b7dfda9d8b21abd2f5848f`.
It freezes four native source files and 1,686 prospective inputs, including
CUPTI, before any new native query. Image-policy001 is SHA256
`11aa12fcccbef6220bf95af3e4ade0f129fdd93df5e0312d290c9b612e3a999a`.

`artifacts/decodergpu-host-20261025/capture-source/host-002/` retains the
bounded Go/fake-header test result. Both lifetime and image callbacks compile
against fake headers only. The observer controls cover copied module bytes and
names, malformed callback data, module truncation/capacity, activity capacity,
callback exceptions and dropped records. The lifetime fixture deliberately
bypasses observation to isolate asynchronous ownership; it does not qualify
CUPTI integration. The retained current-controller controls are in sibling
`controls-001/checks.json`. Thirteen controls passed, including historical
producer/current consumer authentication despite recorded live-source drift,
rejection of old-source current use, raw image/function/context/module/index
mutants, module-only failure and rejection of ungated v1 capture.

Synthetic image observations in these host controls are explicitly labeled.
They copy a pinned cubin as input data and do not claim that CUPTI selected or
executed it. The shared guard export and native constructor qualification,
genuine v2 CUPTI-header build, selected-runtime policy and exact observed
function/module/image binding remain separate acceptance gates. Capture stays
held. A later guard-only Go export requires a new current-consumer freeze over
the immutable v2 native producer, rather than a decoder or model rerun.

## Guard API018 and watcher019 wiring

The accepted v2 producer is `artifacts/tensor/d01-v2-build-001/phase.json`,
SHA256 `5a4bfa0161d337f62e897cb88d083e01b06600fc2ab338000c04e56954abfcd1`.
Its executable SHA256 is
`897f6e25716177bbb6db7a619e256c406935c127c0a45cf92b56da5daea601f5`.
The current wiring changes Go only and does not recompile that executable or
change its four native source files. Its first execution is retained below.

Preparation011 freezes the new CGO0 consumer. Its JSON SHA256 is
`850db81fbbec9a662b0be3285a109669b9cae823e1a8c27594ca2791b62eafa7`,
and its executable SHA256 is
`7e13071b7de297b67e2d5195d7727815579f72d59c012d244aab24bcc13774d4`.
The guarded runtime metadata is
`artifacts/tensor/d01-guard-runtime-001/runtime.json`, SHA256
`476489e03404fabb56449aa2836edf61bc19de25c54f7df2a0559673f9368287`.
It binds 23 prospective ELF nodes, 24 runtime inputs, 21 selected providers and
one rejected alternate. Loader directories are only the selected CUDA library
directory and `/usr/lib`. Where the accepted runtime003 qualification selects a
matching SONAME, D01 requires that exact provider. CUPTI and other additional
unique providers must already belong to the prospective D01 closure.

`guard.json` carries immutable producer/current-consumer custody and the selected
policy. `phases.json` independently pins the exact six names above. Historical
Go origins are authenticated through their selected immutable snapshots, so a
later legitimate Go port does not require another native decoder capture merely
because its source changed. Current capture admission separately checks live Go
inputs and the actual executing consumer. Actual native/runtime code and input
pins remain strict.

The plan retains an immutable copy of the accepted watcher019 source with SHA256
`271cb2c23de4dee042515c41e66ff8ad07593df5852ccfdba046b1838fef80c0`.
`watch-plan.json` declares the exact current consumer command and single decoder
role, with controller/role limits 240/180 seconds. `watch-launch.json` supplies
the expected Python argv, working directory, environment and raw receipt paths
for a separately granted resource wrapper. The actual wrapper must retain its
`runner.CommandReceipt` as the declared `command.json`; the blueprint is not
execution evidence. An owned bounded outer cgroup remains required.

Native success first has status `native_captured_pending_supervisor`. The
read-only `check-capture` import must consume independently pinned native,
watcher and actual supervisor-command receipts. It replays the shared guard,
binds the supervised consumer/native PID, argv and image identities, requires
empty process groups without timeout/ownership errors, rehashes all 32 raw
buffers and recomputes only the prospectively declared spans. It also requires
the actual CUPTI function/context/module/index/image proof. Only that import
returns `native_oracle_qualified`. No qualified native receipt exists yet.

The host turn is retained in
`artifacts/decodergpu-host-20261025/guard-wiring/resource-001/`. All 20 ordinary
test roots and 46 named tests/subcases passed, with zero named-test skips. The
CLI package reported no test files. Only the two explicit fake-C++ compiler
opt-ins were excluded because native source was unchanged and genuinely built.
Vet, current-consumer/immutable-producer authentication, runtime generation and
runtime verification passed. Independently rehashed removal of CUPTI, replacement
of the required image gate and a changed watcher digest were rejected. The
2CPU/3GiB/zero-swap/Tasks64 turn exited zero after 55.974 seconds, with 315.7 MiB
reported peak memory and matching source-before/after pins. These are source and
host results, not a native decoder numerical result or GPU qualification.

The first capture envelope, `artifacts/tensor/d01-native-capture-001`, failed
before creating a native command. The shared strict reader rejected
`policy: allowed: needed: null []string`; the watcher observed no decoder role,
and neither the native nor guard output directory was created. All failure
receipts remain retained. This supplies no decoder execution or numerical data.

D01 now emits dependency-free selected and rejected provider lists as empty
arrays. Runtime generation and verification also pass the actual serialized
guard and independent phase pins through shared `ReplayGuarded` with an absent
run. This host-only preflight must reach the exact expected run-identity error;
an arbitrary parsing or custody error fails preparation. It launches no target.
The regression covers both null dependency arrays and a changed contract pin.
A fresh consumer/runtime freeze and independent delta review are required
before another native capture envelope is admitted.

Preparation012 and runtime002 passed both targeted roots (12 named tests and
subcases, no skips), vet, the immutable-producer seam and actual retained-CLI
runtime generation/verification across the shared serialized boundary. Their
source/host handoff is `guard-wiring/source-handoff-012.json` under
`artifacts/decodergpu-host-20261025`.

The separately admitted `artifacts/tensor/d01-native-capture-002` executed the
four original calls once, exited zero and retained all 32 raw buffers. The
consumer failed the selected-image gate. All four launch/context-UID/FUNCTION
records bind to module 23 and ELF symbol indices 21, 20, 19 and 18, but its
266,280-byte observed ELF differs from both pinned 261,192-byte source images.
All five executable sections differ, including substantial changes to the four
decoder kernels. This is not a harmless whole-file metadata difference.

The unchanged failed receipt, raw images, guard transcript, supervisor identity
and empty cleanup groups are retained. The bounded envelope exited one without
timeout, signals, ownership errors or OOM. Its readback and static section
comparison are in `first-capture/outer-002/` under the same host artifact root.
The numerical buffers are retained but remain unqualified. No normalization,
new image acceptance or native repeat follows from this observation.
