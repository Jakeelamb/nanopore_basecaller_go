# S00 SUP source oracle

S00 is a verification adapter, not a Go caller port. Host preparation and import
never load Torch, Koi or a tensor backend. The native source has not yet been
compiled or executed. S00 has no native PASS.

## Capture design

Two designs were considered. A second encoder implementation could replay the
seven Koi calls after an untouched forward, using the graph's prepared caches.
That design cannot recover cache values before `remove_bits`, and recomputing
them after rounding changes the oracle. The selected design inserts observation
statements into immutable copies of three pinned Dorado translation units.

`Generate` verifies whole-file source hashes, bounds every insertion to its source
function, requires one occurrence of each anchor, and records original block and
anchor hashes. Reversing the declared insertions must reproduce the original
source bytes. Stock files and qualified archives remain untouched.

The reference executable links original Dorado objects only. The observed
executable links separate copied `TxModules`, `ConvStack` and `TxModel` objects
before the same archives. Link maps must prove those distinct TxModules
definitions. There is no symbol interposition or production model bridge.

The first declared producer uses three forwards across two fresh processes:

1. The reference process calls untouched `TxModelImpl::forward` and retains its
   final scores.
2. The same original model and its caches execute the module sequence extracted
   from `TxModel.cpp`. It captures the final convolution, encoder, upsample and
   score boundaries. Its final scores must equal the untouched forward.
3. A fresh observed model executes its first forward. It captures caches before
   and after the actual `remove_bits` call, then each primitive boundary in the
   same pass. Its scores and four coarse boundaries must match the reference.

The second reference forward is needed because the untouched full-model API
returns only scores. It is an observation of original modules using the existing
caches. The observed model has no warmup or cache recomputation. A mismatch stays
failed; it does not create a tolerance or authorize another producer attempt.

## Frozen first case

The bounded case uses HG002 development read
`bb76c31a-36f0-48a0-abd2-107a10e1ade0` from the retained
`artifacts/pod5/fixtures-v3/manifest.json`. It contains 19,856 raw int16 samples.
The source DNA rule trims ten samples. The pinned SUP config supplies
standardisation mean and stdev, and the fixture supplies the calibration float
bits. `hacgpu.Normalisation` applies the maintained source-linked scalar rule.

The actual convolution strides multiply to 12. `BasecallModelConfig.h` defines
transformer chunk granularity as `stride_inner * 64`, giving 768 samples.
`CudaCaller.cpp` defines transformer batch granularity 32. The maintained
`chunk.Fixed` planner, with overlap 600, supplies the first 32 starts
`0,168,...,5208`. The final selected interval ends at 5976. Each chunk has its
own coordinate and normalized half hash. The batch input is 49,152 bytes, SHA256
`67d154004f51b70fd38ee7aadd8e2cda821667b5eb46a1a6ae051602d8be56e8`.
This is a model-stage fixture, not complete-read caller evidence.

`provenance/gpu-target.json` retains default I8 and FP8 options and
`remove_bits=4`. The existing P01 receipt
`artifacts/tensor/gpu-profile-native-20261023T204747Z/profile.json`, frozen at
`67582f569c4e9cbcd3f84c2158584a04537302c1215c1adfaf3a375f3c84e829`,
observed E4M3 support on the selected UUID. Preparation reads that receipt. It
does not query capability or force another precision path.

## Capture contract

`Prepared` is an external trust anchor. It binds the source revision, case,
dispatch, profile, fixture, signal, scaling float bits, trim, all 139 source
weight files, every chunk and the final input. `ReadPrepared` rehashes the pins
and reconstructs all input bits before an importer or future native command.

The fixed FP8 registry contains 191 ordered events. Each encoder contributes
two cache snapshots, one input/counter snapshot and seven operation results.
The other events cover input, five convolutions, encoder tiling, encoder output,
upsampling, the once-scaled half CRF weight and scores. The cache snapshots retain
the unrounded half caches, QKV I8 values and float reciprocals, RoPE, projection,
normalization, shuffled FP8 FC1 and FP8 FC2 caches. They come from actual source
objects at the insertion point.

Each tensor binds its name, dtype, shape, strides, tags, scale tag, storage
offset, logical payload and complete allocated backing. This first fixed case
captures initialized tensors after their source writes, and initialized caches
before and after rounding. Its prospective allocation rule is the entire
initialized allocation. No uninitialized scratch-before-write event is captured.
The importer rejects any exclusion, missing range or mismatch-derived mask.
VCS padding and LUT allocation rules require a separate registry before capture.

Every native primitive event also serializes its actual `KoiTensor` descriptors,
including flattened view shapes, strides, dimension tags, dtype, scale dtype and
scale tag. The importer compares them with a finite source-derived descriptor
registry. Residual-half results are captured after both RMS operations. Entry
counters must be four fresh zero I32 values.

`Build` binds the frozen plan, compiler and full Go controller custody. Selected
Go sources, assembly, headers, embeds and tool binaries come from actual
`go list` and `go env` metadata. Their hashes are recorded before the Go build.
The executing controller, its build argv, clocks, logs and toolchain version
must match. Source/tool drift and another executable reject before native work.
Immutable source/tool copies retain historical producer custody.

Each copied C++ file retains its original source parent through `-iquote`.
A host guard resolves every direct quoted include against the planned search
paths. Wrapped DSOs, all seven explicit `-l` libraries, implicit startup objects
and transitive linker-script `GROUP`, `INPUT` and `AS_NEEDED` inputs are bound
before compilation. Library lookups become exact paths, preserving directory
search order. Deferred compiler queries must confirm the prospective choices.
Five deferred `-M` commands then freeze complete header sets before any `-c`.
Actual `-MD` sets must equal those sets. Both links retain `-t` output and maps;
an input outside the before-compile closure rejects instead of being added later.

The qualified GCC15 manifest binds all 1,622 compiler-distribution files.
An additional `CompilerRuntime` is frozen during host planning, before the first
compiler query or `-M`. It records the C++ driver, cc1plus, collect2, assembler,
linker, nm, the C driver/cc1, the qualified LTO plugin and the finite system
`/usr/lib/bfd-plugins/*.so*` family. Pure Go ELF parsing closes
their interpreters and recursive `DT_NEEDED` dependencies, including cc1plus's
libisl, libmpc, libmpfr and libgmp, and pins the loader cache. These compiler
inputs also belong to the main plan input set. The launcher rebuilds and checks
this contract before and after every query, preprocess, compile and link.
Missing compiler-only DSOs reject before command logs or a child are created.
Tool-path queries must confirm the prospective choices before `-M`; bare tool
names resolve using the declared native PATH. Search-changing compiler options
outside the stock recipe reject. No `-B` override changes GCC's header/library
search. Native environments contain only the declared PATH, locale and
two-thread settings. Command argv, working directories, clocks, exits and raw
logs precede projection.

A separate post-build `RuntimePlan` is mandatory before any model process. It
reads ELF `DT_NEEDED`, RPATH, RUNPATH and interpreter metadata recursively, seeds
the retained P01 runtime and declared CUDA/cuDNN compute/JIT families, and pins
the loader cache. This step loads no native code. `RunCapture`, `Seal` and
`Import` require that frozen pre-model plan; every actual mapped object must be
inside it with unchanged bytes. Unknown mappings remain failed. The first S00
native runtime plan cannot exist until the separately admitted compiler build
succeeds; the system-libm metadata test does not qualify the SUP runtime.

RuntimePlan v2 binds both roles explicitly: `Controller` is the accepted
compiler producer, and `Consumer` is the current frozen Go metadata/capture
controller. `Build` and `Prepared` must match the original accepted build and
input. Historical producer validation still checks every original input,
compiler, generated source, manifest, binary, command, dependency selection,
immutable source/tool snapshot, compiler runtime and link binding. Only the
old live Go source/executing-binary identity is historical. The new consumer
must pass those current-source and executing-binary guards before runtime
planning or capture. Import validates both historical controllers and their
immutable evidence without requiring the importer to be either producer.
`RunBuild` retains its strict current-producer requirement.

The retained profile's explicit canonical loader cache is hash-checked before
it is separated from ELF seeds; the RuntimePlan LoaderCache pin remains
mandatory. Any other claimed non-ELF runtime code is an error. A basename
containing `.so` does not turn cache metadata into code.

JSON keys are exact and required. Duplicate, case-variant, unknown, missing and
null fields reject. Payload paths cannot escape the capture or traverse a
symlink. Nonregular pins reject without blocking on a FIFO. Import requires
externally supplied prepared, build and pre-model runtime-plan SHA256 values.
It does not authenticate those anchors using hashes from the capture itself.

## Commands and remaining admission

Host preparation is authorized. Typical commands, run from the Go repository:

```sh
env GOMAXPROCS=2 GOMEMLIMIT=3GiB CGO_ENABLED=0 GOTOOLCHAIN=local GOPROXY=off GOSUMDB=off go test ./verification/supgpu/...
go run ./verification/supgpu/cmd -unit prepare -model /home/jake/Projects/nanopore_basecaller/models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0 -out artifacts/supgpu/prepared-fixed-NEW
bash verification/supgpu/host-controls.sh PREPARED_JSON artifacts/supgpu/host-controls-NEW
bash verification/supgpu/host-controls.sh --consumer-only BUILD_JSON BUILD_SHA artifacts/supgpu/consumer-host-NEW
```

The host script freezes the Go controller, its selected dependencies and tool
inputs, then uses that exact executable to generate and check the native plan.
`plan-build` requires `-controller` and `-controller-sha256`; a transient
`go run` executable cannot stand in for the frozen controller at native launch.
Its explicit consumer-only mode runs host tests, freezes a new Go consumer,
validates the accepted old build through `check-runtime-consumer`, and tests
consumer-source omission and alternate-binary rejection. It exits before NN
source generation, build planning, compilation or runtime-plan discovery.

`PlanBuild` uses `hacgpu.OracleToolchainForSource` to read the actual stock flags.
It does not call the older `BuildOracle`, which lacks separate observed objects,
depfiles, link maps and raw command receipts. Its current plan contains 35
compiler/tool lookup checks, five dependency preflights, five compile commands and
two link commands. Host planning executes none of them.

The following command requires separate admission for the exact bounded native
phase. The first compiler-only run completed at checkpoint 010 as recorded below:

```sh
FROZEN_HOST -unit build -plan PLAN_JSON -plan-sha256 PLAN_SHA -runtime-admitted
```

After that compiler-only phase, the following metadata command freezes the
prospective runtime without executing a native binary:

```sh
FROZEN_CONSUMER -unit plan-runtime -build BUILD_JSON -build-sha256 BUILD_SHA -controller CONSUMER_JSON -controller-sha256 CONSUMER_SHA -out NEW_RUNTIME_PLAN_JSON
```

The model phase needs its own admission and the reviewed runtime-plan hash:

```sh
FROZEN_CONSUMER -unit source-graph -prepared PREPARED_JSON -prepared-sha256 PREPARED_SHA -build BUILD_JSON -build-sha256 BUILD_SHA -runtime-plan RUNTIME_JSON -runtime-plan-sha256 RUNTIME_SHA -runtime-admitted -out artifacts/supgpu/capture-fixed-NEW
```

The native commands acquire the existing canonical stock lock nonblocking and
retain it through child completion. A grant, current reservations, resource caps
and the lock remain separate requirements. The compiler-only run checked the
native source/header/API and link-map assumptions for this fixed build. Genuine full capture
import and its corrupt-copy controls remain required before expanding to VCS or
other cases. Host tensor/parser controls do not replace that native handshake.

`-unit seal` and `-unit import` use retained files only. An observer projection
failure is repaired from those files, with a fresh immutable analysis copy when
needed. It does not justify another 18-layer inference.

## Retained host evidence

`artifacts/supgpu/host-controls-010/` contains per-command argv, clocks, exits,
stdout, stderr and log hashes. Its 35 named Go tests passed with zero named
failures or skips. The command package has no test files. The same attempt built
the Go checker, accepted the genuine prepared input, generated the six source
files, and froze 2,666 native-plan inputs, 100 linker bindings and 914 selected
Go source/tool snapshots. The actual project closure has 32 Go files and two
embeds, including runner and gpuprofile dependencies. Controls reject missing
quoted-header lookup, an omitted wrapped DSO, an omitted runner source snapshot,
an appended-byte alternate controller, independently rehashed FP8/geometry/
calibration mutations and native build/capture without admission. Pure host
tests also reject changed source/tool bytes and unknown link/runtime inputs.
The actual compiler metadata audit records ten tools, four plugins, 36 ELF
nodes and 37 compiler-runtime pins. An inert ELF fixture exercises dependency
and interpreter closure without machine code. Omitted DSO/interpreter/cache
pins and changed compiler-DSO bytes reject. A launch-boundary test proves the
missing libisl pin rejects before the first query creates logs. CLI controls
independently rehash plans omitting libisl from the top-level inputs, and from
both those inputs and the saved ELF graph; both reject. All 40 raw host logs
match their retained hashes. These observations qualify host/source closure;
native selected-tool queries have not run.

The prepared manifest is
`artifacts/supgpu/prepared-fixed-001/prepared.json`, SHA256
`5d8f024ac5ccbba1655733d7fd43cb3316cf6046871c58af42f555f205b96ae7`.
The current native plan is
`artifacts/supgpu/host-controls-010/build/plan.json`. Its hash and the controller
hashes are retained in that checkpoint and the handoff.

Earlier host attempts remain intact. `generated-fixed-001` failed its unique
anchor check because the fixed and Volta branches both return `untiled_f16`.
Generation now bounds that insertion to the fixed tiled branch. Attempts 002
through 004 and all earlier host controls remain historical snapshots.
Independent review held checkpoint 004 for quoted-header, link-input and Go
producer-custody omissions. Checkpoint 005 failed because `GOENV=off` appears as
an empty value in `go env`; the guard now checks the observed empty value.
Checkpoints 006 and 007 passed their host checks before later runtime-contract
changes. Independent review accepted the earlier source corrections in 008,
then held compiler admission because cc1plus's four math DSOs were unpinned.
Checkpoint 009 adds the separate prospective compiler runtime contract; 010
also freezes BFD's discoverable plugin family and its transitive DSOs. All
earlier checkpoints remain intact. The independent compiler-runtime correction
review passed before the coordinator's compiler-only admission.

## Retained compiler-only evidence

The first admitted run is `artifacts/supgpu/compiler-native-010-001/`.
Its systemd service enforced 6 GiB memory, zero swap, two CPUs and 600 seconds,
with control-group cleanup. Effective limits, raw clocks, exits and logs are
retained. The service exited 0 after 146.728 seconds, reported memory peak
1,781,231,616 bytes and zero OOM events, and its cgroup no longer existed after
wait. The frozen Go launcher held the canonical lock through all compiler
commands and the final closure checks.

`host-controls-010/build/build.json` has status `built`, SHA256
`12628cff8da000af78f9cbfc7ba0c8273da6b6ce26c5dbe0869a7a58eb234296`.
All 35 actual tool/resource lookups matched the prospective contract. Five
`-M` preflights froze 7,635 inputs before compilation. The five actual `-MD`
sets matched; five compiles and two links exited 0. Both actual link traces
and maps stayed within frozen inputs and compiled objects. The separate
reference and observer ELF pins, five object pins, depfiles, maps and all 47
raw compiler command receipts are in the build receipt. No failed build was
repeated and no unknown input was added after compilation to obtain acceptance.

Independent review accepted this completed build as BUILD_ONLY_PASS in
`artifacts/parity-resume-review-20261024/s00-built-review.md`. The first admitted
pure-Go RuntimePlan attempt, `artifacts/supgpu/runtime-plan-010-001/`, exited 1
without publishing a plan: the retained profile's `/etc/ld.so.cache` matched
the `.so` basename filter and was incorrectly passed to ELF traversal. The
failed receipt is preserved; its service/cgroup cleaned up and released the
canonical lock. The metadata consumer needs a separate current-controller
anchor so that repair can reuse the accepted native build without recompiling
it. The approved source correction froze `artifacts/supgpu/consumer-host-011/`:
37 named host tests passed with zero failures/skips, a genuine accepted-build/
new-consumer handshake passed, and independently rehashed omitted-source and
alternate-consumer controls rejected. Its controller has 914 source/tool
snapshots, 32 project Go files and two embeds. The Go-only resource wrapper
retains 3 GiB/swap-zero/two-CPU limits, zero-exit clocks/logs and empty cgroup
cleanup. `build.go`, `controller.go` and the full `RunBuild` body are byte-equal
to checkpoint 010. No NN generation/compilation or runtime-plan generation ran
in this checkpoint. The independent seam review passed before a separately
granted pure-Go metadata retry.

`artifacts/supgpu/runtime-plan-002/plan.json` is the resulting RuntimePlan v2,
SHA256 `2d79a8d2b1a37223a0e4b703c2724d16c88ee36342fb5b5a16d97ad324a85100`.
It binds the old accepted Build/Prepared/compiler Controller and new current
Consumer, with 61 seeds, 62 ELF nodes and 63 inputs including the loader cache.
The declared CUDA/cuDNN/NVRTC/compute-driver families contribute 37 members;
recursive discovery additionally finds libutil.so.1. The member-binding
projection records 420 dependency edges, with 55 having multiple canonical
candidates under the recorded search order. All candidates are pinned; aliases
collapse by canonical path while distinct installed versions remain distinct.
These are prospective resolver bindings, not observed dynamic-loader choices.
The source resolver rejects unresolved names, unsupported expansions and
unexpected pathname dependencies. Capture must reconstruct the same current
plan and reject unknown actual mappings.

The metadata service exited zero in 33.309 seconds under two CPUs, 3 GiB,
swap zero and 180 seconds. Its cgroup was removed and the canonical lock
released. No target, loader, compiler, readelf, ldd, device/GPU or model ran.
Independent metadata review accepted the corpus and held first forwards. Model/target execution,
backend library loading, device/GPU queries, three-forward equality and genuine
capture/import remain held. This metadata result does not close S00 parity.

## Loader guard correction

The 002 metadata review accepted its frozen corpus and held first forwards:
membership alone admitted pinned system cuDNN 9.27, and actual mapping checks
ran only after both producers finished. That hold is preserved in
`artifacts/parity-resume-review-20261024/s00-runtime-plan-review.md`.

The correction adds a verification-only glibc audit DSO with raw Linux x86-64
syscalls and no DT_NEEDED. Its la_version/open/close callbacks use private pipes
to the Go controller. Each loader-object approval is recorded before the audit
callback returns, before object initializers or dlopen can return to its caller.
No symbol binding callback or arithmetic interposition is installed. Missing
or denied approval exits the process. Independent phase pipes stop the main
thread before model construction, before first forward and between/after the
purpose-distinct forwards. The attest-only action exits after pre-model approval,
before CUDA queries, model loading or input upload.

RuntimePlan v3 freezes one selected provider per SONAME. Qualified/P01 cuDNN
9.13 is allowed; system 9.27 and its extra engines remain explicit rejected
alternatives even when their bytes are pinned. Other ambiguous providers require
one P01-observed selection. Selected transitive dependencies are required in
actual pre-model maps. LD_LIBRARY_PATH names the selected qualified directories,
LD_AUDIT names the pinned verifier DSO and LD_BIND_NOW=1 makes initial binding
explicit. Precision flags and model arithmetic remain unchanged.

The parent retains initial maps, every callback/phase map and decision, exact
PID, namespace, sequence/cookie, native stdout/stderr and failures for each
launched process. It denies and drains on rejection. Import replays both transcripts against retained maps
and selected policy, including transient objects absent from final maps.
The finite v2 compiler plan adds one freestanding C compile and shared link:
six compiles, three links; native math sources and the 16 reversible observation
insertions stay unchanged. Older binaries/plans/receipts are not retrofitted.

This is a loader observer for verification, not a performance configuration or
a sandbox against arbitrary anonymous host JIT code. Unknown executable maps
reject at phase boundaries. Every normal loader object, including lazy CUDA/
cuDNN/JIT DSOs, uses the synchronous per-object gate. Metadata planning, real
loader/constructor controls and attestation-only execution require separate
coordinator admission before model forwards.

The separately admitted 012 compiler build completed in
`artifacts/supgpu/compiler-native-012-001/`. Build SHA256
`9ceec402ecda941c831e5c24271f364e4ffa1774bfa716f439176f18dabd2ce8`
binds the two new targets and the freestanding auditor. All 35 queries, six
precompile dependency phases, six compiles and three links passed. The auditor
has no DT_NEEDED or unresolved imports and exports the three required callbacks.
It has not been loaded. The service enforced two CPUs, 6 GiB, zero swap,
64 tasks and 600 seconds. It exited zero after 158.472 seconds with zero OOM
events, an empty removed cgroup and the canonical lock released.

The subsequent bounded static audit checked all 8,713 declared pins, including
source/controller snapshots, build inputs, maps, objects and output binaries.
All 106 raw log hashes from 50 native and three controller commands matched.
`artifacts/supgpu/build-audit-012/audit.json` retains this result. Numerical
sources, FP8=true and the prepared HG002 geometry remain unchanged.

Independent loader-guard source review found L1: direct-child Wait disabled
process-group killing while a descendant still held gate pipes, and a descendant
closing its pipes could survive a successful return. The 013 Go correction
separates leader completion from group cleanup. The drain deadline kills the
group even after Wait. Every completed launch scans actual /proc group members,
kills remaining members and verifies emptiness within a bounded cleanup period.
Guard receipts retain member stat observations, SIGKILL times and final empty
state. Successful import requires clean group completion without a kill.
The enclosing cgroup remains the coordinator's independent lifetime boundary.

Effective static Go controls cover both roles, descendants retaining or closing
gate FDs after rejection, and descendants closing FDs after otherwise-successful
leader exit. The pre-fix run fails all six cases; corrected controls verify the
descendant and group are gone while an unrelated process remains alive. Raw
failure receipts/maps and source-to-copy pins are retained by the maintained
host command. These controls qualify Go cleanup, not glibc constructor ordering.
Consumer 013 reuses the immutable 012 Build through the existing separate
producer/consumer anchors. No NN source regeneration or recompilation is needed
for this Go correction. `artifacts/supgpu/consumer-host-013/` records 55 named
PASS, zero FAIL/SKIP, a genuine accepted-build/current-consumer handshake and
effective omitted-source/changed-binary rejections. Its controller has 917
snapshots, 34 project Go files and three embeds. The current CLI does not import
internal/model. The bounded host service exited zero, removed its cgroup and
released the canonical lock. Real DSO controls, target attestation and all model
forwards remain held pending independent exact-delta review and separate grants.

Independent review accepted the completed 012 build and L1 correction in
`artifacts/parity-resume-review-20261024/s00-guard-cleanup-review.md`. The granted
pure-Go RuntimePlan v3 is `artifacts/supgpu/runtime-plan-003/plan.json`, SHA256
`f1fae0cc36df9c503f5d8554d053276744368214499a947b2f94d493754d25ec`.
It preserves the 012 Build/Prepared/producer and 013 Consumer anchors, with
62 seeds, 63 ELF nodes and 64 input pins. The policy selects 50 providers,
requires 29 at startup and rejects 11 alternatives. All eight selected cuDNN
images are qualified 9.13; all ten system 9.27 images remain rejected. The
selected-binding projection retains 420 dependency edges. This is metadata
evidence; no object was loaded. The 32.658-second service exited zero under
two CPUs, 3 GiB, zero swap and 180 seconds, then removed its cgroup and released
the canonical lock.

Real constructor controls are prepared in the separate test-only controller
`artifacts/supgpu/audit-controls-host-015-001/`. The existing 013 CLI and
guard/policy code, native bridge/auditor and all numerical sources remain
unchanged. The finite plan proposes one CPU CRT/libdl loader and three
freestanding constructor DSOs. Both roles run allowed, pinned known-wrong-tier
and unknown fixtures. The positive must create constructor and dlopen-return
markers. Negatives must reject at an actual saved object-open callback before
either marker, exit 124, drain and leave an empty process group. These fixtures
do not load cuDNN/CUDA and do not substitute for actual target attestation.

Preparation authenticates the accepted auditor and its complete producer once.
The new proof retains the auditor's source/object/depfiles/map/commands and
selected CPU providers. Later control checks consume that proof and the actual
auditor; their input closure excludes NN/Torch/CUDA images. The compiler plan
closes selected GCC assets, tool runtimes and implicit link inputs before its
24 deferred queries, four -M/MD pairs, four compiles and four links. Actual
traces/maps and post-build CPU ELF closure remain required before loading.
Each case revalidates its control runtime, executable/DSOs/auditor and parent
controller manifest/binary. Unknown code and known wrong fixture providers
remain rejected even when their bytes are pinned for custody.

The 015 host wrapper passed four focused current control tests and seven CLI
negatives, froze 1,003 selected Go/test/source/tool snapshots and published the
1,669-input compiler plan. It exited zero in 44.172 seconds under the bounded
3 GiB/two-CPU wrapper. The prior 014 wrapper timed out during repeated NN
metadata hashing after its 58 tests and source-plan checks passed. That attempt
remains unchanged; a fresh continuation completed only its missing negatives
without repeating tests, Go build/freeze or plan generation. The 015 refinement
removes those unconsumed NN hashes from subsequent control checks.

No genuine control compiler query/preprocess/compile, auditor/DSO load, native
target attestation, device/GPU/model work or NN regeneration ran in these host
checkpoints. RuntimePlan 003 and control plan 015 await independent review and
separate native admissions. S00 remains an oracle preparation unit, with no
native arithmetic or port-parity acceptance claimed.
