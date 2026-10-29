# H08 fixed HAC composition

Status: reviewed source and host candidate, integrated into the canonical
feature worktree; native execution remains held. Development worktree
`feature/h08-fixed-hac-20261025` is preserved. Integration added seven files
and changed no inherited source files.

The independent source review is
`artifacts/parity-resume-review-20261024/h08-source-review.md`; guarded
integration is `artifacts/h08-integration-20261025-001/receipt.json`.
The integrated ordinary Go suite and maintained CPU parity pass in
`artifacts/resume-integrated-host-20261025-002/` and
`artifacts/resume-integrated-cpu-parity-20261025-002/`. These are host and CPU
algorithm checks, not native GPU graph qualification.

`HACGPU` composes the existing Go convolution, fused recurrence, and projection
objects. There is no native ForwardHAC operation or stock executable call.
Pinned `CRFModel.cpp:87-145` determines the reservation, input copy, call order,
and returned contiguous half NTC scores.

## Contract and ownership

The first profile is physical N64, chunk192, T32, C1024/K128, fixed I8 recurrence.
The source factorised projection accepts I8 only. The standalone H06 half
recurrence is therefore explicitly rejected as a full-model fallback.

`LoadHACGPUWeights(ctx, engine, directory)` recognizes the pinned HAC config and
loads all 50 named archives through `WeightManifest(HAC)` and the existing
verified CPU loader. It packs the accepted Go caches once. Same-shaped roles
retain archive identity; ordering cannot substitute a foreign layer. The shared
`LoadHACFLSTMWeights` uses this same binding/loader for only the 42 recurrence
archives and returns seven `FLSTMWeightSet` values and their receipts.

`NewHACGPU` owns uploaded caches and borrows its device and streams. Forward and
Close serialize. Forward borrows a contiguous half NCT input until return and
returns an owned score storage alias that keeps the invocation allocation alive
after its other handles close. Cancellation, observer failure, native failure,
or cleanup failure cannot return a successful partial result.
Input is already prepared/standardised at the source CRFModel entry boundary;
this graph does not repeat caller preprocessing.

P03 derives the source's 7,782,400-byte reservation with T35 after padded conv1.
Actual use has T32: input/front, conv0/back, conv1/front, conv2/back, recurrence
temp/front, projection temp/front. Recurrence scratch expires when projection
reuses it. Actual projection temp is 4,718,592 bytes. The existing typed storage
allocator and primitive guards enforce the views' device, extent and alignment
contracts; this Go layer does not claim to measure a device pointer address.

Each invocation creates a fresh pattern-only workspace. It never imports a
retained intermediate or future backing. Go copies input, zeros last-convolution
padding and recurrence sentinels/scratch, runs conv0/1/2, seven recurrent calls,
and one projection, and zeros projection down scratch through the existing
projection implementation.

The all-active qualification case accepts logical input `[64,1,192]`. The
one-logical-chunk policy accepts `[1,1,192]`, internally zero-pads the other 63
input slots, and carries ActiveSlots1/PhysicalSlots64 in the result. Normal
Forward currently rejects this policy with `ErrHACGPUInactiveSlotsUnqualified`.
`ProbeInactiveSlots` runs the same graph implementation for its independent
qualification; a passing probe does not silently enable production admission.

Scores have intrinsic tanh-times-five range [-5,5]. This pinned config has no
clamp sublayer: `BasecallModelConfig::has_clamp` is false and `Decoder.cpp`
selects decoder clamp value 0. Metadata retains Q scale1.02, bias-0.3, blank2,
state length4, and mean-Q start60 without applying an extra clamp.

## Reused observations and launch boundary

`verification/hacgpu/h08/cmd -mode replay` authenticates the independent H01
receipt/capture plus accepted H03 packing digests, H04's 49 native cache
payloads, and H07's three native projection caches. It rejects a changed input
copy. Historical producer custody limits remain historical; this replay does
not repair or rerun those producers.

The host-only `backend-proof` mode consumes the real
`nanopore-tensor-closed-build/v1` producer schema through
`closedbuild.ReadArtifact`. Four independent pins select Build002, Plan005,
Controller005 and its output. The reader authenticates the historical Go
producer's exact dependency selection, raw commands, binaries and complete
source/tool snapshot mapping. It does not require those historical Go originals
to remain current. Native dependencies, external assets, selected compiler ELF
domains/plugins/cache, path bindings, header `-M`/`-MD` equality, link trace/map,
archive members and selected output remain checked. It reads only the consumed
compiler/asset contract data; it does not replay other NN producers or import a
legacy `hacgpu.BuildManifest`.

`Consume` now requires an independently pinned `RuntimeAdmission`. The old
permission flag, H07 Go-only build manifest and map-membership admission are
removed. The admission binds a separate fixed request, exact guarded child argv,
current typed CGO1 compiler artifact, backend compiler proof, qualified runtime
provider policy, CPU13 evidence, phase contract and parent UUID metadata plan.
Missing gates stop before lock acquisition or a native child. No accepted
current CPU13 or graph execution exists yet, so runtime remains held in practice.

The selected-consumer source migration now requires independent evidence and
selection pins throughout this chain. `RuntimeAdmission` is v2, fixed and CPU13
requests are newly versioned v1, numerical consumer receipts are v4, and fake
loader proofs are v2. Old chains are rejected. See
[selected consumers](h08-selected-consumers.md) for exact schema names, fields,
role distinctions, source-only checks and pending producer/runtime gates.

The parent holds the canonical nonblocking lock while calling the maintained
`supgpu.ExecuteGuarded` and `ReplayGuarded`. The child uses one `PhaseClient` at
`pre-model` before the loader, `pre-forward` immediately before the sole graph
call and `post-forward` immediately after it. Startup and each loader object face
the shared initializer veto. Child maps remain raw observations. The parent
runs the separately pinned `nvidia-smi` metadata process before and after the
guarded graph, with a clean exact environment and frozen ELF/runtime closure.
The child never spawns another process or clears the audit environment.

CPU13 is explicitly two evidence tiers. Eleven maintained genuine tensor,
weight, decoder and attention tests run in one audited fixed test target. The
two tests `TestNativeLoaderABIRejections` and
`TestNativeCloseFailureRetainsLibrary` run separately as host-only fake C tests.
Neither opens the genuine backend. Their exact selector, pinned C-only logger,
34 compiler commands, source-before-compiler snapshots, output snapshots and
raw test results are retained separately. Combined acceptance requires all13
named tests and zero skips, rather than calling fake-loader results genuine
backend qualification.

The genuine tests use the maintained `NANOPORE_TENSOR_FIXTURES` interface. Its
default remains the backend directory. H08 prepares a fresh, independently
pinned fixture view by copying the retained archives and native JSON snapshots;
its fixture-only manifest has no compiler-build claim. A symlink identifies the
immutable backend, while `NANOPORE_TENSOR_BACKEND` stays canonical for actual
runtime identity. The real HAC conv0 weight and qualified CUDA13 archive are
checked against authoritative pins. No fixtures or backend producer are rerun.
An external `tensor_test` TestMain overlay supplies only phase/profile wiring;
the maintained test bodies are unchanged. Its distinct fixed `go test -c`
CGO1 compiler profile remains a prerequisite pending compiler-unit integration.

`Forward` and `ForwardObserved` share one implementation; recurrence observation
uses H05's existing production loop. Required all-active evidence is 58 uploaded
cache owners, 13 stage observations, seven complete initialized recurrence
scratch spans and projection down scratch, and complete scores. Full stage and
workspace bytes are retained. Masks are planned before execution. Conv1 padding
and conv2 unwritten rows are excluded. Recurrence compares every initialized row
0..34, including preserved row1; only slack rows35/36 are excluded. The source
capture's direct-write annotations are independently checked.

The second native phase requires a same-build all-active positive, executes one
one-slot probe, compares the first slot at all 13 stages, and proves inactive
input zeros. This is the source-padding independence control. Opaque scratch
for inactive slots is not compared against the all-active run in that phase.
Both phases retain rehashed foreign-layer-cache and changed-score controls.
There is no fallback, oracle rerun, full caller, biological or speed claim.

The coordinator owns native execution and integration. Authenticate the accepted
compiler output without a loader, device, model deserialization or source witness read:

```sh
H08 -mode backend-proof -backend BACKEND -backend-sha BACKEND_SHA \
  -backend-build BUILD_JSON -backend-build-sha BUILD_SHA \
  -backend-plan PLAN_JSON -backend-plan-sha PLAN_SHA \
  -backend-producer CONTROLLER_JSON -backend-producer-sha CONTROLLER_SHA
```

Prospective runtime commands, only after separate coordinator admission, are:

```sh
H08 -mode cpu13 -admission CPU13_ADMISSION -admission-sha CPU13_SHA
H08 -mode consumer -admission ALL64_ADMISSION -admission-sha ALL64_SHA
H08 -mode consumer -admission PADDING_ADMISSION -admission-sha PADDING_SHA
```

Each admission supplies fresh guard and numerical output directories. Padding
requires the independently pinned audited all-active parent receipt, and replay
checks that guard proof before accepting its numerical result. Resource launch
uses the accepted watcher019 with explicit role argv/binary pins, canonical
nonblocking lock, CPUQuota200%, MemoryMax6G, swap0, TasksMax64 and180-second role
bounds. The coordinator freezes the actual paths, hashes, cgroup readback and
watcher plan before execution. These examples grant no native phase.

## Host evidence

`artifacts/h08-host-20261025-003/receipt.json` records seven model tests and five
verifier tests in each CGO mode, vet, the CGO1 controller build, retained replay,
and unadmitted/unclosed launch controls. All required checks passed with no
skips. CGO1 compiled the small existing loader shim; it did not load the genuine
backend or execute CUDA. The controller build freezes 977 actual Go/Cgo/header/
embed/module source pins, six tool hashes, immutable source copies, actual argv,
clocks, raw streams and binary identity, with zero source drift. It reuses the
existing H07 Go custody schema and reader.

The producer-free replay is
`artifacts/tensor/h08-witness-001/replay.json`, SHA256
`8736e2a2d632b45eb034a639263d8531b638fd43b41a01c4170ca67b36226a83`.
It authenticates 90 retained file pins, 13 prospective stage contracts and 58
cache witnesses and rejects a changed input copy. This is witness authentication,
not fresh Go graph arithmetic or native parity.

The CGO1 controller SHA256 is
`03034c23a9e6875534584d17324615cd235b85aa1ea4c7edb15db7cb8a70f7a7`;
its `controller-build.json` SHA256 is
`cf88a09155bbff87568c4aa00e9ac11b186b95d18788c866e90f156e8a01df87`.
These identify the historical isolated candidate. Its six-tool CGO1 custody did
not close the actual GCC15 LTO plugin/native C toolchain inputs; it is not a
current runtime admission proof. A fresh finite current CGO1 closure, prospective
runtime policy and audited native launch guard remain required. This boundary
change is tested with CGO disabled and cannot promote that old controller.

Attempt001 preserves a receipt-observer count mismatch: its regex also selected
the existing P03 root/subtests; retained readback confirms all 36 selected tests
and vet passed. Attempt002 preserves vet failures for unkeyed external literals.
Those literals were corrected before the final frozen attempt003. No failure
caused a native oracle or model rerun.

## Closed producer boundary evidence

`artifacts/backend-artifact-host-20261025-004` retains the genuine backend002
reader positive, 16 top-level tests (23 named passes), vet, and a forced CGO0
H08 CLI build. Both test packages passed; the CLI package has no tests. The
current executable is `8ce3aa0e7a5a13d69a59b30f2d7579a298a3b56f578ab057f4223ab5a7fde653`.
Its separate current closure has 970 source/tool pins; the host selection has
1,047 pins and immutable snapshots, including test sources. The backend proof
authenticates 945 historical producer snapshots and retains 6,178 consumed proof
and native input pins. ABI export checks use the actual consumed `abi.h`, not an
invented historical `loader.c` input. All 42 declarations are exported.

Omitted snapshot, changed source binding, compiler domain and selected output
controls reject in004. Attempt005 uses the same retained executable to finish
modified-log and self-rehashed argv controls, the structural runtime HOLD, and
unchanged current source/membership checks. No Go build, genuine positive,
compiler chain or NN producer was replayed for that observation repair.

Failures remain explicit:001's host list observer rejected Go-generated absolute
test-main cache paths;002 exposed a missing Go import;003 rejected an invented
loader input;004's wrapper collided between a mutant and raw-log filename. In005
the inner commands exit zero as expected, while outer receipt projection fails
on the unused name `run.py`; its actual `resume.py`, raw systemd success/zero exit,
journal clocks and absent cgroup are authenticated by
`outer-observation-repair.json`. The original outer failure is preserved.
The prospective harness now separates `mutants/` from `commands/`; an effective
filesystem regression writes the same basename in both with exclusive creation
and checks path traversal rejection. It executes no producer or Go rebuild.

These observations qualify the historical source/host boundary only. No CXX
query, genuine compilation, output loading, genuine CPU13 fixture, driver/JIT
call, CUDA graph, fresh model arithmetic or source NN execution ran in this unit.
The guard integration's host003 Go results are retained development evidence;
that attempt observed a free lock but failed to acquire it. It is not promoted
to resource-admitted acceptance.

`artifacts/h08-guard-host-20261025-004` acquired the existing lock with O_RDONLY
and LOCK_EX|LOCK_NB. Its actual cgroup caps were CPU200%, MemoryMax3221225472,
swap0 and Tasks64. The twelve H08 tests and fixture-directory test passed. The
wrapper then misclassified the CLI package's no-test-files event as a skipped
test and exited1 before vet. The failed receipt remains unchanged. Supplemental
`observation-repair.json`, SHA256
`ec6a65a8fe33393eb3e11c60e3ceab161678c2f97a7ff10b739f04c318af567c`,
distinguishes named-test results from the empty CLI package and rehashes all1792
originally selected source inputs with zero drift. It replays no tests.

The separately admitted continuation005 ran only CGO0 vet and the maintained
`h08/hostcheck` over those retained JSON bytes. That command delegates exact
package/test classification to `runner.RequireGoTestRoots`. Both commands
exited0, confirming all13 named passes with zero named skips. Its receipt SHA256
is `0488f725be51e414f543a08141887570cfe1ede0eae796d3d93d07ff3adb973f`.
The original source selection, new hostcheck source and five Go tools remained
unchanged. The readonly lock was released at12:16:57.008327UTC; the cgroup was
absent on readback. Its actual caps matched004. No native, fake C or CGO1
execution occurred in either attempt. Current compiler-target, genuine CPU11,
separate fake-loader and graph runtime evidence remain pending.
