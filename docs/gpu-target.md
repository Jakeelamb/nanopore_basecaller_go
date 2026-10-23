# GPU runtime and dispatch profile

The integrated host pass ran all 16 profile test functions, the full ordinary Go
suite, build, vet and formatting checks. Protocol validation, card regeneration
and the dispatch rejection controls also passed. Missing runtime qualification
was rejected before native work. This does not qualify the standalone Koi link,
GPU runtime maps or device observations.

P01 has an implemented and host-verified source checkpoint. Native observations
remain unobserved. No native compiler, GPU query, lock operation, download or
benchmark was executed for this checkpoint. Harness2 remains held after its
terminal failure pending reader/audit. An idle GPU or free lock is not admission.
No native phase has been released.

## Frozen scope

`provenance/gpu-target.json` binds Dorado revision
`8b8fc5d36a9c0baab262a743cb175b5878e38ca6`, HAC v6.0.0 and SUP v5.2.0 config
hashes, expected 50/139 weight counts, source CUDA13 family, retained ABI4 UUID,
default development options, SQK-LSK114 and explicit padded batches. Config pins
come from `provenance/default-cpu-control-plan.json`; UUID comes from canonical
`artifacts/tensor/replay-abi4-cuda-001/runtime-pins.json`. These are prior pins,
not current capability observations. Official CUDA12 remains a separate family.

`BackendIdentity` and `DispatchPlan` contain private value fields and return
copies. Identity includes both config pins, family, UUID, CUDA_VISIBLE_DEVICES
presence/value, device properties, numeric capabilities and default options.
Nil Koi statuses mean unobserved. Success is zero; NOT_SUPPORTED is one.
Invalid-value and internal-error statuses fail. `Bind` requires trusted caller
provenance; it does not authenticate arbitrary user-supplied observations.

## Source predicates

Pointers are relative to pinned `external/dorado-source/dorado/`.

| Decision | Predicate | Source |
| --- | --- | --- |
| HAC fused eligibility | major >=8, SMs >=9; supported C=1024 K=128 | `nn/KoiUtils.cpp:7-43` |
| SUP tiled | supported shape, CUDA, Koi F16 success | `nn/TxModules.cpp:805-834` |
| SUP shape | C=512, heads=8, window=127/128, feedforward=2048 | `nn/TxModules.cpp:809-811` |
| SUP FP8 | E4M3 success and default koi_use_f8=true | `nn/TxModules.cpp:439-440` |
| SUP int8 | default koi_use_i8=true | `nn/TxModules.cpp:833` |
| Rounding | remove_bits=4 | `nn/TxModules.cpp:403-412` |
| VCS | CUDA non-Orin, nonempty devices, model/device predicates | `api/runner_creation.cpp:24-60` |
| VCS default | enabled unless explicitly disabled | `cli/cli_lib/basecaller.cpp:1192` |

SUP F16 unavailability rejects because generic/Volta SUP needs additional cards.
Observed E4M3 unavailability selects non-FP8. Availability selects FP8. Unknown
capability cannot become false. No implicit fallback disables VCS or selected
precision. Fixed chunks are explicit. Batches must be positive multiples of 64
for HAC and 32 for SUP. Automatic batching is not frozen. Both fixed/VCS modes
and all four trim/split combinations are preserved in the 16-request matrix.
That matrix checks profile propagation, not trimming or splitting computation.

## Implemented verification boundary

Only these six owned paths are involved:

- `provenance/gpu-target.json` freezes supported inputs.
- `verification/gpuprofile/profile.go` owns identities, predicates, qualification,
  capture, closed import and frozen-evidence comparison.
- `verification/gpuprofile/profile_test.go` owns source tests.
- `verification/gpuprofile/cmd/main.go` wires source preparation, capture and replay.
- `verification/gpuprofile/native/probe.cpp` is the single standalone native probe.
- `docs/gpu-target.md` records this contract and its limits.

Public interfaces are `DecodeTarget(io.Reader) (Target,error)`,
`CheckRequest(Target,Request) error`, `Bind(Target,Observation) (BackendIdentity,error)`,
`Prepare(BackendIdentity,Request) (DispatchPlan,error)`,
`CaptureProfile(ProbeOptions) (ProbeReceipt,error)`,
`ImportProfile(path,frozenSHA string) (ProbeReceipt,error)`, and
`VerifyFrozen(frozen,candidate ProbeReceipt) error`.
`PredicateHeader` extracts source-pinned native expressions. `ValidateProbe`
compares closed raw observations against Go dispatch and the request matrix.

Capture rejects missing admission, invalid target/qualification, changed accepted
runtime files, unsupported environment, wrong toolkit version, config bytes,
Koi pins or predicate sources before canonical lock acquisition or child execution.
Those paths return a zero receipt and create no attempt. Qualification follows
accepted ABI4 basic/build/backend pins from `verification/koiquant/receipt.go`.
Its small private qualification helper is reproduced locally. No P02 extension,
production tensor operation, shared registration or shared-file edit is required.

After admission and preflight, capture uses existing `runner.Acquire` on the
canonical existing nonblocking host lock, then `runner.NewAttempt` under this
worktree's `artifacts/tensor`. It retains failure attempts. `runner.Execute`
records compiler, compile and probe argv/environment/clocks/exits/stdout/stderr
hashes. Each raw command receipt is written before output interpretation.
Compiler identity must match the accepted build. Depfile inputs, generated
predicate header, probe binary, configs, Koi static archive/header and consumer
sources are pinned and checked again after execution. Actual runtime maps must
include CUDA runtime and driver and match accepted GPU runtime paths/hashes,
including resolved symlink aliases. Foreign Torch/CUDA12 mappings reject.

The probe uses exact pinned Koi eligibility functions, SUP selection expressions,
VCS support function and CLI default expression. Tiny context adapters provide
only the two supported shapes, default development options and CUDA non-Orin
single-device context. It emits actual numeric Koi statuses, device properties,
versions and source-predicate results. It does not load weights, launch model
kernels or establish that a model graph chose or executed those kernels.
Source predicate agreement is a narrower observation than model dispatch parity.

Successful observation status is `predicate_observation_passed`. Raw and
Observation pointers remain null on failure; uninterpreted native bytes remain
in raw logs. Closed import requires a caller-supplied trusted frozen manifest
hash. It rejects missing/null required scalar fields, unknown/case-aliased fields,
duplicate keys, trailing JSON, failed receipts, changed files and projection/raw
or Go/native disagreement. Import revalidates accepted qualification and actual
saved maps without executing a child or acquiring a lock. `VerifyFrozen` rejects
changed capabilities even when the changed observation is independently valid.
It compares against already trusted frozen evidence; it is not authentication.

The import-completeness correction uses capture schema
`nanopore-gpu-profile-capture/v2`. Its required typed `compile` object binds the
accepted compiler, absolute Koi/CUDA roots and attempt directory. Capture and
import share one exact compile argv constructor. Import requires the finite
consumer source set, probe binary, generated predicate header, depfile, Koi
header/archive, CUDA header and saved maps. It reparses the hash-checked depfile,
requires its target to be the recorded probe output, requires all listed header
pins, and requires probe source, generated header, Koi header and CUDA runtime
header membership. A compiler `--version` command cannot replace compilation.
No old receipt or checkpoint is rewritten to fill these new required fields.

These completeness guards run before accepted-qualification replay. Three new
retained-metadata tests give the mutated metadata its deliberately designated
hash, then require the specific semantic guard to reject it. Their baseline
passes the shared compile-closure guard; it is a synthetic metadata fixture,
not fabricated qualified/native evidence. A trusted external frozen hash still
authenticates the chosen exact manifest. Semantic completeness cannot establish
the truth of an arbitrary caller-approved manifest or reconstruct dependencies
deleted from both a malicious depfile and its authenticated history.

## CLI modes after specific release

Source preparation uses `-plan`, `-model`, `-config-sha256`, `-batch` and `-out`.
Options also include `-output fastq|bam`, `-kit SQK-LSK114`,
`-disable-variable-chunk-sizes`, `-no-trim`, and `-disable-read-splitting`.
Unknown flags/models/families reject. Preparation writes null observation and
status `source-prepared-runtime-unobserved` into a fresh directory.

Capture uses the same request flags plus `-runtime-admitted`, `-basic BASIC`,
`-build BUILD`, `-koi-root CUDA13_KOI`, `-cuda-root CUDA_TOOLKIT`,
`-model-root READ_ONLY_MODELS`, and `-out artifacts/tensor/NEW_ATTEMPT`.
The admission flag asserts an actual specific coordinator release. It is not a
scheduler or permission obtained by waiting. Both config files are byte-checked;
weight counts remain expected counts and do not authenticate weight manifests.

Replay uses `-plan provenance/gpu-target.json -replay SAVED/profile.json
-frozen-sha256 TRUSTED_SHA`. It prints the validated receipt and does not recapture.
Replay and runtime-admitted modes are mutually exclusive. All commands are
currently deferred under the source-only grant.

## Deferred checks and remaining closure

Required source tests are:

- `TestUnknownCapabilitiesFailClosed`
- `TestFrozenIdentityNegativeControls`
- `TestCapabilityChangesAlterDispatch`
- `TestTrimSplitFixedVCSMatrix`
- `TestRejectUnsupportedRequests`
- `TestSourcePredicates`
- `TestCaptureRejectsAdmissionAndQualificationBeforeLock`
- `TestClosedProbeRejectsMissingNullUnknownDuplicateAndTrailing`
- `TestNativePredicatesMustAgree`
- `TestFrozenEvidenceRejectsValidCapabilityMutant`
- `TestPinnedPredicateExtraction`
- `TestImportRejectsChangedManifestAndMissingEvidence`
- `TestCorruptQualificationDoesNotAllocateReceipt`
- `TestRetainedMetadataRequiresConsumerProbeAndDependencyPins`
- `TestRetainedMetadataBindsExactCompileArgv`
- `TestRetainedMetadataReparsesDepfileTargetAndRequiredHeaders`

All 16 are written and passed on the host. Required subsequent checks include the
maintained parity command, C++ compile/link, admitted native capture, saved-byte
import, effective frozen-evidence mutant and independent exact-delta/receipt
review. No skipped test qualifies a gate.

Concrete unverified closure is standalone Koi link/toolkit compatibility and
actual mapped runtime agreement with accepted ABI4 inputs. The implementation
attempts that closure and retains its failures; no missing P02 interface is known.
Full weight manifests, per-kernel execution, allocation/live-byte audits and
biological acceptance remain downstream evidence. Declare initialized regions
and exclusions prospectively before those captures. No HG005 selection is used.

The integrated units are committed together on the port branch. Exact
pre-correction small-source snapshots are retained in the authorized outbox under
`go-profile/pre-correction-import-p2/`. This correction addresses the two accepted
P2 static-review findings; it claims no acceptance. Host checks and their hashes
are retained in `artifacts/gpu-plan/host-check-20261023T183053Z/`. Native and
independent review remain outstanding.
