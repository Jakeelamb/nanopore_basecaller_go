# D01 load-only image lineage probe

The two predeclared original cubins were each loaded once. Both native probes
completed one zero-option load, four lookups and zero launches under the audited
five-phase contract. sm_120 produced whole SHA `15145e5b…`; sm_120a produced the
predeclared whole SHA `d296e622…` byte-for-byte. Both old Go receipts remain
failed because the observation reader confused receipt annotations with actual
exec environment variables. No probe is repeated for the reader correction.

The current reader correction is a source candidate awaiting bounded host and
read-only CLI checks. It has not qualified either load observation. Capture002
remains failed; its 32 numerical buffers are outside this correction.

The finite matrix has one original cubin per process: sm_120 or sm_120a, each
261,192 bytes, selected by the original D01 image policy. The archive hash is
`17cf3083c5db538f1e275b0a41ac19560b5986c143634e8a25540b9735f5805f`;
its decoder member is
`930b35afa9e352b412b58e07a43d693630280ed9f422dc324106b2b8e700758f`.
These are provenance inputs, not linked code. The prospective successful output
must be the complete 266,280-byte module with SHA256
`d296e6223866ff4b5386ab0b92932bf5330f3977dbb20abb02c07f43596aefdc`.
A different output fails even if names, SM, symbols or some sections match.

Each case declares `none` or `four-original-functions` lookup mode before
execution. The initial proposed matrix uses four fixed `cuModuleGetFunction`
lookups to permit lazy finalization without executing a function. No attributes,
activity timing or extra load is needed. Both modes use the same exact five
phases: `pre-model`, `pre-load`, `post-load`, `post-lookup`, `post-cleanup`.
The post-lookup checkpoint exists even with zero lookups.

## Observation contract

The application presents its original bytes to exactly one
`cuModuleLoadDataEx`, with zero options. It retains the primary context, sets it
current and records its pointer, CUPTI context UID, actual device UUID, SM and
API driver version. The same selected userspace driver/CUPTI providers and
unchanged default environment/cache policy are required. The probe never
clears caches, sets FORCE/DISABLED options, changes driver policy or selects
old SASS. A version query does not identify the kernel-driver build; provider
pins and a retained `/proc/driver/nvidia/version` snapshot supply that separate
record. Capture002 did not record a device UUID, which limits historical device
identity claims.

The callback observes the whole driver domain and resource module loads. Static
bounded storage holds 256 API records, four 1MiB module copies and one 261,192
byte input copy. Callbacks only copy bytes/metadata and retain error codes; they
never call CUDA APIs, allocate output files or throw across the C boundary.
Every API entry/exit carries a sequence, OS thread, context pointer/UID,
correlation ID and active operation token. The load entry copies the exact
input bytes while parameters are valid; its successful exit copies the returned
CUmodule. Lookup entries retain that CUmodule and the exact requested name;
successful exits retain CUfunction handles.

The installed headers expose no CUmodule-to-CUPTI-module-ID query. Consequently,
a resource can bind only when it is actually delivered inside an observed
same-thread load/lookup callback interval, with the same context and active
token. A lookup interval must consume the original successful load handle.
Asynchronous, unbound, multiple or ambiguous resource callbacks fail. Callback
nesting is observed and tested, never assumed to be guaranteed by CUPTI.

The output JSON and raw entry/module copies are written after callbacks cease,
with explicit close/state checks. Input owners survive exception cleanup. Failed
unload/release/observer cleanup flushes failure logs and exits without destroying
owners. A fresh output-directory check cannot overwrite an older attempt.

The importer rehashes raw files, checks exact input/output extents and requires
one successful paired load, the declared lookup count, actual unload/release
callbacks, complete cleanup, no callback failures and zero launch callbacks.
It binds the resource to its observed operation and returned CUmodule. This
returns only `reproduced_transformation_only`; it cannot modify capture002 or
qualify its 32 numerical buffers. If both original inputs produce identical
output, the combined report must retain source-architecture ambiguity.

The first host turn passed five roots and 32 named tests/subcases, including all
11 fake-native cases, plus vet. Its later Go preparation failed because the new
validator assumed an empty reported `GOEXPERIMENT`; the selected toolchain
reports its `nodwarf5` default. That failed attempt remains retained. The
correction explicitly pins `GOEXPERIMENT=nodwarf5` and rejects other reported
values. A fresh bounded Go-only turn passed the new regression, vet,
preparation002, the actual retained-controller plan check, four independently
rehashed plan mutants and four unadmitted native modes. The unchanged fake C++
evidence was reused without another compilation or fixture execution.

Raw evidence is under `artifacts/decodergpu-host-20261025/lineage/`.
`load-architecture-matrix-001.json` pins both original failed receipts, API entry
copies, whole output copies and cleanup evidence. The immutable standalone
producer is `artifacts/tensor/d01-lineage-build-002/phase.json`, built from
preparation003. Its source and bytes remain unchanged during reader work.

## Read-only observation replay

The v1 contract's `Environment` field retains the historical command-receipt
envelope. The shared guard records CWD and gate-FD annotations there and appends
the native PID to its command receipt. Those annotations are not passed to exec.
The native process receives the eight variables defined by the selected guard
policy. The corrected reader validates the exact receipt annotations separately,
then compares native environ with the exact remaining exec entries. It never
filters observed variables. Extra, duplicate or changed variables, including
CUDA/cache settings, fail.

`ReplayLineageLoad` uses a newly frozen current Go consumer and the old runtime,
build, failed load receipt, guard transcript/maps and supervisor/watch receipts.
It recognizes only the exact old environment-reader error together with native
exit0, successful shared guard replay, controller/supervisor exit1, exact
historical process identities and complete cleanup. It then recomputes the
binding from saved input/output bytes and callbacks. The original whole-image
gate is unchanged, so the sm_120 output remains rejected.

The result is a separate `LineageReplayReport`; it never rewrites or relabels an
old failed receipt. The existing `CheckLineageLoad` path still requires a
successful producer execution. Original producer snapshots and watched
controller identity stay distinct from the new replay consumer's source and
binary custody. A successful replay can qualify only the contemporaneous
load-only observation. It cannot produce a numerical capture receipt.

## Compiler, controller and runtime boundaries

Implementation stays in isolated new `verification/decodergpu/lineage_*.go`,
`lineagecmd/` and new `native/lineage_*` files. The four accepted native source
files, old controllers and captures remain unchanged. Existing private D01
component parsers are reused; the old Koi/CUDART seed and decoder-member link-map
assumptions are not reused for the new standalone target.

The standalone recipe is one C++23 translation unit with `-O2 -DNDEBUG -fPIC
-fno-lto`, selected system/CUDA headers, and absolute pinned `libcuda` and
`libcupti` plus the selected compiler's standard runtime. No Koi, CUDA runtime,
Torch or cuDNN is linked or allowed in the process policy. SHA and phase protocol
includes are immutable copies of their accepted source bytes. The accepted
selected compiler family/plugin closure is an external anchor; all live
selections and loader-cache bindings are checked before its first query.
Maintained phases remain query, complete `-M` header freeze, then one compile
with exact `-MD` equality and one link with closed trace/map evidence. No phase
can execute the built target.

A new CGO0 lineage controller freezes its actual Go dependencies/embeds/module
anchor, selected tools, immutable source copies, exact build commands and binary.
Runtime planning is pure ELF metadata and source-specific selected provider
policy, including CUPTI and the accepted no-import auditor. Shared
`ExecuteGuarded`/`ReplayGuarded` consumes independently pinned five-phase and
policy/custody contracts. Actual serialized-plan validation must reach the
exact absent-run identity boundary before any launch. Accepted watcher019
supervises exact controller/native roles under an independently owned outer
cgroup. The capture controller acquires the existing canonical lock read-only
and nonblocking; its outer wrapper does not own a second lock.

## Planned source/host controls

Two ordinary Go roots cover exact original architecture/lookup contracts and
synthetic callback interval binding. Negatives include wrong original/input
copy/output digest, missing or asynchronous resource, wrong context/thread,
ambiguous modules, resource outside its interval, altered returned load/lookup
handle, missing lookup/cleanup callbacks, nonzero callback errors, empty UUID,
launch evidence despite a zero summary count, and native failure status.

Fake-header controls will compile the actual new source only against tiny
fake CUDA/CUPTI declarations, never genuine CUDA/Torch headers or libraries.
They will exercise callback-lifetime input/output copies, load and lookup
resource association, unrelated-thread/outside-interval failure, overflow,
exception containment, failed cleanup ownership, existing-output refusal and
write/close failure. Positive synthetic payloads copied from retained files are
labeled synthetic and do not constitute driver transformation evidence.

Before any test turn, root receives the exact named selectors/commands, source
pins, required assets and prospective paths. All turns use 2CPU, 3GiB, no swap,
Tasks64 and bounded children. Genuine native phases have separate review and
execution grants. Failed attempts are retained.

## Candidate mechanism and limits

NVIDIA documents runtime cubin re-finalization for CC10+ and distinguishes it
from PTX JIT; this is a possible mechanism, not proof of this invocation.
[CUDA13.4.2 Binary Finalization](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-programming-guide/01-introduction/cuda-platform.html#binary-finalization)

The stable R615 release notes describe a Blackwell nested-divergence correction
and default re-finalization/patching policy for older compiler output. This does
not establish which path produced capture002's module or the exact behavior of
this installed driver build.
[R615.71.09 fixed issues](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-615-71-09/index.html#fixed-issues)

CUPTI promises callback-lifetime parameter/resource storage and associated
module bytes/IDs, not a hash-preserving original-input guarantee. The probe must
observe the association it needs.
[CallbackData](https://docs.nvidia.com/cupti/api/structCUpti__CallbackData.html),
[ModuleResourceData](https://docs.nvidia.com/cupti/api/structCUpti__ModuleResourceData.html)

Exact input-to-output lineage describes the load-only probe. It does not recover
an unrecorded historical internal driver path. The later probe cannot supply
capture002's missing contemporaneous device UUID or kernel-driver observations,
and a reader success cannot qualify its 32 numerical buffers. Historical
numerical acceptance requires separate evidence and review outside this unit.
