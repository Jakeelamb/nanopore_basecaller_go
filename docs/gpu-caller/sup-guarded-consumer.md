# Tiny SUP guarded consumer

The dedicated `./verification/supgpu/graph/cmd` has three modes. Its selected
CGO1 role is `sup-graph-command`, output basename `sup-graph`. Its compiled
artifact is selected through `cgobuild.ReadSUPArtifact(evidence, selection,
expectedRoot)`. Ordinary compilation is not a
selected build proof or runtime qualification.

```
SUP -mode consumer -admission ABSOLUTE_ADMISSION -admission-sha SHA256
SUP -mode guarded-child -request ABSOLUTE_REQUEST -request-sha SHA256
SUP -mode replay -admission ABSOLUTE_ADMISSION -admission-sha SHA256 -receipt ABSOLUTE_WRAPPER -receipt-sha SHA256
```

`SUP` is the absolute selected executable, with identical parent/child bytes
and canonical selected repository cwd. The child command is private to the
existing public audited guard; it opens inherited phase pipes before reading
its request. The parent and child use exact argv. Source capture build/runtime
anchors are data witnesses, never Go build selection or runtime permission.

Private wires live beside their readers. Request schema is
`nanopore-sup-tiny-request/v1`; admission is
`nanopore-sup-tiny-admission/v1`; identity is
`nanopore-sup-tiny-device-plan/v1`. Field names without JSON tags use exact Go
field spelling; shared SUP fields retain their JSON tags. Every field is
required, nulls/duplicates/unknown keys/trailing values reject before typed
decoding. Files must be bounded canonical regular files; symlinks and FIFOs
reject. No normalization repairs a selected policy.

The request independently pins the successful SUP evidence/selection pair,
closed backend four-pin tuple, genuine retained CPU13, four `WitnessAnchors`
(`Capture`, `Prepared`, `SourceBuild`, `SourceRuntime`), model directory,
ordinal and fresh numerical output. Historical CPU13 uses its recorded H08
build/root through the unchanged `h08.ReadCPU13`; this command never relaunches
CPU13 or invokes H08 Consume. All prepared config and 139 weight files must
match before deserialization. Closed backend metadata supplies all 23 actual
runtime providers. Their actual sized pins, SONAME and ordered Needed list,
and the output ELF, must match Allowed. An absent backend SONAME stays absent.

Admission pins the exact guard plan/contract, request and metadata plan, and
fixes command cwd/argv/output. Guard action is `sup-tiny`, observer phases
`pre-model`, `pre-forward`, `post-forward`. Guard custody must contain exact
sized request, selected build pair/output, backend tuple/live inputs/providers,
CPU13, source anchors, prepared inputs/model files, identity plan/tool/closure.
The plan and contract are independently pinned by admission; they cannot
include their own hash. Outputs are disjoint strict descendants of the selected
root's `artifacts`, with canonical existing parents, outside selected inputs.
Parent runtime additionally requires absent guard/numerical/identity outputs.

Admission additionally requires sized `MetadataGuardPlan` and
`MetadataContract` pins. The latter selects exactly
`{"schema":"nanopore-audited-metadata-contract/v1","role":"observer","action":"device-identity","names":[]}`.
Null/omitted names and any phase, role or action variation reject. Numerical
admission explicitly requires the ordinary phase schema. Metadata uses the
same execution core, replay core and unchanged native auditor. At completion,
live and replay both require root, auditor and startup providers in accumulated
approved audit history; a legitimate close does not erase that history. Initial
partial hello maps do not have a full startup-set requirement.

The independent metadata plan selects the actual `nvidia-smi` executable,
seeds, search directories, full ELF closure, exact Args/Environment, nonempty
PCI and driver expectations, and explicit timeout seconds (1 through 60).
`LazyProviders` is an explicit nonnull pin list, possibly empty; `Seeds` must
be exactly the executable plus those providers. The selected metadata policy
contains exactly the closure's non-root images, with actual ELF SONAME/Needed
values. Only dependencies reachable from the executable's closure bindings
are startup-required. Optional providers need not load; every loaded provider
must be preselected. No directory or observed mapping grants new authority.
Both policy roots equal the identity executable. Its auditor must equal the
numerical plan's selected auditor. Loader directories equal `SearchDirs` and
Environment equals shared `SelectedModelEnvironment(policy)`, including
`LD_AUDIT`, `LD_LIBRARY_PATH` and `LD_BIND_NOW`. The metadata plan's Inputs must
include sized identity JSON, executable, closure inputs and auditor. The
numerical plan's Inputs additionally contain the metadata plan/contract pins.
Identity JSON has no backreference to those plans, avoiding hash cycles.
Its query returns index, UUID, PCI and driver. Source UUID is compact lowercase
32hex only; metadata UUID is explicitly parsed as GPU-8-4-4-4-12. The tensor ABI
is not claimed to expose UUID. Before/after metadata must surround the actual
child command and agree with the authenticated SUP source witness.
Metadata commands use the numerical command's selected repository cwd, exact
identity Args and fresh sibling outputs `GUARD_OUTPUT-identity-before` and
`GUARD_OUTPUT-identity-after`. Wrapper Metadata now contains two complete
`GuardedRun` values, including raw command receipts, transcript/maps and log
pins. Both are replayed against admission before CSV and temporal checks.

**External launcher contract:** the root-owned external launcher must acquire
the existing stock `host.lock` read-only with EX|NB **before unit startup** and
retain it until watcher, unit and every retained descendant are terminal and
empty, including parent failure. SUP parent, child, metadata and ExecuteGuarded
never acquire it again. `runner.CheckRoot` only checks existing root intake.
The controller must separately select/review actual resource, identity,
watcher and lock contracts before any native execution. ExecuteGuarded does not
enforce cgroup caps, an end-to-end timeout, or parent-death descendant cleanup.
The canonical API is now `ExecuteGuarded(ctx, plan, contract, command)`.
Nil/already-cancelled contexts reject before output creation. Metadata derives
a timeout context from its selected TimeoutSeconds and caller context. Active
cancellation kills its owned private process group in the guard loop, drains
the pipes, waits and records failure/cleanup. Setpgid and leader Pdeathsig are
shared with numerical execution. Finite outer bounds and descendant admission
remain the launcher's responsibility; no guard acquires the stock lock.
No reusable runtime authorization token or global flag is introduced.

After approved pre-model, the child opens the selected tensor engine, loads
weights, constructs device/stream/model and uploads the prepared F16 NCT
signal. After pre-forward it calls the existing arithmetic comparison exactly
once, then requests post-forward. Cleanup closes input, model (including a
nonnil failed constructor), stream, device, engine and phase pipes, joining
errors. Host weights have no invented close operation. Failure receipts retain
partial comparison and cleanup evidence; no producer is automatically rerun.

The parent replays the public guard using independently selected expectations,
then authenticates the fixed child `consumer.json` and the guard-pinned stdout
Result. Success requires one forward, 191 events, 733 ordered logical/backing
comparisons, original score digest, source/model/build/backend/CPU13 custody,
event descriptor contracts and successful cleanup. Wrapper is
`GUARD_OUTPUT/sup-graph.json`; failure projection is retained in
`GUARD_OUTPUT-identity/parent.json`. Replay does not trust receipt-selected argv,
status or totals. Metadata raw clocks/exits/log pins and guarded run receipts
are retained before numerical projection.

Controller verification after both source owners freeze: format owned files,
run focused boundary controls and `go test ./...`, then maintained
`./verification/verify.sh --source "$PWD/external/dorado-source"` under the
controller's source/runtime policy. `TestGenuineSUPAdmissionReaderOptIn` accepts
`SUP_GRAPH_ADMISSION` and `SUP_GRAPH_ADMISSION_SHA256`, reuses genuine retained
producer artifacts without launch, and rejects omitted providers, fabricated
SONAME and changed backend pins. A skipped fixture supplies no qualification.
Genuine SUP artifact selection, runtime admission and numerical execution
remain separate controller gates.

Integrated verification on 2026-10-29 passed `go test ./...` with CGO1:
868 root tests and 1,916 subcases passed; 92 unselected opt-ins skipped.
The maintained parity command passed its CPU-package/native-algorithm scope.
Two independent same-model review processes completed for each source unit;
the metadata correction, historical-root correction and watcher correction
have no unresolved findings. The retained 191-event source capture also passed
the maintained import command without replaying its native producer.

The real CPU metadata fixture reproduces the original defect after a matched
positive: unselected B returns valid CSV and runs its constructor. The corrected
guard accepts A with saved replay and denies B's actual open before its
constructor. Both cases use the same four compiled ELF files. These checks
qualify that host loader boundary; SUP numerical GPU execution remains pending.

`TestGenuineMetadataProviderBoundaryOptIn` takes independently pinned
`SUP_METADATA_FIXTURE` / `SUP_METADATA_FIXTURE_SHA256`. It runs a real metadata
CLI with explicit same-SONAME lazy providers A/B and the root-selected auditor;
it requires matched A to return valid CSV and replay, and unselected B to be
denied on its actual open before its constructor marker. Fixture sources are
`verification/supgpu/graph/testdata/metadata-{cli,provider}.c`. The exact input
wire and generation/old-boundary procedure are in
[SUP metadata provider controls](sup-metadata-controls.md).
This fixture needs native controller admission; an unselected skip proves
nothing. Actual nvidia-smi remains a separate device/runtime admission.

This admits only N32/chunk768/T64 F8/I8/remove_bits4 numerical consumption.
Full/short/VCS remain held by the source witness/oracle. Planner support is not
native parity, complete calling, biology, performance, HG005 selection or live
readiness. No approval registry is written.
