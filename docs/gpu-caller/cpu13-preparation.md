# CPU13 metadata preparation

`verification/hacgpu/h08/preparecmd` prepares fixtures and CPU13 admission
metadata, and preflights existing CPU13 admissions or fake-loader proofs. It
reads authenticated files and writes fresh preparation files. It never starts
a subprocess, acquires the host lock, loads a backend, initializes a GPU, or
runs a producer. Successful output contains `RuntimeHeld: true`; it supplies
no runtime admission authority or whole-caller qualification.

The existing H08 runtime CLI modes and existing record schemas are unchanged.
The metadata entry points are:

```go
ReadCPU13Admission(pin Pin) (RuntimeAdmission, CPU13Request, error)
ReadFakeLoaderProof(pin, target, build, selection Pin) (FakeProof, error)
ReadPreparationJSON(pin Pin, out any) error
supgpu.ReadGuardPlan(planPin, contractPin supgpu.Pin) (supgpu.GuardPlan, supgpu.PhaseContract, error)
supgpu.ReadRuntimePolicy(selected supgpu.Pin) (supgpu.RuntimePolicy, error)
```

H08 now derives shared guard byte counts by streaming and authenticating the
selected files. A missing, changed, empty, unreadable, or nonregular file returns
an error. Nonblocking open precedes the same-descriptor regular-file check,
hash and byte count, so a FIFO without a writer rejects promptly.
The same sized plan and contract pins reach preflight, execution, and replay;
the shared guard's strict byte checks are unchanged. H08 metadata preflight
uses that shared guard reader to authenticate the complete policy and inputs.

## Invocation

Select a separately qualified `CGO_ENABLED=0` build of this command as
`CPU13_PREPARE`. Run it from the selected source checkout inside the
coordinator's bounded resource and stock-lock envelope:

```sh
"$CPU13_PREPARE" \
  -operation fixture -request "$FIXTURE_OPTIONS" -request-sha "$FIXTURE_OPTIONS_SHA"
"$CPU13_PREPARE" \
  -operation admission -request "$PREPARATION_OPTIONS" -request-sha "$PREPARATION_OPTIONS_SHA"
"$CPU13_PREPARE" \
  -operation cpu13-preflight -request "$PREFLIGHT_REQUEST" -request-sha "$PREFLIGHT_REQUEST_SHA"
"$CPU13_PREPARE" \
  -operation fake-preflight -request "$FAKE_REQUEST" -request-sha "$FAKE_REQUEST_SHA"
```

Each variable must be selected independently before invocation. Output parents
must already exist without symlink aliases. The SHA is of
the complete request file bytes, including whitespace. The CLI accepts only
these flags and operations, rejects positional arguments, unknown JSON fields,
trailing JSON, partial or malformed pins, and nonabsolute or unclean paths.
Errors produce no success JSON. Failed preparation files remain for inspection;
retry with fresh output paths rather than changing or deleting the attempt.

All four CLI operations and the direct `PrepareCPU13Admission` constructor
authenticate their external request through `ReadPreparationJSON`. It checks
the pin shape, opens once with `O_RDONLY|O_NONBLOCK`, and requires a regular
file using that same descriptor before reading. Requests are bounded to 64 MiB;
the read is limited even if the file grows, and its byte count must equal the
descriptor's observed size. The reader hashes the exact bytes it decodes,
including whitespace, and requires a nonempty JSON object with no unknown
fields or trailing values. A FIFO with no writer and a syntactically valid pin
rejects as nonregular before domain validation or output creation. There is no
separate CLI request decoder.

## Typed JSON requests

All H08 pins use the exact fields `Path` and `SHA256`. Paths are absolute and
clean; SHA256 values are full lowercase hexadecimal digests from independently
selected files. Shared `supgpu.Pin` values in policy and guard records instead
use `path`, `sha256`, and `bytes`; byte counts must match authenticated files.
No future target, build, selection, policy, logger, or proof digest is supplied
by this document.

For `fixture`, the request is the existing `h08.FixtureOptions` JSON object:

| Field | Type |
| --- | --- |
| `SourceManifest`, `Archive`, `Backend`, `Weight` | H08 Pin each |
| `Directory` | Fresh absolute fixture directory under the invoking checkout's ignored `artifacts/` |
| `Runtime` | Object mapping each selected canonical provider absolute path to SHA256 |

`PrepareFixtureView` remains authoritative. It authenticates and copies the
eight retained oracle files, pins the source manifest, qualified archive,
HAC conv0 weight and runtime providers, and creates the backend symlink plus
fixture-only manifest, runtime record and `view.json`. Existing directories
reject. Backend basenames that collide with generated metadata reject before
preparation. Output is `{View: FixtureView, Pin: Pin, RuntimeHeld: true}` with
JSON keys spelled as shown.

This operation permits reuse of immutable old oracle payloads when every
declared pin matches. It does not rerun the old producer or make its historical
backend a qualified current backend. Saved fixture views remain unchanged.

For `admission`, the request is `h08.CPU13PreparationOptions`:

| Field | Type / required value |
| --- | --- |
| `Schema` | `nanopore-h08-cpu13-preparation/v1` |
| `ParentBuild`, `ParentBuildSelection` | Selected H08 parent artifact and independent selection Pins |
| `ControllerBuild`, `ControllerBuildSelection` | Selected CPU13 test artifact and independent selection Pins |
| `Fixtures`, `FakeProof`, `Policy` | Existing fixture view, accepted fake proof, separately selected runtime policy Pins |
| `Backend` | Existing `closedbuild.ArtifactOptions`, with `Build`, `Plan`, `Producer`, `Output` Pins |
| `Directory` | Exact existing absolute command working directory |
| `MetadataDirectory` | Fresh strict descendant of `Directory/artifacts/`; parent must exist |
| `GuardOutput` | Distinct absent strict descendant of `Directory/artifacts/` for guarded-runtime output |
| `RequestOutput` | Distinct absent strict descendant of `Directory/artifacts/` for request output |

All selected input paths must be unique. Every output must be clean and absolute,
with an existing canonical parent, and remain inside the ignored artifacts tree.
The artifacts directory itself cannot be selected. Outputs must be absent,
distinct, and cannot contain each other or collide with the selected inputs or
command directory. The policy's `reference` and `observer` must already equal the
selected CPU13 executable, including byte counts. Preparation never retargets
a policy or infers provider allowlists from search directories. Audit, allowed
providers, alternates, loader directories and backend provider closure remain
subject to the existing authoritative validators.

Before output creation or normalization, the constructor passes the original
independently selected policy through `supgpu.ReadRuntimePolicy`, authenticating
its path, SHA256 and derived byte count. That reader reuses the shared closed
JSON decoder and full policy validator. Duplicate observer/provider keys,
wrong-case keys, missing fields, nulls, unknown fields and trailing JSON reject
on the original wire bytes. The shared preparation-options decoder does not
replace this stricter runtime-policy wire boundary.

The constructor authenticates both role-specific artifact/selection pairs,
fixture view, genuine backend artifact, fake proof, and selected policy. It
creates `request.json`, `contract.json`, `plan.json`, and `admission.json` using
exclusive file creation in the fresh metadata directory. The contract fixes
observer action `hac-cpu13` and exactly `pre-model`, `pre-forward`,
`post-forward`. The command uses the existing fixed genuine eleven-test
selector; there is no caller-supplied argv. The plan includes exactly the
eleven selected artifact/fixture/backend/proof/policy inputs, the pinned
preparation-options file, and the emitted CPU13 request, with authenticated
nonzero sizes. The final emitted admission must pass the existing CPU13 reader,
including full shared guard validation, before success is reported.

Output is `h08.CPU13Preparation`: `Admission`, `Request`, `Plan`, `Contract`,
`Policy` are H08 Pins and `RuntimeHeld` is true. Guard and request runtime
directories remain absent; only the metadata directory is created.

For `cpu13-preflight`, the request fields are:

| Field | Type / required value |
| --- | --- |
| `Schema` | `nanopore-h08-cpu13-preflight/v1` |
| `Admission` | Existing CPU13 admission Pin |
| `ParentBuild`, `ParentBuildSelection` | Independently expected H08 pair Pins |
| `ControllerBuild`, `ControllerBuildSelection` | Independently expected CPU13 pair Pins |

It invokes the same CPU13 admission reader and requires exact agreement with
the independently expected pairs. Output is `{Admission: Pin, RuntimeHeld: true}`.

For `fake-preflight`, the fields are `Schema` (exactly
`nanopore-h08-fake-preflight/v1`), `Proof`, `Target`, `Build`, `Selection` (each
an H08 Pin). The selected CPU13 artifact must authenticate and its output must
equal `Target`. The unchanged fake-proof reader authenticates the existing
proof against that independently selected tuple. Output is
`{Proof: Pin, RuntimeHeld: true}`. This is a reader, not a fake-proof producer.

## Verification and remaining gates

The pure metadata controls use inert files. They cover changed request bytes,
partial/missing pins, schemas and wire shapes, collisions, exclusive writes,
policy target mismatch, exact guard sizes, role/phase/policy changes and custody
rejection. New inert controls cover the strict original policy wire, independent
outside-artifacts choices for every output, valid independent output parents,
symlink aliases and nonregular pins. Dedicated entry FIFO controls cover the
public admission constructor and each of the four CLI operations, separately
from the existing guard-pin control. They run only the test binary as a named
self-child with a three-second context deadline and one-second `WaitDelay`,
killing and reaping a regressed blocking reader without leaving a blocked
parent goroutine. They check rejection before preparation/success output.
`preparation_fifo_test.go` and `preparecmd/fifo_test.go` use existing entry APIs
and can be copied alone onto frozen production to reproduce the entry hangs
before integrating the fix. Additional shared-reader controls cover directories,
empty and oversized files, valid typed requests, malformed JSON, and exact-byte
hashing even when only whitespace changes. These new controls are unrun in this
worker assignment; execution and formatting belong to the controller.
Counterexamples use the generic option decoder to show how normalization
previously erased ambiguous keys or an omitted provider scalar before shared
validation. Inert files and synthetic rejection inputs are not native evidence.
The supplied `TestH08GuardAcceptsSameSizedMetadataAsSharedGuard` retains its
meaning: a complete inert plan accepted by the shared reader must also pass
H08's metadata boundary.

Verification commands, subject to the same resource and lock rules:

```sh
gofmt -w verification/hacgpu/h08/guard.go verification/hacgpu/h08/guard_size_test.go \
  verification/hacgpu/h08/preparation.go verification/hacgpu/h08/preparation_test.go \
  verification/hacgpu/h08/preparation_fifo_test.go \
  verification/hacgpu/h08/preparecmd/main.go verification/hacgpu/h08/preparecmd/main_test.go \
  verification/hacgpu/h08/preparecmd/fifo_test.go \
  verification/supgpu/guard_metadata.go verification/supgpu/guard_metadata_test.go
CGO_ENABLED=0 go test -count=1 -run '^TestCPU13PreparationRejectsFIFO$' ./verification/hacgpu/h08
CGO_ENABLED=0 go test -count=1 -run '^TestCPU13PreparationCLIRejectsFIFO$' ./verification/hacgpu/h08/preparecmd
CGO_ENABLED=0 go test -count=1 ./verification/supgpu ./verification/hacgpu/h08/...
go test ./...
```

The opted-in `TestCPU13PreparationGenuineIntegration` requires
`H08_CPU13_PREPARE_INTEGRATION=1`, independently pinned options at
`H08_CPU13_PREPARE_OPTIONS` / `H08_CPU13_PREPARE_OPTIONS_SHA`, and an independent
expected `CPU13Preparation` JSON at `H08_CPU13_PREPARE_EXPECTED` /
`H08_CPU13_PREPARE_EXPECTED_SHA`. Expected output Pins must include full hashes
and paths for all four emitted files and the selected policy. The test compares
all of them and rereads the admission. Every attempt requires fresh outputs.
An ordinary skip provides no positive metadata qualification. No complete real
artifact positive was available or exercised in this worker scope.

Fresh source-current H08 and CPU13 compilation/selection, genuine C-only
fake-run logger/proof production, and admitted guarded CPU13 execution remain
separate required work. The maintained parity command also remains required
under controller admission. Metadata success does not establish loader/runtime
acceptance, GPU parity, biological validity, performance, or whole-caller
qualification. Preserve old receipts and failed attempts unchanged.

## Retained metadata evidence

The original H08 adapter rejects a correctly sized inert plan after the shared
guard accepts those same bytes (`h08-byte-handoff-original-002`). The corrected
byte-preserving adapter passes that regression
(`h08-byte-handoff-corrected-001`). Before the policy/output/FIFO corrections,
the retained CGO0 H08 run passed 93 root tests and skipped nine unselected
opt-ins in `h08-metadata-integrated-host-001`. These are historical metadata
and source checks; they do not qualify the source after these corrections.

A retained forced CGO0 build of the preparation CLI predates these corrections
and retains 1,088 selected source
and tool pins, immutable snapshots, before/after package selections and raw
build logs under `artifacts/tensor/cpu13-current-metadata-preparer-001`.
Its executable SHA256 is
`ff3f6462e6fc442301a2225d2c33a3361d16960fa2573775f327cf5d95f4a328`.
The actual `fixture` operation reuses the unchanged independently pinned oracle
manifest `13f995b4a4dec3898e209b8abc79dc5ef0a31c25e8f5cb23dc11eb79082a4ed1`
and authenticates all eight copied files, archive, weight and selected ABI5
backend bytes. The resulting `view.json` SHA256 is
`8526c8fa99c40b14e8a8dbf5aca9a4cac8adac25c5c9f078622a5de74a82e8fc`.
This unchanged view remains authenticated fixture data. The retained helper is
not a source-current full preflight for the corrected unit; the controller must
qualify a new CGO0 metadata CLI after final source selection. Reusing this view
does not create a new producer receipt.

The retained `cpu13-abi5-fixture-view-host-001` exits zero. A rehashed request
selecting a changed backend digest exits one at the retained-byte hash gate,
before creating its output (`cpu13-abi5-fixture-bad-backend-host-001`). Both run
under 2 CPUs, 3 GiB, zero swap, 64 tasks and an external 180-second deadline.
No target, native compiler, fixture producer, backend load or GPU operation runs.
The complete admission constructor and its real-artifact integration remain
unqualified until the fresh H08/CPU13 compiler tuples and fake proof exist.

The policy/output/FIFO regressions retain a differential check in
`cpu13-metadata-original-boundaries-001`: the two exact frozen original
production files reproduce the blocking FIFO, accepted runtime-output escape,
and weak-policy normalization. The corrected production files pass all three
roots in `cpu13-metadata-corrected-boundaries-001`, with no native producer.
The corrected CGO0 H08 tree, including the separate fake-capture host tests,
passes 106 root tests with eleven unselected opt-ins skipped in
`h08-metadata-integrated-host-002`. The shared guard/policy tests pass four
roots in `h08-supgpu-metadata-host-002`. Full real-artifact admission and runtime
qualification remain pending.

Those prior FIFO receipts cover `guardPin`, not the external request reads.
The shared request boundary and dedicated entry FIFO tests require fresh
controller-owned old/new differential runs and source-current host checks.
The historical helper and fixture-view pins above remain unchanged; a new
CGO0 metadata helper must be qualified after final source selection. No entry
FIFO execution, new helper qualification, genuine admission or runtime result
is claimed by this source correction.

The remaining external-request hang is retained in
`cpu13-request-fifo-original-001`: the direct constructor and all four CLI
operations reach their bounded self-child deadlines on unchanged frozen002
production. The shared reader passes both regression roots and every operation
in `cpu13-request-fifo-corrected-001`. The metadata-only CGO0 H08 tree passes
102 root tests with nine unselected opt-ins skipped in
`h08-metadata-integrated-host-003`. No compiler, selected target, backend or GPU
runs in these request-boundary tests.
