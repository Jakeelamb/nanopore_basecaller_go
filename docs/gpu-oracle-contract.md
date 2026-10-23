# GPU oracle contract

P05 source checkpoint. No native capture, GPU qualification or runtime
acceptance has run for this module. The existing koiquant schemas remain frozen.
The new types describe the shared handoff and do not reinterpret old manifests.
The integrated host pass ran the contract tests, the full ordinary Go suite, the
verifier CLI and the synthetic inventory import, replay and rejection controls.
These observations do not qualify native execution or genuine POD5 provenance.
Receipts are in `artifacts/gpu-plan/host-check-20261023T183053Z/`.

`gpuoracle.Prepared`, `Frozen`, `Imported` and `Accepted` are separate identities.
Each persisted identity has mandatory `Schema string` and `Stage Stage` fields.
Schema is exactly `nanopore-gpu-oracle/v1`. Stages are respectively prepared,
frozen, imported and accepted. `ValidateIdentity() error` checks only this pair.
`Decode` rejects a schema or stage mismatch for its concrete target type.
A preparation pins source, inputs and a prospective range declaration. A frozen
capture pins that preparation, producer closure, qualification, payloads and the
maintained runner command receipts. An import pins frozen capture, consumer code,
required named tests and their receipt. Acceptance separately pins import,
runtime qualification and independent verdict. Structural decoding is not semantic
qualification. Each adapter must validate its exact expected pin set and schema
before consumption. Self-produced candidate output cannot serve as native oracle.

`Decode` accepts only these concrete contract shapes and `Allocation`. Every field
is mandatory with exact spelling. Unknown, duplicate, missing and null values are
rejected, including nested values. Empty arrays are structurally legal; adapters
must reject empty required closures. `Pin.Check` verifies retained file bytes.
Consumers must check current consumer source pins and all input pins before reuse.

An allocation partitions all retained bytes into ordered, disjoint, nonempty
ranges. Kinds are live, initialized, guard and uninitialized. Each range cites a
source pin. Freeze declarations before capture. Compare live, initialized and
guard bytes exactly; retain but exclude only prospectively declared uninitialized
storage. Never infer exclusions from observed mismatches. Range validation checks
coverage and bounds, not whether the source actually supports the declaration.

Native handlers must call `Admit` with fresh coordinator intake and a concrete
qualification validator before launching any command. Exported
`Admit(intake bool, qualification func() error) (func() error, error)` selects
`runner.HostLock` internally. Production callers cannot substitute another path.
An unexported package helper supplies temporary or missing paths only for package
rejection tests. Intake and qualification precede acquisition on both paths.
Lock acquisition is nonblocking
and never creates or truncates. Retain its close function through all native child
completion. Import handlers must consume retained files without producers, GPU
initialization or lock acquisition. A CLI flag alone is not qualification.

Use the maintained runner receipts with argv, environment, UTC clocks, exits and
raw stdout/stderr hashes. Preserve failed and partial attempts in fresh ignored
artifact directories, even when projection fails. Never overwrite a frozen attempt.
Repair observers from saved bytes. Require successful command exits and exact
named-test accounting through `RequireTests`; empty gates, no-match and skips fail.

The caller dispatcher has a closed `Unit` identity independent of `Mode`.
Units are schedule-check, postcall-check, qualify, biological, benchmark and
inventory. Modes remain import, replay and native. `Request` contains Unit, Mode,
Input, Output and Admitted. Each handler has signature
`func(gpucaller.Request) error`. `UnitHandlers` contains Import, Replay and Native
handlers. `Handlers` explicitly contains ScheduleCheck, PostcallCheck, Qualify,
Biological, Benchmark and Inventory registrations of type UnitHandlers.
`Dispatch(Request, Handlers) error` selects exactly one unit and mode, with no
fallback. CLI requires `-unit` and `-mode`. Integration registers I04 Inventory Import/Replay
in the Linux-only verifier CLI; Inventory Native remains absent. Missing, unwired
and unknown units or modes fail closed. Host import
and replay handlers must not launch native work even when Admitted is true.
Native dispatch requires Admitted; its handler must still validate qualification
and acquire admission through `gpuoracle.Admit`. P05 serializes registration
patches; I04 and other owners supply adapters without editing this registry.
Handlers must validate fresh output paths and full operation-specific contracts.
No complete caller/model/decoder wrapper is introduced.

Host checks after explicit coordinator release passed: the ordinary `go test ./...`
suite, the four new verification packages, the protocol CLI and the dispatch
rejection controls. Still deferred are maintained parity, held temporary-lock
zero-child rejection, genuine-file import with an effective corrupt-copy control,
stale consumer-pin rejection and independent review. Synthetic source tests do not
establish the genuine fixture handshake.
HG005 is excluded from candidate selection. Native and biological gates stay open.
