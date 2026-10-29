# H08 selected compiled-artifact consumers

The six H08 consumer call sites require independently pinned evidence and
selection through `cgobuild.ReadH08Artifact(evidence, selection)` or
`cgobuild.ReadCPU13Artifact(evidence, selection)`. They use only the opaque
artifact's `Output()` pin. There is no inferred selector, default, raw-reader
fallback, caller-supplied origin/status, or additional CLI selection flag.
Artifact acceptance alone grants no runtime authority.

| Wire type | Required Schema | Selected build fields |
| --- | --- | --- |
| RuntimeAdmission | `nanopore-h08-runtime-admission/v2` | `ParentBuild`, `ParentBuildSelection` |
| FixedRequest | `nanopore-h08-fixed-request/v1` | `ControllerBuild`, `ControllerBuildSelection` |
| CPU13Request | `nanopore-h08-cpu13-request/v1` | `ControllerBuild`, `ControllerBuildSelection` |
| ConsumerReceipt | `nanopore-hac-fixed-consumer/v4` | `ControllerBuild`, `ControllerBuildSelection`, both in `Inputs` |
| FakeProof | `nanopore-h08-fake-loader-proof/v2` | `ControllerBuild`, `ControllerBuildSelection`, both in `Inputs` |

Every field above is mandatory; each pin contains an absolute `Path` and selected
`SHA256`. Schema/version checks reject the previous wires. The role readers
authenticate the complete selected chain and reject missing, null/zero, partial,
changed, swapped and wrong-role selectors before native work. The existing
H08 JSON decoder remains unchanged; this migration adds schema checks and
mandatory role-reader validation rather than a new general JSON decoder.

For both all-active and padding, the admission's parent evidence and selection
must equal the fixed request's controller evidence and selection. Both are
required in the guard plan's `Inputs`. Padding's prior all-active numerical
receipt must also name the same pair. The executing parent and child must each
match the selected H08 output. Missing selection stops the child before creating
its numerical attempt.

For CPU13, the parent pair selects `h08-command`, while the request pair selects
`cpu13-tensor-tests`. Each goes through its own role reader. All four pins are
required guard inputs. The fake proof binds the CPU13 test pair and output,
retains exactly 34 C-only compilation records, and requires the same two named
fake loader tests. Genuine TestMain validates the request schema and CPU13
selector before reading fixtures, installing backend environment variables,
requesting a phase checkpoint, or calling the maintained tests.

`GuardedConsumer` (`nanopore-h08-guarded-consumer/v1`) and `CPU13Receipt`
(`nanopore-h08-cpu13/v1`) retain their wrapper schemas. Their readers require the
new admission/request chain. Both numerical parent projections check the v4
schema, evidence/selection equality and `Inputs` membership before accepting
the numerical result. CLI flags and exact child/test argv remain unchanged:
admissions still travel through `-admission/-admission-sha`, fixed child requests
through `-request/-request-sha`, and CPU13 through
`-h08-request/-h08-request-sha`.

The successful and observed selection contracts remain owned by
[cgobuild](../../verification/hacgpu/h08/cgobuild/README.md#select-compiled-artifacts).
Successful selections name the independent complete producer tuple. Observed
selections bind the preselected request and independently witnessed post-exit
execution/cleanup. The retained observation profile is H08 only; it cannot
substitute for a CPU13 build. Legacy strict raw readers remain in cgobuild for
historical tools.

The controller ran the CGO0 H08 package tree under 2 CPU, 3 GiB, swap 0 and 64
tasks, holding the stock host lock. All 81 root tests passed; six separately
selected genuine-data opt-ins skipped. Raw receipts and source snapshots are in
`artifacts/pstack-herdr-polish-20261028/h08-consumer-host-001/` in the canonical
workspace. The six raw reader calls are gone. The host tests cover selector rejection at admissions,
actual-controller and child boundaries; rehashed version/pair mismatches;
guard membership for both role pairs; numerical selection/schema/membership;
and fake-proof selection/membership. Child controls supply real private pipes,
send no approval, check that no phase was requested, and keep the test callback
and backend environment untouched. Failed build fixtures and synthetic host
fake-proof logs are test inputs, not native execution evidence.

The focused command is:

```sh
CGO_ENABLED=0 go test ./verification/hacgpu/h08 -run '^TestH08' -count=1
CGO_ENABLED=0 go test ./verification/hacgpu/h08/...
```

Run it under controller-owned resource admission. Repository-wide checks and
maintained parity are recorded in the [integrated evidence](h08-retained-observation.md).
Opt-in genuine artifact
tests require their separately selected inputs; skips do not supply acceptance.
Reuse unchanged retained observation/full-proof evidence when all pins match;
this migration does not request another 297-operation replay.

No positive successful H08/CPU13 tuple or full admitted runtime fixture is known
for the migrated source. Early selector rejections and membership positives do
not establish end-to-end admission. CPU13 admission's target-reader and final
ReadCPU13/RunCPU13 fake-proof projections still need genuine matching parent,
test target, fixture, backend and guarded runtime inputs for complete positive
integration. No successful historical fixture is invented.

External admission/request/fake-proof constructors are not tracked here. The
controller must create new versioned selected wires for future genuine work;
old receipts remain immutable. Fresh role-specific CGO1 plans, query receipts,
builds and successful selected tuples must bind the changed consumer source.
The CGO0 observer sealer targets cgobuild/cmd and is unchanged. Future CPU13
source-bound targets still use the maintained TestMain overlay. Fresh fake
loader proof, genuine guarded CPU13, approved GPU graph/runtime and biological
or performance qualification remain separate gates. Old52ae embeds the old
reader and remains unexecuted; failed005 remains failed. The GPU approval
registry is unchanged and empty. H08 and the whole port remain open.
