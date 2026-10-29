# H08 retained compiled observation

The complete retained build proof, fresh CGO0 observer seal, witnessed observer
and independent selected reader have passed. These results qualify selected
compiled metadata. Runtime remains held, the H08 card remains open, and
`approvedGPUBundles` is empty.

This record uses the controller's retained evidence and the completed receipts
read during task t-9c67d1. No test, build, compiler, native program or device ran
for this documentation task. See the [operator contract](../../verification/hacgpu/h08/cgobuild/README.md#retained-compiled-proof),
[sealer](../../verification/hacgpu/h08/cgobuild/tools/README.md) and
[launcher](../../verification/hacgpu/h08/cgobuild/tools/OBSERVE.md).

## Accepted evidence

Canonical paths below use these absolute roots:

```text
ROOT=/home/jake/Projects/nanopore_basecaller_go/artifacts/pstack-herdr-polish-20261028
CORE=/home/jake/Projects/nanopore_basecaller_go_worktrees/cgo-build-proof-20261028
GO=/home/jake/Projects/nanopore_basecaller_go
```

| Evidence | Retained entry point | Result |
| --- | --- | --- |
| Full common body | ROOT/h08-retained-full-002/inner-receipt.json and stdout.jsonl | Exit 0, source unchanged. All 297 operations, 289 accepted and 8 expected exits. |
| Final host controls | ROOT/h08-host-cleaned-001/inner-receipt.json and stdout.jsonl | Exit 0, source unchanged. 63 root tests pass; six separate genuine/fixture opt-ins skip. |
| Sealer and launcher glue | ROOT/h08-sealer-glue.sh and h08-launcher-glue.sh, with controller-retained logs | 7 sealer and 9 launcher mocked controls passed. These are synthetic controls. |
| Actual CGO0 seal | CORE/artifacts/tensor/cgo-observer-seal-001/summary.json | CGO0_SEALED. Maintained check-controller passed and the rehashed missing-snapshot descriptor rejected. No native commands. |
| Actual observer and root witness | CORE/artifacts/tensor/h08-observer-execution-001/summary.json | OBSERVED_SELECTED. Real observer exit 0 and external post-exit cleanup precede selection. |
| Selected reader | ROOT/h08-selected-positive-001/inner-receipt.json and stdout.jsonl | TestGenuineSelectedObservation exited 0, source unchanged, finished 2026-10-29T00:08:11.930143Z. |

The full body authenticates 346 historical process records and 97 owned groups.
It covers 16 exact noncanonical resource answers, eight link operations, seven
maps and one expected unsupported probe. The native preimage beginning 415967
becomes output 52ae02 only through the derived 32-byte Go FIPS HMAC update.
The common validator checks whole-file equality after that update. The fixed
profile uses Go 1.27.3 on Linux/amd64 and the pinned linker below. No native
producer was repeated.

The actual observer had PID 1010466, start identity 2996346 and owned unit
`h08-observe-51479f54b1d0413181a70b4c8d02e3f5.service`. The root checked the
selected executable, argv, cwd and exact eight-entry environment. It recorded
PID/start absence, an empty owned group, an absent cgroup and raw systemd
inactive/dead/MainPID0/emptyControlGroup before external selection.
Limits were outer 600s, service 575s, work 550s, 2 CPU, 3 GiB, swap 0 and 64 tasks.
Root raw launch evidence remains in ROOT/h08-observer-run-001. These process
identities describe that completed attempt and are not live status claims.

## Selected pins

The request contains the full independent historical producer tuple, handoff,
outer receipt, cleanup, failure stderr and two algorithm-reference pins.
The seal descriptor contains the current observer's separate source/tool closure.
The launcher summary contains the observation, selection, execution and cleanup
paths and hashes. Those files are evidence indexes for their referenced payloads.

| Artifact | Path | SHA256 |
| --- | --- | --- |
| Original failed receipt | GO/artifacts/tensor/h08-cgo-command-build-005/receipt.json | f487edce8e9256f5c4cd432337fa95080d7894f94e2d9bdeb26c8d3ed6aed82b |
| Native preimage | GO/artifacts/tensor/h08-cgo-command-build-005/inputs/415967eaec4b7eae8f43bc8568aafc0e2cce19ed298370e36d8c3dc7ca1f2e59 | 415967eaec4b7eae8f43bc8568aafc0e2cce19ed298370e36d8c3dc7ca1f2e59 |
| Saved output | GO/artifacts/tensor/h08-cgo-command-plan-006/work/h08 | 52ae02a2370d718123bc7088d3de1d5d483122323eaa5c7e794a854df1917c69 |
| Handoff | GO/artifacts/h08-cgo-build-handoff-20261025-005/handoff.json | c0981c0bdb5f96536462b407e3d41ef7a1b7809bb3ff4a415fc2d33d75a197f5 |
| Request | ROOT/h08-observation-request-001.json | ee9a79be2d5a9ccd754cabe2e0873cf1f696be371aac86bf6a03eeeae6f7231b |
| Observer controller | CORE/artifacts/tensor/cgo-observer-seal-001/controller.json | eba4439ed319d192c6597f5a17d320b92ab9c2a15efc00d3151e5dd6b5d454df |
| Observer executable | CORE/artifacts/tensor/cgo-observer-seal-001/cgo-plan | a229f70b4ed8c5c92b1ee9ed4b85e7bfc0998cc7758fce17efe15206179dfcfc |
| Observation | CORE/artifacts/tensor/h08-compiled-observation-001/observation.json | 24dced3e7c2004301696a528eb3fb301c89273d4450dee6be99f488a2d343801 |
| Selection | CORE/artifacts/tensor/h08-observer-execution-001/selection.json | 87b0c99f4ae2b74e10a9838409ef0d89498ac072deb8437083f4516e978a2a5b |
| Execution | CORE/artifacts/tensor/h08-observer-execution-001/execution.json | 083dee6508d8db7bd05803e016d36783dacf23d4872088d63ecccb5013364ce8 |
| Cleanup | CORE/artifacts/tensor/h08-observer-execution-001/cleanup.json | 33a1d84468c9220cbf41a13c791d2b955aa90c3f5077ab80dba29dd35d3bc74e |
| Go linker | /usr/lib/go/pkg/tool/linux_amd64/link | 97df419b7cf8f4bb8e76c3623b8a1d371c6818533c4be04f0a3563ff6169843d |

## Cleanup corrections and current integrated acceptance

The full source review found two cleanup defects. The original sealer signaled
its numeric process group after reaping its owner. The original launcher could
release the stock host lock after a failed stop while its detached service still
ran. Each new mocked regression fails on its original source with the intended
ownership assertion in ROOT/h08-{sealer,launcher}-old-regression-001/.
The corrected sealer passes 18 mocked controls; the corrected launcher passes
19. ROOT/h08-{sealer,launcher}-cleanup-controls-001/ retains those executions.

The sealer now retains the unreaped child through group cleanup and reaps once.
The launcher keeps exclusion until independently confirmed service/process/group/
cgroup absence, or until its external watchdog ends the attempt. A waiter exit,
failed stop or bus error supplies no absence proof. Parent handshake, arm and
inner Go startup share the ten-second startup deadline.

The corrected tools also passed fresh real execution. Seal002 is CGO0_SEALED,
with its maintained positive and missing-snapshot rejection. Observer002 is
OBSERVED_SELECTED; ROOT/h08-selected-positive-002 independently accepts its
selected proof. These runs did not repeat the native producer or execute the
saved output. The unchanged Go proof retains the three separately passing
semantic negative families recorded above.

| Current artifact | Path | SHA256 |
| --- | --- | --- |
| Controller | CORE/artifacts/tensor/cgo-observer-seal-002/controller.json | 57f1d578bd2de3df180789dd1c626da89ad7bf94fbf40cf3b7e409c0f4ab0081 |
| Request | ROOT/h08-observation-request-002.json | adad9727f2bdfcb5440dffe216dd7dfe3052578d57f0651c679b775d76e5d867 |
| Observation | CORE/artifacts/tensor/h08-compiled-observation-002/observation.json | 9cf95a6186676212bad2c5c15500b433dd9992156adf20cf3008c9dbd956cd0c |
| Selection | CORE/artifacts/tensor/h08-observer-execution-002/selection.json | e3ab1e0087d0f46336ee30cdfe777ebb5b7fdc0f56a53fbae88e1adb31e97b5d |

The current observer executable has the same a229f70b digest as seal001. The
execution and cleanup pins are in CORE/artifacts/tensor/h08-observer-execution-002/
summary.json. ROOT/h08-seal-observer-002/ and h08-observer-run-002/ retain the
outer clocks, bounds, logs and exit zero. The initial selected pins above remain
the historical001 chain, with their original source and raw receipts unchanged.

The integrated six-consumer source passed `go test ./...`: 795 root tests pass,
80 separately selected opt-ins skip, and no test fails. ROOT/h08-final-host-002/
retains that run. Maintained parity also passes in ROOT/h08-final-parity-002/,
with CPU package and native algorithm scope. Full positive source-current H08
and CPU13 target/runtime acceptance remains open.

## Preserved failures

Each attempt remains under ROOT with its original receipts and raw logs:

| Attempt | Reason |
| --- | --- |
| h08-retained-links-001 | The query binding accepted too narrow a GCC spelling. |
| h08-retained-links-002 | A test asserted the wrong operation role. |
| h08-retained-full-001 | The external-driver guard accepted only the bare spelling. |
| h08-host-extld-001 | A test attempted to write its read-only fixture. |

These were reader or test failures, not GPU numerical failures. Original build005
remains failed with its original hash. Strict raw APIs still reject it. The
observation records a separate authenticated proof and does not promote the
original receipt to success. The saved output embeds the old reader and remains
unexecuted. It cannot become the current runtime consumer through this proof.

## Remaining work and limits

The output, native-preimage and proof-summary corruption families each pass their
own genuine-data rejection check after all owned ancestors are rehashed. Receipts
are ROOT/h08-selected-negative-{output,native,summary}-001/. Each reruns the full
proof within its separate 600s bound and reaches the serialized-proof comparison.
Rewritten execution ancestors are adversarial fixtures, not new execution
witnesses. The initial proof-only `go test ./...` passed with 789 root tests
passing and 80 separate opt-ins skipped in ROOT/h08-final-host-001; its maintained
parity also passed in ROOT/h08-final-parity-001. The current integrated checks
are recorded above. Independent review is required before this unit is delivered.

All six consumer call sites now require independent evidence and selection;
the [consumer contract](h08-selected-consumers.md) records the versions and host
rejection checks. Fresh source-current target builds and complete positive
runtime admissions remain open. Compiled-artifact acceptance does not
establish native target execution, GPU graph acceptance, CPU13, all64, padding,
fixed profiles, full-caller acceptance, biological accuracy or performance.
Source readiness, compiled-artifact proof, runtime admission and whole-port
readiness remain separate. Dorado, stock and HG005 are unchanged.
