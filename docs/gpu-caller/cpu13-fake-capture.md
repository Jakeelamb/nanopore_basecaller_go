# CPU13 fake-loader capture

`verification/hacgpu/h08/fakecapture/cmd` is a finite-purpose static Go0
producer/logger for the two maintained fake-loader tests. It uses the selected
CPU13 test binary, not copied test bodies. It executes the selected C compiler
only for the maintained five-argument shared-library recipe. No genuine backend,
CUDA program, model, stock caller, fixture producer or weight is selected by
this request. Genuine CPU13, GPU qualification and biological evidence remain
separate work.

Capture requires both of these existing interfaces:

- `h08.ReadFakeLoaderProof(pin, target, build, selection Pin) (FakeProof, error)`,
  the public metadata reader for independently selected proof pins.
- The separately reviewed reader correction from 34 to **39** compiler records.
  The maintained source has four initial rejection cases, 34 missing-symbol
  cases and one close-failure compile. This arithmetic is not execution evidence.
  The producer retains every invocation and rejects other counts.

## Preparation by the controller

1. Integrate those dependencies. Build this command with `CGO_ENABLED=0` using
   the independently admitted Go0 preparation/build workflow. Pin its actual
   executable bytes. Capture checks its own live executable against `Logger`,
   Go build information for `CGO_ENABLED=0`, and absence of an ELF interpreter
   or DT_NEEDED entries. No runtime self-compilation occurs.
2. Produce and independently select a successful, source-current
   `cpu13-tensor-tests` tuple through cgobuild. `ReadCPU13Artifact` must accept
   it. This producer accepts `nanopore-h08-successful-selection/v1` only;
   historical observed or failed outputs cannot substitute. Root independently
   pins the selected target output. Old failed005 and saved52ae are untouched.
3. Choose the exact accepted CPU13 `Plan.CC` as `Compiler`; this is not an
   arbitrary `cc` from the shell. The full selected artifact reader authenticates
   the build before target execution. Its authenticated receipt's Plan supplies
   the compiler identity and the allowed closed environment subset.
4. Create the canonical parent directory below `Root/artifacts/tensor`, but
   leave `Out` absent. No parent directory is created implicitly. Pins must be
   distinct canonical absolute regular files, with complete lowercase SHA256.
   Inputs and request must be outside `Out`. Retain request bytes and obtain its
   SHA independently, before choosing CLI flags.

Request JSON has exactly these fields and this casing. Uppercase symbolic
values below are placeholders, not usable pins:

```json
{
  "Schema": "nanopore-h08-fake-capture-request/v1",
  "ControllerBuild": {"Path": "/ABS/SUCCESSFUL-CPU13-RECEIPT", "SHA256": "INDEPENDENT_SHA256"},
  "ControllerBuildSelection": {"Path": "/ABS/SUCCESSFUL-CPU13-SELECTION", "SHA256": "INDEPENDENT_SHA256"},
  "Target": {"Path": "/ABS/SELECTED-tensor.test", "SHA256": "INDEPENDENT_SHA256"},
  "Logger": {"Path": "/ABS/SELECTED-static-fakecapture", "SHA256": "INDEPENDENT_SHA256"},
  "Compiler": {"Path": "/ABS/ACCEPTED-Plan.CC", "SHA256": "INDEPENDENT_SHA256"},
  "Root": "/ABS/GO-PORT-SOURCE-ROOT",
  "Out": "/ABS/GO-PORT-SOURCE-ROOT/artifacts/tensor/FRESH-CAPTURE"
}
```

Unknown, omitted, duplicate, case-folded, null, trailing or changed request
fields reject. So do partial/changed pins, wrong role/selection, independently
wrong output/compiler, symlink or relative/nonclean paths and output collisions.
`fakecapture.Preflight(requestPin)` is a read-only metadata API that performs
the artifact authentication and creates no output. It grants no admission.

## Required root launcher and lock lifetime

Root owns the launcher; this package does not install it. There is deliberately
no raw `timeout` wrapper or duplicate stock-lock acquisition in the producer.
Before **Go0 interpreter/binary startup**, root must:

- Open the existing
  `/home/jake/Projects/nanopore_basecaller/results/stock/host.lock` with
  `O_RDONLY` and acquire `LOCK_EX|LOCK_NB`. Missing/contended lock blocks launch.
- Install the independently supervised systemd/cgroup-v2 envelope, with
  effective CPU quota at most 2 CPUs, memory at most 3 GiB, swap 0 and at most
  64 tasks. The intended settings are `CPUQuota=200%`, `MemoryMax=3G`,
  `MemorySwapMax=0`, `TasksMax=64`. Use cgroup-wide termination, including
  descendants (`KillMode=control-group`), with an independently bounded stop
  path. A direct-parent kill is insufficient.
- Arm the external **600 second** watchdog before starting Go. Metadata
  authentication, Go startup, target, logger, proof readback and cleanup are all
  inside that outer bound. Root must retain raw launcher commands, clocks,
  statuses, watchdog/unit identity and effective cgroup readback.
- Keep all descendants in that owned cgroup. Do not use detached services or
  permit compiler descendants to escape it. Keep the lock owner alive outside
  any early-return child cleanup path.

After success, failure, cancellation or timeout, root terminates remaining work
and proves the unit/cgroup **and all descendants** strictly dead before closing
the lock descriptor. Keep the lock through this verification. Retain timeout
and cleanup failures. A producer exit, file receipt, free-looking lock or
reaped leader does not establish completed cleanup. Never signal a process
group using a reaped/reusable PID.

The command below is only the payload **inside** that already installed root
envelope, with working directory equal to Request.Root:

```sh
/ABS/SELECTED-static-fakecapture \
  -request /ABS/SELECTED-request.json \
  -request-sha INDEPENDENT_REQUEST_SHA256 \
  -admit-native-phase
```

There are no positional arguments or arbitrary command/environment options.
The admission flag records root's explicit native-phase decision; it does not
prove that a watchdog or lock exists. Capture calls `closedbuild.Limits()` and
`runner.CheckRoot()` at its boundary and records effective limits. Helpers
repeat the limits check, and command startup rechecks them. These checks happen
after startup and cannot replace the preinstalled envelope. Outer launch and
cleanup proof remain root's independent responsibility.

## Capture and private cc session

Capture creates fresh mode0700 `bin`, `logs`, `tmp` and `compiles` directories
below `Out`, and `bin/cc` as a symlink to the selected Logger. It writes a
create-exclusive mode0400 `session.json`, pinning the request, authenticated
Plan, logger/compiler/target, actual capture PID/start tick/cgroup, root/output,
closed compiler environment and the shared target deadline. The target gets
only this explicit environment:

```text
PATH=Out/bin
LANG=C
LC_ALL=C
TZ=UTC
TMPDIR=Out/tmp
GOMAXPROCS=2
GOMEMLIMIT=3GiB
H08_FAKE_SESSION=Out/session.json
H08_FAKE_SESSION_SHA=actual session SHA256
```

There is no inherited shell environment, LD_PRELOAD, LD_AUDIT, loader path,
backend pin or fixture/weight/runtime variable. The selected target's maintained
fake TestMain installs `/fake-only/no-backend.so` and empties the genuine
weight/fixtures/runtime-manifest variables. Exact target argv is:

```text
TARGET -h08-profile fake -test.v -test.count=1 -test.run ^(TestNativeLoaderABIRejections|TestNativeCloseFailureRetainsLibrary)$
```

The target has one **180 second** deadline, including all helper and compiler
work. Each compiler inherits the remaining shared deadline, so no compiler can
consume a fresh budget beyond the target cap. Root's 600 second limit covers
the metadata work outside that target window too.

The `cc` entry is chosen by basename, then requires the private absolute symlink
and exactly `PATH=Out/bin`. The maintained target uses `exec.Command("cc", ...)`,
so its literal `argv[0]` is `cc`; the absolute private spelling is also accepted.
No `-mode cc` launcher exists. The helper authenticates session/request pins,
its actual Go0 executable and compiler, session Plan linkage and exact closed
environment. It checks the live capture PID/start/cgroup/executable and direct
ancestry `capture -> selected target -> helper`, including target executable
bytes and common cgroup. A stale or public session cannot authorize a command.

Only these five arguments are accepted:

```text
-shared -fPIC CLEAN_ABSOLUTE_SOURCE.c -o CLEAN_ABSOLUTE_OUTPUT.so
```

Source/output must be inside `Out/tmp` with canonical parents. Source must be
regular C, not a symlink or C++ suffix. Only the three standard-header directives
used by these tests (`stdint.h`, `stddef.h`, `stdio.h`) are admitted. Other
directives, framework includes and C++ constructs reject. The compiler process
receives exactly six argv entries including Compiler.Path. Its environment is
the authenticated Plan's exact PATH/LANG/LC_ALL/TZ subset, with TMPDIR replaced
by the session-private target temp root. Go build controls, loader and framework
variables are excluded.

Each helper independently allocates `compiles/cc-*` with `MkdirTemp`; there is
no shared mutable aggregate JSON. It snapshots/pins source before compiler
startup, reserves output with O_EXCL, and rejects collisions. It retains full
argv/environment/cwd, before-launch metadata, actual process identity/cgroup,
clocks/exit and raw stdout/stderr with hashes. It snapshots output before
returning to the target, and rechecks source/output and request/session/compiler
pins. The target's Go TempDir cleanup may then remove the original files.
`FakeCompile.Source` and `.Output` keep their original paths/hashes;
`.SourceBefore` and `.OutputSnapshot` name retained readable snapshots.

Failed compiler attempts retain their directory, source snapshot, logs, exit
and any readable output snapshot, and return nonzero to the target. Rejections
before a trusted session is known remain in the target's raw output; the helper
does not write to an untrusted directory. A killed helper can leave an incomplete
directory, which blocks acceptance. Cancellation kills only owned unreaped
direct children via Go's CommandContext; Pdeathsig aids direct-child cleanup.
Root's cgroup watchdog still owns every remaining descendant. No killpg occurs.

Immediately after a successful `Start`, the logger writes `born.json` with the
actual child PID and the UTC time it observed successful startup, before live
identity or executable inspection. This clock is not a kernel birth timestamp.
`process.json` records the birth, any observed PID/parent/start tick/cgroup,
authentication status and observation error. An observation failure preserves
the known PID and available partial facts, kills/reaps the owned child, and
retains its actual Wait exit and raw streams. It remains unqualified even if
Wait reports zero. A Start failure records its error without a child birth.
No missing tick, cgroup or executable observation is inferred or retried. This
fast/dead-child case is source-defined; no native scheduling failure has been
observed.

## Proof publication and retained evidence

The target's `logs/command.json` and process/launch metadata are retained before
projection. Capture inventories every leaf record, including failures, without
truncating to39. It requires zero target exit, exactly39 complete successful
records and unique original outputs. It emits typed v2
`provisional-fake-proof.json`, including target/build/selection/logger/compiler,
request/session/Plan custody, actual commands and all compiles. Its two proposed
test results are assertions until `h08.ReadFakeLoaderProof` checks the exact
raw RUN/PASS set and rejects missing/extra/duplicate/skip/fail results.

Only after that public reader succeeds does capture publish `fake-proof.json`
and a successful `summary.json` with proof pin and `RuntimeHeld=true`. Every
write is create-exclusive; outputs are never reused or overwritten. A cancelled
or partial run has a failed summary and no successful final capture receipt.
A proof file alone is not acceptance: require the independently pinned successful
summary as well. Root reads completed evidence using
`fakecapture.ReadAccepted(requestPin, summaryPin)`, which authenticates the
selected artifact again, session, all leaf records and the public fake-proof
reader. Root's post-unit cleanup evidence is still separately required.

The package's own target summary and compiler records pin `launch.json`,
`born.json`, `process.json` and `command.json`. One private typed reader requires
those exact files and compares cwd, executable, argv, environment, launch and
completion fields, birth/process observation, clocks and raw-log pins against
the enclosing record. Publication and accepted readback both use it. Missing,
changed or rehashed-but-inconsistent receipts reject acceptance. These are new
unpublished FakeCapture v1 records with no compatibility fallback; the public
H08 FakeProof API and schema remain unchanged.

Evidence layout:

| Path below Out | Meaning |
| --- | --- |
| `session.json` | Pinned private request/parent/compiler/target binding |
| `logs/{launch,born,process,command}.json` | Pinned target launch, successful-Start birth, process observation and completion |
| `logs/{stdout,stderr}` | Raw target streams |
| `compiles/cc-*/source.c` | Source bytes retained before compiler |
| `compiles/cc-*/output.so` | Output retained before target TempDir cleanup |
| `compiles/cc-*/{launch,born,process,command,compile}.json` | Per-invocation pinned metadata, birth/observation and raw receipt |
| `compiles/cc-*/{stdout,stderr}` | Raw compiler streams |
| `provisional-fake-proof.json` | Clearly provisional reader candidate; retained on reader failure |
| `fake-proof.json` | Public-reader-accepted proof; requires successful summary |
| `summary.json` | Final capture status, limits, parent/target, raw target receipt, leaf pins and proof pin |

Early boundary failure cannot safely create an output summary in an already
occupied or untrusted directory. Its failed CLI JSON/error/exit must be retained
by root's outer raw logs. Abrupt watchdog termination can leave no summary;
missing completion is unqualified. Keep all such attempts.

## Controller validation

No commands in this section were run by the source-only worker. Root owns
formatting, compilation and every check under its admission controls. After
dependency integration, format only the new Go files, then run the focused
ordinary tests with CGO disabled and repository checks including `go test ./...`
and the maintained parity command. Metadata tests use inert files and invented
process identities; they are reader controls, not process evidence. Lifecycle
regressions use only a Go self-child with a per-call observation-failure seam
and an absent-cwd Start failure. Neither executes a compiler, selected CPU13
target or backend/GPU work. The no-command controls inspect absent
launch/process evidence on cancellation, changed executable pins and wrong argv.

After root captures one real selected target, independently pin this fixture:

```json
{
  "Schema": "nanopore-h08-fake-capture-integration/v1",
  "Request": {"Path": "/ABS/REAL-request.json", "SHA256": "INDEPENDENT_SHA256"},
  "Summary": {"Path": "/ABS/REAL-CAPTURE/summary.json", "SHA256": "INDEPENDENT_SHA256"}
}
```

Set `H08_FAKE_CAPTURE_FIXTURE` and `H08_FAKE_CAPTURE_FIXTURE_SHA256` to the
independently selected fixture, then run these exact entries under root's
bounded host-check envelope:

```sh
CGO_ENABLED=0 go test ./verification/hacgpu/h08/fakecapture -run '^TestGenuineAcceptedCapture$' -count=1 -v
CGO_ENABLED=0 go test ./verification/hacgpu/h08/fakecapture -run '^TestGenuineCaptureRejectionControls$' -count=1 -v
```

These integration entries read the actual retained capture and start no native
program. The second retains derived invalid proof/log controls in a fresh
`fake-capture-rejections-*` sibling directory and exercises missing/extra compile,
selection/logger/compiler, missing snapshots, exit, result and raw
skip/fail/duplicate/extra-test rejection, plus selected-tuple wrong
output/compiler/role preflight controls with no attempt creation. Missing fixture variables honestly skip;
partial variables fail. Invalid derived candidates are control data, not runtime
receipts. The real target capture itself is the separately admitted CLI payload
above, never an ordinary unit test.

The genuine controls also read the selected target and an actual compiler leaf
through the private receipt reader, then retain missing, changed and rehashed
inconsistent launch/birth/process/command controls in separate directories.
They preserve replaced/removed derived bytes and an invalid expected-record
index. Root's original capture is never changed or deleted, and no producer is
repeated. Ordinary metadata tests exercise the same reader/control cases with
inert records.

Known limits: neither admission flags nor in-process limit checks authenticate
root's lock/watchdog installation; root must supply that evidence. The producer
does not attest cgroup cleanup, immutable filesystem isolation against a root
writer, compiler syscall sandboxing, genuine backend loading, whole-caller
computation, GPU behavior, performance or biological correctness. Successful
artifact/proof reading and source-only delivery grant no further runtime work.

## Retained source checks

The controller formatted the new package and ran its eight host boundary tests
in `cpu13-fake-capture-host-001`; both genuine capture opt-ins skipped without a
selected real fixture. The integrated `go test ./...` passed 817 root tests with
85 opt-ins skipped, and maintained parity passed in `h08-final-host-007` and
`h08-final-parity-007`. These checks ran under 2 CPUs, 3 GiB, zero swap and 64
tasks. No native fake capture or successful CPU13 target tuple exists yet.
Source review, a qualified static Go0 logger build, genuine compiler artifacts,
actual capture, retained readback controls and independent outer cleanup remain
required before this producer supplies accepted fake-loader evidence.
The receipt/birth corrections and their new tests postdate those retained host
checks. They have not been formatted, compiled or executed by this source-only
worker; controller formatting, focused/full Go checks, parity and fresh review
are still required.

The receipt and child-birth corrections pass the isolated CGO0 package checks
in `cpu13-fake-capture-host-002`: fourteen root passes and two unselected genuine
opt-ins skipped. The self-child controls exercise Go process ownership and
failed identity observation; the inert receipt controls authenticate missing,
changed and inconsistent retained metadata. These are host checks. No selected
CPU13 target, C compiler, genuine backend or GPU runs in these tests.
