# Independent compiled observation

`observe_compiled.py` accepts only an independently hashed, preselected request
and a fresh receipt directory. The request already selects the current sealed
CGO0 ObserverController, ObserverRoot and fresh observer output. Obtain the
actual controller pin from the separate seal; no placeholder controller is
accepted. The maintained Go observer owns complete controller/retained-proof
validation. This launcher neither builds nor runs native/query/plan/GPU modes.

The root must arm this **external 600-second watchdog before Python starts**.
Use absolute Python/script/request paths and an independently supplied request
hash. Do not derive a digest from a pathname or observer output. This command
is required supervision, not an attestation made by the script:

```sh
systemd-run --user --wait --pipe --unit="h08-observe-root-UNIQUE" \
  --property=RuntimeMaxSec=600s --property=TimeoutStartSec=5s \
  --property=TimeoutStopSec=5s --property=KillMode=control-group \
  --property=SendSIGKILL=yes --property=CPUQuota=200% \
  --property=MemoryMax=3G --property=MemorySwapMax=0 --property=TasksMax=64 \
  /ABSOLUTE/PYTHON -I /ABSOLUTE/TOOLS/observe_compiled.py \
  --request /ABSOLUTE/REQUEST.json --request-sha256 INDEPENDENT_64HEX \
  --receipt-dir /SELECTED/ObserverRoot/artifacts/tensor/FRESH-RECEIPTS
```

The interpreter must be provisioned before this command. No uv resolution,
installation, compiler, native library or device startup belongs in this attempt.
A systemd runtime limit begins at activation, before Python startup. Its stop
allowance is separately bounded at five seconds. Never widen either envelope.
The caller retains this outer command's clocks, raw logs, exit and cleanup.
Use a fresh unique root unit and fresh request/output/receipt paths per attempt.

The parent reads effective cgroup limits before hashing inputs. It authenticates
request/controller/executable, reserves receipts, and opens the canonical stock
`/home/jake/Projects/nanopore_basecaller/results/stock/host.lock` O_RDONLY with
LOCK_EX|LOCK_NB. It retains the descriptor through all service/waiter cleanup.
No inner stage or observer reacquires it. It refuses symlink aliases, output
reuse and overlapping receipt/observer directories. The repository's
`/artifacts/*` ignore rule covers these generated payloads.

Within ten seconds of parent startup, it starts a unique transient user service
with RuntimeMaxSec=575s, TimeoutStartSec=5s, TimeoutStopSec=5s,
KillMode=control-group, SendSIGKILL=yes, CPUQuota=200%, MemoryMax=3G,
MemorySwapMax=0, TasksMax=64. These properties precede private Python startup.
The private stage rechecks effective limits and runtime pins; a parent-verified
interpreter/cgroup/config-argv handshake precedes Go startup. The observer's
single work wait is at most 550s; a direct unreaped child receives at most five
seconds for failure cleanup. RuntimeMaxSec bounds all descendants. Parent work
wait ends at 570s from startup. The external watchdog remains the final bound,
including interpreter startup and unresolved cleanup.

The parent requires the verified inner handshake and arm within ten seconds of
startup. The pinned private configuration carries the same monotonic deadline;
the inner stage checks it immediately before Go startup. Failed cleanup retains
the stock-lock descriptor until owned inner/observer identities, groups and
cgroup are absent and exact inactive/dead/MainPID0/emptyControlGroup properties
are confirmed. A failed stop, bus error, failed probe or exited waiter does not
prove absence. Unknown startup supplies no safe local expiry deadline. Unresolved
cleanup keeps the descriptor open until the external 600-second watchdog ends
the parent. Only confirmed absence permits ordinary waiter cleanup and lock close.

The service uses exactly cwd=ObserverRoot and this actual observer argv:

```text
EXE -mode observe-compiled -observation-request REQUEST -observation-request-sha REQUEST_SHA -out OUT -host-admitted
```

Its complete cleared environment, in wire/exec order, is:

```text
CGO_ENABLED=0 GOTOOLCHAIN=local GOWORK=off GOPROXY=off GOSUMDB=off GOENV=off GOMAXPROCS=2 GOMEMLIMIT=3GiB
```

No HOME, home or CODEX_HOME override is introduced. Intent captures argv, cwd,
request and selected executable pin before Popen. The parent independently
checks the live group leader's PID, stat field22, raw single cgroup-v2 line,
`/proc/PID/exe` pathname and bytes, argv, environment and cwd. A child that exits
too quickly to witness is rejected. Logs retain real UTC clocks, integer exit
and actual stdout/stderr SHA256, including failures.

The private stage waits for the parent after recording completion. The parent
suspends its still-live owned `systemd-run --wait --pipe` waiter before releasing
the private stage. The waiter's unit reference keeps the dead transient unit
inspectable; do not add --collect. Once the service exits, the parent verifies
original PID/start identity absence, empty process group and absent cgroup. It
never signals the observer's numeric process group. A reused group or ambiguous
identity prevents selection. It then captures exactly:

```text
systemctl --user show UNIT.service -p ActiveState -p SubState -p MainPID -p ControlGroup
```

Raw properties must be precisely inactive/dead/MainPID0/emptyControlGroup, one
newline-terminated occurrence each. The CommandReceipt environment_overrides
is []. The waiter is resumed and must actually exit zero before selection.
Failure stops only the independently identified owned service, then waits/reaps
its owned launcher. No unconditional kill of a potentially reused PID/PGID is
used. An unreaped Popen child can safely be killed directly on timeout.

Success authenticates stdout as exactly one returned Pin and hashes the actual
observation bytes. It rechecks independent request/controller/executable and
original failed receipt (including Status=failed), then writes exclusive,
read-only, atomically published Cleanup, Execution, ObservedSelection in that
order, with the private Go wire fields and runner.CommandReceipt names.
`summary.json` carries Observation, Selection, Execution, Cleanup and the
unchanged OriginalFailedReceipt pins. These root records are external to the
observer. No Python retained-proof algorithm is substituted for Go validation.

`outer-intent.json`, `lock-held.json`, `config.json`, `service-intent.json`,
`independent-inner.json`, `inner-effective.json`, `observer.intent.json`,
`spawn.json`, `independent-process.json`, `observer.command.json`, raw logs,
`inner-result.json`, cleanup command/logs, `service-result.json` and
`outer-result.json` preserve the attempt. Failed partial outputs remain; do not
repair them. A refusal before a trusted fresh receipt path exists emits a JSON
error to stderr (argparse selector errors exit2). The supervising root must
retain those outer logs. No failure emits a successful selection.
Unresolved cleanup retains `cleanup-pending.json` and numbered
`failure-show-N` command/log files. Confirmed failure cleanup adds
`failure-cleanup.json`. A watchdog-ended attempt can lack final service/outer
records; those missing records prove neither cleanup nor selection.

Controller-only mocked controls (not executed in this source-only assignment):

```sh
uv run --no-project python -m unittest discover \
  -s verification/hacgpu/h08/cgobuild/tools -p test_observe_compiled.py
```

These use temporary synthetic payloads and mocked lock/service/process boundaries.
They cover missing/incorrect selectors, rehashed bad executable digests, existing
outputs, foreign executable/argv, failed child/cleanup, raw output mismatch,
exact wire keys/order and a complete fake launcher positive. Synthetic selections
and copied/rehashed records are negative fixtures, never execution witnesses.
Failure controls also cover an exited waiter, failed stop, bus errors and failed
absence probes while exclusion remains held, plus late parent/inner startup.
The root must separately execute these controls, the genuine sealed observer,
the genuine selected reader and three separately bounded semantic negatives.
No runtime, GPU, basecalling, biological or performance qualification is claimed.
