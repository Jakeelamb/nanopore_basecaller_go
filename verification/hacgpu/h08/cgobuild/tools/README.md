# Bounded CGO0 controller seal

`seal_controller.py` builds only the fixed Go controller and asks the resulting
binary's maintained `check-controller` mode to validate its exact Controller
schema. It then rehashes a descriptor with one snapshot removed and requires
exit 1 with `custody incomplete`. It runs no native queries, CGO1 build, plan,
admission mode, output load, or GPU operation.

The supervising controller must freeze source first, arm an independent
180-second cgroup watchdog **before interpreter startup**, and hold the existing
stock host lock O_RDONLY with exclusive nonblocking flock. This script does not
acquire or reacquire that lock, create a service, or certify those prerequisites.
The controller must retain the lock through confirmed service and cgroup cleanup,
including failed attempts.
Run it within effective 2 CPU / 3 GiB / swap 0 / Tasks 64 limits. It reads the
cgroup ancestor chain before bulk work and before each child.

Inside that already bounded and locked execution:

```sh
uv run --no-project python verification/hacgpu/h08/cgobuild/tools/seal_controller.py \
  --root "$CHECKOUT" \
  --tools-controller "$PINNED_OLD_CONTROLLER" \
  --tools-controller-sha256 "$INDEPENDENT_OLD_CONTROLLER_SHA256" \
  --out "$CHECKOUT/artifacts/tensor/cgo-controller-seal-NEW"
```

All selectors are required. Supply the old controller digest independently; do
not infer it from a filename or trust its current mutable pathname. The supplied
controller's authenticated Tools map supplies every expected Go tool/resource
hash, including `/usr/lib/go/bin/go`, assembler/compiler/linker, VERSION,
go.env and assembly headers. No download, tool fallback, HEAD restriction,
package wildcard, output reuse, hash cache or retry is supported. The actual
selected Go1.27.3 installation is `/usr/lib/go`. No Git operation is performed.

The two `go list -deps -json` calls select the fixed command's source and module
closure. Selection membership and bytes must agree after the exact forced
`go build -a -x -work -p=2 -pgo=off -trimpath -buildvcs=false -o OUTPUT TARGET`.
Every selected file and authenticated tool receives a read-only snapshot;
all live pins are rechecked before subsequent children and after verification.
The maintained Go checker owns descriptor, receipt and static ELF validation.
The glue does not claim that a mocked executable proves any of those properties.

Every attempted command directory retains the intended argv/environment,
PID/PGID (null if launch was refused), UTC interval, exit, raw streams and hashes,
and any launcher/timeout error. A non-reaping Linux wait retains the child identity
through group cleanup; the child is reaped once afterward. No numeric group is
signaled after that identity is released. One shared 175-second deadline covers work and child waits after
imports. Raw receipt finalization may outlive that internal work deadline;
the independent watchdog is the hard bound. Internal timing cannot certify
interpreter startup or external cleanup. Failed attempt directories remain in
place; always select a new output directory.

`controller.json`, `missing-snapshot.json`, `summary.json`, and command receipts
are the evidence entry points. The descriptor contains source/tool/snapshot,
list, executable and build-receipt identities. `tools-controller-pin.json`
records the independent input metadata identity. Payloads stay under ignored
`artifacts/tensor`, outside Git. This is CGO0 controller custody only, with no
native, runtime, biological or performance qualification.

Focused controls (to be run by the supervising controller; not executed during
this source-only assignment):

```sh
uv run --no-project python -m unittest discover \
  -s verification/hacgpu/h08/cgobuild/tools -p 'test_seal_controller.py'
```

These invoke the CLI entry point with temporary source/tool fixtures and mocked
process launch/cgroup reads. They check missing selectors, root/source/output
refusal, independent digest refusal, bounded-resource refusal, source/tool
changes before child, exact allowed launches, and a rehashed corruption control.
Mocked ownership controls also cover normal and expected exits, timeout,
interruption, lost ownership, signal failure and one reap after group cleanup.
They touch no host services and do not compile or execute Go.
