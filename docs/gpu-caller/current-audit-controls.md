# Current effective loader controls

`supgpu-audit-control effective-run` runs the six CPU loader/constructor cases
with a separately frozen current test controller and unchanged retained native
fixtures. It does not rebuild the auditor or fixtures. The historical producer
remains pinned metadata; its Go guard is not the executing guard.

After independently installing finite supervision/cgroup limits, an outer raw
command receipt and cleanup, and a 60-second bound for each of the six children,
the root controller supplies these selections from the repository root:

```sh
"$CURRENT_TEST_BINARY" supgpu-audit-control effective-run \
  -native-admitted \
  -controller "$CURRENT_CONTROLLER_JSON" -controller-sha256 "$CURRENT_CONTROLLER_SHA256" \
  -build /home/jake/Projects/nanopore_basecaller_go/artifacts/supgpu/audit-controls-host-015-001/native/build.json \
  -build-sha256 3492dde8dec0e5e3e8f4aac401e8120fb9fdb0e0a38025ff23b3ec706bd011f5 \
  -out "$FRESH_OUTPUT_DIRECTORY"
```

The current binary/controller/hash and output directory are new root selections,
not known historical pins. No supervisor is installed by this action. Admission
is checked before reading paths or creating output. The action authenticates
`readAuditController(..., true)` before reading the retained build with
`readAuditControlBuildForController(..., false)`. Existing compiler provenance
checks remain intact: finite plan equality, before-compile inputs, command argv,
environment and logs, pre/actual dependency equality, link-map inputs, object and
output pins, constructor ELF shape, auditor ELF/proof and reconstructed runtime.
Current and retained controller snapshots of loader.c, constructor.c and audit.c
must match the consumed sources by SHA256 and byte count. All of these checks
repeat before each child; drift rejects rather than recompiling or widening policy.

The action acquires the existing read-only EX|NB stock lock once. It calls
`executeAuditControlCase` and the shared current guard in this order:
reference allowed, reference wrong-known-tier, reference unknown, observer
allowed, observer wrong-known-tier, observer unknown. The existing marker,
exit124, saved object-open veto and callback/guard evaluator remains authoritative.

Output creation is exclusive. Each case retains command.json, stdout/stderr,
guard/maps and applicable markers. Exclusive progress-01.json through
progress-06.json retain provisional failed status and the observed prefix.
effective-run.json uses `nanopore-supgpu-current-effective-audit-run/v1` and records
the current controller/binary, historical producer, original native build/runtime,
auditor/proof/sources, ordered inventory, executed cases, clocks and failure error.
Only successful evaluation of all six cases sets passed. A killed controller may
leave only a prefix; the outer raw receipt remains required. Input rejection
before output creation is reported by the outer command receipt.

Historical run015 failed after two cases; run016 continued the remaining four.
Together they witness the historical guard, not the changed current guard's live
observation behavior. This new command enables a fresh current effective witness;
source changes and host-only tests do not establish one. These CPU controls grant
no GPU, numerical parity, complete caller or biological qualification.

Root validation: format both changed Go files, run the `^TestAuditControl` host
tests and project-required checks, build/freeze a current test controller using
the maintained audit-controls.sh host-build receipt/selection recipe, then run
the admitted command only under the independently selected outer supervision.
Do not use the old binary/controller, invoke compile, replay the old compiler, or
interpret inert host source-gate fixtures as native integration evidence.

Before native admission, run the read-only boundary check from the current
controller's source root, with the same explicit selections used by the run:

```sh
"$CURRENT_TEST_BINARY" supgpu-audit-control effective-check \
  -controller "$CURRENT_CONTROLLER_JSON" -controller-sha256 "$CURRENT_CONTROLLER_SHA256" \
  -build /home/jake/Projects/nanopore_basecaller_go/artifacts/supgpu/audit-controls-host-015-001/native/build.json \
  -build-sha256 3492dde8dec0e5e3e8f4aac401e8120fb9fdb0e0a38025ff23b3ec706bd011f5
```

This returns typed metadata with status `RuntimeHeld`; it creates no output,
acquires no lock and starts no child. `-out` is rejected. Check and run share
selection, SHA verification and `readEffectiveAuditInputs`; successful checking
provides no runtime authority. Retain the genuine check's raw exit and logs plus
wrong selected pin, rehashed wrong root/path and changed native-source negatives.

Each controller's root is reconstructed from its independently pinned Go-list
module/package metadata: the exact module identity, main module Dir/Root/GoMod,
package Dir, unique ordinary SUPGPU package and consistent project roots must
agree. Missing, duplicate, conflicting or wrong-module evidence rejects with no
cwd fallback. The current consumer additionally requires its root to equal cwd,
its live snapshots to match and its binary to be the actual executing binary.
Historical reconstruction preserves original canonical paths and validated
snapshot copies. Compiler source selections use that producer's explicit root
and exact original snapshot membership/hash/bytes; full plan equality and all
existing receipt checks remain required.

Root should first retain the stable `TestAuditControlSavedProducerRoot` failure
against the old helper (copy only audit_root_test.go or overlay the old helper),
then format all four Go test files and retain the corrected host regression.
Freeze a new current controller, retain the genuine two-root effective-check
and rejection controls, run full Go/parity checks and the fresh configured panel,
then consider separately supervised native admission.
