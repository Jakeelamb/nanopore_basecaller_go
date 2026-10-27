# Audited loader and phase guard

`verification/supgpu` exposes one verification guard for source-specific C++ and Go children. It observes actual executable mappings at every loader request and phase, checks exact selected code hashes, and approves each loader object before its initializer can run. Unknown code and declared wrong providers stop the target. Its saved transcript replays through the same decision function.

The caller owns native-phase authorization, current producer/controller custody, prospective ELF/runtime selection, resource limits and the exact phase sequence. The guard does not choose a model, compiler, dataset, GPU or numerical variant. Source/host tests do not authorize a native load or qualify a caller.

## Parent interface

Write immutable JSON using these exported types, then store their independently computed pins in the caller's approved admission metadata:

```go
contract := supgpu.PhaseContract{
    Schema: "nanopore-audited-phase-contract/v1",
    Role: "reference", Action: "decoder",
    Names: []string{"pre-model", "pre-forward", "post-back-guide",
        "post-beam-search", "post-posterior", "post-staged"},
}
plan := supgpu.GuardPlan{
    Schema: "nanopore-audited-guard-plan/v1",
    Policy: selectedRuntimePolicy,
    Contract: independentlyPinnedContract,
    Inputs: authenticatedCustodyInputs,
}
command := supgpu.GuardCommand{
    Directory: absoluteCWD,
    Output: freshGuardOutputDirectory,
    Args: exactSelectedTargetArgs,
}
run, err := supgpu.ExecuteGuarded(expectedPlanPin, expectedContractPin, command)
// Retain run and its failure evidence before handling err.
actualCode, err := supgpu.ReplayGuarded(expectedPlanPin, expectedContractPin, command, run)
```

The plan contains the actual typed `RuntimePolicy` and declared input pins, including the caller's immutable source/controller and auditor-producer proof. The caller must construct that policy from its approved source-specific compiler/link/runtime bindings. An observed mapping or a self-declared receipt cannot select a provider. Preserve P01 qualification where it applies; freeze every prospective additional provider before launching. Known wrong providers remain in `Alternates` and cannot become allowed merely because they were frozen.

`Output` is a fresh directory for guard evidence. It can be a new subdirectory of an already prepared attempt. The child may write its numerical output to another independently expected argument path. The exact target argv, cwd, output path, raw logs, runtime plan pin and phase contract are bound during replay. A caller must not obtain the expected pins or expected command from the receipt being checked.

Every contract starts with `pre-model`, contains one to32 unique names and uses at most64 lower-case ASCII letters, digits or hyphens per name. The action uses the same bounded spelling. The role is `reference` or `observer`; it selects the matching runtime root. A single-target caller can pin the same binary in both root fields. The generic core follows the supplied complete list. It never derives D01/H06/H08 phases from SUP action names.

Rehashed, shortened or reordered transcripts still face the original external phase contract. A rehashed receipt cannot substitute a shorter contract, change the command, remove map evidence, change object hashes or hide a nonempty process group. Failure maps, raw command logs and group cleanup remain available after rejection.

The loader request schema and private FD3/FD4 protocol are unchanged. Original SUP v1 transcripts continue through the legacy SUP adapter, which supplies the original SUP phase lists to the same core. Historical012/017 compiler/capture controllers remain immutable historical snapshots; this interface does not claim their original sources are current.

## Go child

CGO1 Go children use the same inherited FD5/FD6 phase protocol as C++:

```go
phases, err := supgpu.OpenPhaseClient()
if err != nil { return err }
defer phases.Close()
if err := phases.Checkpoint("pre-model"); err != nil { return err }
// Only now cross the loader/device/model boundary admitted for this child.
```

Use one client for the process. Call every expected checkpoint in order, before entering the corresponding stage. `Checkpoint` waits for the parent's actual-map check and exact acknowledgement. Missing pipes, a denial or an unexpected acknowledgement must stop the child. The client cannot change the parent's independent expected phase list.

## Future process supervision

`verification/supgpu/tools/watch_roles.py` supervises exact role children of a Go controller. Linux attaches a child to its creating OS task, so the helper enumerates every `task/*/children` list. It binds each role to full argv, the actual executable SHA256, PID/private PGID and kernel start ticks. The lifetime bound starts at process birth. Group cleanup remains independent of leader exit and inherited output pipes.

Pin the helper, its exact role plan and the enclosing resource wrapper before launch. The outer cgroup remains necessary for complete attempt cleanup. Host controls use an actual CGO0 Go child created on a different locked OS thread, enforce a short birth-relative deadline, and preserve an unrelated process group.

Capture002's old watcher inspected only the Go main task and recorded no roles. Its actual command durations were below180s and its outer600s cgroup limit was real. That historical gap is preserved. The corrected helper does not establish retroactive per-role enforcement or justify repeating completed inference.
