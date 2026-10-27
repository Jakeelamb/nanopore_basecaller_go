package decodergpu

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type GuardRuntimeOptions struct {
	Build, Consumer, AuditorProof File
	Out, Capture                  string
}

func decoderCommand(r RuntimePlan, b CompilerPhase, p ClosedBuildPlan) supgpu.GuardCommand {
	prepared := filepath.Dir(p.Preparation.Path)
	return supgpu.GuardCommand{Directory: r.WorkingDirectory, Output: filepath.Join(r.CaptureDirectory, "guard"), Args: []string{b.Output.Path, "--fixed-hac-staged-oracle", filepath.Join(prepared, "params.bin"), filepath.Join(prepared, "chunks.bin"), filepath.Join(prepared, "scores-original.f16"), filepath.Join(r.CaptureDirectory, "native")}}
}

func immutableGoCustody(inputs map[string]string, p Preparation) error {
	for path, sha := range p.Go.SnapshotPins {
		inputs[path] = sha
	}
	for _, c := range p.Go.Commands {
		if err := commandEvidence(c); err != nil {
			return err
		}
		inputs[c.Stdout] = c.StdoutSHA
		inputs[c.Stderr] = c.StderrSHA
	}
	inputs[p.Go.Controller] = p.Go.ControllerSHA
	return nil
}

func guardCustody(r RuntimePlan, b CompilerPhase, p ClosedBuildPlan, auditorInputs map[string]string) (map[string]string, error) {
	return guardCustodyScoped(r, b, p, auditorInputs, nil)
}

func guardCustodyScoped(r RuntimePlan, b CompilerPhase, p ClosedBuildPlan, auditorInputs map[string]string, proofs *numericProofs) (map[string]string, error) {
	inputs := maps.Clone(r.Inputs)
	if numericBuild(p) {
		proof, e := proofs.historical(r.NumericLineage)
		if e != nil {
			return nil, e
		}
		maps.Copy(inputs, proof.Inputs)
		if e = addPin(inputs, r.KernelVersion); e != nil {
			return nil, e
		}
	}
	maps.Copy(inputs, auditorInputs)
	current, err := readPreparation(r.ConsumerPreparation, false, proofs)
	if err != nil {
		return nil, err
	}
	old, err := readPreparation(p.Preparation, false, proofs)
	if err != nil {
		return nil, err
	}
	for _, prep := range []Preparation{current, old} {
		if err = immutableGoCustody(inputs, prep); err != nil {
			return nil, err
		}
		for original, want := range prep.Inputs {
			path := original
			if copy, ok := p.SourceBindings[original]; ok {
				if p.SnapshotPins[copy] != want {
					return nil, fmt.Errorf("consumer/producer native source differs")
				}
				path = copy
			}
			inputs[path] = want
		}
	}
	for original, want := range b.Inputs {
		path := original
		if copy, ok := p.SourceBindings[original]; ok {
			if p.SnapshotPins[copy] != want {
				return nil, fmt.Errorf("producer source binding changed")
			}
			path = copy
		}
		inputs[path] = want
	}
	maps.Copy(inputs, b.Objects)
	for _, f := range []File{r.Build, r.Plan, p.Preparation, r.ConsumerPreparation, r.Controller, r.Consumer, r.Watcher, r.WatcherInterpreter, r.PhaseContract, b.Output, b.Depfile, b.BeforeCompile, b.LinkMap, b.Prior} {
		if err = addPin(inputs, f); err != nil {
			return nil, err
		}
	}
	for _, c := range b.Commands {
		if err = commandEvidence(c); err != nil {
			return nil, err
		}
		inputs[c.Stdout] = c.StdoutSHA
		inputs[c.Stderr] = c.StderrSHA
	}
	for _, code := range append(slices.Clone(r.Policy.Allowed), r.Policy.Alternates...) {
		inputs[code.Object.Path] = code.Object.SHA256
	}
	return inputs, checkPins(inputs)
}

func makeGuardPlan(r RuntimePlan) (g supgpu.GuardPlan, err error) {
	contract, err := guardPin(r.PhaseContract)
	if err != nil {
		return g, err
	}
	g = supgpu.GuardPlan{Schema: "nanopore-audited-guard-plan/v1", Policy: r.Policy, Contract: contract, Inputs: []supgpu.Pin{}}
	for _, path := range slices.Sorted(maps.Keys(r.Custody)) {
		pin, e := guardPin(File{path, r.Custody[path]})
		if e != nil {
			return g, e
		}
		g.Inputs = append(g.Inputs, pin)
	}
	return g, nil
}

func checkSerializedGuard(r RuntimePlan, b CompilerPhase, p ClosedBuildPlan) error {
	guard, err := guardPin(r.Guard)
	if err != nil {
		return err
	}
	phases, err := guardPin(r.PhaseContract)
	if err != nil {
		return err
	}
	_, err = supgpu.ReplayGuarded(guard, phases, decoderCommand(r, b, p), supgpu.GuardedRun{})
	if err == nil || err.Error() != "guarded run differs from independently expected invocation" {
		return fmt.Errorf("serialized decoder guard did not reach absent-run boundary: %v", err)
	}
	return nil
}

func PlanGuardedRuntime(o GuardRuntimeOptions) (r RuntimePlan, err error) {
	for _, p := range []*File{&o.Build, &o.Consumer, &o.AuditorProof} {
		*p, err = normalizeAnchor(*p)
		if err != nil {
			return r, err
		}
	}
	proofs := newNumericProofs()
	var declared CompilerPhase
	if err = readJSON(o.Build, &declared); err != nil {
		return r, err
	}
	var declaredPlan ClosedBuildPlan
	if err = readJSON(declared.Plan, &declaredPlan); err != nil {
		return r, err
	}
	if declaredPlan.Schema != "nanopore-decoder-closed-build/v3" && o.Build.SHA256 != decoderBuildSHA {
		return r, fmt.Errorf("this capture admits only the accepted v2 decoder build")
	}
	r, err = makeRuntimeScoped(o.Build, o.Consumer, proofs)
	if err != nil {
		return r, err
	}
	b, p, err := readBuiltPhaseScoped(o.Build, false, proofs)
	if err != nil {
		return r, err
	}
	if !numericBuild(p) && (b.Output.SHA256 != decoderBinarySHA || p.Schema != "nanopore-decoder-closed-build/v2") || p.Recipe.CUPTI.Path == "" {
		return r, fmt.Errorf("accepted observed producer/CUPTI identity differs")
	}
	a, q, auditInputs, err := readAuditor(o.AuditorProof)
	if err != nil {
		return r, err
	}
	r.Schema = "nanopore-decoder-runtime-plan/v3"
	r.AuditProof = o.AuditorProof
	r.Watcher, err = artifactPin("verification/supgpu/tools/watch_roles.py")
	if err != nil {
		return r, err
	}
	if r.Watcher.SHA256 != roleWatcherSHA {
		return r, fmt.Errorf("accepted watcher019 changed")
	}
	r.WorkingDirectory, err = os.Getwd()
	if err != nil {
		return r, err
	}
	r.WorkingDirectory, err = filepath.EvalSymlinks(r.WorkingDirectory)
	if err != nil {
		return r, err
	}
	r.CaptureDirectory, err = filepath.Abs(o.Capture)
	if err != nil {
		return r, err
	}
	if o.Capture == "" {
		return r, fmt.Errorf("prospective fresh capture path required")
	}
	if _, e := os.Stat(r.CaptureDirectory); !os.IsNotExist(e) {
		return r, fmt.Errorf("prospective capture path must not exist")
	}
	r.Seeds = append(r.Seeds, nativeproof.Pin{Path: a.Audit.Path, SHA256: a.Audit.SHA256})
	slices.SortFunc(r.Seeds, func(a, b nativeproof.Pin) int {
		if a.Path < b.Path {
			return -1
		}
		if a.Path > b.Path {
			return 1
		}
		return 0
	})
	r.ELF, err = nativeproof.CloseELF(r.Seeds, r.SearchDirs)
	if err != nil {
		return r, err
	}
	if err = checkHostELF(r.ELF); err != nil {
		return r, err
	}
	for _, pin := range r.ELF.Inputs {
		r.Inputs[pin.Path] = pin.SHA256
	}
	binary, err := guardPin(b.Output)
	if err != nil {
		return r, err
	}
	r.Policy, err = selectedDecoderPolicy(r.ELF, binary, a.Audit, q.Policy, r.SearchDirs)
	if err != nil {
		return r, err
	}
	dir, err := runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	body, err := os.ReadFile(r.Watcher.Path)
	if err != nil {
		return r, err
	}
	watcherPath := filepath.Join(dir, "watch_roles.py")
	if err = os.WriteFile(watcherPath, body, 0400); err != nil {
		return r, err
	}
	r.Watcher, err = artifactPin(watcherPath)
	if err != nil {
		return r, err
	}
	r.WatcherInterpreter, err = artifactPin("/usr/bin/python3")
	if err != nil {
		return r, err
	}
	if numericBuild(p) {
		old, e := readPreparation(p.Preparation, false, proofs)
		if e != nil {
			return r, e
		}
		current, e := readPreparation(r.ConsumerPreparation, false, proofs)
		if e != nil {
			return r, e
		}
		if current.NumericLineage != old.NumericLineage {
			return r, fmt.Errorf("numeric producer/consumer lineage differs")
		}
		r.Schema = "nanopore-decoder-runtime-plan/v4"
		r.NumericLineage = old.NumericLineage
		kernel, e := os.ReadFile("/proc/driver/nvidia/version")
		if e != nil {
			return r, e
		}
		kernelPath := filepath.Join(dir, "driver-version.txt")
		if e = os.WriteFile(kernelPath, kernel, 0400); e != nil {
			return r, e
		}
		r.KernelVersion, e = artifactPin(kernelPath)
		if e != nil {
			return r, e
		}
		if e = verifyNumericConditionsScoped(r, false, proofs); e != nil {
			return r, e
		}
	}
	phasePath := filepath.Join(dir, "phases.json")
	if err = writeJSON(phasePath, decoderPhases()); err != nil {
		return r, err
	}
	r.PhaseContract, err = artifactPin(phasePath)
	if err != nil {
		return r, err
	}
	r.Custody, err = guardCustodyScoped(r, b, p, auditInputs, proofs)
	if err != nil {
		return r, err
	}
	g, err := makeGuardPlan(r)
	if err != nil {
		return r, err
	}
	guardPath := filepath.Join(dir, "guard.json")
	if err = writeJSON(guardPath, g); err != nil {
		return r, err
	}
	r.Guard, err = artifactPin(guardPath)
	if err != nil {
		return r, err
	}
	if err = checkSerializedGuard(r, b, p); err != nil {
		return r, err
	}
	if err = proofs.finish(); err != nil {
		return r, err
	}
	r.Limits = []string{"Exact selected standalone decoder/CUPTI providers; unknown code is vetoed before initializers.", "Every six-phase checkpoint requires actual selected mappings; guard replay uses independently pinned contract/command.", "Only original four decoder calls, once each; no timing activity or warmup.", "Actual FUNCTION/context/module/ELF-index/image evidence is required, never inferred from device SM.", "Watcher019 source and exact future role argv are pinned; outer owned cgroup remains required."}
	runtimePath := filepath.Join(dir, "runtime.json")
	if err = writeJSON(runtimePath, r); err != nil {
		return r, err
	}
	runtimePin, err := artifactPin(runtimePath)
	if err != nil {
		return r, err
	}
	watchPath := filepath.Join(dir, "watch-plan.json")
	if err = writeJSON(watchPath, expectedWatchPlan(r, runtimePin, b, p)); err != nil {
		return r, err
	}
	watchPin, err := artifactPin(watchPath)
	if err != nil {
		return r, err
	}
	return r, writeJSON(filepath.Join(dir, "watch-launch.json"), expectedSupervisor(r, watchPin))
}

func verifyGuardRuntime(r RuntimePlan, b CompilerPhase, p ClosedBuildPlan, current bool) error {
	return verifyGuardRuntimeScoped(r, b, p, current, nil)
}

func verifyGuardRuntimeScoped(r RuntimePlan, b CompilerPhase, p ClosedBuildPlan, current bool, proofs *numericProofs) error {
	if !numericRuntimeIdentity(r, b, p) || r.ImageGate != passiveImageGate || r.Plan != b.Plan || r.Controller != p.Controller {
		return fmt.Errorf("observed producer/runtime identity differs")
	}
	if numericBuild(p) {
		if err := verifyNumericConditionsScoped(r, current, proofs); err != nil {
			return err
		}
	}
	consumer, err := readPreparation(r.ConsumerPreparation, current, proofs)
	if err != nil {
		return err
	}
	if numericBuild(p) && consumer.NumericLineage != r.NumericLineage {
		return fmt.Errorf("numeric current consumer policy differs")
	}
	if r.Consumer != (File{consumer.Go.Controller, consumer.Go.ControllerSHA}) {
		return fmt.Errorf("runtime consumer binding differs")
	}
	a, q, auditInputs, err := readAuditor(r.AuditProof)
	if err != nil {
		return err
	}
	seeds, dirs, err := runtimeSeeds(b.Output, p.Recipe)
	if err != nil {
		return err
	}
	seeds = append(seeds, nativeproof.Pin{Path: a.Audit.Path, SHA256: a.Audit.SHA256})
	slices.SortFunc(seeds, func(a, b nativeproof.Pin) int {
		if a.Path < b.Path {
			return -1
		}
		if a.Path > b.Path {
			return 1
		}
		return 0
	})
	closure, err := nativeproof.CloseELF(seeds, dirs)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(seeds, r.Seeds) || !slices.Equal(dirs, r.SearchDirs) || !reflect.DeepEqual(closure, r.ELF) {
		return fmt.Errorf("decoder/auditor runtime closure differs")
	}
	cache, err := artifactPin("/etc/ld.so.cache")
	if err != nil {
		return err
	}
	inputs := map[string]string{cache.Path: cache.SHA256}
	for _, pin := range closure.Inputs {
		inputs[pin.Path] = pin.SHA256
	}
	if cache != r.LoaderCache || !maps.Equal(inputs, r.Inputs) {
		return fmt.Errorf("runtime inputs/cache differ")
	}
	binary, err := guardPin(b.Output)
	if err != nil {
		return err
	}
	policy, err := selectedDecoderPolicy(closure, binary, a.Audit, q.Policy, dirs)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(policy, r.Policy) {
		return fmt.Errorf("selected decoder provider policy changed")
	}
	var phases supgpu.PhaseContract
	if err = readJSON(r.PhaseContract, &phases); err != nil {
		return err
	}
	if !reflect.DeepEqual(phases, decoderPhases()) {
		return fmt.Errorf("exact six-phase decoder contract changed")
	}
	if err = r.Watcher.verify(); err != nil {
		return err
	}
	if r.Watcher.SHA256 != roleWatcherSHA || r.Watcher.Path != filepath.Join(filepath.Dir(r.Guard.Path), "watch_roles.py") {
		return fmt.Errorf("immutable watcher019 identity differs")
	}
	if err = r.WatcherInterpreter.verify(); err != nil {
		return err
	}
	custody, err := guardCustodyScoped(r, b, p, auditInputs, proofs)
	if err != nil {
		return err
	}
	if !maps.Equal(custody, r.Custody) {
		return fmt.Errorf("guard producer/current-consumer custody changed")
	}
	expected, err := makeGuardPlan(r)
	if err != nil {
		return err
	}
	var saved supgpu.GuardPlan
	if err = readJSON(r.Guard, &saved); err != nil {
		return err
	}
	if !reflect.DeepEqual(expected, saved) {
		return fmt.Errorf("independent guard plan changed")
	}
	if !filepath.IsAbs(r.CaptureDirectory) || !filepath.IsAbs(r.WorkingDirectory) {
		return fmt.Errorf("capture command directories must be absolute")
	}
	return checkSerializedGuard(r, b, p)
}
