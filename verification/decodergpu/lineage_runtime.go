package decodergpu

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type LineageRuntime struct {
	Schema, Directory, Output, WorkingDirectory                            string
	Build, ConsumerPlan, AuditorProof, Guard, Phases, Watcher, Interpreter File
	KernelVersion                                                          File
	CaseIndex                                                              int
	Contract                                                               LineageContract
	ELF                                                                    nativeproof.ELFClosure
	Seeds                                                                  []nativeproof.Pin
	SearchDirs                                                             []string
	LoaderCache                                                            File
	Policy                                                                 supgpu.RuntimePolicy
	Custody                                                                map[string]string
}
type LineageRuntimeOptions struct {
	Build, ConsumerPlan, AuditorProof File
	CaseIndex                         int
	Out, Output                       string
}

func readLineageBuild(anchor File) (b CompilerPhase, p LineagePlan, err error) {
	if err = readJSON(anchor, &b); err != nil {
		return b, p, err
	}
	if b.Schema != "nanopore-decoder-lineage-compiler/v1" || b.Phase != "build" || b.Status != "passed" || b.Error != "" || len(b.Commands) != 2 || len(b.Objects) != 1 {
		return b, p, fmt.Errorf("lineage immutable build status/shape differs")
	}
	if err = readJSON(b.Plan, &p); err != nil {
		return b, p, err
	}
	if err = verifyLineagePlan(p, false, false); err != nil {
		return b, p, err
	}
	var header CompilerPhase
	if err = readJSON(b.Prior, &header); err != nil {
		return b, p, err
	}
	if header.Schema != b.Schema || header.Plan != b.Plan || header.Phase != "headers" || header.Status != "passed" || header.Error != "" || len(header.Commands) != 1 || !maps.Equal(b.Inputs, header.Inputs) || b.BeforeCompile != header.BeforeCompile || !slices.Equal(b.Dependencies, header.Dependencies) {
		return b, p, fmt.Errorf("lineage build/header proof differs")
	}
	for _, command := range append(slices.Clone(b.Commands), header.Commands...) {
		if err = commandEvidence(command); err != nil {
			return b, p, err
		}
	}
	object := filepath.Join(b.Directory, "lineage.o")
	binary := filepath.Join(b.Directory, "lineage-probe")
	dep := filepath.Join(b.Directory, "actual.d")
	linkMap := filepath.Join(b.Directory, "link.map")
	compile := append([]string{p.Recipe.Compiler.Path}, p.Recipe.Flags...)
	compile = append(compile, "-MD", "-MF", dep, "-c", p.Source, "-o", object)
	link := append([]string{p.Recipe.Compiler.Path, object}, p.Recipe.Libraries...)
	link = append(link, "-Wl,-t", "-Wl,-Map="+linkMap, "-o", binary)
	if !slices.Equal(b.Commands[0].Args, compile) || !slices.Equal(b.Commands[1].Args, link) || b.Output.Path != binary || b.Depfile.Path != dep || b.LinkMap.Path != linkMap || b.Objects[object] == "" {
		return b, p, fmt.Errorf("lineage build actual command/output roles differ")
	}
	for _, file := range []File{b.Output, b.Depfile, b.LinkMap, b.BeforeCompile, header.Depfile} {
		if err = file.verify(); err != nil {
			return b, p, err
		}
	}
	var frozen map[string]string
	if err = readJSON(b.BeforeCompile, &frozen); err != nil {
		return b, p, err
	}
	if !maps.Equal(frozen, b.Inputs) {
		return b, p, fmt.Errorf("lineage before-compile proof differs")
	}
	actual, err := dependencySet(dep, p.WorkingDirectory)
	if err != nil {
		return b, p, err
	}
	if err = verifyDependencies(b.Dependencies, actual); err != nil {
		return b, p, err
	}
	actual, err = dependencySet(header.Depfile.Path, p.WorkingDirectory)
	if err != nil {
		return b, p, err
	}
	if err = verifyDependencies(b.Dependencies, actual); err != nil {
		return b, p, err
	}
	inputs := maps.Clone(p.Inputs)
	for _, path := range b.Dependencies {
		want, ok := b.Inputs[path]
		if !ok {
			return b, p, fmt.Errorf("lineage frozen header omitted")
		}
		inputs[path] = want
	}
	if !maps.Equal(inputs, b.Inputs) {
		return b, p, fmt.Errorf("lineage build/header input membership differs")
	}
	if err = checkProducerInputs(b.Inputs, ClosedBuildPlan{Directory: p.Directory, SourceBindings: p.SourceBindings, SnapshotPins: p.SnapshotPins}, false); err != nil {
		return b, p, err
	}
	if err = checkPins(b.Objects); err != nil {
		return b, p, err
	}
	preprocess := append([]string{p.Recipe.Compiler.Path}, p.Recipe.Flags...)
	preprocess = append(preprocess, "-M", "-MT", "lineage-probe", "-MF", header.Depfile.Path, p.Source)
	if !slices.Equal(header.Commands[0].Args, preprocess) {
		return b, p, fmt.Errorf("lineage retained -M command differs")
	}
	var query CompilerPhase
	if err = readJSON(header.Prior, &query); err != nil {
		return b, p, err
	}
	queries := compilerQueries(lineageCompiler(p))
	if query.Schema != b.Schema || query.Plan != b.Plan || query.Phase != "query" || query.Status != "passed" || query.Error != "" || len(query.Commands) != len(queries) || !maps.Equal(query.Inputs, p.Inputs) {
		return b, p, fmt.Errorf("lineage retained query proof differs")
	}
	for i, q := range queries {
		command := query.Commands[i]
		if !slices.Equal(command.Args, []string{p.Recipe.Compiler.Path, q.Flag}) {
			return b, p, fmt.Errorf("lineage retained query argv differs")
		}
		if err = commandEvidence(command); err != nil {
			return b, p, err
		}
		body, e := os.ReadFile(command.Stdout)
		if e != nil {
			return b, p, e
		}
		if err = verifyQuery(string(body), q, p.WorkingDirectory); err != nil {
			return b, p, err
		}
	}
	trace, err := os.ReadFile(b.Commands[1].Stdout)
	if err != nil {
		return b, p, err
	}
	if err = h07.VerifyLinkTrace(string(trace), p.WorkingDirectory, lineageLinkInputs(p), b.Objects); err != nil {
		return b, p, err
	}
	body, err := os.ReadFile(linkMap)
	if err != nil {
		return b, p, err
	}
	err = lineageLinkMap(string(body), p.WorkingDirectory, lineageLinkInputs(p), b.Objects)
	return b, p, err
}
func lineagePhaseContract() supgpu.PhaseContract {
	return supgpu.PhaseContract{Schema: "nanopore-audited-phase-contract/v1", Role: "reference", Action: "decoder-lineage", Names: lineagePhases()}
}
func lineageCommand(r LineageRuntime, b CompilerPhase) supgpu.GuardCommand {
	return supgpu.GuardCommand{Directory: r.WorkingDirectory, Output: filepath.Join(r.Output, "guard"), Args: []string{b.Output.Path, "--decoder-load-lineage", r.Contract.Architecture, string(r.Contract.Lookup), r.Contract.Input.Image.Path, filepath.Join(r.Output, "native")}}
}
func lineageWatchPlan(r LineageRuntime, runtime File, b CompilerPhase, consumer LineagePlan) WatchPlan {
	return WatchPlan{Controller: []string{consumer.Go.Controller, "-unit", "load", "-runtime", runtime.Path, "-runtime-sha256", runtime.SHA256, "-admitted", "-out", r.Output}, Limit: 240, Roles: []WatchRole{{Name: "lineage", Args: lineageCommand(r, b).Args, SHA256: b.Output.SHA256, Limit: 180}}}
}
func lineageWatchInvocation(r LineageRuntime, watch File) SupervisorInvocation {
	dir := r.Output + "-supervisor"
	return SupervisorInvocation{Args: []string{r.Interpreter.Path, "-B", r.Watcher.Path, watch.Path, filepath.Join(dir, "watcher.json"), watch.SHA256}, Environment: []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "TZ=UTC", "GOMAXPROCS=2", "GOMEMLIMIT=3GiB", "PYTHONDONTWRITEBYTECODE=1"}, Directory: r.WorkingDirectory, Stdout: filepath.Join(dir, "stdout"), Stderr: filepath.Join(dir, "stderr"), Receipt: filepath.Join(dir, "command.json")}
}

func lineageRuntimeShape(b CompilerPhase, p LineagePlan, consumer LineagePlan, proof File) (r LineageRuntime, err error) {
	_, _, original, _, err := lineageAnchors(p.Producer, p.ProducerRuntime, p.Capture)
	if err != nil {
		return r, err
	}
	audit, qualification, auditorInputs, err := readAuditor(proof)
	if err != nil {
		return r, err
	}
	r = LineageRuntime{Schema: "nanopore-decoder-load-runtime/v1", AuditorProof: proof, WorkingDirectory: p.WorkingDirectory, SearchDirs: slices.Clone(original.SearchDirs), Custody: map[string]string{}}
	r.Seeds = []nativeproof.Pin{{Path: b.Output.Path, SHA256: b.Output.SHA256}, {Path: audit.Audit.Path, SHA256: audit.Audit.SHA256}}
	for _, code := range original.Policy.Allowed {
		if code.SONAME == "libcuda.so.1" || strings.HasPrefix(code.SONAME, "libcupti.so") || strings.HasPrefix(code.SONAME, "libnvidia-") {
			r.Seeds = append(r.Seeds, nativeproof.Pin{Path: code.Object.Path, SHA256: code.Object.SHA256})
		}
	}
	slices.SortFunc(r.Seeds, func(a, b nativeproof.Pin) int { return strings.Compare(a.Path, b.Path) })
	r.ELF, err = nativeproof.CloseELF(r.Seeds, r.SearchDirs)
	if err != nil {
		return r, err
	}
	if err = checkHostELF(r.ELF); err != nil {
		return r, err
	}
	binary, err := guardPin(b.Output)
	if err != nil {
		return r, err
	}
	r.Policy, err = selectedDecoderPolicy(r.ELF, binary, audit.Audit, qualification.Policy, r.SearchDirs)
	if err != nil {
		return r, err
	}
	for _, code := range r.Policy.Allowed {
		if strings.HasPrefix(code.SONAME, "libcudart") {
			return r, fmt.Errorf("lineage runtime must exclude CUDA runtime")
		}
	}
	r.LoaderCache, err = artifactPin("/etc/ld.so.cache")
	if err != nil {
		return r, err
	}
	maps.Copy(r.Custody, auditorInputs)
	for _, source := range []LineagePlan{p, consumer} {
		for path, want := range source.Inputs {
			if copy, ok := source.SourceBindings[path]; ok {
				path = copy
			}
			r.Custody[path] = want
		}
		for path, want := range source.Go.SnapshotPins {
			r.Custody[path] = want
		}
	}
	for path, want := range b.Inputs {
		if copy, ok := p.SourceBindings[path]; ok {
			path = copy
		}
		r.Custody[path] = want
	}
	maps.Copy(r.Custody, b.Objects)
	for _, node := range r.ELF.Inputs {
		r.Custody[node.Path] = node.SHA256
	}
	for _, command := range b.Commands {
		if err = commandEvidence(command); err != nil {
			return r, err
		}
		r.Custody[command.Stdout] = command.StdoutSHA
		r.Custody[command.Stderr] = command.StderrSHA
	}
	for _, file := range []File{b.Plan, b.Prior, b.Output, b.BeforeCompile, b.Depfile, b.LinkMap, r.LoaderCache, proof} {
		if err = addPin(r.Custody, file); err != nil {
			return r, err
		}
	}
	return r, nil
}
func lineageGuardPlan(r LineageRuntime) (g supgpu.GuardPlan, err error) {
	contract, err := guardPin(r.Phases)
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
func lineageSerializedGuard(r LineageRuntime, b CompilerPhase) error {
	guard, err := guardPin(r.Guard)
	if err != nil {
		return err
	}
	contract, err := guardPin(r.Phases)
	if err != nil {
		return err
	}
	_, err = supgpu.ReplayGuarded(guard, contract, lineageCommand(r, b), supgpu.GuardedRun{})
	if err == nil || err.Error() != "guarded run differs from independently expected invocation" {
		return fmt.Errorf("lineage serialized guard did not reach exact absent-run boundary: %v", err)
	}
	return nil
}

func PlanLineageRuntime(o LineageRuntimeOptions) (r LineageRuntime, err error) {
	for _, file := range []*File{&o.Build, &o.ConsumerPlan, &o.AuditorProof} {
		*file, err = normalizeAnchor(*file)
		if err != nil {
			return r, err
		}
	}
	b, p, err := readLineageBuild(o.Build)
	if err != nil {
		return r, err
	}
	var consumer LineagePlan
	if err = readJSON(o.ConsumerPlan, &consumer); err != nil {
		return r, err
	}
	if err = verifyLineagePlan(consumer, true, true); err != nil {
		return r, err
	}
	if o.CaseIndex < 0 || o.CaseIndex >= len(p.Cases) || !reflect.DeepEqual(p.Cases, consumer.Cases) {
		return r, fmt.Errorf("lineage prospective case/consumer differs")
	}
	r, err = lineageRuntimeShape(b, p, consumer, o.AuditorProof)
	if err != nil {
		return r, err
	}
	r.Build = o.Build
	r.ConsumerPlan = o.ConsumerPlan
	r.CaseIndex = o.CaseIndex
	r.Contract = p.Cases[o.CaseIndex]
	r.Output, err = filepath.Abs(o.Output)
	if err != nil {
		return r, err
	}
	if o.Output == "" {
		return r, fmt.Errorf("lineage fresh prospective output required")
	}
	if _, e := os.Stat(r.Output); !os.IsNotExist(e) {
		return r, fmt.Errorf("lineage prospective output already exists")
	}
	r.Directory, err = runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	kernel, err := os.ReadFile("/proc/driver/nvidia/version")
	if err != nil {
		return r, err
	}
	kernelPath := filepath.Join(r.Directory, "driver-version.txt")
	if err = os.WriteFile(kernelPath, kernel, 0400); err != nil {
		return r, err
	}
	r.KernelVersion, err = artifactPin(kernelPath)
	if err != nil {
		return r, err
	}
	watcher, err := artifactPin("verification/supgpu/tools/watch_roles.py")
	if err != nil || watcher.SHA256 != roleWatcherSHA {
		return r, fmt.Errorf("lineage accepted watcher019 differs: %v", err)
	}
	body, err := os.ReadFile(watcher.Path)
	if err != nil {
		return r, err
	}
	watcherCopy := filepath.Join(r.Directory, "watch_roles.py")
	if err = os.WriteFile(watcherCopy, body, 0400); err != nil {
		return r, err
	}
	r.Watcher, err = artifactPin(watcherCopy)
	if err != nil {
		return r, err
	}
	r.Interpreter, err = artifactPin("/usr/bin/python3")
	if err != nil {
		return r, err
	}
	phasePath := filepath.Join(r.Directory, "phases.json")
	if err = writeJSON(phasePath, lineagePhaseContract()); err != nil {
		return r, err
	}
	r.Phases, err = artifactPin(phasePath)
	if err != nil {
		return r, err
	}
	for _, file := range []File{r.Build, r.ConsumerPlan, r.Phases, r.Watcher, r.Interpreter, r.KernelVersion} {
		if err = addPin(r.Custody, file); err != nil {
			return r, err
		}
	}
	g, err := lineageGuardPlan(r)
	if err != nil {
		return r, err
	}
	guardPath := filepath.Join(r.Directory, "guard.json")
	if err = writeJSON(guardPath, g); err != nil {
		return r, err
	}
	r.Guard, err = artifactPin(guardPath)
	if err != nil {
		return r, err
	}
	if err = lineageSerializedGuard(r, b); err != nil {
		return r, err
	}
	path := filepath.Join(r.Directory, "runtime.json")
	if err = writeJSON(path, r); err != nil {
		return r, err
	}
	runtime, err := artifactPin(path)
	if err != nil {
		return r, err
	}
	watchPath := filepath.Join(r.Directory, "watch-plan.json")
	if err = writeJSON(watchPath, lineageWatchPlan(r, runtime, b, consumer)); err != nil {
		return r, err
	}
	watch, err := artifactPin(watchPath)
	if err != nil {
		return r, err
	}
	return r, writeJSON(filepath.Join(r.Directory, "watch-launch.json"), lineageWatchInvocation(r, watch))
}

func readLineageRuntime(anchor File, current bool) (r LineageRuntime, b CompilerPhase, p LineagePlan, consumer LineagePlan, err error) {
	if err = readJSON(anchor, &r); err != nil {
		return
	}
	if r.Schema != "nanopore-decoder-load-runtime/v1" || r.Directory != filepath.Dir(anchor.Path) || anchor.Path != filepath.Join(r.Directory, "runtime.json") {
		err = fmt.Errorf("lineage runtime metadata identity differs")
		return
	}
	b, p, err = readLineageBuild(r.Build)
	if err != nil {
		return
	}
	if err = readJSON(r.ConsumerPlan, &consumer); err != nil {
		return
	}
	if err = verifyLineagePlan(consumer, current, current); err != nil {
		return
	}
	if r.CaseIndex < 0 || r.CaseIndex >= len(p.Cases) || !reflect.DeepEqual(p.Cases, consumer.Cases) || !reflect.DeepEqual(r.Contract, p.Cases[r.CaseIndex]) {
		err = fmt.Errorf("lineage source/case/consumer contract differs")
		return
	}
	expected, e := lineageRuntimeShape(b, p, consumer, r.AuditorProof)
	if e != nil {
		err = e
		return
	}
	expected.Directory = r.Directory
	expected.Output = r.Output
	expected.Build = r.Build
	expected.ConsumerPlan = r.ConsumerPlan
	expected.CaseIndex = r.CaseIndex
	expected.Contract = p.Cases[r.CaseIndex]
	expected.Phases = r.Phases
	expected.Guard = r.Guard
	expected.Watcher = r.Watcher
	expected.Interpreter = r.Interpreter
	expected.KernelVersion = r.KernelVersion
	if !filepath.IsAbs(r.Output) || r.Output == r.Directory || r.Phases.Path != filepath.Join(r.Directory, "phases.json") || r.Guard.Path != filepath.Join(r.Directory, "guard.json") || r.Watcher.Path != filepath.Join(r.Directory, "watch_roles.py") || r.Watcher.SHA256 != roleWatcherSHA {
		err = fmt.Errorf("lineage phase/guard/watcher/output paths differ")
		return
	}
	if r.KernelVersion.Path != filepath.Join(r.Directory, "driver-version.txt") {
		err = fmt.Errorf("lineage kernel-driver version snapshot identity differs")
		return
	}
	if current {
		raw, e := os.ReadFile("/proc/driver/nvidia/version")
		if e != nil {
			err = e
			return
		}
		if digest(raw) != r.KernelVersion.SHA256 {
			err = fmt.Errorf("lineage live kernel-driver version changed")
			return
		}
	}
	var phases supgpu.PhaseContract
	if err = readJSON(r.Phases, &phases); err != nil {
		return
	}
	if !reflect.DeepEqual(phases, lineagePhaseContract()) {
		err = fmt.Errorf("lineage exact five-phase contract differs")
		return
	}
	for _, file := range []File{r.Build, r.ConsumerPlan, r.Phases, r.Watcher, r.Interpreter, r.KernelVersion} {
		if err = addPin(expected.Custody, file); err != nil {
			return
		}
	}
	if !reflect.DeepEqual(expected, r) {
		err = fmt.Errorf("lineage selected runtime/guard custody reconstructed differently")
		return
	}
	expectedGuard, e := lineageGuardPlan(r)
	if e != nil {
		err = e
		return
	}
	var saved supgpu.GuardPlan
	if err = readJSON(r.Guard, &saved); err != nil {
		return
	}
	if !reflect.DeepEqual(expectedGuard, saved) {
		err = fmt.Errorf("lineage serialized selected guard plan differs")
		return
	}
	err = lineageSerializedGuard(r, b)
	return
}
func CheckLineageRuntime(anchor File) error {
	anchor, err := normalizeAnchor(anchor)
	if err != nil {
		return err
	}
	_, _, _, _, err = readLineageRuntime(anchor, true)
	return err
}
