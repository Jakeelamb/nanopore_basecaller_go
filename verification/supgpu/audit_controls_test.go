package supgpu

import (
	"debug/elf"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

var auditControlSources = []string{"verification/supgpu/native/controls/loader.c", "verification/supgpu/native/controls/constructor.c"}

type auditControlPlan struct {
	Schema          string         `json:"schema"`
	Controller      Pin            `json:"controller"`
	ProducerProof   Pin            `json:"auditor_producer_proof"`
	AcceptedRuntime Pin            `json:"accepted_runtime"`
	AcceptedBuild   Pin            `json:"accepted_build"`
	CompilerPlan    Pin            `json:"compiler_plan"`
	Audit           Pin            `json:"audit"`
	Out             string         `json:"output_directory"`
	Inputs          []Pin          `json:"inputs"`
	Sources         []Pin          `json:"sources"`
	Queries         []ToolQuery    `json:"queries"`
	Preprocess      []BuildCommand `json:"preprocess"`
	Commands        []BuildCommand `json:"commands"`
	Outputs         []string       `json:"outputs"`
}
type auditControlBuild struct {
	Schema       string                  `json:"schema"`
	Status       string                  `json:"status"`
	Plan         Pin                     `json:"plan"`
	Preflight    []runner.CommandReceipt `json:"preflight_commands"`
	Commands     []runner.CommandReceipt `json:"commands"`
	PreDeps      []Pin                   `json:"precompile_depfiles"`
	Depfiles     []Pin                   `json:"depfiles"`
	Dependencies [][]string              `json:"dependencies"`
	Objects      []Pin                   `json:"objects"`
	Maps         []Pin                   `json:"maps"`
	Outputs      []Pin                   `json:"outputs"`
	Runtime      Pin                     `json:"runtime"`
	BeforeInputs Pin                     `json:"before_compile_inputs"`
}
type auditControlCase struct {
	Role              string                `json:"role"`
	Kind              string                `json:"kind"`
	Command           runner.CommandReceipt `json:"command"`
	Guard             Pin                   `json:"guard"`
	Constructor       Pin                   `json:"constructor_marker"`
	ConstructorAbsent bool                  `json:"constructor_absent"`
	Returned          Pin                   `json:"returned_marker"`
	ReturnedAbsent    bool                  `json:"returned_absent"`
	Rejection         string                `json:"rejection"`
}
type auditControlRun struct {
	Schema string             `json:"schema"`
	Status string             `json:"status"`
	Build  Pin                `json:"build"`
	Cases  []auditControlCase `json:"cases"`
}

// A separate CGO0 test executable exposes the control CLI. Ordinary Go tests
// retain their usual entry point; frozen model-consumer source is unchanged.
func TestMain(m *testing.M) {
	if len(os.Args) > 1 && os.Args[1] == "supgpu-audit-control" {
		if err := auditControlCommand(os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(runner.ExitCode(err))
		}
		return
	}
	os.Exit(m.Run())
}
func auditControlAdmission(unit string, admitted bool) error {
	if (unit == "compile" || unit == "run" || unit == "continue" || unit == "effective-run") && !admitted {
		return fmt.Errorf("native audit %s requires separate coordinator admission", unit)
	}
	return nil
}
func auditControlCommand(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("audit control action required")
	}
	unit := args[0]
	flags := flag.NewFlagSet("supgpu-audit-control", flag.ContinueOnError)
	dir := flags.String("dir", "", "controller build directory")
	controller := flags.String("controller", "", "frozen test controller")
	controllerSHA := flags.String("controller-sha256", "", "controller SHA256")
	runtimePath := flags.String("runtime-plan", "", "accepted metadata runtime plan")
	runtimeSHA := flags.String("runtime-plan-sha256", "", "runtime plan SHA256")
	proofPath := flags.String("proof", "", "frozen auditor producer proof")
	proofSHA := flags.String("proof-sha256", "", "auditor proof SHA256")
	planPath := flags.String("plan", "", "control plan")
	planSHA := flags.String("plan-sha256", "", "control plan SHA256")
	buildPath := flags.String("build", "", "control build")
	buildSHA := flags.String("build-sha256", "", "control build SHA256")
	prefix := flags.String("prefix", "", "immutable accepted partial run")
	prefixSHA := flags.String("prefix-sha256", "", "accepted partial run SHA256")
	out := flags.String("out", "", "fresh output directory")
	admitted := flags.Bool("native-admitted", false, "separate root admission for this exact phase")
	if err := flags.Parse(args[1:]); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected audit control arguments")
	}
	if err := auditControlAdmission(unit, *admitted); err != nil {
		return err
	}
	var value any
	var err error
	switch unit {
	case "freeze":
		value, err = freezeAuditController(*dir)
	case "proof":
		cp, e := pin(*controller)
		if e != nil {
			return e
		}
		if cp.SHA256 != *controllerSHA {
			return fmt.Errorf("test controller manifest changed")
		}
		rp, e := pin(*runtimePath)
		if e != nil {
			return e
		}
		if rp.SHA256 != *runtimeSHA {
			return fmt.Errorf("accepted runtime manifest changed")
		}
		value, err = prepareAuditorProof(cp, rp, *out)
	case "plan":
		value, err = prepareAuditControlPlan(*controller, *controllerSHA, *proofPath, *proofSHA, *out)
	case "check-plan":
		value, _, err = readAuditControlPlan(*planPath, *planSHA)
	case "compile":
		value, err = compileAuditControls(*planPath, *planSHA)
	case "run":
		value, err = runAuditControls(*buildPath, *buildSHA, *out)
	case "effective-run":
		value, err = runEffectiveAuditControls(*controller, *controllerSHA, *buildPath, *buildSHA, *out)
	case "effective-check":
		if *out != "" {
			return fmt.Errorf("effective-check does not accept output directory")
		}
		value, err = checkEffectiveAuditControls(*controller, *controllerSHA, *buildPath, *buildSHA)
	case "continuation-plan":
		value, err = prepareAuditContinuation(*controller, *controllerSHA, *buildPath, *buildSHA, *prefix, *prefixSHA, *out)
	case "check-continuation":
		value, _, _, err = readAuditContinuation(*planPath, *planSHA)
	case "continue":
		value, err = continueAuditControls(*planPath, *planSHA, *out)
	default:
		return fmt.Errorf("unknown audit control action %q", unit)
	}
	if err != nil {
		return err
	}
	raw, err := Marshal(value)
	if err == nil {
		fmt.Println(string(raw))
	}
	return err
}

func auditPlanRecipe(controller, proofPin Pin, out string) (p auditControlPlan, compiler BuildPlan, err error) {
	return auditPlanRecipeForController(controller, proofPin, out, true)
}

func auditPlanRecipeForController(controller, proofPin Pin, out string, current bool) (p auditControlPlan, compiler BuildPlan, err error) {
	p = auditControlPlan{Schema: "nanopore-supgpu-audit-control-plan/v2", Controller: controller, ProducerProof: proofPin, Out: out, Inputs: []Pin{}, Sources: []Pin{}}
	c, e := readAuditController(controller, current)
	if e != nil {
		err = e
		return
	}
	proof, e := readAuditorProof(proofPin)
	if e != nil {
		err = e
		return
	}
	if proof.Controller != controller {
		err = fmt.Errorf("auditor proof controller differs")
		return
	}
	p.AcceptedBuild, p.AcceptedRuntime, p.CompilerPlan, p.Audit = proof.AcceptedBuild, proof.AcceptedRuntime, proof.CompilerPlan, proof.Audit
	compiler = BuildPlan{Compiler: proof.Compiler, CompilerRuntime: proof.CompilerRuntime}
	seen := map[string]bool{}
	add := func(v Pin) {
		if !seen[v.Path] {
			seen[v.Path] = true
			p.Inputs = append(p.Inputs, v)
		}
	}
	assets, e := compilerAssetPins(proof.QualifiedBuildDir)
	if e != nil {
		err = e
		return
	}
	for _, v := range append(assets, proof.CompilerRuntime.Inputs...) {
		add(v)
	}
	libs, bindings, queries, e := linkClosure([]string{"-ldl"}, out, proof.Compiler.Path)
	if e != nil {
		err = e
		return
	}
	if len(libs) != 1 {
		err = fmt.Errorf("finite libdl selection differs")
		return
	}
	for _, b := range bindings {
		add(b.Input)
	}
	p.Queries = append(compilerQueries(proof.CompilerRuntime), queries...)
	for _, v := range []Pin{proofPin, controller, proof.Audit} {
		add(v)
	}
	p.Sources, err = auditControllerNativeSources(c)
	if err != nil {
		return
	}
	for _, v := range p.Sources {
		add(v)
	}
	for _, path := range []string{"/usr/include/asm/unistd_64.h", "/usr/include/dlfcn.h", "/usr/include/bits/fcntl-linux.h"} {
		var v Pin
		v, err = pin(path)
		if err != nil {
			return
		}
		add(v)
	}
	p.Preprocess, p.Commands, p.Outputs = auditControlCommands(out, proof.Compiler.Path, p.Sources, libs[0])
	sort.Slice(p.Inputs, func(i, j int) bool { return p.Inputs[i].Path < p.Inputs[j].Path })
	compiler.Inputs, compiler.Preprocess, compiler.Commands = p.Inputs, p.Preprocess, p.Commands
	if err = verifyCompilerRuntime(compiler); err != nil {
		return
	}
	return
}

func auditControllerNativeSources(c GoController) ([]Pin, error) {
	if err := verifyPin("", c.Dependencies); err != nil {
		return nil, err
	}
	deps, err := os.ReadFile(c.Dependencies.Path)
	if err != nil {
		return nil, err
	}
	root, err := controllerSourceRoot(deps, "github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu")
	if err != nil {
		return nil, err
	}
	sources := []Pin{}
	for _, path := range auditControlSources {
		original := filepath.Join(root, path)
		var selected Pin
		count := 0
		for _, s := range c.Snapshots {
			if s.Original == original {
				selected = s.Copy
				count++
			}
		}
		if count != 1 {
			return nil, fmt.Errorf("native source original membership differs: %s", original)
		}
		if err = verifyPin("", selected); err != nil {
			return nil, err
		}
		v, e := pin(original)
		if e != nil {
			return nil, e
		}
		if v.SHA256 != selected.SHA256 || v.Bytes != selected.Bytes {
			return nil, fmt.Errorf("native source differs from producer snapshot: %s", original)
		}
		sources = append(sources, v)
	}
	return sources, nil
}

func auditControlCommands(out, driver string, sources []Pin, dl string) ([]BuildCommand, []BuildCommand, []string) {
	pre := []BuildCommand{}
	commands := []BuildCommand{}
	objects := []string{}
	outputs := []string{filepath.Join(out, "loader"), filepath.Join(out, "providers", "qualified", "libsupgpu_control_provider.so.1"), filepath.Join(out, "providers", "wrong", "libsupgpu_control_provider.so.1"), filepath.Join(out, "providers", "unknown", "libsupgpu_control_unknown.so.1")}
	for i := 0; i < 4; i++ {
		source := sources[0].Path
		if i != 0 {
			source = sources[1].Path
		}
		obj := filepath.Join(out, fmt.Sprintf("control-%d.o", i))
		objects = append(objects, obj)
		base := []string{driver, "-x", "c", "-O2", "-fPIC", "-ffreestanding", "-fno-builtin", "-fno-stack-protector"}
		if i != 0 {
			base = append(base, fmt.Sprintf("-DCONTROL_VARIANT=%d", i))
		}
		args := append(append([]string{}, base...), "-MD", "-MF", obj+".d", "-c", source, "-o", obj)
		commands = append(commands, BuildCommand{out, args, obj + ".d", ""})
		args = append(append([]string{}, base...), "-M", "-MF", obj+".before.d", "-MT", obj, source)
		pre = append(pre, BuildCommand{out, args, obj + ".before.d", ""})
	}
	for i, output := range outputs {
		args := []string{driver}
		if i != 0 {
			soname := "libsupgpu_control_provider.so.1"
			if i == 3 {
				soname = "libsupgpu_control_unknown.so.1"
			}
			args = append(args, "-shared", "-nostdlib", "-nodefaultlibs", "-nostartfiles", "-Wl,-soname,"+soname)
		}
		args = append(args, objects[i])
		if i == 0 {
			args = append(args, dl)
		}
		args = append(args, "-Wl,-t,-Map,"+output+".map", "-o", output)
		commands = append(commands, BuildCommand{out, args, "", output + ".map"})
	}
	return pre, commands, outputs
}
func prepareAuditControlPlan(controllerPath, controllerSHA, proofPath, proofSHA, out string) (auditControlPlan, error) {
	cp, err := pin(controllerPath)
	if err != nil {
		return auditControlPlan{}, err
	}
	if cp.SHA256 != controllerSHA {
		return auditControlPlan{}, fmt.Errorf("audit controller manifest changed")
	}
	rp, err := pin(proofPath)
	if err != nil {
		return auditControlPlan{}, err
	}
	if rp.SHA256 != proofSHA {
		return auditControlPlan{}, fmt.Errorf("accepted runtime manifest changed")
	}
	out, err = filepath.Abs(out)
	if err != nil {
		return auditControlPlan{}, err
	}
	p, _, err := auditPlanRecipe(cp, rp, out)
	if err != nil {
		return p, err
	}
	if err = os.Mkdir(out, 0755); err != nil {
		return p, err
	}
	for _, output := range p.Outputs[1:] {
		if err = os.MkdirAll(filepath.Dir(output), 0755); err != nil {
			return p, err
		}
	}
	return p, writeJSON(filepath.Join(out, "plan.json"), p)
}
func readAuditControlPlan(path, sha string) (p auditControlPlan, compiler BuildPlan, err error) {
	return readAuditControlPlanForController(path, sha, true)
}

func readAuditControlPlanForController(path, sha string, current bool) (p auditControlPlan, compiler BuildPlan, err error) {
	if err = readPinnedJSON(path, sha, &p); err != nil {
		return
	}
	expected, c, e := auditPlanRecipeForController(p.Controller, p.ProducerProof, p.Out, current)
	if e != nil {
		err = e
		return
	}
	if !reflect.DeepEqual(expected, p) {
		err = fmt.Errorf("finite audit compiler plan changed")
		return
	}
	return p, c, nil
}

func compileAuditControls(path, sha string) (b auditControlBuild, err error) {
	b = auditControlBuild{Schema: "nanopore-supgpu-audit-control-build/v1", Status: "failed"}
	p, compiler, err := readAuditControlPlan(path, sha)
	if err != nil {
		return b, err
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return b, err
	}
	defer lock.Close()
	b.Plan, err = pin(path)
	if err != nil {
		return b, err
	}
	defer func() {
		e := writeJSON(filepath.Join(p.Out, "build.json"), b)
		if err == nil {
			err = e
		}
	}()
	verifyInputs := func() error {
		for _, v := range p.Inputs {
			if e := verifyPin("", v); e != nil {
				return e
			}
		}
		_, e := readAuditController(p.Controller, true)
		return e
	}
	if err = verifyInputs(); err != nil {
		return b, err
	}
	for i, q := range p.Queries {
		r, e := executeCompiler(compiler, p.Out, p.Out, fmt.Sprintf("query-%02d", i), q.Argv)
		b.Preflight = append(b.Preflight, r)
		if e != nil {
			return b, e
		}
		raw, e := os.ReadFile(r.Stdout)
		if e != nil {
			return b, e
		}
		if e = sameQueryResult(string(raw), q, p.Out); e != nil {
			return b, e
		}
	}
	allowed := map[string]bool{}
	for _, v := range p.Inputs {
		allowed[v.Path] = true
	}
	for i, c := range p.Preprocess {
		r, e := executeCompiler(compiler, p.Out, p.Out, fmt.Sprintf("deps-%02d", i), c.Argv)
		b.Preflight = append(b.Preflight, r)
		if e != nil {
			return b, e
		}
		v, e := pin(c.Depfile)
		if e != nil {
			return b, e
		}
		b.PreDeps = append(b.PreDeps, v)
		deps, e := sortedDependencies(c.Depfile, c.Directory)
		if e != nil {
			return b, e
		}
		for _, path := range deps {
			if !allowed[path] {
				return b, fmt.Errorf("unfrozen control preprocessing input %s", path)
			}
		}
		b.Dependencies = append(b.Dependencies, deps)
	}
	if err = writeJSON(filepath.Join(p.Out, "before-compile-inputs.json"), p.Inputs); err != nil {
		return b, err
	}
	b.BeforeInputs, err = pin(filepath.Join(p.Out, "before-compile-inputs.json"))
	if err != nil {
		return b, err
	}
	for i, c := range p.Commands {
		if err = verifyInputs(); err != nil {
			return b, err
		}
		r, e := executeCompiler(compiler, p.Out, p.Out, fmt.Sprintf("build-%02d", i), c.Argv)
		b.Commands = append(b.Commands, r)
		if e != nil {
			return b, e
		}
		if c.Depfile != "" {
			v, e := pin(c.Depfile)
			if e != nil {
				return b, e
			}
			b.Depfiles = append(b.Depfiles, v)
			deps, e := sortedDependencies(c.Depfile, c.Directory)
			if e != nil {
				return b, e
			}
			if !reflect.DeepEqual(deps, b.Dependencies[i]) {
				return b, fmt.Errorf("control compile dependencies differ")
			}
			v, e = pin(c.Argv[len(c.Argv)-1])
			if e != nil {
				return b, e
			}
			b.Objects = append(b.Objects, v)
		}
		if c.LinkMap != "" {
			v, e := pin(c.LinkMap)
			if e != nil {
				return b, e
			}
			b.Maps = append(b.Maps, v)
			for _, observation := range []struct {
				path  string
				isMap bool
			}{{r.Stdout, false}, {c.LinkMap, true}} {
				raw, e := os.ReadFile(observation.path)
				if e != nil {
					return b, e
				}
				if e = verifyLinkInputs(string(raw), p.Out, p.Inputs, b.Objects, observation.isMap); e != nil {
					return b, e
				}
			}
		}
	}
	for _, path := range p.Outputs {
		v, e := pin(path)
		if e != nil {
			return b, e
		}
		b.Outputs = append(b.Outputs, v)
	}
	runtime, e := auditControlRuntime(p, b)
	if e != nil {
		return b, e
	}
	runtimePath := filepath.Join(p.Out, "runtime.json")
	if err = writeJSON(runtimePath, runtime); err != nil {
		return b, err
	}
	b.Runtime, err = pin(runtimePath)
	if err != nil {
		return b, err
	}
	if err = verifyInputs(); err != nil {
		return b, err
	}
	b.Status = "built"
	return b, nil
}

func auditControlRuntime(p auditControlPlan, b auditControlBuild) (RuntimePlan, error) {
	proof, err := readAuditorProof(p.ProducerProof)
	if err != nil {
		return RuntimePlan{}, err
	}
	seeds := append(append([]Pin{}, b.Outputs...), p.Audit)
	nodes, inputs, err := elfClosure(seeds, proof.SearchDirs)
	if err != nil {
		return RuntimePlan{}, err
	}
	r := RuntimePlan{Schema: "nanopore-supgpu-audit-control-runtime/v1", Build: p.AcceptedBuild, Controller: p.Controller, Consumer: p.Controller, Family: "cpu-only-real-loader-controls/v1", Seeds: seeds, Nodes: nodes, Inputs: append(inputs, proof.Cache), LoaderCache: proof.Cache, SearchDirs: proof.SearchDirs}
	r.Policy = RuntimePolicy{Schema: "nanopore-supgpu-selected-runtime/v1", Lazy: "audit-every-loader-object-before-initializers/v1", Reference: b.Outputs[0], Observer: b.Outputs[0], Audit: p.Audit, Allowed: []RuntimeCode{}, Alternates: []RuntimeCode{}, Directories: []string{filepath.Dir(b.Outputs[1].Path)}}
	expectedSonames := []string{"libsupgpu_control_provider.so.1", "libsupgpu_control_provider.so.1", "libsupgpu_control_unknown.so.1"}
	for i, v := range b.Outputs[1:] {
		if err = verifyConstructorFixture(v, expectedSonames[i]); err != nil {
			return r, err
		}
	}
	r.Policy.Allowed = append(r.Policy.Allowed, RuntimeCode{b.Outputs[1], expectedSonames[0], []string{}, "control-qualified-provider", false})
	r.Policy.Alternates = append(r.Policy.Alternates, RuntimeCode{b.Outputs[2], expectedSonames[1], []string{}, "control-known-wrong-tier", false})
	cpu := auditCPUName
	for _, n := range nodes {
		control := n.Object == p.Audit
		for _, v := range b.Outputs {
			control = control || n.Object == v
		}
		if control {
			continue
		}
		found := false
		for _, c := range proof.CPU {
			if c.Object == n.Object && cpu[c.SONAME] {
				c.Startup = true
				r.Policy.Allowed = append(r.Policy.Allowed, c)
				found = true
				dir := filepath.Dir(c.Object.Path)
				if !containsString(r.Policy.Directories, dir) {
					r.Policy.Directories = append(r.Policy.Directories, dir)
				}
				break
			}
		}
		if !found {
			return r, fmt.Errorf("control runtime is outside selected CPU providers: %s", n.Object.Path)
		}
	}
	return r, nil
}

func verifyConstructorFixture(p Pin, soname string) error {
	if err := verifyPin("", p); err != nil {
		return err
	}
	f, err := elf.Open(p.Path)
	if err != nil {
		return err
	}
	defer f.Close()
	names, err := f.DynString(elf.DT_SONAME)
	if err != nil || !reflect.DeepEqual(names, []string{soname}) {
		return fmt.Errorf("constructor fixture SONAME differs")
	}
	needed, err := f.ImportedLibraries()
	if err != nil {
		return err
	}
	if len(needed) != 0 {
		return fmt.Errorf("constructor fixture unexpectedly imports libraries")
	}
	symbols, err := f.DynamicSymbols()
	if err != nil {
		return err
	}
	for _, s := range symbols {
		if s.Name != "" && s.Section == elf.SHN_UNDEF {
			return fmt.Errorf("constructor fixture has unresolved import %s", s.Name)
		}
	}
	init := f.Section(".init_array")
	if init == nil || init.Size != 8 {
		return fmt.Errorf("fixture must have exactly one init-array constructor")
	}
	return nil
}

func readAuditControlBuild(path, sha string) (b auditControlBuild, p auditControlPlan, r RuntimePlan, err error) {
	return readAuditControlBuildForController(path, sha, true)
}

func readAuditControlBuildForController(path, sha string, current bool) (b auditControlBuild, p auditControlPlan, r RuntimePlan, err error) {
	if err = readPinnedJSON(path, sha, &b); err != nil {
		return
	}
	var compiler BuildPlan
	p, compiler, err = readAuditControlPlanForController(b.Plan.Path, b.Plan.SHA256, current)
	if err != nil {
		return
	}
	_ = compiler
	if b.Schema != "nanopore-supgpu-audit-control-build/v1" || b.Status != "built" || len(b.Preflight) != len(p.Queries)+4 || len(b.Commands) != 8 || len(b.Objects) != 4 || len(b.Maps) != 4 || len(b.Depfiles) != 4 || len(b.PreDeps) != 4 || len(b.Dependencies) != 4 || len(b.Outputs) != 4 {
		err = fmt.Errorf("incomplete control build")
		return
	}
	all := append(append(append(append([]Pin{}, b.Objects...), b.Maps...), b.Outputs...), b.Runtime)
	all = append(all, b.Depfiles...)
	all = append(all, b.PreDeps...)
	all = append(all, b.BeforeInputs)
	for _, v := range all {
		if err = verifyPin("", v); err != nil {
			return
		}
	}
	var before []Pin
	if err = readPinnedJSON(b.BeforeInputs.Path, b.BeforeInputs.SHA256, &before); err != nil {
		return
	}
	if !reflect.DeepEqual(before, p.Inputs) {
		err = fmt.Errorf("control before-compile input freeze differs")
		return
	}
	for i, cmd := range b.Preflight {
		var args []string
		if i < len(p.Queries) {
			args = p.Queries[i].Argv
		} else {
			args = p.Preprocess[i-len(p.Queries)].Argv
		}
		if !reflect.DeepEqual(cmd.Args, args) {
			err = fmt.Errorf("control preflight command differs")
			return
		}
		if err = verifyCommand(cmd); err != nil {
			return
		}
		if !reflect.DeepEqual(cmd.Environment, append([]string{"SUPGPU_COMMAND_CWD=" + p.Out}, nativeEnvironment()...)) {
			err = fmt.Errorf("control native environment differs")
			return
		}
		if i < len(p.Queries) {
			var raw []byte
			raw, err = os.ReadFile(cmd.Stdout)
			if err != nil {
				return
			}
			if err = sameQueryResult(string(raw), p.Queries[i], p.Out); err != nil {
				return
			}
		}
	}
	for i, cmd := range b.Commands {
		c := p.Commands[i]
		if !reflect.DeepEqual(cmd.Args, c.Argv) || !reflect.DeepEqual(cmd.Environment, append([]string{"SUPGPU_COMMAND_CWD=" + p.Out}, nativeEnvironment()...)) {
			err = fmt.Errorf("control compile/link command differs")
			return
		}
		if err = verifyCommand(cmd); err != nil {
			return
		}
		if i < 4 {
			if b.Objects[i].Path != c.Argv[len(c.Argv)-1] || b.Depfiles[i].Path != c.Depfile || b.PreDeps[i].Path != p.Preprocess[i].Depfile {
				err = fmt.Errorf("control object/dep binding differs")
				return
			}
			var pre, actual []string
			pre, err = sortedDependencies(b.PreDeps[i].Path, p.Out)
			if err != nil {
				return
			}
			actual, err = sortedDependencies(b.Depfiles[i].Path, p.Out)
			if err != nil {
				return
			}
			if !reflect.DeepEqual(pre, actual) || !reflect.DeepEqual(pre, b.Dependencies[i]) {
				err = fmt.Errorf("control dependency evidence differs")
				return
			}
		}
		if i >= 4 {
			if b.Maps[i-4].Path != c.LinkMap {
				err = fmt.Errorf("control map binding differs")
				return
			}
			for _, o := range []struct {
				path  string
				isMap bool
			}{{cmd.Stdout, false}, {c.LinkMap, true}} {
				var raw []byte
				raw, err = os.ReadFile(o.path)
				if err != nil {
					return
				}
				if err = verifyLinkInputs(string(raw), p.Out, p.Inputs, b.Objects, o.isMap); err != nil {
					return
				}
			}
		}
	}
	for i, v := range b.Outputs {
		if v.Path != p.Outputs[i] {
			err = fmt.Errorf("control output binding differs")
			return
		}
	}
	if err = readPinnedJSON(b.Runtime.Path, b.Runtime.SHA256, &r); err != nil {
		return
	}
	expected, e := auditControlRuntime(p, b)
	if e != nil {
		err = e
		return
	}
	if !reflect.DeepEqual(r, expected) {
		err = fmt.Errorf("prospective control runtime changed")
		return
	}
	return
}

func runAuditControls(path, sha, out string) (run auditControlRun, err error) {
	run = auditControlRun{Schema: "nanopore-supgpu-audit-control-run/v1", Status: "failed", Cases: []auditControlCase{}}
	b, p, r, err := readAuditControlBuild(path, sha)
	if err != nil {
		return run, err
	}
	run.Build, err = pin(path)
	if err != nil {
		return run, err
	}
	out, err = filepath.Abs(out)
	if err != nil {
		return run, err
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return run, err
	}
	defer lock.Close()
	if err = os.Mkdir(out, 0755); err != nil {
		return run, err
	}
	defer func() {
		e := writeJSON(filepath.Join(out, "run.json"), run)
		if err == nil {
			err = e
		}
	}()
	parent, e := readAuditController(p.Controller, true)
	if e != nil {
		return run, e
	}
	for _, role := range []string{"reference", "observer"} {
		for i, kind := range []string{"allowed", "wrong-known-tier", "unknown"} {
			for _, v := range append(append([]Pin{}, r.Inputs...), p.Audit, p.Controller, parent.Binary, b.Runtime) {
				if err = verifyPin("", v); err != nil {
					return run, err
				}
			}
			dir := filepath.Join(out, role+"-"+kind)
			if err = os.Mkdir(dir, 0755); err != nil {
				return run, err
			}
			c, caseError := executeAuditControlCase(dir, role, kind, i, b, r)
			run.Cases = append(run.Cases, c)
			if err = writeAuditControlProgress(out, run); err != nil {
				return run, err
			}
			if caseError != nil {
				return run, caseError
			}
		}
	}
	run.Status = "passed"
	return run, nil
}

func TestAuditControlNativeAdmissionPrecedesInputs(t *testing.T) {
	for _, unit := range []string{"compile", "run"} {
		err := auditControlCommand([]string{unit, "-plan", "/must/not/open", "-build", "/must/not/open", "-out", "/must/not/create"})
		if err == nil || !strings.Contains(err.Error(), "requires separate coordinator admission") {
			t.Fatalf("native control touched inputs without admission: %v", err)
		}
	}
}
func TestAuditControlFiniteRecipes(t *testing.T) {
	dir := t.TempDir()
	sources := []Pin{{Path: filepath.Join(dir, "loader.c")}, {Path: filepath.Join(dir, "constructor.c")}}
	pre, commands, outputs := auditControlCommands(dir, "/unexecuted/compiler", sources, "/frozen/libdl.a")
	if len(pre) != 4 || len(commands) != 8 || len(outputs) != 4 {
		t.Fatal("finite CPU control shape differs")
	}
	for i := 0; i < 4; i++ {
		if commands[i].Depfile == "" || pre[i].Depfile == "" || commands[i].Argv[len(commands[i].Argv)-1] != filepath.Join(dir, fmt.Sprintf("control-%d.o", i)) {
			t.Fatal("control pre/actual dependency binding differs")
		}
	}
	for _, c := range commands {
		for _, arg := range c.Argv {
			if strings.Contains(arg, "cuda") || strings.Contains(arg, "torch") || strings.Contains(arg, "TxModel") {
				t.Fatal("GPU/model input entered CPU loader control")
			}
		}
	}
}

func executeAuditControlCase(dir, role, kind string, i int, b auditControlBuild, r RuntimePlan) (c auditControlCase, err error) {
	constructor := filepath.Join(dir, "constructor.marker")
	returned := filepath.Join(dir, "returned.marker")
	c = auditControlCase{Role: role, Kind: kind}
	c.Command, err = executeAttested(dir, dir, role, "attest-only", []string{b.Outputs[0].Path, b.Outputs[i+1].Path, constructor, returned}, r, b.Runtime, readCodeMaps)
	actualError := err
	if e := writeJSON(filepath.Join(dir, "command.json"), c.Command); e != nil {
		return c, e
	}
	c.Guard, err = pin(filepath.Join(dir, role+"-guard.json"))
	if err != nil {
		return c, err
	}
	var g GuardReceipt
	if err = readPinnedJSON(c.Guard.Path, c.Guard.SHA256, &g); err != nil {
		return c, err
	}
	if i == 0 {
		if actualError != nil {
			return c, actualError
		}
		c.Constructor, err = pin(constructor)
		if err != nil {
			return c, err
		}
		c.Returned, err = pin(returned)
		if err != nil {
			return c, err
		}
		raw, e := os.ReadFile(constructor)
		if e != nil || string(raw) != "allowed constructor\n" {
			return c, fmt.Errorf("allowed real constructor marker differs")
		}
		if _, err = verifyGuard(g, r, b.Runtime, role, "attest-only"); err != nil {
			return c, err
		}
	} else {
		c.Rejection = ""
		if actualError != nil {
			c.Rejection = actualError.Error()
		}
		_, e := os.Stat(constructor)
		c.ConstructorAbsent = os.IsNotExist(e)
		_, e = os.Stat(returned)
		c.ReturnedAbsent = os.IsNotExist(e)
		expected := "known wrong runtime tier"
		if i == 2 {
			expected = "unknown mapped code"
		}
		if actualError == nil || !strings.Contains(actualError.Error(), expected) || !c.ConstructorAbsent || !c.ReturnedAbsent || c.Command.Exit != 124 || !g.Group.Empty {
			return c, fmt.Errorf("real %s constructor veto failed: error=%v ctorAbsent=%v returnedAbsent=%v exit=%d", kind, actualError, c.ConstructorAbsent, c.ReturnedAbsent, c.Command.Exit)
		}
		rejected := false
		for _, d := range g.Decisions {
			rejected = rejected || (!d.Approved && d.Request.Kind == "open" && d.Request.Path == b.Outputs[i+1].Path && d.Maps.Path != "")
		}
		if !rejected {
			return c, fmt.Errorf("no saved rejection at real object-open callback")
		}
		if _, e = verifyGuard(g, r, b.Runtime, role, "attest-only"); e == nil {
			return c, fmt.Errorf("denied real loader transcript qualified import")
		}
	}
	return c, nil
}

func TestAuditControlFreezesAbsoluteGeneratedTestMain(t *testing.T) {
	dir := t.TempDir()
	tools := filepath.Join(dir, "tools")
	if err := os.Mkdir(tools, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "pkg", "include"), 0700); err != nil {
		t.Fatal(err)
	}
	generated := filepath.Join(dir, "generated-test-main.go")
	deps, err := Marshal(map[string]any{"Dir": filepath.Join(dir, "verification/supgpu"), "Root": dir, "ImportPath": "github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu", "Module": map[string]any{"Path": "github.com/Jakeelamb/nanopore_basecaller_go", "Main": true, "Dir": dir, "GoMod": filepath.Join(dir, "go.mod")}, "GoFiles": []string{generated}})
	if err != nil {
		t.Fatal(err)
	}
	env, err := Marshal(goEnvironment{GOROOT: dir, GOTOOLDIR: tools, GOOS: "linux", GOARCH: "amd64", GOAMD64: "v1", CGO_ENABLED: "0", GOWORK: "off"})
	if err != nil {
		t.Fatal(err)
	}
	paths, err := auditTestInputs(deps, env, "/unexecuted/go")
	if err != nil {
		t.Fatal(err)
	}
	if !containsString(paths, generated) {
		t.Fatal("actual generated absolute test main was omitted/misjoined")
	}
}

func TestAuditControlCPUProvidersExcludeNNImages(t *testing.T) {
	cpu := RuntimeCode{Object: Pin{Path: "/selected/libc.so.6"}, SONAME: "libc.so.6"}
	nn := RuntimeCode{Object: Pin{Path: "/unconsumed/libtorch_cuda.so"}, SONAME: "libtorch_cuda.so"}
	rejected := RuntimeCode{Object: Pin{Path: "/rejected/libc.so.6"}, SONAME: "libc.so.6"}
	selected, dirs := auditCPUProviders(RuntimePlan{Policy: RuntimePolicy{Allowed: []RuntimeCode{cpu, nn}, Alternates: []RuntimeCode{rejected}}})
	if !reflect.DeepEqual(selected, []RuntimeCode{cpu}) || !reflect.DeepEqual(dirs, []string{"/selected"}) {
		t.Fatalf("unconsumed NN/rejected runtime entered CPU controls: %+v %v", selected, dirs)
	}
}
