package decodergpu

import (
	"context"
	"debug/elf"
	"errors"
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type CompilerPhaseOptions struct {
	Plan, Prior File
	Phase, Out  string
	Admitted    bool
}
type CompilerPhase struct {
	Schema, Phase, Status, Error, Started, Finished, Directory string
	Plan, Prior                                                File
	Inputs, Objects                                            map[string]string
	Commands                                                   []runner.CommandReceipt
	Dependencies                                               []string
	Depfile, BeforeCompile, Output, LinkMap                    File
}

func nativeEnvironment() []string {
	return []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "TZ=UTC", "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2", "CUDA_VISIBLE_DEVICES=0"}
}

// No command logs or child exist until this selected-tool/plugin boundary
// passes. It is checked again on both success and failure after the child.
func verifyCompilerBoundary(p ClosedBuildPlan) error {
	c, err := compilerClosure(p.Recipe.Compiler)
	if err != nil {
		return err
	}
	if !compilerClosuresEqual(c, p.Compiler) {
		return fmt.Errorf("selected compiler runtime changed before native command")
	}
	for path, want := range c.Inputs {
		if p.Inputs[path] != want {
			return fmt.Errorf("compiler ELF/plugin/cache input omitted: %s", path)
		}
	}
	return nil
}
func checkedCompilerCommand(check func() error, run func() (runner.CommandReceipt, error)) (runner.CommandReceipt, error) {
	if err := check(); err != nil {
		return runner.CommandReceipt{}, err
	}
	r, err := run()
	return r, errors.Join(err, check())
}
func executeNative(cwd, out, label string, args, environment []string) (r runner.CommandReceipt, err error) {
	if err = requireNativeLimits(); err != nil {
		return r, err
	}
	r = runner.CommandReceipt{Args: slices.Clone(args), Environment: append([]string{"D01_COMMAND_CWD=" + cwd}, environment...), Started: time.Now().UTC().Format(time.RFC3339Nano), Exit: -1, Stdout: filepath.Join(out, label+".stdout"), Stderr: filepath.Join(out, label+".stderr")}
	stdout, err := os.OpenFile(r.Stdout, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return r, err
	}
	stderr, err := os.OpenFile(r.Stderr, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		stdout.Close()
		return r, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	cmd.Dir = cwd
	cmd.Env = slices.Clone(environment)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
	cmd.WaitDelay = 2 * time.Second
	err = cmd.Run()
	r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
	if cmd.ProcessState != nil {
		r.Exit = cmd.ProcessState.ExitCode()
	}
	err = errors.Join(err, ctx.Err(), stdout.Close(), stderr.Close())
	r.StdoutSHA, _ = runner.SHA(r.Stdout)
	r.StderrSHA, _ = runner.SHA(r.Stderr)
	return r, err
}

func checkNativeLimits(cpu, memory, swap string) error {
	parts := strings.Fields(cpu)
	if len(parts) != 2 || parts[0] == "max" {
		return fmt.Errorf("native CPUQuota must be bounded at two cores")
	}
	quota, e := strconv.ParseUint(parts[0], 10, 64)
	if e != nil {
		return e
	}
	period, e := strconv.ParseUint(parts[1], 10, 64)
	if e != nil || period == 0 || quota == 0 || float64(quota)/float64(period) > 2 {
		return fmt.Errorf("native CPUQuota exceeds two cores")
	}
	limit, e := strconv.ParseUint(strings.TrimSpace(memory), 10, 64)
	if e != nil || limit == 0 || limit > 3<<30 {
		return fmt.Errorf("native MemoryMax must be at most 3GiB")
	}
	if strings.TrimSpace(swap) != "0" {
		return fmt.Errorf("native MemorySwapMax must be zero")
	}
	return nil
}
func requireNativeLimits() error {
	raw, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		return err
	}
	relative := ""
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.HasPrefix(line, "0::/") {
			relative = strings.TrimPrefix(line, "0::/")
		}
	}
	if relative == "" || strings.Contains(relative, "..") {
		return fmt.Errorf("bounded cgroup v2 required before native child")
	}
	dir := filepath.Join("/sys/fs/cgroup", relative)
	values := []string{}
	for _, name := range []string{"cpu.max", "memory.max", "memory.swap.max"} {
		b, e := os.ReadFile(filepath.Join(dir, name))
		if e != nil {
			return e
		}
		values = append(values, string(b))
	}
	return checkNativeLimits(values[0], values[1], values[2])
}

func compilerQueries(p ClosedBuildPlan) []ResourceQuery {
	q := []ResourceQuery{}
	for _, t := range p.Compiler.Tools {
		if t.Name != "driver" && t.Name != "gcc-driver" {
			q = append(q, ResourceQuery{"-print-prog-name=" + t.Name, t.Executable})
		}
	}
	q = append(q, ResourceQuery{"-print-file-name=liblto_plugin.so", p.Compiler.Plugins[0]})
	return append(q, p.Recipe.Resources...)
}
func verifyQuery(raw string, q ResourceQuery, cwd string) error {
	name := strings.TrimSpace(raw)
	if name == "" || strings.ContainsAny(name, "\r\n") {
		return fmt.Errorf("compiler resource response is not a single path")
	}
	if !filepath.IsAbs(name) {
		if strings.ContainsRune(name, '/') {
			name = filepath.Join(cwd, name)
		} else {
			var err error
			if strings.HasPrefix(q.Flag, "-print-prog-name=") {
				p, e := resolveFile(name, []string{"/usr/bin", "/bin"})
				name, err = p.Path, e
			} else {
				err = fmt.Errorf("compiler did not resolve resource %s", name)
			}
			if err != nil {
				return err
			}
		}
	}
	got, err := artifactPin(name)
	if err != nil {
		return err
	}
	if got != q.Expected {
		return fmt.Errorf("compiler selected unexpected resource: %s", name)
	}
	return nil
}
func dependencySet(path, cwd string) ([]string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return h07.ParseDependencies(string(raw), cwd)
}
func verifyDependencies(before, after []string) error {
	if !slices.Equal(before, after) {
		return fmt.Errorf("actual -MD dependencies differ from frozen -M set")
	}
	return nil
}

func readCompilerPhase(anchor File, plan File, want string) (r CompilerPhase, err error) {
	return readCompilerPhaseMode(anchor, plan, want, nil)
}

func readCompilerPhaseMode(anchor File, plan File, want string, producer *ClosedBuildPlan) (r CompilerPhase, err error) {
	if err = readJSON(anchor, &r); err != nil {
		return r, err
	}
	if r.Schema != "nanopore-decoder-compiler-phase/v1" || r.Phase != want || r.Status != "passed" || r.Error != "" || r.Plan != plan || len(r.Commands) == 0 {
		return r, fmt.Errorf("wrong or unsuccessful prior compiler phase")
	}
	for _, c := range r.Commands {
		if err = commandEvidence(c); err != nil {
			return r, err
		}
	}
	if producer == nil {
		err = checkPins(r.Inputs)
	} else {
		err = checkProducerInputs(r.Inputs, *producer, false)
	}
	if err != nil {
		return r, err
	}
	if err = checkPins(r.Objects); err != nil {
		return r, err
	}
	if want == "headers" {
		if r.BeforeCompile.Path == "" || r.Depfile.Path == "" || len(r.Dependencies) == 0 {
			return r, fmt.Errorf("precompile header freeze absent")
		}
		if err = r.BeforeCompile.verify(); err != nil {
			return r, err
		}
		if err = r.Depfile.verify(); err != nil {
			return r, err
		}
		var frozen map[string]string
		if err = readJSON(r.BeforeCompile, &frozen); err != nil {
			return r, err
		}
		if !maps.Equal(frozen, r.Inputs) {
			return r, fmt.Errorf("before-compile input manifest differs")
		}
	}
	return r, nil
}

// RunCompilerPhase is the maintained finite controller: query, then header
// freeze, then compile/link. Each release names exact prior immutable receipts.
// A compiler-only release can never invoke the native stage executable.
func RunCompilerPhase(o CompilerPhaseOptions) (r CompilerPhase, err error) {
	if !o.Admitted {
		return r, fmt.Errorf("D01 %s requires separate coordinator admission", o.Phase)
	}
	if o.Phase != "query" && o.Phase != "headers" && o.Phase != "build" {
		return r, fmt.Errorf("unknown D01 compiler phase")
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return r, err
	}
	defer lock.Close()
	o.Plan, err = normalizeAnchor(o.Plan)
	if err != nil {
		return r, err
	}
	if o.Phase != "query" {
		o.Prior, err = normalizeAnchor(o.Prior)
		if err != nil {
			return r, err
		}
	}
	var p ClosedBuildPlan
	if err = readJSON(o.Plan, &p); err != nil {
		return r, err
	}
	if err = verifyClosedPlan(p); err != nil {
		return r, err
	}
	var prior CompilerPhase
	if o.Phase != "query" {
		want := "query"
		if o.Phase == "build" {
			want = "headers"
		}
		prior, err = readCompilerPhase(o.Prior, o.Plan, want)
		if err != nil {
			return r, err
		}
	}
	out, err := runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	r = CompilerPhase{Schema: "nanopore-decoder-compiler-phase/v1", Phase: o.Phase, Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Directory: out, Plan: o.Plan, Prior: o.Prior, Inputs: maps.Clone(p.Inputs), Objects: map[string]string{}}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
		}
		save := writeJSON(filepath.Join(out, "phase.json"), r)
		if err == nil {
			err = save
		}
	}()
	execute := func(label string, args []string) (runner.CommandReceipt, error) {
		return checkedCompilerCommand(func() error { return verifyCompilerBoundary(p) }, func() (runner.CommandReceipt, error) {
			c, e := executeNative(p.BuildDir, out, label, args, nativeEnvironment())
			r.Commands = append(r.Commands, c)
			return c, e
		})
	}
	if o.Phase == "query" {
		for i, q := range compilerQueries(p) {
			c, e := execute(fmt.Sprintf("query-%02d", i), []string{p.Recipe.Compiler.Path, q.Flag})
			if e != nil {
				return r, e
			}
			raw, e := os.ReadFile(c.Stdout)
			if e != nil {
				return r, e
			}
			if e = verifyQuery(string(raw), q, p.BuildDir); e != nil {
				return r, e
			}
		}
	} else if o.Phase == "headers" {
		if len(prior.Commands) != len(compilerQueries(p)) {
			return r, fmt.Errorf("tool/resource query phase incomplete")
		}
		for i, q := range compilerQueries(p) {
			if !slices.Equal(prior.Commands[i].Args, []string{p.Recipe.Compiler.Path, q.Flag}) {
				return r, fmt.Errorf("prior tool query argv changed")
			}
			body, e := os.ReadFile(prior.Commands[i].Stdout)
			if e != nil {
				return r, e
			}
			if e = verifyQuery(string(body), q, p.BuildDir); e != nil {
				return r, e
			}
		}
		dep := filepath.Join(out, "prospective.d")
		args := append([]string{p.Recipe.Compiler.Path}, p.Recipe.Flags...)
		args = append(args, "-M", "-MT", "decoder-oracle", "-MF", dep, p.Source)
		if _, err = execute("headers", args); err != nil {
			return r, err
		}
		r.Dependencies, err = dependencySet(dep, p.BuildDir)
		if err != nil {
			return r, err
		}
		if !slices.Contains(r.Dependencies, p.Source) {
			return r, fmt.Errorf("preprocessor omitted the immutable native TU")
		}
		for _, path := range r.Dependencies {
			if err = filePin(r.Inputs, path); err != nil {
				return r, err
			}
		}
		r.Depfile, err = artifactPin(dep)
		if err != nil {
			return r, err
		}
		before := filepath.Join(out, "before-compile-inputs.json")
		if err = writeJSON(before, r.Inputs); err != nil {
			return r, err
		}
		r.BeforeCompile, err = artifactPin(before)
		if err != nil {
			return r, err
		}
	} else {
		if len(prior.Commands) != 1 {
			return r, fmt.Errorf("header preflight command count differs")
		}
		expected := append([]string{p.Recipe.Compiler.Path}, p.Recipe.Flags...)
		expected = append(expected, "-M", "-MT", "decoder-oracle", "-MF", prior.Depfile.Path, p.Source)
		if !slices.Equal(expected, prior.Commands[0].Args) {
			return r, fmt.Errorf("header preflight argv changed")
		}
		actualPre, e := dependencySet(prior.Depfile.Path, p.BuildDir)
		if e != nil {
			return r, e
		}
		if e = verifyDependencies(prior.Dependencies, actualPre); e != nil {
			return r, e
		}
		for path, want := range p.Inputs {
			if prior.Inputs[path] != want {
				return r, fmt.Errorf("header freeze omitted build-plan input: %s", path)
			}
		}
		for _, path := range prior.Dependencies {
			if _, ok := prior.Inputs[path]; !ok {
				return r, fmt.Errorf("header freeze omitted dependency: %s", path)
			}
		}
		r.Inputs = maps.Clone(prior.Inputs)
		r.Dependencies = slices.Clone(prior.Dependencies)
		r.BeforeCompile = prior.BeforeCompile
		if err = checkPins(r.Inputs); err != nil {
			return r, err
		}
		dep := filepath.Join(out, "actual.d")
		object := filepath.Join(out, "oracle.o")
		args := append([]string{p.Recipe.Compiler.Path}, p.Recipe.Flags...)
		args = append(args, "-MD", "-MF", dep, "-c", p.Source, "-o", object)
		if _, err = execute("compile", args); err != nil {
			return r, err
		}
		actual, e := dependencySet(dep, p.BuildDir)
		if e != nil {
			return r, e
		}
		if err = verifyDependencies(r.Dependencies, actual); err != nil {
			return r, err
		}
		if err = checkPins(r.Inputs); err != nil {
			return r, err
		}
		f, e := elf.Open(object)
		if e != nil {
			return r, e
		}
		valid := f.Type == elf.ET_REL && f.Class == elf.ELFCLASS64 && f.Machine == elf.EM_X86_64
		f.Close()
		if !valid {
			return r, fmt.Errorf("compiled object is outside x86_64 relocatable domain")
		}
		if err = filePin(r.Objects, object); err != nil {
			return r, err
		}
		r.Depfile, err = artifactPin(dep)
		if err != nil {
			return r, err
		}
		binary := filepath.Join(out, "decoder-oracle")
		linkMap := filepath.Join(out, "link.map")
		link := append([]string{p.Recipe.Compiler.Path, object}, p.Recipe.Libraries...)
		link = append(link, "-Wl,-t", "-Wl,-Map="+linkMap, "-o", binary)
		c, e := execute("link", link)
		if e != nil {
			return r, e
		}
		trace, e := os.ReadFile(c.Stdout)
		if e != nil {
			return r, e
		}
		if err = h07.VerifyLinkTrace(string(trace), p.BuildDir, r.Inputs, r.Objects); err != nil {
			return r, err
		}
		body, e := os.ReadFile(linkMap)
		if e != nil {
			return r, e
		}
		if err = verifyDecoderLinkMap(string(body), p.BuildDir, r.Inputs, r.Objects, p.Recipe.Koi.Path); err != nil {
			return r, err
		}
		r.LinkMap, err = artifactPin(linkMap)
		if err != nil {
			return r, err
		}
		r.Output, err = artifactPin(binary)
		if err != nil {
			return r, err
		}
	}
	if err = checkPins(r.Inputs); err != nil {
		return r, err
	}
	if err = checkPins(r.Objects); err != nil {
		return r, err
	}
	if err = verifyClosedPlan(p); err != nil {
		return r, err
	}
	r.Status = "passed"
	return r, nil
}

func verifyDecoderLinkMap(body, cwd string, inputs, objects map[string]string, koi string) error {
	loads := []string{}
	decoder := false
	for _, line := range strings.Split(body, "\n") {
		fields := strings.Fields(line)
		if len(fields) == 2 && fields[0] == "LOAD" {
			loads = append(loads, fields[1])
		}
		if strings.Contains(line, koi+"(decoder_lib.cu.o)") {
			decoder = true
		}
	}
	if err := h07.VerifyLinkTrace(strings.Join(loads, "\n"), cwd, inputs, objects); err != nil {
		return err
	}
	if !decoder {
		return fmt.Errorf("link map does not identify original Koi decoder archive member")
	}
	return nil
}

func readBuiltPhase(anchor File) (b CompilerPhase, p ClosedBuildPlan, err error) {
	return readBuiltPhaseMode(anchor, true)
}

func readBuiltPhaseMode(anchor File, current bool) (b CompilerPhase, p ClosedBuildPlan, err error) {
	return readBuiltPhaseScoped(anchor, current, nil)
}

func readBuiltPhaseScoped(anchor File, current bool, proofs *numericProofs) (b CompilerPhase, p ClosedBuildPlan, err error) {
	if err = readJSON(anchor, &b); err != nil {
		return b, p, err
	}
	if err = readJSON(b.Plan, &p); err != nil {
		return b, p, err
	}
	if err = verifyClosedPlanModeScoped(p, current, proofs); err != nil {
		return b, p, err
	}
	var historical *ClosedBuildPlan
	if !current {
		historical = &p
	}
	b, err = readCompilerPhaseMode(anchor, b.Plan, "build", historical)
	if err != nil {
		return b, p, err
	}
	header, err := readCompilerPhaseMode(b.Prior, b.Plan, "headers", historical)
	if err != nil {
		return b, p, err
	}
	if !maps.Equal(b.Inputs, header.Inputs) || b.BeforeCompile != header.BeforeCompile || !slices.Equal(b.Dependencies, header.Dependencies) || len(b.Commands) != 2 || len(b.Objects) != 1 {
		return b, p, fmt.Errorf("closed build differs from frozen header/command contract")
	}
	object := filepath.Join(b.Directory, "oracle.o")
	binary := filepath.Join(b.Directory, "decoder-oracle")
	dep := filepath.Join(b.Directory, "actual.d")
	linkMap := filepath.Join(b.Directory, "link.map")
	compile := append([]string{p.Recipe.Compiler.Path}, p.Recipe.Flags...)
	compile = append(compile, "-MD", "-MF", dep, "-c", p.Source, "-o", object)
	link := append([]string{p.Recipe.Compiler.Path, object}, p.Recipe.Libraries...)
	link = append(link, "-Wl,-t", "-Wl,-Map="+linkMap, "-o", binary)
	if !slices.Equal(compile, b.Commands[0].Args) || !slices.Equal(link, b.Commands[1].Args) || b.Output.Path != binary || b.Depfile.Path != dep || b.LinkMap.Path != linkMap || b.Objects[object] == "" {
		return b, p, fmt.Errorf("native build argv/output roles changed")
	}
	for _, f := range []File{b.Output, b.Depfile, b.LinkMap} {
		if err = f.verify(); err != nil {
			return b, p, err
		}
	}
	actual, err := dependencySet(b.Depfile.Path, p.BuildDir)
	if err != nil {
		return b, p, err
	}
	if err = verifyDependencies(header.Dependencies, actual); err != nil {
		return b, p, err
	}
	trace, err := os.ReadFile(b.Commands[1].Stdout)
	if err != nil {
		return b, p, err
	}
	if err = h07.VerifyLinkTrace(string(trace), p.BuildDir, b.Inputs, b.Objects); err != nil {
		return b, p, err
	}
	body, err := os.ReadFile(b.LinkMap.Path)
	if err != nil {
		return b, p, err
	}
	if err = verifyDecoderLinkMap(string(body), p.BuildDir, b.Inputs, b.Objects, p.Recipe.Koi.Path); err != nil {
		return b, p, err
	}
	return b, p, nil
}
