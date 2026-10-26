package decodergpu

import (
	"debug/elf"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func lineageCompiler(p LineagePlan) ClosedBuildPlan {
	return ClosedBuildPlan{Recipe: NativeRecipe{Compiler: p.Recipe.Compiler, Resources: p.Recipe.Resources}, Compiler: p.Compiler, Inputs: p.Inputs}
}
func lineageLinkInputs(p LineagePlan) map[string]string {
	inputs := maps.Clone(p.Recipe.Inputs)
	maps.Copy(inputs, p.Compiler.Inputs)
	return inputs
}
func lineageLinkMap(body, cwd string, inputs, objects map[string]string) error {
	loads := []string{}
	for _, line := range strings.Split(body, "\n") {
		fields := strings.Fields(line)
		if len(fields) == 2 && fields[0] == "LOAD" {
			name := fields[1]
			for _, bad := range []string{"libkoi", "libcudart", "libtorch", "libc10", "libcudnn"} {
				if strings.Contains(filepath.Base(name), bad) {
					return fmt.Errorf("non-lineage library in link map: %s", name)
				}
			}
			loads = append(loads, name)
		}
	}
	if len(loads) == 0 {
		return fmt.Errorf("lineage link map has no LOAD records")
	}
	return h07.VerifyLinkTrace(strings.Join(loads, "\n"), cwd, inputs, objects)
}
func readLineagePrior(anchor, plan File, phase string) (r CompilerPhase, err error) {
	if err = readJSON(anchor, &r); err != nil {
		return r, err
	}
	if r.Schema != "nanopore-decoder-lineage-compiler/v1" || r.Plan != plan || r.Phase != phase || r.Status != "passed" || r.Error != "" || len(r.Commands) == 0 {
		return r, fmt.Errorf("lineage prior phase identity/status differs")
	}
	for _, c := range r.Commands {
		if err = commandEvidence(c); err != nil {
			return r, err
		}
	}
	if err = checkPins(r.Inputs); err != nil {
		return r, err
	}
	if err = checkPins(r.Objects); err != nil {
		return r, err
	}
	return r, nil
}

func RunLineageCompiler(o CompilerPhaseOptions) (r CompilerPhase, err error) {
	if !o.Admitted || (o.Phase != "query" && o.Phase != "headers" && o.Phase != "build") {
		return r, fmt.Errorf("lineage exact compiler phase requires separate admission")
	}
	if err = requireNativeLimits(); err != nil {
		return r, err
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
	var p LineagePlan
	if err = readJSON(o.Plan, &p); err != nil {
		return r, err
	}
	if err = verifyLineagePlan(p, true, true); err != nil {
		return r, err
	}
	var prior CompilerPhase
	if o.Phase != "query" {
		o.Prior, err = normalizeAnchor(o.Prior)
		if err != nil {
			return r, err
		}
		want := "query"
		if o.Phase == "build" {
			want = "headers"
		}
		prior, err = readLineagePrior(o.Prior, o.Plan, want)
		if err != nil {
			return r, err
		}
	}
	out, err := runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	r = CompilerPhase{Schema: "nanopore-decoder-lineage-compiler/v1", Phase: o.Phase, Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Directory: out, Plan: o.Plan, Prior: o.Prior, Inputs: maps.Clone(p.Inputs), Objects: map[string]string{}}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
		}
		if e := writeJSON(filepath.Join(out, "phase.json"), r); err == nil {
			err = e
		}
	}()
	component := lineageCompiler(p)
	execute := func(label string, args []string) (runner.CommandReceipt, error) {
		return checkedCompilerCommand(func() error { return verifyCompilerBoundary(component) }, func() (runner.CommandReceipt, error) {
			c, e := executeNative(p.WorkingDirectory, out, label, args, nativeEnvironment())
			r.Commands = append(r.Commands, c)
			return c, e
		})
	}
	queries := compilerQueries(component)
	if o.Phase == "query" {
		for i, q := range queries {
			c, e := execute(fmt.Sprintf("query-%02d", i), []string{p.Recipe.Compiler.Path, q.Flag})
			if e != nil {
				return r, e
			}
			body, e := os.ReadFile(c.Stdout)
			if e != nil {
				return r, e
			}
			if e = verifyQuery(string(body), q, p.WorkingDirectory); e != nil {
				return r, e
			}
		}
	} else if o.Phase == "headers" {
		if len(prior.Commands) != len(queries) {
			return r, fmt.Errorf("lineage compiler query count differs")
		}
		for i, q := range queries {
			if !slices.Equal(prior.Commands[i].Args, []string{p.Recipe.Compiler.Path, q.Flag}) {
				return r, fmt.Errorf("lineage prior query command differs")
			}
			body, e := os.ReadFile(prior.Commands[i].Stdout)
			if e != nil {
				return r, e
			}
			if e = verifyQuery(string(body), q, p.WorkingDirectory); e != nil {
				return r, e
			}
		}
		dep := filepath.Join(out, "prospective.d")
		args := append([]string{p.Recipe.Compiler.Path}, p.Recipe.Flags...)
		args = append(args, "-M", "-MT", "lineage-probe", "-MF", dep, p.Source)
		if _, err = execute("headers", args); err != nil {
			return r, err
		}
		r.Dependencies, err = dependencySet(dep, p.WorkingDirectory)
		if err != nil {
			return r, err
		}
		if !slices.Contains(r.Dependencies, p.Source) {
			return r, fmt.Errorf("lineage immutable TU omitted by -M")
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
		if len(prior.Commands) != 1 || len(prior.Dependencies) == 0 {
			return r, fmt.Errorf("lineage precompile header evidence absent")
		}
		expected := append([]string{p.Recipe.Compiler.Path}, p.Recipe.Flags...)
		expected = append(expected, "-M", "-MT", "lineage-probe", "-MF", prior.Depfile.Path, p.Source)
		if !slices.Equal(expected, prior.Commands[0].Args) {
			return r, fmt.Errorf("lineage -M argv differs")
		}
		actual, e := dependencySet(prior.Depfile.Path, p.WorkingDirectory)
		if e != nil {
			return r, e
		}
		if e = verifyDependencies(prior.Dependencies, actual); e != nil {
			return r, e
		}
		var frozen map[string]string
		if err = readJSON(prior.BeforeCompile, &frozen); err != nil {
			return r, err
		}
		if !maps.Equal(frozen, prior.Inputs) {
			return r, fmt.Errorf("lineage before-compile manifest differs")
		}
		want := maps.Clone(p.Inputs)
		for _, path := range prior.Dependencies {
			if err = filePin(want, path); err != nil {
				return r, err
			}
		}
		if !maps.Equal(want, prior.Inputs) {
			return r, fmt.Errorf("lineage -M input membership differs")
		}
		r.Inputs = maps.Clone(prior.Inputs)
		r.Dependencies = slices.Clone(prior.Dependencies)
		r.BeforeCompile = prior.BeforeCompile
		dep := filepath.Join(out, "actual.d")
		object := filepath.Join(out, "lineage.o")
		args := append([]string{p.Recipe.Compiler.Path}, p.Recipe.Flags...)
		args = append(args, "-MD", "-MF", dep, "-c", p.Source, "-o", object)
		if _, err = execute("compile", args); err != nil {
			return r, err
		}
		actual, err = dependencySet(dep, p.WorkingDirectory)
		if err != nil {
			return r, err
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
		valid := f.Class == elf.ELFCLASS64 && f.Machine == elf.EM_X86_64 && f.Type == elf.ET_REL
		f.Close()
		if !valid {
			return r, fmt.Errorf("lineage object is outside x86_64 relocatable domain")
		}
		if err = filePin(r.Objects, object); err != nil {
			return r, err
		}
		r.Depfile, err = artifactPin(dep)
		if err != nil {
			return r, err
		}
		binary := filepath.Join(out, "lineage-probe")
		linkMap := filepath.Join(out, "link.map")
		link := append([]string{p.Recipe.Compiler.Path, object}, p.Recipe.Libraries...)
		link = append(link, "-Wl,-t", "-Wl,-Map="+linkMap, "-o", binary)
		command, e := execute("link", link)
		if e != nil {
			return r, e
		}
		trace, e := os.ReadFile(command.Stdout)
		if e != nil {
			return r, e
		}
		linkInputs := lineageLinkInputs(p)
		if err = h07.VerifyLinkTrace(string(trace), p.WorkingDirectory, linkInputs, r.Objects); err != nil {
			return r, err
		}
		body, e := os.ReadFile(linkMap)
		if e != nil {
			return r, e
		}
		if err = lineageLinkMap(string(body), p.WorkingDirectory, linkInputs, r.Objects); err != nil {
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
	r.Status = "passed"
	return r, nil
}

func CheckLineagePlan(anchor File) error {
	anchor, err := normalizeAnchor(anchor)
	if err != nil {
		return err
	}
	var p LineagePlan
	if err = readJSON(anchor, &p); err != nil {
		return err
	}
	return verifyLineagePlan(p, true, true)
}
