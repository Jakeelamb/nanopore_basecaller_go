package h06

import (
	"debug/elf"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type BuildOptions struct {
	Plan     nativeproof.Pin
	Out      string
	Admitted bool
}
type BuildReceipt struct {
	Schema, Status, Error, Started, Finished string
	Plan                                     nativeproof.Pin
	Inputs, Objects                          map[string]string
	Limits                                   map[string]string
	Dependencies                             map[string][]string
	Depfiles, Outputs, LinkMaps              map[string]nativeproof.Pin
	BeforeCompile                            nativeproof.Pin
	Commands                                 []runner.CommandReceipt
	RuntimeHeld                              bool
}

func verifyBuildQuery(body string, q linkQuery, cwd string, inputs map[string]string) error {
	name := strings.TrimSpace(body)
	if name == "" || strings.ContainsAny(name, "\r\n") {
		return fmt.Errorf("compiler query did not return one path")
	}
	if !filepath.IsAbs(q.Expected) {
		if name != q.Expected || !strings.HasPrefix(q.Argv[1], "-print-file-name=") {
			return fmt.Errorf("unresolved compiler resource differs")
		}
		return nil
	}
	if !filepath.IsAbs(name) {
		if strings.ContainsRune(name, '/') {
			name = filepath.Join(cwd, name)
		} else {
			if !strings.HasPrefix(q.Argv[1], "-print-prog-name=") {
				return fmt.Errorf("required compiler resource unresolved")
			}
			name = filepath.Join("/usr/bin", name)
		}
	}
	got, err := buildPin(name)
	if err != nil {
		return err
	}
	expected, err := buildPin(q.Expected)
	if err != nil {
		return err
	}
	if got != expected || inputs[expected.Path] != expected.SHA256 {
		return fmt.Errorf("compiler selected an unpinned resource: %s", name)
	}
	return nil
}
func readBuildDeps(path, cwd string) ([]string, error) {
	body, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return h07.ParseDependencies(string(body), cwd)
}
func checkBuildDependencies(before, after []string) error {
	if !slices.Equal(before, after) {
		return fmt.Errorf("actual -MD dependencies differ from frozen -M set")
	}
	return nil
}

func checkHalfLinkMap(body, target string) error {
	flstm := "libdorado_nn.a(FLSTMStack.cpp.o)"
	wm := "libdorado_nn.a(WorkingMemory.cpp.o)"
	if !strings.Contains(body, wm) {
		return fmt.Errorf("link map lacks stock WorkingMemory implementation")
	}
	if target == "reference" && !strings.Contains(body, flstm) {
		return fmt.Errorf("reference link lacks untouched stock FLSTM implementation")
	}
	if target == "observed" && strings.Contains(body, flstm) {
		return fmt.Errorf("observed link selected a second archived FLSTM implementation")
	}
	return nil
}

// BuildOracle compiles and links two executables. It has no target-execution
// path. The immutable plan names the current controller and external contracts.
func BuildOracle(o BuildOptions) (r BuildReceipt, err error) {
	if !o.Admitted {
		return r, fmt.Errorf("%w: H06 native compiler work requires separate root review and admission", runner.ErrAdmissionBlocked)
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return r, err
	}
	defer lock.Close()
	if err = requireBuildLimits(); err != nil {
		return r, err
	}
	var p BuildPlan
	if err = readBuildJSON(o.Plan, &p); err != nil {
		return r, err
	}
	if err = CheckBuildPlan(p); err != nil {
		return r, err
	}
	out, err := runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	r = BuildReceipt{Schema: "nanopore-hac-half-native-build/v1", Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Plan: o.Plan, Inputs: maps.Clone(p.Inputs), Objects: map[string]string{}, Dependencies: map[string][]string{}, Depfiles: map[string]nativeproof.Pin{}, Outputs: map[string]nativeproof.Pin{}, LinkMaps: map[string]nativeproof.Pin{}, RuntimeHeld: true}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
		}
		err = errors.Join(err, writeBuildJSON(filepath.Join(out, "build.json"), r))
	}()
	r.Limits, err = readBuildLimits()
	if err != nil {
		return r, err
	}
	environment := []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "TZ=UTC", "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2"}
	check := func() error {
		if e := checkCompilerContract(p.Compiler, r.Inputs); e != nil {
			return e
		}
		return checkBuildPins(r.Inputs)
	}
	execute := func(label string, args []string) (runner.CommandReceipt, error) {
		return checkedBuildCommand(check, func() (runner.CommandReceipt, error) {
			c, e := buildCommand(p.BuildDir, out, label, args, environment)
			r.Commands = append(r.Commands, c)
			return c, e
		})
	}
	for i, q := range p.Queries {
		c, e := execute(fmt.Sprintf("query-%02d", i), q.Argv)
		if e != nil {
			return r, e
		}
		body, e := os.ReadFile(c.Stdout)
		if e != nil {
			return r, e
		}
		if e = verifyBuildQuery(string(body), q, p.BuildDir, r.Inputs); e != nil {
			return r, e
		}
	}
	for _, unit := range p.Units {
		dep := filepath.Join(out, unit.Name+"-before.d")
		args := append([]string{p.Driver.Path}, unit.Flags...)
		args = append(args, "-M", "-MT", unit.Name, "-MF", dep, unit.Source.Path)
		if _, err = execute("headers-"+unit.Name, args); err != nil {
			return r, err
		}
		deps, e := readBuildDeps(dep, p.BuildDir)
		if e != nil {
			return r, e
		}
		source, e := filepath.EvalSymlinks(unit.Source.Path)
		if e != nil {
			return r, e
		}
		if !slices.Contains(deps, source) {
			return r, fmt.Errorf("preprocessor omitted selected H06 source")
		}
		r.Dependencies[unit.Name] = deps
		for _, path := range deps {
			pin, e := buildPin(path)
			if e != nil {
				return r, e
			}
			if e = addBuildPin(r.Inputs, pin); e != nil {
				return r, e
			}
		}
		pin, e := buildPin(dep)
		if e != nil {
			return r, e
		}
		r.Depfiles[unit.Name+"-before"] = pin
	}
	before := filepath.Join(out, "before-compile-inputs.json")
	if err = writeBuildJSON(before, r.Inputs); err != nil {
		return r, err
	}
	if r.BeforeCompile, err = buildPin(before); err != nil {
		return r, err
	}
	for _, unit := range p.Units {
		object := filepath.Join(out, unit.Name+".o")
		dep := filepath.Join(out, unit.Name+"-actual.d")
		args := append([]string{p.Driver.Path}, unit.Flags...)
		args = append(args, "-MD", "-MF", dep, "-c", unit.Source.Path, "-o", object)
		if _, err = execute("compile-"+unit.Name, args); err != nil {
			return r, err
		}
		actual, e := readBuildDeps(dep, p.BuildDir)
		if e != nil {
			return r, e
		}
		if e = checkBuildDependencies(r.Dependencies[unit.Name], actual); e != nil {
			return r, e
		}
		f, e := elf.Open(object)
		if e != nil {
			return r, e
		}
		valid := f.Type == elf.ET_REL && f.Class == elf.ELFCLASS64 && f.Machine == elf.EM_X86_64
		f.Close()
		if !valid {
			return r, fmt.Errorf("H06 compiler output is not x86_64 relocatable ELF")
		}
		pin, e := buildPin(object)
		if e != nil {
			return r, e
		}
		r.Objects[pin.Path] = pin.SHA256
		pin, e = buildPin(dep)
		if e != nil {
			return r, e
		}
		r.Depfiles[unit.Name+"-actual"] = pin
	}
	for _, target := range p.Targets {
		if err = checkBuildPins(r.Objects); err != nil {
			return r, err
		}
		binary := filepath.Join(out, "h06-"+target.Name)
		linkMap := filepath.Join(out, target.Name+".map")
		args := []string{p.Driver.Path}
		for _, name := range target.Units {
			args = append(args, filepath.Join(out, name+".o"))
		}
		args = append(args, p.Libraries...)
		args = append(args, "-Wl,-t", "-Wl,-Map="+linkMap, "-o", binary)
		c, e := execute("link-"+target.Name, args)
		if e != nil {
			return r, e
		}
		trace, e := os.ReadFile(c.Stdout)
		if e != nil {
			return r, e
		}
		if e = h07.VerifyLinkTrace(string(trace), p.BuildDir, r.Inputs, r.Objects); e != nil {
			return r, e
		}
		body, e := os.ReadFile(linkMap)
		if e != nil {
			return r, e
		}
		if e = checkHalfLinkMap(string(body), target.Name); e != nil {
			return r, e
		}
		metadata, e := nativeproof.InspectELF(binary)
		if e != nil {
			return r, e
		}
		if metadata.Class != elf.ELFCLASS64 || metadata.Machine != elf.EM_X86_64 {
			return r, fmt.Errorf("H06 executable is outside x86_64")
		}
		r.Outputs[target.Name] = metadata.Pin
		pin, e := buildPin(linkMap)
		if e != nil {
			return r, e
		}
		r.LinkMaps[target.Name] = pin
	}
	if err = check(); err != nil {
		return r, err
	}
	if err = checkBuildPins(r.Objects); err != nil {
		return r, err
	}
	if err = CheckBuildPlan(p); err != nil {
		return r, err
	}
	r.Status = "built_target_unexecuted"
	return r, nil
}
