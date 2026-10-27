package closedbuild

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
	Plan     Pin
	Out      string
	Admitted bool
}
type Build struct {
	Schema, Status, Error, Started, Finished string
	Plan                                     Pin
	Inputs, Objects                          map[string]string
	Dependencies                             map[string][]string
	Depfiles, Outputs, LinkMaps              map[string]Pin
	BeforeCompile                            Pin
	ArchiveMembers                           map[string]string
	Commands                                 []runner.CommandReceipt
	Limits                                   map[string]string
	RuntimeHeld                              bool
}

func depfile(path, cwd string) ([]string, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return nil, e
	}
	return h07.ParseDependencies(string(b), cwd)
}
func queryResult(body string, q Query, cwd string, inputs map[string]string) error {
	name := strings.TrimSpace(body)
	if name == "" || strings.ContainsAny(name, "\r\n") {
		return fmt.Errorf("compiler query did not return one path")
	}
	if !filepath.IsAbs(name) {
		if strings.ContainsRune(name, '/') {
			name = filepath.Join(cwd, name)
		} else if strings.HasPrefix(q.Argv[1], "-print-prog-name=") {
			name = filepath.Join("/usr/bin", name)
		} else {
			return fmt.Errorf("required compiler resource unresolved")
		}
	}
	p, e := PinFile(name)
	if e != nil {
		return e
	}
	expected, e := PinFile(q.Expected)
	if e != nil {
		return e
	}
	if p != expected || inputs[p.Path] != p.SHA256 {
		return fmt.Errorf("compiler selected unpinned resource %s", name)
	}
	return nil
}
func nativeSymbols(path string) ([]string, error) {
	f, e := elf.Open(path)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	symbols, e := f.DynamicSymbols()
	if e != nil {
		return nil, e
	}
	names := []string{}
	for _, s := range symbols {
		if s.Section != elf.SHN_UNDEF {
			names = append(names, s.Name)
		}
	}
	slices.Sort(names)
	return names, nil
}
func Compile(o BuildOptions) (r Build, err error) {
	if !o.Admitted {
		return r, fmt.Errorf("%w: closed native compilation requires separate reviewed compiler admission", runner.ErrAdmissionBlocked)
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return r, err
	}
	defer lock.Close()
	limits, err := Limits()
	if err != nil {
		return r, err
	}
	var p Plan
	if err = read(o.Plan, &p); err != nil {
		return r, err
	}
	if err = CheckPlan(p); err != nil {
		return r, err
	}
	if _, err = ReadController(p.Options.Controller, true); err != nil {
		return r, err
	}
	out, err := runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	r = Build{Schema: "nanopore-tensor-closed-build/v1", Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Plan: o.Plan, Inputs: maps.Clone(p.Inputs), Objects: map[string]string{}, Dependencies: map[string][]string{}, Depfiles: map[string]Pin{}, Outputs: map[string]Pin{}, LinkMaps: map[string]Pin{}, Limits: limits, RuntimeHeld: true}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
		}
		err = errors.Join(err, Write(filepath.Join(out, "build.json"), r))
	}()
	check := func() error {
		return errors.Join(p.Compiler.Check(), checkPathBindings(p.PathBindings), checkPins(r.Inputs))
	}
	env := []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "TZ=UTC", "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2"}
	execute := func(label string, args []string) (runner.CommandReceipt, error) {
		if e := check(); e != nil {
			return runner.CommandReceipt{}, e
		}
		c, e := command(out, label, p.Directory, args, env)
		r.Commands = append(r.Commands, c)
		return c, errors.Join(e, check())
	}
	for i, q := range p.Queries {
		c, e := execute(fmt.Sprintf("query-%02d", i), q.Argv)
		if e != nil {
			return r, e
		}
		b, e := os.ReadFile(c.Stdout)
		if e != nil {
			return r, e
		}
		if e = queryResult(string(b), q, p.Directory, r.Inputs); e != nil {
			return r, e
		}
	}
	for _, u := range p.Units {
		name := u.Recipe.Name
		path := filepath.Join(out, name+"-before.d")
		args := append([]string{p.Recipe.Options.Compiler}, u.Recipe.Flags...)
		args = append(args, "-M", "-MT", name, "-MF", path, u.Source.Path)
		if _, err = execute("headers-"+name, args); err != nil {
			return r, err
		}
		deps, e := depfile(path, p.Directory)
		if e != nil {
			return r, e
		}
		if !slices.Contains(deps, u.Source.Path) {
			return r, fmt.Errorf("preprocessor omitted selected source")
		}
		r.Dependencies[name] = deps
		for _, path := range deps {
			q, e := PinFile(path)
			if e != nil {
				return r, e
			}
			if e = add(r.Inputs, q); e != nil {
				return r, e
			}
		}
		q, e := PinFile(path)
		if e != nil {
			return r, e
		}
		r.Depfiles[name+"-before"] = q
	}
	r.ArchiveMembers, err = archiveMembers(p.Recipe.Options.Torch, p.Recipe.Options.Archive, r.Inputs)
	if err != nil {
		return r, err
	}
	before := filepath.Join(out, "before-compile-inputs.json")
	if err = Write(before, r.Inputs); err != nil {
		return r, err
	}
	if r.BeforeCompile, err = PinFile(before); err != nil {
		return r, err
	}
	for _, u := range p.Units {
		name := u.Recipe.Name
		object := filepath.Join(out, name+".o")
		actual := filepath.Join(out, name+"-actual.d")
		args := append([]string{p.Recipe.Options.Compiler}, u.Recipe.Flags...)
		args = append(args, "-MD", "-MF", actual, "-c", u.Source.Path, "-o", object)
		if _, err = execute("compile-"+name, args); err != nil {
			return r, err
		}
		deps, e := depfile(actual, p.Directory)
		if e != nil {
			return r, e
		}
		if !slices.Equal(deps, r.Dependencies[name]) {
			return r, fmt.Errorf("actual -MD differs from frozen -M: %s", name)
		}
		f, e := elf.Open(object)
		if e != nil {
			return r, e
		}
		valid := f.Type == elf.ET_REL && f.Class == elf.ELFCLASS64 && f.Machine == elf.EM_X86_64
		f.Close()
		if !valid {
			return r, fmt.Errorf("compiler output not x86_64 ET_REL")
		}
		q, e := PinFile(object)
		if e != nil {
			return r, e
		}
		r.Objects[q.Path] = q.SHA256
		q, e = PinFile(actual)
		if e != nil {
			return r, e
		}
		r.Depfiles[name+"-actual"] = q
	}
	for _, u := range p.Units {
		name := u.Recipe.Name
		binary := filepath.Join(out, filepath.Base(u.Recipe.Output))
		linkMap := filepath.Join(out, name+".map")
		if err = checkPins(r.Objects); err != nil {
			return r, err
		}
		args := append([]string{p.Recipe.Options.Compiler}, u.Recipe.Flags...)
		if u.Recipe.Shared {
			args = append(args, "-shared")
		}
		args = append(args, filepath.Join(out, name+".o"), "-o", binary)
		args = append(args, u.Libraries...)
		args = append(args, "-Wl,-t", "-Wl,-Map="+linkMap)
		c, e := execute("link-"+name, args)
		if e != nil {
			return r, e
		}
		b, e := os.ReadFile(c.Stdout)
		if e != nil {
			return r, e
		}
		if e = h07.VerifyLinkTrace(string(b), p.Directory, r.Inputs, r.Objects); e != nil {
			return r, e
		}
		m, e := nativeproof.InspectELF(binary)
		if e != nil {
			return r, e
		}
		if m.Class != elf.ELFCLASS64 || m.Machine != elf.EM_X86_64 {
			return r, fmt.Errorf("output ELF outside x86_64")
		}
		if name == "backend" {
			symbols, e := nativeSymbols(binary)
			if e != nil {
				return r, e
			}
			for _, symbol := range []string{"nt_cuda_decoder_guide_program", "nt_cuda_decoder_guide_frame", "nt_cuda_sup_primitive", "nt_cuda_decoder_post_numbers", "nt_cuda_decoder_beam_program", "nt_cuda_decoder_beam_numbers"} {
				if !slices.Contains(symbols, symbol) {
					return r, fmt.Errorf("current ABI5 backend missing %s", symbol)
				}
			}
		}
		r.Outputs[name] = m.Pin
		q, e := PinFile(linkMap)
		if e != nil {
			return r, e
		}
		r.LinkMaps[name] = q
	}
	if err = check(); err != nil {
		return r, err
	}
	if err = checkPins(r.Objects); err != nil {
		return r, err
	}
	if err = CheckPlan(p); err != nil {
		return r, err
	}
	r.Status = "built_outputs_unexecuted"
	return r, nil
}
