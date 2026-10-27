package closedbuild

import (
	"debug/elf"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild/recipe"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

// ArtifactOptions are four independent admission anchors. Changing a manifest's
// own digest cannot change its admitted plan, producer or selected output.
type ArtifactOptions struct{ Build, Plan, Producer, Output Pin }

// BackendArtifact is a compiler proof, never runtime qualification. Inputs names
// the files authenticated now; ProducerSnapshots binds historical names to their
// immutable bytes. Historical Go originals are deliberately absent from Inputs.
type BackendArtifact struct {
	Schema                        string
	Build, Plan, Producer, Output Pin
	Inputs                        map[string]string
	ProducerSnapshots             map[string]Pin
	Runtime                       []nativeproof.ELFMetadata
	RuntimeHeld                   bool
}

func artifactCommand(r runner.CommandReceipt, args, env []string, previous, finish string) error {
	a, e := time.Parse(time.RFC3339Nano, r.Started)
	b, f := time.Parse(time.RFC3339Nano, r.Finished)
	lo, g := time.Parse(time.RFC3339Nano, previous)
	hi, h := time.Parse(time.RFC3339Nano, finish)
	if e != nil || f != nil || g != nil || h != nil || a.Before(lo) || b.Before(a) || b.After(hi) || r.Exit != 0 || !slices.Equal(r.Args, args) || !slices.Equal(r.Environment, env) {
		return fmt.Errorf("command argv/environment/exit/clocks differ")
	}
	return checkPins(map[string]string{r.Stdout: r.StdoutSHA, r.Stderr: r.StderrSHA})
}

func producerSources(c Controller, base string) error {
	if c.Schema != "nanopore-tensor-closed-controller/v1" || c.Status != "built" || c.Error != "" || len(c.Lists) != 2 || len(c.Commands) != 5 || len(c.Binaries) != 2 || len(c.Inputs) == 0 || len(c.Snapshots) != len(c.Inputs) || len(c.Tools) == 0 {
		return fmt.Errorf("incomplete historical producer custody")
	}
	selected := maps.Clone(c.Tools)
	var before []string
	for i, lp := range c.Lists {
		var raw []byte
		if e := verify(lp); e != nil {
			return e
		}
		raw, e := os.ReadFile(lp.Path)
		if e != nil {
			return e
		}
		names, e := selectedGo(raw)
		if e != nil {
			return e
		}
		if i == 0 {
			before = names
		} else if !slices.Equal(before, names) {
			return fmt.Errorf("historical source selection differs")
		}
		for _, path := range names {
			if c.Inputs[path] == "" {
				return fmt.Errorf("selected historical source omitted: %s", path)
			}
			selected[path] = c.Inputs[path]
		}
	}
	if !maps.Equal(selected, c.Inputs) {
		return fmt.Errorf("historical producer source/tool binding differs")
	}
	for path, want := range c.Inputs {
		p, ok := c.Snapshots[path]
		if !ok || p.SHA256 != want || p.Path != filepath.Join(base, "sources", want) {
			return fmt.Errorf("historical snapshot/source binding differs: %s", path)
		}
		if e := verify(p); e != nil {
			return e
		}
	}
	return nil
}

// readProducer never consults the mutable historical source or Go tool paths.
// Its independently admitted manifest, complete selection and snapshots are the
// history. This is not a claim that a current consumer has that build closure.
func readProducer(pin Pin) (c Controller, err error) {
	if err = read(pin, &c); err != nil {
		return c, err
	}
	base := filepath.Dir(c.Binaries["closed"].Path)
	if err = producerSources(c, base); err != nil {
		return c, err
	}
	goBin := filepath.Join(c.GoRoot, "bin/go")
	list := []string{goBin, "list", "-deps", "-json", "./verification/tensor/closedbuild/cmd", "./verification/tensor/build"}
	argv := [][]string{
		{goBin, "env", "-json", "GOROOT", "GOTOOLDIR", "CGO_ENABLED", "GOOS", "GOARCH"}, list,
		{goBin, "build", "-a", "-x", "-work", "-p=2", "-pgo=off", "-trimpath", "-buildvcs=false", "-o", filepath.Join(base, "closed"), "./verification/tensor/closedbuild/cmd"},
		{goBin, "build", "-a", "-x", "-work", "-p=2", "-pgo=off", "-trimpath", "-buildvcs=false", "-o", filepath.Join(base, "maintained"), "./verification/tensor/build"}, list,
	}
	if len(c.Commands[0].Environment) == 0 || !strings.HasPrefix(c.Commands[0].Environment[0], "CLOSED_BUILD_CWD=/") {
		return c, fmt.Errorf("historical producer directory absent")
	}
	env := []string{c.Commands[0].Environment[0], "PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "TZ=UTC", "GOROOT=" + c.GoRoot, "GOTOOLCHAIN=local", "GOPROXY=off", "GOSUMDB=off", "GOENV=off", "GOWORK=off", "GOFLAGS=", "GOOS=linux", "GOARCH=amd64", "GOAMD64=v1", "CGO_ENABLED=0", "GOMAXPROCS=2", "GOMEMLIMIT=3GiB", "GOCACHE=" + filepath.Join(base, "cache"), "GOTMPDIR=" + filepath.Join(base, "tmp"), "GOPATH=" + filepath.Join(base, "gopath")}
	previous := c.Started
	for i, cmd := range c.Commands {
		if err = artifactCommand(cmd, argv[i], env, previous, c.Finished); err != nil {
			return c, fmt.Errorf("historical producer command %d: %w", i, err)
		}
		previous = cmd.Finished
	}
	for i, n := range []int{1, 4} {
		if c.Lists[i] != (Pin{Path: c.Commands[n].Stdout, SHA256: c.Commands[n].StdoutSHA}) {
			return c, fmt.Errorf("historical list/raw command binding differs")
		}
	}
	var ge map[string]string
	body, e := os.ReadFile(c.Commands[0].Stdout)
	if e != nil {
		return c, e
	}
	if e = json.Unmarshal(body, &ge); e != nil {
		return c, e
	}
	if !maps.Equal(ge, map[string]string{"GOROOT": c.GoRoot, "GOTOOLDIR": filepath.Join(c.GoRoot, "pkg/tool/linux_amd64"), "CGO_ENABLED": "0", "GOOS": "linux", "GOARCH": "amd64"}) {
		return c, fmt.Errorf("historical Go environment differs")
	}
	for _, name := range []string{"closed", "maintained"} {
		p := c.Binaries[name]
		if p.Path != filepath.Join(base, name) {
			return c, fmt.Errorf("historical binary binding differs")
		}
		m, e := nativeproof.InspectELF(p.Path)
		if e != nil || m.Pin != p || m.Interpreter != "" || len(m.Needed) != 0 {
			return c, fmt.Errorf("historical producer binary differs: %v", e)
		}
	}
	return c, nil
}

func linkPaths(body, cwd string, linkMap bool) (map[string]bool, error) {
	set := map[string]bool{}
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if linkMap {
			if !strings.HasPrefix(line, "LOAD ") {
				continue
			}
			line = strings.TrimPrefix(line, "LOAD ")
		}
		if line == "" {
			continue
		}
		if !filepath.IsAbs(line) {
			line = filepath.Join(cwd, line)
		}
		path, e := filepath.EvalSymlinks(line)
		if e != nil {
			return nil, e
		}
		set[path] = true
	}
	if len(set) == 0 {
		return nil, fmt.Errorf("empty link input evidence")
	}
	return set, nil
}

// ReadArtifact authenticates the actual compiler-only producer without running
// commands or loading ELF code. It intentionally does not call CheckPlan, whose
// live Go-source check is appropriate at compilation, not for historical reuse.
func ReadArtifact(o ArtifactOptions) (a BackendArtifact, err error) {
	for _, p := range []Pin{o.Build, o.Plan, o.Producer, o.Output} {
		if len(p.SHA256) != 64 || !filepath.IsAbs(p.Path) {
			return a, fmt.Errorf("independent artifact anchors required")
		}
	}
	var b Build
	var p Plan
	if err = read(o.Build, &b); err != nil {
		return a, err
	}
	if b.Schema != "nanopore-tensor-closed-build/v1" || b.Status != "built_outputs_unexecuted" || b.Error != "" || !b.RuntimeHeld || b.Plan != o.Plan || len(b.Outputs) != 1 || b.Outputs["backend"] != o.Output {
		return a, fmt.Errorf("admitted backend build/output differs")
	}
	if err = read(o.Plan, &p); err != nil {
		return a, err
	}
	if p.Schema != "nanopore-tensor-closed-plan/v1" || !p.RuntimeHeld || p.Options.Controller != o.Producer || !slices.Equal(p.Options.Profiles, []string{"backend"}) || len(p.Units) != 1 || p.ToolkitVersion != 13040 {
		return a, fmt.Errorf("admitted backend plan/producer differs")
	}
	c, err := readProducer(o.Producer)
	if err != nil {
		return a, err
	}
	if c.Commands[0].Environment[0] != "CLOSED_BUILD_CWD="+p.Directory {
		return a, fmt.Errorf("native plan/historical producer source directory differs")
	}
	wantRecipe, e := recipe.New(p.Recipe.Options)
	if e != nil || !equal(p.Recipe, wantRecipe) {
		return a, fmt.Errorf("maintained backend recipe differs")
	}
	u := p.Units[0]
	wantUnit, e := p.Recipe.Unit("backend", "")
	if e != nil || !equal(u.Recipe, wantUnit) || u.Source.Path != filepath.Join(p.Directory, u.Recipe.Source) || p.Inputs[u.Source.Path] != u.Source.SHA256 {
		return a, fmt.Errorf("backend native source binding differs")
	}
	var contract CompilerContract
	if err = read(p.Options.CompilerContract, &contract); err != nil {
		return a, err
	}
	if !equal(contract.Compiler, p.Compiler) || contract.Recipe.Compiler.Path != p.Recipe.Options.Compiler {
		return a, fmt.Errorf("compiler contract/domain differs")
	}
	if err = p.Compiler.Check(); err != nil {
		return a, err
	}
	if err = checkPathBindings(p.PathBindings); err != nil {
		return a, err
	}
	if len(b.Commands) != len(p.Queries)+3 || len(b.Dependencies) != 1 || len(b.Depfiles) != 2 || len(b.Objects) != 1 || len(b.LinkMaps) != 1 {
		return a, fmt.Errorf("incomplete backend command/output evidence")
	}
	base := filepath.Dir(o.Output.Path)
	beforeDep := filepath.Join(base, "backend-before.d")
	actualDep := filepath.Join(base, "backend-actual.d")
	object := filepath.Join(base, "backend.o")
	linkMap := filepath.Join(base, "backend.map")
	if o.Output.Path != filepath.Join(base, u.Recipe.Output) || b.Depfiles["backend-before"].Path != beforeDep || b.Depfiles["backend-actual"].Path != actualDep || b.LinkMaps["backend"].Path != linkMap || b.Objects[object] == "" {
		return a, fmt.Errorf("backend output paths differ")
	}
	for _, q := range []Pin{b.BeforeCompile, b.Depfiles["backend-before"], b.Depfiles["backend-actual"], b.LinkMaps["backend"]} {
		if err = verify(q); err != nil {
			return a, err
		}
	}
	deps, e := depfile(beforeDep, p.Directory)
	if e != nil {
		return a, e
	}
	actual, e := depfile(actualDep, p.Directory)
	if e != nil {
		return a, e
	}
	if !slices.Equal(deps, actual) || !slices.Equal(deps, b.Dependencies["backend"]) || !slices.Contains(deps, u.Source.Path) {
		return a, fmt.Errorf("backend -M/-MD/source dependencies differ")
	}
	inputs := maps.Clone(p.Inputs)
	for _, path := range deps {
		if b.Inputs[path] == "" {
			return a, fmt.Errorf("backend dependency omitted")
		}
		inputs[path] = b.Inputs[path]
	}
	var before map[string]string
	if err = read(b.BeforeCompile, &before); err != nil {
		return a, err
	}
	if !maps.Equal(inputs, b.Inputs) || !maps.Equal(before, b.Inputs) {
		return a, fmt.Errorf("before-compile input closure differs")
	}
	a = BackendArtifact{Schema: "nanopore-tensor-backend-artifact/v1", Build: o.Build, Plan: o.Plan, Producer: o.Producer, Output: o.Output, Inputs: map[string]string{}, ProducerSnapshots: maps.Clone(c.Snapshots), Runtime: p.Runtime, RuntimeHeld: true}
	native := map[string]bool{}
	for _, path := range deps {
		native[path] = true
	}
	for path, want := range b.Inputs {
		// A project header selected by CGO0 is still a live native input.
		ext := filepath.Ext(path)
		historical := c.Inputs[path] == want && !native[path] && (!slices.Contains([]string{".h", ".c", ".cpp", ".inc"}, ext) || strings.HasPrefix(path, c.GoRoot+"/"))
		if historical {
			continue
		}
		if err = verify(Pin{Path: path, SHA256: want}); err != nil {
			return a, err
		}
		a.Inputs[path] = want
	}
	for _, q := range c.Snapshots {
		a.Inputs[q.Path] = q.SHA256
	}
	for _, q := range []Pin{o.Build, o.Plan, o.Producer, o.Output, b.BeforeCompile, b.Depfiles["backend-before"], b.Depfiles["backend-actual"], b.LinkMaps["backend"]} {
		a.Inputs[q.Path] = q.SHA256
	}
	if err = checkPins(b.Objects); err != nil {
		return a, err
	}
	maps.Copy(a.Inputs, b.Objects)
	args := [][]string{}
	for _, q := range p.Queries {
		args = append(args, q.Argv)
	}
	headerArgs := append([]string{p.Recipe.Options.Compiler}, u.Recipe.Flags...)
	headerArgs = append(headerArgs, "-M", "-MT", "backend", "-MF", beforeDep, u.Source.Path)
	compileArgs := append([]string{p.Recipe.Options.Compiler}, u.Recipe.Flags...)
	compileArgs = append(compileArgs, "-MD", "-MF", actualDep, "-c", u.Source.Path, "-o", object)
	linkArgs := append([]string{p.Recipe.Options.Compiler}, u.Recipe.Flags...)
	linkArgs = append(linkArgs, "-shared", object, "-o", o.Output.Path)
	linkArgs = append(linkArgs, u.Libraries...)
	linkArgs = append(linkArgs, "-Wl,-t", "-Wl,-Map="+linkMap)
	args = append(args, headerArgs, compileArgs, linkArgs)
	env := []string{"CLOSED_BUILD_CWD=" + p.Directory, "PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "TZ=UTC", "OMP_NUM_THREADS=2", "MKL_NUM_THREADS=2"}
	previous := b.Started
	for i, cmd := range b.Commands {
		if err = artifactCommand(cmd, args[i], env, previous, b.Finished); err != nil {
			return a, fmt.Errorf("backend command %d: %w", i, err)
		}
		previous = cmd.Finished
		a.Inputs[cmd.Stdout], a.Inputs[cmd.Stderr] = cmd.StdoutSHA, cmd.StderrSHA
		if i < len(p.Queries) {
			body, e := os.ReadFile(cmd.Stdout)
			if e != nil {
				return a, e
			}
			if err = queryResult(string(body), p.Queries[i], p.Directory, b.Inputs); err != nil {
				return a, err
			}
		}
	}
	trace, e := os.ReadFile(b.Commands[len(b.Commands)-1].Stdout)
	if e != nil {
		return a, e
	}
	if err = h07.VerifyLinkTrace(string(trace), p.Directory, b.Inputs, b.Objects); err != nil {
		return a, err
	}
	mapBody, e := os.ReadFile(linkMap)
	if e != nil {
		return a, e
	}
	t, e := linkPaths(string(trace), p.Directory, false)
	if e != nil {
		return a, e
	}
	l, e := linkPaths(string(mapBody), p.Directory, true)
	if e != nil || !maps.Equal(t, l) {
		return a, fmt.Errorf("link map/trace input sets differ: %v", e)
	}
	members, e := archiveMembers(p.Recipe.Options.Torch, p.Recipe.Options.Archive, b.Inputs)
	if e != nil || !maps.Equal(members, b.ArchiveMembers) {
		return a, fmt.Errorf("Torch archive member custody differs: %v", e)
	}
	for _, target := range []struct {
		Path string
		Kind elf.Type
	}{{object, elf.ET_REL}, {o.Output.Path, elf.ET_DYN}} {
		f, e := elf.Open(target.Path)
		if e != nil {
			return a, e
		}
		valid := f.Type == target.Kind && f.Class == elf.ELFCLASS64 && f.Data == elf.ELFDATA2LSB && f.Machine == elf.EM_X86_64
		f.Close()
		if !valid {
			return a, fmt.Errorf("backend ELF kind/architecture differs")
		}
	}
	if err = verify(o.Output); err != nil {
		return a, err
	}
	providers := map[string]string{}
	for _, m := range p.Runtime {
		providers[m.Pin.Path] = m.Pin.SHA256
	}
	runtime, e := prospectiveRuntime(providers)
	if e != nil || !equal(runtime, p.Runtime) {
		return a, fmt.Errorf("prospective runtime domain differs: %v", e)
	}
	byName := map[string]bool{}
	for _, m := range runtime {
		byName[m.SONAME], byName[filepath.Base(m.Pin.Path)] = true, true
	}
	output, e := nativeproof.InspectELF(o.Output.Path)
	if e != nil {
		return a, e
	}
	for _, name := range output.Needed {
		if !byName[name] {
			return a, fmt.Errorf("backend DT_NEEDED provider absent: %s", name)
		}
	}
	symbols, e := nativeSymbols(o.Output.Path)
	if e != nil {
		return a, e
	}
	abi := filepath.Join(p.Directory, "internal/tensor/abi.h")
	if !slices.Contains(deps, abi) || b.Inputs[abi] == "" {
		return a, fmt.Errorf("consumed native ABI header omitted")
	}
	body, e := os.ReadFile(abi)
	if e != nil {
		return a, e
	}
	loads := regexp.MustCompile(`(?m)^(?:uint32_t|int)\s+(nt_[a-z0-9_]+)\s*\(`).FindAllSubmatch(body, -1)
	if len(loads) == 0 {
		return a, fmt.Errorf("empty native ABI contract")
	}
	for _, m := range loads {
		if !slices.Contains(symbols, string(m[1])) {
			return a, fmt.Errorf("backend lacks required ABI export %s", m[1])
		}
	}
	return a, nil
}
