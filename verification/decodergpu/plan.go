package decodergpu

import (
	"debug/elf"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type Command struct {
	Phase string
	Argv  []string
	Held  bool
}
type BuildPlan struct {
	Toolchain           hacgpu.OracleBuildToolchain
	Inputs              map[string]string
	Commands            []Command
	RequiredClosure     []string
	NativeExecutionHeld bool
}

// LinkReferences includes file and -l references hidden inside -Wl arguments.
// Search/rpath/output controls are not inputs. Unknown bare tokens fail closed.
func LinkReferences(args []string) (files, libraries []string, err error) {
	var tokens []string
	for _, arg := range args {
		if strings.HasPrefix(arg, "-Wl,") {
			tokens = append(tokens, strings.Split(arg[4:], ",")...)
		} else {
			tokens = append(tokens, arg)
		}
	}
	skip := false
	libraryPending := false
	for _, arg := range tokens {
		if libraryPending {
			if arg == "" || strings.HasPrefix(arg, "-") {
				return nil, nil, fmt.Errorf("linker -l lacks a library name")
			}
			libraries = append(libraries, "-l"+arg)
			libraryPending = false
			continue
		}
		if arg == "-l" || arg == "--library" {
			libraryPending = true
			continue
		}
		if strings.HasPrefix(arg, "--library=") {
			name := strings.TrimPrefix(arg, "--library=")
			if name == "" {
				return nil, nil, fmt.Errorf("empty linker library")
			}
			libraries = append(libraries, "-l"+name)
			continue
		}
		if strings.HasPrefix(arg, "-T") && len(arg) > 2 {
			files = append(files, arg[2:])
			continue
		}
		if strings.HasPrefix(arg, "--script=") {
			files = append(files, strings.TrimPrefix(arg, "--script="))
			continue
		}
		if skip {
			skip = false
			continue
		}
		switch arg {
		case "-rpath", "--rpath", "-rpath-link", "--rpath-link", "-Map", "--Map", "-o", "-L", "--library-path":
			skip = true
			continue
		}
		if strings.HasPrefix(arg, "-l") {
			libraries = append(libraries, arg)
			continue
		}
		if strings.HasPrefix(arg, "-") {
			continue
		}
		if strings.Contains(arg, ".so") || strings.HasSuffix(arg, ".a") || strings.HasSuffix(arg, ".o") || strings.HasSuffix(arg, ".ld") {
			files = append(files, arg)
			continue
		}
		return nil, nil, fmt.Errorf("unclassified linker token %q", arg)
	}
	if skip || libraryPending {
		return nil, nil, fmt.Errorf("linker option lacks its argument")
	}
	return files, libraries, nil
}
func canonical(path string) (string, error) {
	p, e := filepath.Abs(path)
	if e != nil {
		return "", e
	}
	return filepath.EvalSymlinks(p)
}
func filePin(pins map[string]string, path string) error {
	p, e := canonical(path)
	if e != nil {
		return e
	}
	info, e := os.Stat(p)
	if e != nil {
		return e
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("closure input is not regular: %s", p)
	}
	got, e := runner.SHA(p)
	if e != nil {
		return e
	}
	if prior, ok := pins[p]; ok && prior != got {
		return fmt.Errorf("closure input changed: %s", p)
	}
	pins[p] = got
	return nil
}
func checkPins(pins map[string]string) error {
	for p, want := range pins {
		info, e := os.Stat(p)
		if e != nil || !info.Mode().IsRegular() {
			return fmt.Errorf("frozen input is not regular: %s", p)
		}
		got, e := runner.SHA(p)
		if e != nil {
			return e
		}
		if got != want {
			return fmt.Errorf("frozen input changed: %s", p)
		}
	}
	return nil
}

// ExpandLinkerScripts follows nested script references without invoking ld.
// Resolution is supplied by the separately admitted compiler resource phase.
func ExpandLinkerScripts(pins map[string]string, resolve func(string, string) (string, error)) error {
	queue := make([]string, 0, len(pins))
	for p := range pins {
		queue = append(queue, p)
	}
	slices.Sort(queue)
	seen := map[string]bool{}
	for len(queue) > 0 {
		p := queue[0]
		queue = queue[1:]
		if seen[p] {
			continue
		}
		seen[p] = true
		f, e := os.Open(p)
		if e != nil {
			return e
		}
		var prefix [8]byte
		n, e := f.Read(prefix[:])
		f.Close()
		if e != nil && e != io.EOF {
			return e
		}
		if strings.HasPrefix(string(prefix[:n]), "\x7fELF") || strings.HasPrefix(string(prefix[:n]), "!<arch>\n") {
			continue
		}
		if !strings.Contains(filepath.Base(p), ".so") && !strings.HasSuffix(p, ".ld") && !strings.HasSuffix(p, ".a") {
			continue
		}
		info, e := os.Stat(p)
		if e != nil {
			return e
		}
		if info.Size() > 1<<20 {
			return fmt.Errorf("linker script text exceeds bounded size")
		}
		body, e := os.ReadFile(p)
		if e != nil {
			return e
		}
		refs, e := h07.LinkerScriptInputs(string(body))
		if e != nil {
			return e
		}
		for _, ref := range refs {
			next, e := resolve(ref, p)
			if e != nil {
				return e
			}
			if e = filePin(pins, next); e != nil {
				return e
			}
			next, e = canonical(next)
			if e != nil {
				return e
			}
			queue = append(queue, next)
		}
	}
	return nil
}
func PlanBuild(buildDir, out string) (p BuildPlan, err error) {
	p = BuildPlan{Inputs: map[string]string{}, NativeExecutionHeld: true, RequiredClosure: []string{
		"actual compiler driver and cc1plus/as/collect2/ld executable bytes and their ELF loaders/DSO closures",
		"compiler specs, multilib search dirs, start/end CRT objects, gcc/stdC++/libc/libm libraries and transitive linker scripts",
		"preprocess -M complete system/project/header closure frozen BEFORE compile, exact -MD equality after compile",
		"all direct and -Wl-wrapped DSOs/archives/-l inputs resolved with real search flags; transitive GROUP/INPUT/AS_NEEDED/INCLUDE",
		"closed -Wl,-t and link map: every consumed input already pinned before compile; unknown trace is failure, never retro-add",
		"prospective ELF interpreter + transitive runtime dependencies (including CUDA driver/runtime); actual mapped paths/hash/SM120 checked before stage calls",
		"all controller selected Go/Cgo/asm/embedded dependency files, Go tool/environment and retained executable before native tools",
	}}
	p.Toolchain, err = hacgpu.OracleToolchainForSource(buildDir, "basecall/decode/CUDADecoder.cpp")
	if err != nil {
		return p, err
	}
	compiler, err := exec.LookPath(p.Toolchain.Compiler)
	if err != nil {
		return p, err
	}
	if err = filePin(p.Inputs, compiler); err != nil {
		return p, err
	}
	source, err := filepath.Abs("verification/decodergpu/native/oracle.cpp")
	if err != nil {
		return p, err
	}
	for _, path := range []string{source, "verification/decodergpu/native/sha256.inc", "verification/decodergpu/native/runtime_gate.inc", "verification/decodergpu/native/image_observer.inc", filepath.Join(buildDir, "compile_commands.json"), filepath.Join(buildDir, "build.ninja")} {
		if err = filePin(p.Inputs, path); err != nil {
			return p, err
		}
	}
	headerFound := false
	for i, arg := range p.Toolchain.Includes {
		include := ""
		if strings.HasPrefix(arg, "-I") && len(arg) > 2 {
			include = arg[2:]
		} else if (arg == "-isystem" || arg == "-I") && i+1 < len(p.Toolchain.Includes) {
			include = p.Toolchain.Includes[i+1]
		}
		if include == "" {
			continue
		}
		if !filepath.IsAbs(include) {
			include = filepath.Join(buildDir, include)
		}
		header := filepath.Join(include, "koi.h")
		if info, e := os.Stat(header); e == nil && info.Mode().IsRegular() {
			if err = filePin(p.Inputs, header); err != nil {
				return p, err
			}
			actual, e := canonical(header)
			if e != nil {
				return p, e
			}
			if p.Inputs[actual] != "8b4115c2e92ebf1acf1db2aa76bdc9b3a22768cb595b92363f88e44ff92d3b88" {
				return p, fmt.Errorf("prospective Koi header differs from selected decoder archive")
			}
			headerFound = true
			break
		}
	}
	if !headerFound {
		return p, fmt.Errorf("matching Koi header absent from retained include search")
	}

	// The retained stock line is provenance. Execution is exclusively the
	// maintained closed controller, whose standalone recipe needs no Torch.
	recipe, err := observedRecipe(p.Toolchain, buildDir)
	if err != nil {
		return p, err
	}
	for path, pin := range recipe.Inputs {
		p.Inputs[path] = pin
	}
	return p, checkPins(p.Inputs)
}

type goPackage struct {
	Dir, ImportPath                                                            string
	GoFiles, CgoFiles, CFiles, CXXFiles, HFiles, SFiles, SysoFiles, EmbedFiles []string
	Error                                                                      any
	DepsErrors                                                                 []any
}
type GoClosure struct {
	Pins                      map[string]string
	Selection                 map[string]string
	Snapshots, SnapshotPins   map[string]string
	Commands                  []runner.CommandReceipt
	Controller, ControllerSHA string
}

func FreezeGoController(dir, out string) (g GoClosure, err error) {
	g.Pins = map[string]string{}
	g.Selection = map[string]string{}
	add := func(path string) error {
		if err := filePin(g.Pins, path); err != nil {
			return err
		}
		actual, e := canonical(path)
		if e != nil {
			return e
		}
		g.Selection[path] = actual
		return nil
	}
	goBin, err := exec.LookPath("go")
	if err != nil {
		return g, err
	}
	if err = add(goBin); err != nil {
		return g, err
	}
	overrides := []string{"GOENV=off", "GOWORK=off", "GOTOOLCHAIN=local", "GOPROXY=off", "GOSUMDB=off", "GOFLAGS=", "CGO_ENABLED=0", "GOOS=linux", "GOARCH=amd64", "GOAMD64=v1", "GOEXPERIMENT=", "GODEBUG="}
	execute := func(label string, args ...string) (runner.CommandReceipt, error) {
		rec, e := runner.Execute(out, label, overrides, append([]string{goBin}, args...)...)
		g.Commands = append(g.Commands, rec)
		return rec, e
	}
	rec, err := execute("go-dependencies", "list", "-deps", "-json", "-pgo=off", "./verification/decodergpu/cmd")
	if err != nil {
		return g, err
	}
	raw, err := os.ReadFile(rec.Stdout)
	if err != nil {
		return g, err
	}
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	count := 0
	for {
		var p goPackage
		e := dec.Decode(&p)
		if e == io.EOF {
			break
		}
		if e != nil {
			return g, e
		}
		if p.Error != nil || len(p.DepsErrors) > 0 {
			return g, fmt.Errorf("Go dependency resolution failed: %s", p.ImportPath)
		}
		for _, list := range [][]string{p.GoFiles, p.CgoFiles, p.CFiles, p.CXXFiles, p.HFiles, p.SFiles, p.SysoFiles, p.EmbedFiles} {
			for _, name := range list {
				if err = add(filepath.Join(p.Dir, name)); err != nil {
					return g, err
				}
			}
		}
		count++
	}
	if count == 0 {
		return g, fmt.Errorf("empty actual Go dependency closure")
	}
	for _, name := range []string{"go.mod", "go.sum"} {
		path := filepath.Join(dir, name)
		if _, e := os.Stat(path); e == nil {
			if err = add(path); err != nil {
				return g, err
			}
		}
	}
	envReceipt, e := execute("go-environment", "env", "-json")
	if e != nil {
		return g, e
	}
	envRaw, e := os.ReadFile(envReceipt.Stdout)
	if e != nil {
		return g, e
	}
	var env map[string]string
	if e = json.Unmarshal(envRaw, &env); e != nil {
		return g, e
	}
	toolPaths := []string{goBin}
	for _, name := range []string{"compile", "asm", "link"} {
		path := filepath.Join(env["GOTOOLDIR"], name)
		if err = add(path); err != nil {
			return g, err
		}
		toolPaths = append(toolPaths, path)
	}
	includeFiles, e := os.ReadDir(filepath.Join(env["GOROOT"], "pkg", "include"))
	if e != nil {
		return g, e
	}
	for _, entry := range includeFiles {
		if entry.Type().IsRegular() {
			if err = add(filepath.Join(env["GOROOT"], "pkg", "include", entry.Name())); err != nil {
				return g, err
			}
		}
	}

	for _, path := range toolPaths {
		f, e := elf.Open(path)
		if e != nil {
			return g, e
		}
		dynamic := false
		for _, program := range f.Progs {
			if program.Type == elf.PT_INTERP || program.Type == elf.PT_DYNAMIC {
				dynamic = true
			}
		}
		f.Close()
		if dynamic {
			return g, fmt.Errorf("host Go tool requires an explicit runtime closure: %s", path)
		}
	}
	if err = checkPins(g.Pins); err != nil {
		return g, err
	}
	g.Snapshots, g.SnapshotPins, err = snapshotPins(g.Pins, filepath.Join(out, "go-inputs"))
	if err != nil {
		return g, err
	}
	g.Controller = filepath.Join(out, "decoder-controller")
	if _, err = execute("go-controller-build", "build", "-x", "-pgo=off", "-trimpath", "-buildvcs=true", "-o", g.Controller, "./verification/decodergpu/cmd"); err != nil {
		return g, err
	}
	if err = checkPins(g.Pins); err != nil {
		return g, err
	}
	if err = checkPins(g.SnapshotPins); err != nil {
		return g, err
	}
	g.ControllerSHA, err = runner.SHA(g.Controller)
	return g, err
}
