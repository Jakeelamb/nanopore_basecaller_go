package cgobuild

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

func flagValue(args []string, key string) string {
	for i, a := range args {
		if a == key && i+1 < len(args) {
			return args[i+1]
		}
		if strings.HasPrefix(a, key+"=") {
			return strings.TrimPrefix(a, key+"=")
		}
	}
	return ""
}
func knownPackage(p Plan, name string) bool {
	_, _, ok := testDescription(p, name)
	return ok
}
func cgoLease(p Plan, args []string, cwd string, derived []GeneratedSet) (Lease, error) {
	l := Lease{Schema: "nanopore-h08-go-tool-lease/v1", Kind: "cgo", Directory: cwd, Args: slices.Clone(args), Inputs: map[string]string{}}
	if input := flagValue(args, "-dynimport"); input != "" {
		out := flagValue(args, "-dynout")
		dir := filepath.Dir(absArg(cwd, out))
		for _, g := range derived {
			for _, q := range g.Files {
				if filepath.Dir(q.Path) == dir {
					l.Package = g.Package
					l.ObjectDir = dir
				}
			}
		}
		if l.Package == "" {
			return l, fmt.Errorf("dynimport outside completed Cgo generation")
		}
		if !privateOutput(p, cwd, out) {
			return l, fmt.Errorf("dynimport output outside private work")
		}
		pkg, e := packageByName(p, l.Package)
		if e != nil {
			return l, e
		}
		want := []string{"-dynpackage", filepath.Base(pkg.Directory), "-dynimport", input, "-dynout", out}
		if pkg.ImportPath == "runtime/cgo" {
			want = append(want, "-dynlinker")
		}
		if !exactArgs(args, want) {
			return l, fmt.Errorf("unknown dynimport arguments")
		}
		q, e := closedPin(absArg(cwd, input))
		if e != nil || !under(p.Options.WorkRoot, q.Path) {
			return l, fmt.Errorf("invalid dynimport input")
		}
		l.Inputs[q.Path] = q.SHA256
		l.Kind = "dynimport"
		return l, nil
	}
	l.Package = flagValue(args, "-importpath")
	pkg, e := packageByName(p, l.Package)
	if e != nil {
		return l, e
	}
	if cwd != pkg.Directory {
		return l, fmt.Errorf("Cgo working directory differs")
	}
	obj := flagValue(args, "-objdir")
	if obj == "" {
		return l, fmt.Errorf("Cgo object directory missing")
	}
	l.ObjectDir = absArg(cwd, obj)
	actual, e := filepath.EvalSymlinks(l.ObjectDir)
	if e != nil || actual != l.ObjectDir || !under(p.Options.WorkRoot, actual) {
		return l, fmt.Errorf("Cgo object directory outside private work")
	}
	want := []string{"-objdir", obj, "-importpath", pkg.ImportPath}
	if pkg.ImportPath == "runtime/cgo" {
		want = append(want, "-import_runtime_cgo=false", "-import_syscall=false")
	}
	ld := append([]string{"-O2", "-g"}, pkg.LDFlags...)
	quoted := []string{}
	for _, v := range ld {
		quoted = append(quoted, strconv.Quote(v))
	}
	want = append(want, "-ldflags="+strings.Join(quoted, " "), "--")
	want = append(want, pkg.CPPFlags...)
	want = append(want, "-I", obj, "-O2", "-g")
	want = append(want, pkg.CFlags...)
	for _, file := range pkg.Cgo {
		want = append(want, filepath.Join(pkg.Directory, file))
	}
	if !exactArgs(args, want) {
		return l, fmt.Errorf("Cgo source/flags differ from fixed package recipe")
	}
	for _, name := range append(append(slices.Clone(pkg.Cgo), pkg.Headers...), pkg.C...) {
		path := filepath.Join(pkg.Directory, name)
		if p.Inputs[path] == "" {
			return l, fmt.Errorf("Cgo input omitted")
		}
		l.Inputs[path] = p.Inputs[path]
	}
	return l, nil
}

func toolInputs(p Plan, driver Pin, name string, args []string, cwd string) (map[string]string, error) {
	return toolInputsEvidence(p, driver, name, args, cwd, nil, nil, nil)
}

// buildArgs emits strconv.Quote(driver.Path) inside -extld=. Go's quoted.Split
// preserves quotes inside a field; cmd/link's quoted.Flag removes the outer
// pair without unescaping. Restrict the selected path so both admitted forms
// denote exactly one executable, with no arguments or escape interpretation.
func selectedExternalLinker(value string, driver Pin) bool {
	if !filepath.IsAbs(driver.Path) || strings.ContainsAny(driver.Path, " \t\r\n\x00\"'\\") {
		return false
	}
	return value == driver.Path || value == strconv.Quote(driver.Path)
}

func toolInputsEvidence(p Plan, driver Pin, name string, args []string, cwd string, recorded *operation, bindings, outputs map[string]string) (map[string]string, error) {
	inputs := map[string]string{}
	resolve := func(path string) (Pin, error) {
		path = absArg(cwd, path)
		if recorded != nil {
			canonical := recorded.Bindings[path]
			q := Pin{Path: canonical, SHA256: recorded.Inputs[canonical]}
			snap, ok := recorded.Snapshots[canonical]
			if !ok || snap.SHA256 != q.SHA256 || canonical == "" {
				return q, fmt.Errorf("Go tool input binding omitted")
			}
			return q, verifyPin(snap)
		}
		q, e := closedPin(path)
		if e == nil && bindings != nil {
			bindings[path] = q.Path
		}
		return q, e
	}
	private := func(path string) (Pin, error) {
		q, e := resolve(path)
		if e != nil || !under(p.Options.WorkRoot, absArg(cwd, path)) || !under(p.Options.WorkRoot, q.Path) {
			return Pin{}, fmt.Errorf("Go tool input outside private work")
		}
		return q, nil
	}
	body := func(path string) ([]byte, error) {
		q, e := resolve(path)
		if e != nil {
			return nil, e
		}
		if recorded != nil {
			return os.ReadFile(recorded.Snapshots[q.Path].Path)
		}
		return os.ReadFile(q.Path)
	}
	output := func(path string) bool {
		path = absArg(cwd, path)
		if recorded != nil {
			return under(p.Options.WorkRoot, path) && under(p.Options.WorkRoot, recorded.Outputs[path])
		}
		if !privateOutput(p, cwd, path) {
			return false
		}
		dir, _ := filepath.EvalSymlinks(filepath.Dir(path))
		if outputs != nil {
			outputs[path] = filepath.Join(dir, filepath.Base(path))
		}
		return true
	}
	if name == "pack" {
		if len(args) < 3 || args[0] != "r" || !output(args[1]) {
			return nil, fmt.Errorf("unadmitted Go pack operation/output")
		}
		// r reads the existing archive as well as the supplied objects.
		archive := absArg(cwd, args[1])
		_, existing := func() (string, bool) {
			if recorded != nil {
				v, ok := recorded.Bindings[archive]
				return v, ok
			}
			_, e := os.Lstat(archive)
			return "", e == nil
		}()
		if existing {
			q, e := private(archive)
			if e != nil {
				return nil, e
			}
			inputs[q.Path] = q.SHA256
		}
		for _, file := range args[2:] {
			q, e := private(file)
			if e != nil {
				return nil, e
			}
			inputs[q.Path] = q.SHA256
		}
		return inputs, nil
	}
	values := map[string]bool{"-o": true, "-p": true, "-trimpath": true, "-buildid": true, "-goversion": true, "-importcfg": true, "-embedcfg": true, "-asmhdr": true, "-symabis": true, "-I": true, "-D": true, "-X": true}
	single := []string{"-std", "-complete", "-nolocalimports", "-pack", "-gensymabis", "-shared", "-dynlink", "-w", "-s", "-v", "-linkshared"}
	inline := []string{"-lang=", "-c=", "-d=", "-buildmode=exe", "-linkmode=external", "-compressdwarf=false", "-extld=", "-buildid=", "-goversion=", "-p=", "-trimpath=", "-o=", "-X="}
	for i := 0; i < len(args); i++ {
		a := args[i]
		configKind := ""
		if strings.HasPrefix(a, "-") {
			if values[a] {
				if i+1 == len(args) {
					return nil, fmt.Errorf("missing Go tool flag value")
				}
				key := a
				i++
				a = args[i]
				if (key == "-o" || key == "-asmhdr") && !output(a) {
					return nil, fmt.Errorf("Go tool output outside private work: %s", a)
				}
				if key == "-o" || key == "-asmhdr" {
					continue
				}
				if key == "-importcfg" || key == "-embedcfg" || key == "-symabis" {
					configKind = key
					if _, e := private(a); e != nil {
						return nil, fmt.Errorf("Go helper input outside private work")
					}
					if key == "-embedcfg" {
						data, e := body(a)
						if e != nil {
							return nil, e
						}
						var cfg struct {
							Patterns map[string][]string
							Files    map[string]string
						}
						if e = json.Unmarshal(data, &cfg); e != nil || cfg.Patterns == nil || cfg.Files == nil {
							return nil, fmt.Errorf("invalid Go embedcfg")
						}
						for _, file := range cfg.Files {
							q, e := resolve(file)
							if e != nil || p.Inputs[q.Path] != q.SHA256 {
								return nil, fmt.Errorf("Go embed payload outside selected inputs")
							}
							inputs[q.Path] = q.SHA256
						}
					}
				} else if key == "-I" {
					dir, e := filepath.EvalSymlinks(absArg(cwd, a))
					if e != nil || dir != filepath.Join(p.Options.GoRoot, "pkg/include") && !under(p.Options.WorkRoot, dir) {
						return nil, fmt.Errorf("Go include directory outside selected/private roots")
					}
					continue
				} else {
					continue
				}
			} else if slices.Contains(single, a) {
				continue
			} else {
				known := false
				for _, prefix := range inline {
					if strings.HasPrefix(a, prefix) {
						known = true
						break
					}
				}
				if !known {
					return nil, fmt.Errorf("unknown Go %s argument %s", name, a)
				}
				if strings.HasPrefix(a, "-extld=") {
					if name != "link" || !selectedExternalLinker(strings.TrimPrefix(a, "-extld="), driver) {
						return nil, fmt.Errorf("foreign external linker before Go tool child")
					}
					continue
				}
				if strings.HasPrefix(a, "-o=") && !output(strings.TrimPrefix(a, "-o=")) {
					return nil, fmt.Errorf("Go tool output outside private work")
				}
				continue
			}
		}
		path := absArg(cwd, a)
		q, e := resolve(path)
		if e != nil {
			return nil, e
		}
		if p.Inputs[q.Path] != q.SHA256 && !under(p.Options.WorkRoot, q.Path) {
			return nil, fmt.Errorf("Go tool input outside selected source/private work: %s", q.Path)
		}
		inputs[q.Path] = q.SHA256
		if configKind == "-importcfg" {
			body, e := body(path)
			if e != nil {
				return nil, e
			}
			for _, line := range strings.Split(string(body), "\n") {
				if !strings.HasPrefix(line, "packagefile ") {
					continue
				}
				_, file, ok := strings.Cut(strings.TrimPrefix(line, "packagefile "), "=")
				if !ok {
					return nil, fmt.Errorf("invalid importcfg")
				}
				q, e := private(file)
				if e != nil || !under(p.Options.WorkRoot, q.Path) {
					return nil, fmt.Errorf("Go import archive outside private work")
				}
				inputs[q.Path] = q.SHA256
			}
		}
	}
	return inputs, nil
}

func InterposeTool(session Pin, args []string) (err error) {
	s, p, _, e := load(session)
	if e != nil {
		return e
	}
	if s.Phase != "build" {
		return reject(s, args, fmt.Errorf("Go tools unavailable in metadata query phase"))
	}
	if len(args) > 0 && args[0] == s.Driver.Path {
		w := compilerIdentityWrapper{"nanopore-h08-wrapped-compiler-id/v1", s.Driver, slices.Clone(args), map[string]string{"GOROOT": os.Getenv("GOROOT"), "CGO_LDFLAGS": os.Getenv("CGO_LDFLAGS")}}
		if e = validateCompilerIdentity(p, s, w); e != nil {
			return reject(s, args, e)
		}
		stdin, e := compilerStdin(args[1:], os.Stdin)
		if e != nil {
			return reject(s, args, e)
		}
		_, e = interposeCCWrapped(s, p, args[1:], stdin, nil, true, nativeBoundary, &w)
		return e
	}
	d, e := opDir(s, "tool")
	if e != nil {
		return e
	}
	op := operation{Schema: "nanopore-h08-cgo-operation/v1", Status: "rejected", OriginalArgs: slices.Clone(args), Expected: []int{0}, Inputs: operationInputs(p, s), Bindings: map[string]string{}, Outputs: map[string]string{}}
	defer func() {
		if err != nil {
			op.Error = err.Error()
			err = reject(s, args, err)
		}
		err2 := write(filepath.Join(d, "operation.json"), op)
		if err == nil {
			err = err2
		}
	}()
	if len(args) < 2 {
		return fmt.Errorf("Go tool executable/arguments missing")
	}
	tool, e := closedPin(args[0])
	if e != nil || p.GoTools[tool.Path] != tool.SHA256 {
		return fmt.Errorf("unselected Go tool")
	}
	name := filepath.Base(tool.Path)
	actual := args[1:]
	op.Role = "go-tool-" + name
	op.Inputs[tool.Path] = tool.SHA256
	metadata := exactArgs(actual, []string{"-V=full"})
	if metadata {
		op.Role = "go-tool-id"
	} else if !slices.Contains([]string{"cgo", "compile", "asm", "pack", "link"}, name) {
		return fmt.Errorf("Go tool not used by fixed H08 build")
	}
	if e = checkPins(p.Inputs); e != nil {
		return e
	}
	cwd, e := os.Getwd()
	if e != nil {
		return e
	}
	op.Directory = cwd
	op.Package = os.Getenv("TOOLEXEC_IMPORTPATH")
	sp := session
	var lease *Lease
	var generatedPackage Package
	if !metadata && name == "cgo" {
		derived, e := derivedRead(s)
		if e != nil {
			return e
		}
		l, e := cgoLease(p, actual, cwd, derived)
		if e != nil {
			return e
		}
		l.Tool = tool
		lease = &l
		generatedPackage, e = packageByName(p, l.Package)
		if e != nil {
			return e
		}
		maps.Copy(op.Inputs, l.Inputs)
	} else if !metadata {
		pkg := os.Getenv("TOOLEXEC_IMPORTPATH")
		if !knownPackage(p, pkg) {
			return fmt.Errorf("Go tool request outside selected package graph: %s", pkg)
		}
		if name == "link" {
			lease = &Lease{Schema: "nanopore-h08-go-tool-lease/v1", Kind: "link", Package: pkg, Directory: cwd, Tool: tool, Args: slices.Clone(actual), Inputs: map[string]string{}}
		}
		value := ""
		extra := map[string]string{}
		if name == "link" {
			if op.Selection, e = closedPin(filepath.Join(s.Directory, "go-selection.json")); e != nil {
				return e
			}
			g, selected, e := readGoSelection(p, s, sp, op.Selection)
			if e != nil {
				return e
			}
			value, extra = g.DefaultGODEBUG, selected
		}
		if e = validateLinkAssignments(p, name, actual, value); e != nil {
			return e
		}
		inputs, e := toolInputsEvidence(p, s.Driver, name, actual, cwd, nil, op.Bindings, op.Outputs)
		if e != nil {
			return e
		}
		maps.Copy(inputs, extra)
		maps.Copy(op.Inputs, inputs)
		if name == "compile" {
			if e := validateMainInputs(p, op.Package, actual, inputs); e != nil {
				return e
			}
		}
		if lease != nil {
			maps.Copy(lease.Inputs, inputs)
		}
	}
	op.ToolOverrides = map[string]string{"GOROOT": os.Getenv("GOROOT"), "CGO_LDFLAGS": os.Getenv("CGO_LDFLAGS")}
	env, e := toolEnvironment(p, s, sp, name, metadata, lease, op.ToolOverrides)
	if e != nil {
		return e
	}
	if lease != nil {
		op.Lease = lease
		path := filepath.Join(d, "lease.json")
		if e = write(path, *lease); e != nil {
			return e
		}
		q, e := closedPin(path)
		if e != nil {
			return e
		}
		env = append(env, "H08_CGO_LEASE="+q.Path, "H08_CGO_LEASE_SHA="+q.SHA256)
	}
	op.Snapshots, e = copySnapshots(filepath.Join(s.Directory, "inputs"), op.Inputs)
	if e != nil {
		return e
	}
	if e = write(filepath.Join(d, "before.json"), op.Inputs); e != nil {
		return e
	}
	op.Before, e = closedPin(filepath.Join(d, "before.json"))
	if e != nil {
		return e
	}
	// Revalidate paths after snapshot/receipt work and immediately before the
	// selected tool. A replacement final symlink must not reach its open call.
	if !metadata && name != "cgo" {
		inputs, e := toolInputs(p, s.Driver, name, actual, cwd)
		if e != nil {
			return e
		}
		for path, hash := range inputs {
			if op.Inputs[path] != hash {
				return fmt.Errorf("Go tool input binding changed before child")
			}
		}
	} else if lease != nil {
		derived, e := derivedRead(s)
		if e != nil {
			return e
		}
		if _, e = cgoLease(p, actual, cwd, derived); e != nil {
			return e
		}
	}
	cmd, proc, e := runProgram(context.Background(), d, cwd, args, env, nil, true, false)
	op.Commands = append(op.Commands, cmd)
	op.Processes = append(op.Processes, proc)
	if e != nil {
		return e
	}
	after := maps.Clone(op.Inputs)
	if name == "pack" && len(actual) > 1 {
		delete(after, absArg(cwd, actual[1]))
	}
	if e = checkPins(after); e != nil {
		return e
	}
	if lease != nil && name == "cgo" {
		producer, e := closedPin(filepath.Join(d, "command.json"))
		if e != nil {
			return e
		}
		stage := "translation"
		if lease.Kind == "dynimport" {
			stage = "dynimport"
		}
		g, e := CaptureGenerated(generatedPackage, stage, p.Options.WorkRoot, lease.ObjectDir, filepath.Join(s.Directory, "inputs"), producer)
		if e != nil {
			return e
		}
		file := strings.ReplaceAll(lease.Package, "/", "_") + "-" + stage + ".json"
		if e = write(filepath.Join(s.Directory, "derived", file), g); e != nil {
			return e
		}
	}
	if out := flagValue(actual, "-o"); out != "" {
		if !privateOutput(p, cwd, out) {
			return fmt.Errorf("Go tool output path changed")
		}
		q, e := closedPin(absArg(cwd, out))
		if e != nil {
			return e
		}
		op.Output = q
		frozen, e := copySnapshots(filepath.Join(s.Directory, "inputs"), map[string]string{q.Path: q.SHA256})
		if e != nil {
			return e
		}
		op.OutputSnapshot = frozen[q.Path]
	}
	op.Status = "accepted"
	return nil
}
