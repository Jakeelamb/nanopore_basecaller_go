package cgobuild

import (
	"context"
	"debug/elf"
	"errors"
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"unicode"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func nativeBoundary(p Plan) error {
	if e := ValidateSpecification(p); e != nil {
		return e
	}
	return errors.Join(p.Compiler.Check(), checkPins(p.Inputs))
}
func expectedError(e error, code int, want []int) error {
	if !slices.Contains(want, code) {
		return fmt.Errorf("native exit %d outside source-defined profile %v: %w", code, want, e)
	}
	if e == nil {
		return nil
	}
	if multi, ok := e.(interface{ Unwrap() []error }); ok {
		for _, child := range multi.Unwrap() {
			if x := expectedError(child, code, want); x != nil {
				return x
			}
		}
		return nil
	}
	if exit, ok := e.(*exec.ExitError); ok && exit.ExitCode() == code {
		return nil
	}
	return e
}
func dependencyNames(body, cwd string) ([]string, error) {
	body = strings.ReplaceAll(strings.ReplaceAll(body, "\\\r\n", ""), "\\\n", "")
	colon, escaped := -1, false
	for i, c := range body {
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' {
			escaped = true
			continue
		}
		if c == ':' {
			colon = i
			break
		}
	}
	if colon < 1 {
		return nil, fmt.Errorf("native dependency target omitted")
	}
	var token strings.Builder
	var names []string
	flush := func() {
		if token.Len() > 0 {
			names = append(names, absArg(cwd, token.String()))
			token.Reset()
		}
	}
	escaped = false
	for _, c := range body[colon+1:] {
		if escaped {
			token.WriteRune(c)
			escaped = false
		} else if c == '\\' {
			escaped = true
		} else if unicode.IsSpace(c) {
			flush()
		} else {
			token.WriteRune(c)
		}
	}
	if escaped {
		return nil, fmt.Errorf("unfinished dependency escape")
	}
	flush()
	slices.Sort(names)
	return slices.Compact(names), nil
}
func parseDep(path, cwd string, bindings map[string]string) ([]string, error) {
	body, e := os.ReadFile(path)
	if e != nil {
		return nil, e
	}
	names, e := dependencyNames(string(body), cwd)
	if e != nil {
		return nil, e
	}
	var out []string
	for _, name := range names {
		actual, e := filepath.EvalSymlinks(name)
		if e != nil {
			return nil, e
		}
		bindings[name] = actual
		out = append(out, actual)
	}
	slices.Sort(out)
	return slices.Compact(out), nil
}
func checkHeader(p Plan, path string, derived []GeneratedSet) error {
	lib := filepath.Dir(toolPin(p.Compiler, "cc1").Path)
	if under("/usr/include", path) || under(filepath.Join(lib, "include"), path) || under(filepath.Join(lib, "include-fixed"), path) {
		return nil
	}
	if p.Inputs[path] != "" {
		return nil
	}
	for _, g := range derived {
		for _, q := range g.Files {
			if path == q.Path {
				return CheckGeneratedLive(g)
			}
		}
	}
	return fmt.Errorf("preflight selected header outside fixed source/sysroot closure: %s", path)
}
func preprocessorArgs(args []string, dep string) []string {
	out := []string{}
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "-o" {
			i++
			continue
		}
		if strings.HasPrefix(a, "-o") && len(a) > 2 {
			continue
		}
		if a == "-c" || a == "-E" || a == "-dM" {
			continue
		}
		out = append(out, a)
	}
	return append(out, "-M", "-MT", "native", "-MF", dep)
}
func compileProbeArgs(args []string, object, dep string) []string {
	out := []string{}
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "-o" {
			i++
			continue
		}
		if strings.HasPrefix(a, "-o") && len(a) > 2 {
			continue
		}
		if strings.HasPrefix(a, "-Wl,") || a == "-ldl" || a == "-lpthread" {
			continue
		}
		out = append(out, a)
	}
	return append(out, "-c", "-o", object, "-MD", "-MF", dep)
}
func linkProbeArgs(args []string, object string) []string {
	out := []string{}
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "-x" {
			i++
			continue
		}
		if a == "-" || filepath.Ext(a) == ".c" {
			continue
		}
		out = append(out, a)
	}
	return append(out, object)
}
func sameLinkEvidence(trace, linkMap, cwd string, inputs, objects map[string]string) error {
	if e := h07.VerifyLinkTrace(trace, cwd, inputs, objects); e != nil {
		return e
	}
	t, e := linkPathsLocal(trace, cwd, false)
	if e != nil {
		return e
	}
	m, e := linkPathsLocal(linkMap, cwd, true)
	if e != nil {
		return e
	}
	if !maps.Equal(t, m) {
		return fmt.Errorf("link trace/map inputs differ")
	}
	return nil
}
func linkPathsLocal(body, cwd string, isMap bool) (map[string]bool, error) {
	out := map[string]bool{}
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if isMap {
			if !strings.HasPrefix(line, "LOAD ") {
				continue
			}
			line = strings.TrimPrefix(line, "LOAD ")
		}
		if line == "" {
			continue
		}
		q, e := closedPin(absArg(cwd, line))
		if e != nil {
			return nil, e
		}
		out[q.Path] = true
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("empty link input evidence")
	}
	return out, nil
}

func verifyQueryAnswer(q Query, command runner.CommandReceipt, cwd string) error {
	if q.Expected.Path == "" {
		return nil
	}
	body, e := os.ReadFile(command.Stdout)
	if e != nil {
		return e
	}
	name := strings.TrimSpace(string(body))
	if !filepath.IsAbs(name) {
		if strings.Contains(name, "/") {
			name = filepath.Join(cwd, name)
		} else if strings.HasPrefix(q.Argv[1], "-print-prog-name=") {
			name = filepath.Join("/usr/bin", name)
		} else {
			return fmt.Errorf("required CC resource query unresolved")
		}
	}
	actual, e := closedPin(name)
	if e != nil || actual != q.Expected {
		return fmt.Errorf("CC selected undeclared resource: %s", name)
	}
	return nil
}

// interposeCC has one production boundary. Tests replace only its compiler ELF
// checker with a sealed Go fake's closure; argument/source/resource rejection,
// snapshots, -M/MD, spawning and receipts remain the real implementation.
func interposeCC(s Session, p Plan, args []string, stdin []byte, lease *Lease, forward bool, boundary func(Plan) error) (op operation, err error) {
	return interposeCCWrapped(s, p, args, stdin, lease, forward, boundary, nil)
}

func interposeCCWrapped(s Session, p Plan, args []string, stdin []byte, lease *Lease, forward bool, boundary func(Plan) error, wrapper *compilerIdentityWrapper) (op operation, err error) {
	d, e := opDir(s, "cc")
	if e != nil {
		return op, e
	}
	op = operation{Schema: "nanopore-h08-cgo-operation/v1", Status: "rejected", OriginalArgs: slices.Clone(args), Inputs: operationInputs(p, s), Snapshots: map[string]Pin{}, DependencyBindings: map[string]string{}}
	defer func() {
		if err != nil {
			op.Error = err.Error()
			err = errors.Join(err, reject(s, args, err))
		}
		err = errors.Join(err, write(filepath.Join(d, "operation.json"), op))
	}()
	if err = ValidateSpecification(p); err != nil {
		return op, err
	}
	derived, e := derivedRead(s)
	if e != nil {
		return op, e
	}
	cwd, e := os.Getwd()
	if e != nil {
		return op, e
	}
	r, e := classifyCC(p, cwd, args, stdin, lease, derived)
	if e != nil {
		return op, e
	}
	op.Role = r.Role
	op.Directory, op.Lease, op.Request = cwd, lease, &r
	op.Expected = r.Expected
	if wrapper != nil {
		if e = validateCompilerIdentity(p, s, *wrapper); e != nil {
			return op, e
		}
		if r.Role != "metadata" || !slices.Equal(args, wrapper.Args[1:]) {
			return op, fmt.Errorf("wrapped compiler identity is not metadata")
		}
		path := filepath.Join(d, "wrapper.json")
		if e = write(path, *wrapper); e != nil {
			return op, e
		}
		if op.Wrapper, e = closedPin(path); e != nil {
			return op, e
		}
		op.Inputs[op.Wrapper.Path] = op.Wrapper.SHA256
		op.Inputs[wrapper.Driver.Path] = wrapper.Driver.SHA256
	}
	maps.Copy(op.Inputs, r.Inputs)
	if lease != nil {
		if err = verifyPin(lease.Tool); err != nil {
			return op, err
		}
		if err = checkPins(lease.Inputs); err != nil {
			return op, err
		}
		maps.Copy(op.Inputs, lease.Inputs)
	}
	if err = boundary(p); err != nil {
		return op, err
	}
	env := environment(p)
	env = append(env, "TERM=dumb")
	// All native invocations receive an immutable before-child input manifest.
	execute := func(label string, childArgs []string, data []byte, emit bool) (runner.CommandReceipt, error) {
		if e := boundary(p); e != nil {
			return runner.CommandReceipt{}, e
		}
		now, e := classifyCC(p, cwd, args, stdin, lease, derived)
		if e != nil || !reflect.DeepEqual(r, now) {
			return runner.CommandReceipt{}, fmt.Errorf("native request path/input binding changed before child: %v", e)
		}
		if e := checkPins(op.Inputs); e != nil {
			return runner.CommandReceipt{}, e
		}
		for _, g := range derived {
			if e := CheckGeneratedLive(g); e != nil {
				return runner.CommandReceipt{}, e
			}
		}
		snaps, e := copySnapshots(filepath.Join(s.Directory, "inputs"), op.Inputs)
		if e != nil {
			return runner.CommandReceipt{}, e
		}
		maps.Copy(op.Snapshots, snaps)
		cd := filepath.Join(d, label)
		if e = os.Mkdir(cd, 0700); e != nil {
			return runner.CommandReceipt{}, e
		}
		if e = write(filepath.Join(cd, "before.json"), op.Inputs); e != nil {
			return runner.CommandReceipt{}, e
		}
		q, e := closedPin(filepath.Join(cd, "before.json"))
		if e != nil {
			return runner.CommandReceipt{}, e
		}
		op.Before = q
		now, e = classifyCC(p, cwd, args, stdin, lease, derived)
		if e != nil || !reflect.DeepEqual(r, now) {
			return runner.CommandReceipt{}, fmt.Errorf("native request path/input binding changed at spawn: %v", e)
		}
		cmd, proc, e := runProgram(context.Background(), cd, r.Directory, append([]string{p.CC.Path}, childArgs...), env, data, emit, true)
		op.Commands = append(op.Commands, cmd)
		op.Processes = append(op.Processes, proc)
		if x := checkPins(op.Inputs); x != nil {
			return cmd, x
		}
		return cmd, e
	}
	if r.Role == "metadata" {
		cmd, e := execute("actual", r.Args, nil, forward)
		if e != nil {
			return op, e
		}
		for _, q := range p.Queries {
			if exactArgs(r.Args, q.Argv[1:]) {
				if e = verifyQueryAnswer(q, cmd, cwd); e != nil {
					return op, e
				}
			}
		}
		op.Status = "accepted"
		return op, nil
	}
	if slices.Contains(r.Args, "-") {
		path := filepath.Join(d, "stdin.c")
		if e = os.WriteFile(path, stdin, 0400); e != nil {
			return op, e
		}
		q, e := closedPin(path)
		if e != nil {
			return op, e
		}
		op.Inputs[q.Path] = q.SHA256
		op.Stdin = q
	}
	var deps []string
	if r.Source.Path != "" || slices.Contains(r.Args, "-") {
		before := filepath.Join(d, "before.d")
		pre, preErr := execute("preflight", preprocessorArgs(r.Args, before), stdin, false)
		deps, e = parseDep(before, cwd, op.DependencyBindings)
		if preErr != nil || e != nil {
			if r.Role != "go-flag-probe" && r.Role != "go-link-probe" {
				return op, fmt.Errorf("native preflight failed: %v / %v", preErr, e)
			}
			if x := expectedError(preErr, pre.Exit, []int{0, 1}); x != nil {
				return op, x
			}
			op.DepUnavailable = "source-defined unsupported Go flag probe; retained preflight/actual diagnostics"
		} else {
			if r.Source.Path != "" && !slices.Contains(deps, r.Source.Path) {
				return op, fmt.Errorf("preflight omitted selected native source")
			}
			op.Dependencies = deps
			op.PreflightDep, e = closedPin(before)
			if e != nil {
				return op, e
			}
			for _, path := range deps {
				if path != r.Source.Path {
					if e = checkHeader(p, path, derived); e != nil {
						return op, e
					}
				}
				q, e := closedPin(path)
				if e != nil {
					return op, e
				}
				op.Inputs[q.Path] = q.SHA256
			}
		}
	}
	actualDep := filepath.Join(d, "actual.d")
	actualArgs := append(slices.Clone(r.Args), "-MD", "-MF", actualDep)
	objects := map[string]string{}
	if r.Link && (r.Source.Path != "" || slices.Contains(r.Args, "-")) {
		object := filepath.Join(d, "probe.o")
		cmd, e := execute("probe-compile", compileProbeArgs(r.Args, object, actualDep), stdin, forward)
		if x := expectedError(e, cmd.Exit, r.Expected); x != nil {
			return op, x
		}
		if cmd.Exit != 0 {
			op.Status = "accepted_expected_exit"
			if op.DepUnavailable == "" {
				return op, fmt.Errorf("supported preflight but compiler probe failed before link")
			}
			return op, nil
		}
		q, e := closedPin(object)
		if e != nil {
			return op, e
		}
		objects[q.Path] = q.SHA256
		op.Inputs[q.Path] = q.SHA256
		actualArgs = linkProbeArgs(r.Args, object)
	}
	if r.Link {
		actualArgs = append(actualArgs, "-Wl,-t", "-Wl,-Map="+filepath.Join(d, "link.map"))
	}
	cmd, actualErr := execute("actual", actualArgs, stdin, forward)
	if e = expectedError(actualErr, cmd.Exit, r.Expected); e != nil {
		return op, e
	}
	if op.DepUnavailable != "" && cmd.Exit == 0 {
		return op, fmt.Errorf("failed preflight cannot qualify a successful actual compile")
	}
	if actual, e := parseDep(actualDep, cwd, op.DependencyBindings); e == nil {
		if op.DepUnavailable != "" || !slices.Equal(deps, actual) {
			return op, fmt.Errorf("actual native -MD differs from frozen -M")
		}
		op.ActualDep, e = closedPin(actualDep)
		if e != nil {
			return op, e
		}
	} else if r.Role == "cgo-diagnostics" && cmd.Exit == 1 {
		op.DepUnavailable = "source-defined Cgo diagnostic compile did not emit an actual depfile"
	} else if op.DepUnavailable == "" && !r.Link {
		return op, fmt.Errorf("actual native depfile absent: %v", e)
	}
	if r.Link {
		mapPath := filepath.Join(d, "link.map")
		trace, e := os.ReadFile(cmd.Stdout)
		if e != nil {
			return op, e
		}
		body, mapErr := os.ReadFile(mapPath)
		if mapErr == nil && len(strings.TrimSpace(string(trace))) > 0 {
			if e = sameLinkEvidence(string(trace), string(body), cwd, op.Inputs, objects); e != nil {
				return op, e
			}
			op.LinkMap, e = closedPin(mapPath)
			if e != nil {
				return op, e
			}
		} else if !slices.Contains(r.Expected, 1) || cmd.Exit != 1 {
			return op, fmt.Errorf("native link evidence absent")
		}
	}
	if cmd.Exit == 0 && r.Output != "" && r.Output != "/dev/null" && r.Role != "cgo-defines" {
		f, e := elf.Open(r.Output)
		if e != nil {
			return op, e
		}
		valid := f.Class == elf.ELFCLASS64 && f.Machine == elf.EM_X86_64
		if !r.Link {
			valid = valid && f.Type == elf.ET_REL
		}
		f.Close()
		if !valid {
			return op, fmt.Errorf("native output kind/architecture differs")
		}
		op.Output, e = closedPin(r.Output)
		if e != nil {
			return op, e
		}
		frozen, e := copySnapshots(filepath.Join(s.Directory, "inputs"), map[string]string{op.Output.Path: op.Output.SHA256})
		if e != nil {
			return op, e
		}
		op.OutputSnapshot = frozen[op.Output.Path]
	}
	op.Status = "accepted"
	if cmd.Exit != 0 {
		op.Status = "accepted_expected_exit"
	}
	return op, nil
}

func InterposeCC(session Pin, args []string) (int, error) {
	s, p, _, e := load(session)
	if e != nil {
		return 125, e
	}
	lease, e := readLease()
	if e != nil {
		return 125, reject(s, args, e)
	}
	if lease != nil {
		if p.GoTools[lease.Tool.Path] != lease.Tool.SHA256 || lease.Kind != "cgo" && lease.Kind != "link" || filepath.Base(lease.Tool.Path) != lease.Kind {
			return 125, reject(s, args, fmt.Errorf("native request has no selected active Go-tool producer"))
		}
	}
	stdin, e := compilerStdin(args, os.Stdin)
	if e != nil {
		return 125, reject(s, args, e)
	}
	op, e := interposeCC(s, p, args, stdin, lease, true, nativeBoundary)
	if e != nil {
		return 125, e
	}
	if len(op.Commands) == 0 {
		return 125, fmt.Errorf("native operation has no child")
	}
	return op.Commands[len(op.Commands)-1].Exit, nil
}
