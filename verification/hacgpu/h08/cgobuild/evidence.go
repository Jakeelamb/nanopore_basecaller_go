package cgobuild

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func generatedProducer(g GeneratedSet, operations map[string]operation) error {
	parent, ok := operations[filepath.Dir(g.Producer.Path)]
	if !ok || parent.Role != "go-tool-cgo" || parent.Lease == nil || parent.Lease.Package != g.Package || len(parent.Commands) != 1 || parent.Commands[0].Exit != 0 || filepath.Base(g.Producer.Path) != "command.json" {
		return fmt.Errorf("generated set has no exact successful Cgo producer")
	}
	var command runner.CommandReceipt
	if e := read(g.Producer, &command); e != nil || !reflect.DeepEqual(command, parent.Commands[0]) {
		return fmt.Errorf("generated producer raw command differs")
	}
	stage := "translation"
	if parent.Lease.Kind == "dynimport" {
		stage = "dynimport"
	}
	if stage != g.Stage {
		return fmt.Errorf("generated stage/producer differs")
	}
	for _, q := range g.Files {
		if filepath.Dir(q.Path) != parent.Lease.ObjectDir {
			return fmt.Errorf("generated set outside producer object directory")
		}
	}
	return nil
}

func recordedDependencies(pin Pin, op operation) ([]string, error) {
	if e := verifyPin(pin); e != nil {
		return nil, e
	}
	body, e := os.ReadFile(pin.Path)
	if e != nil {
		return nil, e
	}
	names, e := dependencyNames(string(body), op.Directory)
	if e != nil {
		return nil, e
	}
	var out []string
	for _, name := range names {
		canonical := op.DependencyBindings[name]
		if canonical == "" || op.Inputs[canonical] == "" {
			return nil, fmt.Errorf("dependency lacks before-child snapshot binding")
		}
		out = append(out, canonical)
	}
	slices.Sort(out)
	return slices.Compact(out), nil
}

func beforeCommand(c runner.CommandReceipt, args, env []string, inputs map[string]string, op operation) error {
	if !slices.Equal(c.Args, args) || !slices.Equal(c.Environment, env) {
		return fmt.Errorf("actual selected command/environment differs from request")
	}
	var actual runner.CommandReceipt
	q, e := closedPin(filepath.Join(filepath.Dir(c.Stdout), "command.json"))
	if e != nil {
		return e
	}
	if e = read(q, &actual); e != nil || !reflect.DeepEqual(actual, c) {
		return fmt.Errorf("command raw receipt binding differs")
	}
	q, e = closedPin(filepath.Join(filepath.Dir(c.Stdout), "before.json"))
	if e != nil {
		return e
	}
	var before map[string]string
	if e = read(q, &before); e != nil || !maps.Equal(before, inputs) {
		return fmt.Errorf("complete before-child input set differs")
	}
	for path, hash := range before {
		if op.Inputs[path] != hash || op.Snapshots[path].SHA256 != hash {
			return fmt.Errorf("before-child snapshot differs")
		}
	}
	return nil
}

func recordedLinkPaths(body string, isMap bool, op operation, resources attestedQueryResources) (map[string]bool, error) {
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
		canonical := line
		if selected, ok := resources.exactAnswerBindings[line]; ok {
			canonical = selected.Path
			if op.Inputs[canonical] != selected.SHA256 {
				return nil, fmt.Errorf("query resource differs from link input")
			}
			if bound := op.Request.Bindings[line]; bound != "" && bound != canonical {
				return nil, fmt.Errorf("conflicting query/request binding")
			}
		} else if bound := op.Request.Bindings[line]; bound != "" {
			canonical = bound
		} else {
			if !filepath.IsAbs(line) || filepath.Clean(line) != line {
				return nil, fmt.Errorf("unselected raw link spelling: %s", line)
			}
			canonical = absArg(op.Directory, line)
		}
		if op.Inputs[canonical] == "" || op.Snapshots[canonical].SHA256 != op.Inputs[canonical] {
			return nil, fmt.Errorf("link used input outside before-child snapshots")
		}
		out[canonical] = true
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("empty recorded link evidence")
	}
	return out, nil
}
func actualLinkObjects(op operation) (map[string]string, error) {
	required := maps.Clone(op.Request.Inputs)
	if op.Request.Link && (op.Request.Source.Path != "" || slices.Contains(op.Request.Args, "-")) {
		if len(op.Commands) != 3 || op.Commands[1].Exit != 0 {
			return nil, fmt.Errorf("split link lacks successful probe compile")
		}
		last := op.Commands[2]
		dir := filepath.Dir(filepath.Dir(last.Stdout))
		object := filepath.Join(dir, "probe.o")
		want := append(linkProbeArgs(op.Request.Args, object), "-Wl,-t", "-Wl,-Map="+filepath.Join(dir, "link.map"))
		if len(last.Args) == 0 || !slices.Equal(last.Args[1:], want) {
			return nil, fmt.Errorf("split link transformed argv differs")
		}
		if op.Inputs[object] == "" || op.Snapshots[object].SHA256 != op.Inputs[object] {
			return nil, fmt.Errorf("probe object before-link pin omitted")
		}
		delete(required, op.Request.Source.Path)
		required[object] = op.Inputs[object]
	}
	return required, nil
}

func recordedLinkEvidence(trace, body string, op operation, resources attestedQueryResources) error {
	a, e := recordedLinkPaths(trace, false, op, resources)
	if e != nil {
		return e
	}
	b, e := recordedLinkPaths(body, true, op, resources)
	if e != nil {
		return e
	}
	if !maps.Equal(a, b) {
		return fmt.Errorf("retained link trace/map differ")
	}
	required, e := actualLinkObjects(op)
	if e != nil {
		return e
	}
	if op.Request.Link && (op.Request.Source.Path != "" || slices.Contains(op.Request.Args, "-")) {
		if a[op.Request.Source.Path] || op.Stdin.Path != "" && a[op.Stdin.Path] {
			return fmt.Errorf("split link used compile source instead of object")
		}
	}
	for path := range required {
		if !a[path] {
			return fmt.Errorf("link evidence omitted requested object")
		}
	}
	return nil
}

func verifyNativeOperation(p Plan, s Session, op operation, dir string, derived []GeneratedSet, resources attestedQueryResources) error {
	if op.Request == nil || !filepath.IsAbs(op.Directory) {
		return fmt.Errorf("native request/cwd omitted")
	}
	lastExit := op.Commands[len(op.Commands)-1].Exit
	if op.Status == "accepted" && lastExit != 0 || op.Status == "accepted_expected_exit" && lastExit != 1 {
		return fmt.Errorf("accepted status/exit differ")
	}
	if op.Lease != nil {
		l := op.Lease
		if l.Schema != "nanopore-h08-go-tool-lease/v1" || p.GoTools[l.Tool.Path] != l.Tool.SHA256 || filepath.Base(l.Tool.Path) != l.Kind || l.Kind != "cgo" && l.Kind != "link" || !knownPackage(p, l.Package) || l.Directory != op.Directory {
			return fmt.Errorf("native request producer lease differs")
		}
	}
	var stdin []byte
	if op.Stdin.Path != "" {
		if e := verifyPin(op.Stdin); e != nil {
			return e
		}
		var e error
		stdin, e = os.ReadFile(op.Stdin.Path)
		if e != nil {
			return e
		}
		if op.Stdin.Path != filepath.Join(dir, "stdin.c") {
			return fmt.Errorf("stdin evidence path differs")
		}
	}
	r, e := classifyRequest(p, op.Directory, op.OriginalArgs, stdin, op.Lease, derived, &op)
	if e != nil || !reflect.DeepEqual(r, *op.Request) || op.Role != r.Role || !slices.Equal(op.Expected, r.Expected) {
		return fmt.Errorf("recorded native request classification differs: %v", e)
	}
	inputs := operationInputs(p, s)
	maps.Copy(inputs, r.Inputs)
	if op.Selection.Path != "" || op.ToolOverrides != nil {
		return fmt.Errorf("unexpected Go tool fields on native operation")
	}
	if op.Wrapper.Path != "" {
		var wrapper compilerIdentityWrapper
		if op.Wrapper.Path != filepath.Join(dir, "wrapper.json") || r.Role != "metadata" {
			return fmt.Errorf("compiler identity wrapper domain differs")
		}
		if e = read(op.Wrapper, &wrapper); e != nil {
			return e
		}
		if e = validateCompilerIdentity(p, s, wrapper); e != nil || !slices.Equal(wrapper.Args[1:], op.OriginalArgs) {
			return fmt.Errorf("compiler identity wrapper request differs: %v", e)
		}
		inputs[op.Wrapper.Path] = op.Wrapper.SHA256
		inputs[wrapper.Driver.Path] = wrapper.Driver.SHA256
	}
	if op.Lease != nil {
		maps.Copy(inputs, op.Lease.Inputs)
	}
	if r.Role == "metadata" {
		if op.Stdin.Path != "" {
			return fmt.Errorf("unexpected retained metadata stdin")
		}
	} else if slices.Contains(r.Args, "-") {
		if op.Stdin.Path == "" {
			return fmt.Errorf("compiler stdin not retained")
		}
		inputs[op.Stdin.Path] = op.Stdin.SHA256
	} else if op.Stdin.Path != "" {
		return fmt.Errorf("unexpected retained stdin")
	}
	env := append(environment(p), "TERM=dumb")
	i := 0
	step := func(label string, args []string, expected []int) error {
		if i >= len(op.Commands) {
			return fmt.Errorf("native command step omitted")
		}
		c := op.Commands[i]
		i++
		if filepath.Dir(c.Stdout) != filepath.Join(dir, label) || !slices.Contains(expected, c.Exit) {
			return fmt.Errorf("native step identity/exit differs")
		}
		return beforeCommand(c, append([]string{p.CC.Path}, args...), env, inputs, op)
	}
	if r.Role == "metadata" {
		if op.Directory != p.Options.Root || op.Lease != nil || len(op.Commands) != 1 || !maps.Equal(inputs, op.Inputs) {
			return fmt.Errorf("metadata phase boundary differs")
		}
		if e = step("actual", r.Args, []int{0}); e != nil {
			return e
		}
		for _, q := range p.Queries {
			if slices.Equal(r.Args, q.Argv[1:]) {
				return verifyQueryAnswer(q, op.Commands[0], op.Directory)
			}
		}
		return fmt.Errorf("unselected metadata query")
	}
	if r.Source.Path != "" || slices.Contains(r.Args, "-") {
		if e = step("preflight", preprocessorArgs(r.Args, filepath.Join(dir, "before.d")), []int{0, 1}); e != nil {
			return e
		}
		if op.PreflightDep.Path != "" {
			if op.PreflightDep.Path != filepath.Join(dir, "before.d") || op.Commands[0].Exit != 0 {
				return fmt.Errorf("preflight dependency identity differs")
			}
			deps, e := recordedDependencies(op.PreflightDep, op)
			if e != nil {
				return e
			}
			if !slices.Equal(deps, op.Dependencies) || r.Source.Path != "" && !slices.Contains(deps, r.Source.Path) {
				return fmt.Errorf("selected source/preflight dependencies differ")
			}
			for _, path := range deps {
				if path != r.Source.Path {
					if e := checkHeader(p, path, derived); e != nil {
						return e
					}
				}
				inputs[path] = op.Inputs[path]
			}
		} else if r.Role != "go-flag-probe" && r.Role != "go-link-probe" || op.DepUnavailable != "source-defined unsupported Go flag probe; retained preflight/actual diagnostics" {
			return fmt.Errorf("preflight equality proof omitted")
		}
	}
	actualDep := filepath.Join(dir, "actual.d")
	actualArgs := append(slices.Clone(r.Args), "-MD", "-MF", actualDep)
	if r.Link && (r.Source.Path != "" || slices.Contains(r.Args, "-")) {
		object := filepath.Join(dir, "probe.o")
		if e = step("probe-compile", compileProbeArgs(r.Args, object, actualDep), r.Expected); e != nil {
			return e
		}
		if op.Commands[i-1].Exit != 0 {
			if op.DepUnavailable == "" || i != len(op.Commands) || !maps.Equal(inputs, op.Inputs) {
				return fmt.Errorf("failed probe operation incomplete")
			}
			return nil
		}
		if op.PreflightDep.Path == "" || op.DepUnavailable != "" {
			return fmt.Errorf("successful probe compile lacks preflight equality proof")
		}
		if op.Inputs[object] == "" || op.Snapshots[object].SHA256 != op.Inputs[object] {
			return fmt.Errorf("probe object before-link pin omitted")
		}
		if e = verifyPin(op.Snapshots[object]); e != nil {
			return e
		}
		inputs[object] = op.Inputs[object]
		actualArgs = linkProbeArgs(r.Args, object)
	}
	if r.Link {
		actualArgs = append(actualArgs, "-Wl,-t", "-Wl,-Map="+filepath.Join(dir, "link.map"))
	}
	if e = step("actual", actualArgs, r.Expected); e != nil {
		return e
	}
	if i != len(op.Commands) || !maps.Equal(inputs, op.Inputs) {
		return fmt.Errorf("extra command/input proof differs")
	}
	last := op.Commands[i-1]
	if op.ActualDep.Path != "" {
		if op.ActualDep.Path != actualDep || op.PreflightDep.Path == "" {
			return fmt.Errorf("actual depfile identity differs")
		}
		deps, e := recordedDependencies(op.ActualDep, op)
		if e != nil {
			return e
		}
		if !slices.Equal(deps, op.Dependencies) {
			return fmt.Errorf("actual -MD differs from retained -M")
		}
	} else if !r.Link && (last.Exit == 0 || r.Role != "cgo-diagnostics" && r.Role != "go-flag-probe") {
		return fmt.Errorf("actual dependency evidence omitted")
	}
	if op.LinkMap.Path != "" {
		if !r.Link || op.LinkMap.Path != filepath.Join(dir, "link.map") {
			return fmt.Errorf("link map path differs")
		}
		trace, e := os.ReadFile(last.Stdout)
		if e != nil {
			return e
		}
		body, e := os.ReadFile(op.LinkMap.Path)
		if e != nil {
			return e
		}
		if e = recordedLinkEvidence(string(trace), string(body), op, resources); e != nil {
			return e
		}
	} else if r.Link && last.Exit == 0 {
		return fmt.Errorf("successful link evidence omitted")
	}
	if last.Exit == 0 && r.Output != "" && r.Role != "cgo-defines" {
		if op.Output.Path != r.OutputTarget {
			return fmt.Errorf("native output differs from exact -o target")
		}
	} else if op.Output.Path != "" {
		return fmt.Errorf("unexpected native output evidence")
	}
	return nil
}

func verifyToolOperation(p Plan, s Session, sp Pin, op operation, dir string, derived []GeneratedSet) error {
	if op.Wrapper.Path != "" {
		return fmt.Errorf("native wrapper cannot authorize a Go tool")
	}
	if len(op.Commands) != 1 || len(op.OriginalArgs) < 2 || !filepath.IsAbs(op.Directory) {
		return fmt.Errorf("Go tool operation incomplete")
	}
	if filepath.Dir(op.Commands[0].Stdout) != dir || op.Status != "accepted" || !slices.Equal(op.Expected, []int{0}) {
		return fmt.Errorf("Go tool raw operation binding differs")
	}
	args := op.OriginalArgs
	tool := Pin{Path: args[0], SHA256: p.GoTools[args[0]]}
	name := filepath.Base(tool.Path)
	if tool.SHA256 == "" {
		return fmt.Errorf("unselected recorded Go tool")
	}
	inputs := operationInputs(p, s)
	inputs[tool.Path] = tool.SHA256
	metadata := slices.Equal(args[1:], []string{"-V=full"})
	if (metadata || name != "link") && op.Selection.Path != "" {
		return fmt.Errorf("unexpected Go selection on another tool")
	}
	if metadata {
		if op.Role != "go-tool-id" || op.Lease != nil {
			return fmt.Errorf("Go tool metadata role differs")
		}
	} else {
		if op.Role != "go-tool-"+name || !slices.Contains([]string{"cgo", "compile", "asm", "pack", "link"}, name) {
			return fmt.Errorf("unselected Go tool role")
		}
		if name == "cgo" {
			l := op.Lease
			if l == nil || l.Tool != tool || !slices.Equal(l.Args, args[1:]) || l.Directory != op.Directory || l.Schema != "nanopore-h08-go-tool-lease/v1" {
				return fmt.Errorf("Cgo producer lease differs")
			}
			actual, e := cgoLease(p, args[1:], op.Directory, derived)
			if e != nil {
				return e
			}
			actual.Tool = tool
			if !reflect.DeepEqual(actual, *l) {
				return fmt.Errorf("Cgo lease/source recipe differs")
			}
			maps.Copy(inputs, l.Inputs)
		} else {
			if !knownPackage(p, op.Package) {
				return fmt.Errorf("Go tool outside actual package graph")
			}
			value := ""
			extra := map[string]string{}
			if name == "link" {
				if op.Selection.Path == "" {
					return fmt.Errorf("Go linker selection omitted")
				}
				g, selected, e := readGoSelection(p, s, sp, op.Selection)
				if e != nil {
					return e
				}
				value, extra = g.DefaultGODEBUG, selected
			}
			if e := validateLinkAssignments(p, name, args[1:], value); e != nil {
				return e
			}
			a, e := toolInputsEvidence(p, s.Driver, name, args[1:], op.Directory, &op, nil, nil)
			if e != nil {
				return e
			}
			maps.Copy(a, extra)
			maps.Copy(inputs, a)
			if name == "compile" {
				if e = validateMainInputs(p, op.Package, args[1:], a); e != nil {
					return e
				}
			}
			if name == "link" {
				if op.Lease == nil || op.Lease.Kind != "link" || op.Lease.Tool != tool || !slices.Equal(op.Lease.Args, args[1:]) || !maps.Equal(op.Lease.Inputs, a) {
					return fmt.Errorf("Go link producer lease differs")
				}
			} else if op.Lease != nil {
				return fmt.Errorf("unexpected Go producer lease")
			}
		}
	}
	if !maps.Equal(inputs, op.Inputs) {
		return fmt.Errorf("Go tool before-child set differs")
	}
	env, e := toolEnvironment(p, s, sp, name, metadata, op.Lease, op.ToolOverrides)
	if e != nil {
		return e
	}
	if op.Lease != nil {
		q, e := closedPin(filepath.Join(dir, "lease.json"))
		if e != nil {
			return e
		}
		var l Lease
		if e = read(q, &l); e != nil || !reflect.DeepEqual(l, *op.Lease) {
			return fmt.Errorf("Go lease artifact binding differs")
		}
		env = append(env, "H08_CGO_LEASE="+q.Path, "H08_CGO_LEASE_SHA="+q.SHA256)
	}
	if e := beforeCommand(op.Commands[0], args, env, inputs, op); e != nil {
		return e
	}
	if output := flagValue(args[1:], "-o"); output != "" && op.Output.Path != op.Outputs[absArg(op.Directory, output)] {
		return fmt.Errorf("Go output differs from exact -o target")
	}
	return nil
}
