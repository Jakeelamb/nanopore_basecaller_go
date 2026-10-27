package cgobuild

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"time"
)

func verifyActualSelection(p Plan, body []byte) error {
	d := json.NewDecoder(bytes.NewReader(body))
	seen := map[string]bool{}
	baseSeen := map[string]bool{}
	for {
		var a struct {
			ImportPath, Dir, ForTest, Name                                            string
			GoFiles, CgoFiles, CFiles, HFiles, SFiles, SysoFiles, EmbedFiles, Imports []string
			Error                                                                     *struct{ Err string }
		}
		e := d.Decode(&a)
		if e == io.EOF {
			break
		}
		if e != nil {
			return e
		}
		if a.Error != nil {
			return fmt.Errorf("actual Go selection failed: %s", a.Error.Err)
		}
		if a.ImportPath == "unsafe" || a.ImportPath == "C" {
			continue
		}
		if seen[a.ImportPath] {
			return fmt.Errorf("duplicate actual package selection")
		}
		seen[a.ImportPath] = true
		if p.Role == CPU13Tests && a.ImportPath == TensorPackage+".test" {
			if a.Dir != filepath.Join(p.Options.Root, "internal/tensor") || a.ForTest != "" || a.Name != "main" || len(a.GoFiles) != 1 || len(a.CgoFiles)+len(a.CFiles)+len(a.HFiles)+len(a.SFiles)+len(a.SysoFiles)+len(a.EmbedFiles) != 0 {
				return fmt.Errorf("generated CPU13 test main selection differs")
			}
			q, e := privateInput(p, a.GoFiles[0])
			if e != nil || !under(filepath.Join(p.Options.WorkRoot, "cache"), q.Path) || q.SHA256 != p.CPU13.MainSHA256 {
				return fmt.Errorf("generated list testmain differs from pinned Go template")
			}
			continue
		}
		if p.Role == CPU13Tests && a.ImportPath == TensorPackage+"_test"+testSuffix {
			tensor, ok := sourcePackage(p, TensorPackage)
			files := append(slices.Clone(tensor.XTestGo), OverlayName)
			slices.Sort(files)
			imports := append(slices.Clone(tensor.XTestImports), "os", "testing", Module+"/verification/hacgpu/h08")
			slices.Sort(imports)
			imports = slices.Compact(imports)
			actualImports := make([]string, len(a.Imports))
			for i, name := range a.Imports {
				actualImports[i] = strings.TrimSuffix(name, testSuffix)
			}
			slices.Sort(actualImports)
			if !slices.Equal(actualImports, imports) {
				return fmt.Errorf("external CPU13 source/overlay imports differ")
			}
			if !ok || a.Dir != filepath.Join(p.Options.Root, "internal/tensor") || a.ForTest != TensorPackage || a.Name != "tensor_test" || !slices.Equal(a.GoFiles, files) || len(a.CgoFiles)+len(a.CFiles)+len(a.HFiles)+len(a.SFiles)+len(a.SysoFiles)+len(a.EmbedFiles) != 0 {
				return fmt.Errorf("external CPU13 source/overlay selection differs")
			}
			continue
		}
		base, variant, ok := testDescription(p, a.ImportPath)
		if !ok || variant && a.ForTest != TensorPackage || !variant && a.ForTest != "" {
			return fmt.Errorf("unadmitted actual test variant")
		}
		var wanted *Package
		for i := range p.Packages {
			if p.Packages[i].ImportPath == base {
				wanted = &p.Packages[i]
			}
		}
		if wanted == nil || a.Dir != wanted.Directory {
			return fmt.Errorf("actual Cgo package selection differs: %s", a.ImportPath)
		}
		baseSeen[base] = true
		goFiles := wanted.Go
		if variant && base == TensorPackage {
			goFiles = append(slices.Clone(goFiles), wanted.TestGo...)
		}
		for _, pair := range [][2][]string{{a.GoFiles, goFiles}, {a.CgoFiles, wanted.Cgo}, {a.CFiles, wanted.C}, {a.HFiles, wanted.Headers}, {a.SFiles, wanted.Assembly}, {a.SysoFiles, wanted.Syso}, {a.EmbedFiles, wanted.Embed}} {
			if !slices.Equal(pair[0], pair[1]) {
				return fmt.Errorf("actual selected source members differ: %s", a.ImportPath)
			}
		}
	}
	if len(baseSeen) != len(p.Packages) {
		return fmt.Errorf("actual selected package graph incomplete")
	}
	if p.Role == CPU13Tests {
		if !seen[TensorPackage+".test"] || !seen[TensorPackage+"_test"+testSuffix] {
			return fmt.Errorf("CPU13 generated/external main packages omitted")
		}
		for _, name := range p.CPU13.TestCopies {
			if !seen[name+testSuffix] {
				return fmt.Errorf("CPU13 required test-copy package omitted")
			}
		}
	}
	return nil
}

func checkCommand(cArgs []string, expected []int, cStarted, cFinished string, stdout, stdoutSHA, stderr, stderrSHA string, exit int) error {
	a, e := time.Parse(time.RFC3339Nano, cStarted)
	b, f := time.Parse(time.RFC3339Nano, cFinished)
	if len(cArgs) == 0 || e != nil || f != nil || b.Before(a) || !slices.Contains(expected, exit) {
		return fmt.Errorf("command argv/clocks/exit incomplete")
	}
	if e = verifyPin(Pin{Path: stdout, SHA256: stdoutSHA}); e != nil {
		return e
	}
	return verifyPin(Pin{Path: stderr, SHA256: stderrSHA})
}
func VerifyQueries(pin, plan Pin) error {
	var r Receipt
	if e := read(pin, &r); e != nil {
		return e
	}
	if r.Schema != "nanopore-h08-cgo-execution/v1" || r.Status != "metadata_queries_passed" || r.Plan != plan || r.Error != "" || !r.RuntimeHeld {
		return fmt.Errorf("accepted query phase required")
	}
	var p Plan
	if e := read(plan, &p); e != nil {
		return e
	}
	return ValidateReceipt(p, r)
}

// ValidateReceipt rejects partial proof even when the outer Go process returned
// success. In particular a tolerated failed unknown probe cannot qualify it.
func ValidateReceipt(p Plan, r Receipt) error {
	if r.Schema != "nanopore-h08-cgo-execution/v1" || !r.RuntimeHeld || r.Error != "" || r.Status != "metadata_queries_passed" && r.Status != "compiled_unexecuted" {
		return fmt.Errorf("native phase receipt incomplete")
	}
	if r.Role != p.Role {
		return fmt.Errorf("receipt/plan build role differs")
	}
	var s Session
	if e := read(r.Session, &s); e != nil {
		return e
	}
	if s.Plan != r.Plan || s.Controller != r.Controller {
		return fmt.Errorf("phase/session binding differs")
	}
	if s.Schema != "nanopore-h08-cgo-session/v1" || !filepath.IsAbs(s.Directory) {
		return fmt.Errorf("phase session schema/directory differs")
	}
	if e := noRejections(s); e != nil {
		return e
	}
	if len(r.Operations) == 0 {
		return fmt.Errorf("native/tool operation proof omitted")
	}
	if r.Status == "compiled_unexecuted" {
		if r.Selection.Path == "" || len(r.Commands) != 2 {
			return fmt.Errorf("actual Go selection producer omitted")
		}
		g, _, e := readGoSelection(p, s, r.Session, r.Selection)
		if e != nil {
			return e
		}
		if e = selectionProducerMatches(g, r.Commands[0]); e != nil {
			return e
		}
	} else if r.Selection.Path != "" {
		return fmt.Errorf("metadata phase cannot carry a Go selection")
	}
	roles := map[string]int{}
	queries := map[string]bool{}
	var generated []GeneratedSet
	for _, pin := range r.Generated {
		var g GeneratedSet
		if e := read(pin, &g); e != nil {
			return e
		}
		pkg, e := packageByName(p, g.Package)
		if e != nil {
			return e
		}
		if e = CheckGenerated(pkg, g.Stage, g.Files); e != nil {
			return e
		}
		if len(g.Bindings) != len(g.Files) || len(g.Snapshots) != len(g.Files) {
			return fmt.Errorf("generated snapshot/binding set incomplete")
		}
		for name, q := range g.Files {
			if !under(p.Options.WorkRoot, q.Path) || g.Snapshots[name].SHA256 != q.SHA256 {
				return fmt.Errorf("generated input outside snapshot/work domain")
			}
			if e = verifyPin(g.Snapshots[name]); e != nil {
				return e
			}
		}
		for path, target := range g.Bindings {
			if !under(p.Options.WorkRoot, path) || !under(p.Options.WorkRoot, target) || g.Files[filepath.Base(path)].Path != target {
				return fmt.Errorf("generated path binding differs")
			}
		}
		generated = append(generated, g)
	}
	operations := map[string]operation{}
	for _, pin := range r.Operations {
		var op operation
		if e := read(pin, &op); e != nil {
			return e
		}
		if op.Schema != "nanopore-h08-cgo-operation/v1" || op.Error != "" || op.Status != "accepted" && op.Status != "accepted_expected_exit" || len(op.Commands) == 0 || len(op.Expected) == 0 || len(op.Inputs) == 0 || len(op.Snapshots) != len(op.Inputs) {
			return fmt.Errorf("operation before-input/command proof incomplete")
		}
		dir := filepath.Dir(pin.Path)
		if !under(filepath.Join(s.Directory, "commands"), pin.Path) || filepath.Base(pin.Path) != "operation.json" {
			return fmt.Errorf("operation outside phase command domain")
		}
		if _, duplicate := operations[dir]; duplicate {
			return fmt.Errorf("duplicate operation proof")
		}
		if strings.HasPrefix(op.Role, "go-tool-") {
			if e := verifyToolOperation(p, s, r.Session, op, dir, generated); e != nil {
				return e
			}
		} else if e := verifyNativeOperation(p, s, op, dir, generated); e != nil {
			return e
		}
		operations[dir] = op
		for path, want := range op.Inputs {
			q, ok := op.Snapshots[path]
			if !ok || q.SHA256 != want {
				return fmt.Errorf("operation snapshot binding omitted")
			}
			if e := verifyPin(q); e != nil {
				return e
			}
		}
		for i, c := range op.Commands {
			want := op.Expected
			if i < len(op.Commands)-1 {
				want = []int{0, 1}
			}
			if e := checkCommand(c.Args, want, c.Started, c.Finished, c.Stdout, c.StdoutSHA, c.Stderr, c.StderrSHA, c.Exit); e != nil {
				return e
			}
			var before map[string]string
			bp, e := closedPin(filepath.Join(filepath.Dir(c.Stdout), "before.json"))
			if e != nil {
				return e
			}
			if e = read(bp, &before); e != nil || len(before) == 0 {
				return fmt.Errorf("before-child input manifest omitted")
			}
			for path, want := range before {
				q, ok := op.Snapshots[path]
				if !ok || q.SHA256 != want {
					return fmt.Errorf("before-child input not frozen in operation")
				}
			}
			if i == len(op.Commands)-1 && op.Before != bp {
				return fmt.Errorf("last before-child manifest binding differs")
			}
		}
		if op.Role == "native-tu" || op.Role == "cgo-dwarf" || op.Role == "cgo-defines" {
			if op.ActualDep.Path == "" || op.DepUnavailable != "" {
				return fmt.Errorf("successful TU/header equality proof omitted")
			}
		}
		if op.ActualDep.Path != "" {
			if e := verifyPin(op.ActualDep); e != nil {
				return e
			}
		}
		if op.Output.Path != "" {
			if op.OutputSnapshot.SHA256 != op.Output.SHA256 {
				return fmt.Errorf("intermediate output snapshot differs")
			}
			if e := verifyPin(op.OutputSnapshot); e != nil {
				return e
			}
		}
		if op.LinkMap.Path != "" {
			if e := verifyPin(op.LinkMap); e != nil {
				return e
			}
		}
		if op.Role == "cgo-dynlink" || op.Role == "go-link" {
			if op.LinkMap.Path == "" || op.Output.Path == "" || op.Commands[len(op.Commands)-1].Exit != 0 {
				return fmt.Errorf("complete successful link evidence omitted")
			}
		}
		roles[op.Role]++
		if op.Role == "metadata" {
			for _, q := range p.Queries {
				if exactArgs(op.OriginalArgs, q.Argv[1:]) {
					if queries[q.Name] && r.Status == "metadata_queries_passed" {
						return fmt.Errorf("duplicate metadata query")
					}
					queries[q.Name] = true
				}
			}
		}
	}
	if r.Status == "metadata_queries_passed" {
		if s.Phase != "queries" || len(r.Commands) != 0 || len(r.Generated) != 0 || r.Output.Path != "" || len(r.Operations) != len(p.Queries) || len(queries) != len(p.Queries) {
			return fmt.Errorf("finite metadata query chain incomplete")
		}
		return nil
	}
	if s.Phase != "build" || r.Output.Path == "" || roles["native-tu"] == 0 || roles["cgo-dynlink"] != 2 || roles["go-link"] != 1 || roles["go-tool-cgo"] != 4 || roles["go-tool-link"] != 1 || len(r.Generated) != 4 {
		return fmt.Errorf("current Cgo build/final generated proof incomplete")
	}
	if e := validateTarget(p); e != nil {
		return e
	}
	if p.Role == CPU13Tests {
		if r.OverlaySnapshot.SHA256 != digest([]byte(p.CPU13.OverlayJSON)) {
			return fmt.Errorf("CPU13 overlay snapshot omitted/changed")
		}
		if e := verifyPin(r.OverlaySnapshot); e != nil {
			return e
		}
		count := 0
		for _, op := range operations {
			if op.Role == "go-tool-compile" && op.Package == TensorPackage+".test" {
				count++
			}
		}
		if count != 1 {
			return fmt.Errorf("CPU13 generated testmain compiler custody incomplete")
		}
	} else if r.OverlaySnapshot.Path != "" {
		return fmt.Errorf("ordinary H08 artifact cannot carry CPU13 overlay")
	}
	if e := verifyPin(r.Output); e != nil {
		return e
	}
	if e := finalOutputBindings(p, r, operations); e != nil {
		return e
	}
	for _, op := range operations {
		if op.Request != nil && op.Lease != nil {
			found := false
			for _, parent := range operations {
				if strings.HasPrefix(parent.Role, "go-tool-") && reflect.DeepEqual(parent.Lease, op.Lease) {
					found = true
				}
			}
			if !found {
				return fmt.Errorf("native lease has no actual Go-tool producer")
			}
		}
	}
	if len(r.Commands) != 2 {
		return fmt.Errorf("actual Cgo selection/build commands omitted")
	}
	for _, cmd := range r.Commands {
		if e := checkCommand(cmd.Args, []int{0}, cmd.Started, cmd.Finished, cmd.Stdout, cmd.StdoutSHA, cmd.Stderr, cmd.StderrSHA, cmd.Exit); e != nil {
			return e
		}
	}
	selection, e := os.ReadFile(r.Commands[0].Stdout)
	if e != nil {
		return e
	}
	if e = verifyActualSelection(p, selection); e != nil {
		return e
	}
	if !slices.Equal(r.Commands[0].Args, listArgs(p)) || !slices.Equal(r.Commands[1].Args, buildArgs(p, s, r.Session)) || !slices.Equal(r.Commands[0].Environment, phaseEnvironment(p, s, r.Session)) || !slices.Equal(r.Commands[1].Environment, phaseEnvironment(p, s, r.Session)) {
		return fmt.Errorf("actual current consumer command binding differs")
	}
	seen := map[string]bool{}
	for _, pin := range r.Generated {
		var g GeneratedSet
		if e := read(pin, &g); e != nil {
			return e
		}
		pkg, e := packageByName(p, g.Package)
		if e != nil {
			return e
		}
		if e = CheckGenerated(pkg, g.Stage, g.Files); e != nil {
			return e
		}
		if e = verifyPin(g.Producer); e != nil {
			return e
		}
		if e = generatedProducer(g, operations); e != nil {
			return e
		}
		key := g.Package + "/" + g.Stage
		if seen[key] {
			return fmt.Errorf("duplicate generated source proof")
		}
		seen[key] = true
	}
	for _, pkg := range []string{Module + "/internal/tensor", "runtime/cgo"} {
		for _, stage := range []string{"translation", "dynimport"} {
			if !seen[pkg+"/"+stage] {
				return fmt.Errorf("generated source stage omitted")
			}
		}
	}
	return nil
}

func ReadBuild(pin Pin) (Receipt, error)      { return ReadH08Build(pin) }
func ReadH08Build(pin Pin) (Receipt, error)   { return readBuildRole(pin, H08Command) }
func ReadCPU13Build(pin Pin) (Receipt, error) { return readBuildRole(pin, CPU13Tests) }
func readBuildRole(pin Pin, role BuildRole) (Receipt, error) {
	var r Receipt
	if e := read(pin, &r); e != nil {
		return r, e
	}
	if r.Role != role {
		return r, fmt.Errorf("compiled artifact has wrong fixed build role: want %s", role)
	}
	var p Plan
	if e := read(r.Plan, &p); e != nil {
		return r, e
	}
	if p.Role != role {
		return r, fmt.Errorf("compiled artifact plan has wrong fixed build role")
	}
	if e := validateTarget(p); e != nil {
		return r, e
	}
	if e := ValidateReceipt(p, r); e != nil {
		return r, e
	}
	if r.Status != "compiled_unexecuted" {
		return r, fmt.Errorf("CGO1 compiled artifact required")
	}
	return r, nil
}

// Explicit failure marker remains authoritative if Go tolerates a helper exit.
func RejectionCount(session Pin) (int, error) {
	var s Session
	if e := read(session, &s); e != nil {
		return 0, e
	}
	files, e := os.ReadDir(filepath.Join(s.Directory, "rejected"))
	if e != nil {
		return 0, e
	}
	n := 0
	for _, file := range files {
		if strings.HasPrefix(file.Name(), "request-") {
			n++
		}
	}
	return n, nil
}
