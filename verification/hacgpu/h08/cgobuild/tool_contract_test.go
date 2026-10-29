package cgobuild

import (
	"context"
	"encoding/json"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func TestWrappedCompilerIdentityRetainsProofAndRejectsOtherRequests(t *testing.T) {
	p, s, _, log := fakeFixture(t)
	t.Chdir(p.Options.Root)
	s.Phase, s.Driver = "build", p.CC
	p.Queries = []Query{{Name: "cc-tool-id", Argv: append([]string{p.CC.Path}, "-###", "-x", "c", "-c", "-")}}
	w := compilerIdentityWrapper{"nanopore-h08-wrapped-compiler-id/v1", s.Driver, append([]string{s.Driver.Path}, "-###", "-x", "c", "-c", "-"), sourceToolOverrides(p, "cc-id", true, nil)}
	op, e := interposeCCWrapped(s, p, w.Args[1:], nil, nil, false, fakeBoundary, &w)
	if e != nil || spawns(log) != 1 || op.Wrapper.Path == "" || op.Inputs[op.Wrapper.Path] != op.Wrapper.SHA256 || op.Inputs[w.Driver.Path] != w.Driver.SHA256 {
		t.Fatalf("exact nested identity proof: %v", e)
	}
	dir := filepath.Dir(op.Wrapper.Path)
	if e = verifyNativeOperation(p, s, op, dir, nil, attestedQueryResources{}); e != nil {
		t.Fatal(e)
	}
	for _, kind := range []string{"foreign tool", "driver hash", "C++", "extra flag", "query phase", "foreign environment", "stdin"} {
		t.Run(kind, func(t *testing.T) {
			copy := w
			copy.Args, copy.Overrides = slices.Clone(w.Args), maps.Clone(w.Overrides)
			session := s
			var stdin []byte
			switch kind {
			case "foreign tool":
				copy.Args[0] = "/foreign/tool"
			case "driver hash":
				copy.Driver.SHA256 = strings.Repeat("0", 64)
			case "C++":
				copy.Args[3] = "c++"
			case "extra flag":
				copy.Args = append(copy.Args, "-foreign")
			case "query phase":
				session.Phase = "queries"
			case "foreign environment":
				copy.Overrides["GOROOT"] = "/foreign"
			case "stdin":
				stdin = []byte("int foreign;")
			}
			before := spawns(log)
			if _, e := interposeCCWrapped(session, p, copy.Args[1:], stdin, nil, false, fakeBoundary, &copy); e == nil || spawns(log) != before {
				t.Fatalf("nested variant reached fake compiler: %v", e)
			}
		})
	}
	missing := op
	missing.Wrapper = Pin{}
	if verifyNativeOperation(p, s, missing, dir, nil, attestedQueryResources{}) == nil {
		t.Fatal("reader accepted omitted wrapper provenance")
	}
	missing = op
	missing.Snapshots = maps.Clone(op.Snapshots)
	delete(missing.Snapshots, w.Driver.Path)
	if verifyNativeOperation(p, s, missing, dir, nil, attestedQueryResources{}) == nil {
		t.Fatal("reader accepted omitted wrapper driver snapshot")
	}
}

func TestFixedCgoRecipesUseAbsoluteInputsForBothPackages(t *testing.T) {
	for _, family := range []string{"runtime/cgo", TensorPackage} {
		t.Run(family, func(t *testing.T) {
			p, _, lease, log := fakeFixture(t)
			pkg := p.Packages[0]
			pkg.ImportPath = family
			pkg.LDFlags = []string{"-ldl"}
			if family == "runtime/cgo" {
				pkg.Cgo = []string{"cgo.go"}
				pkg.CFlags = []string{"-Wall", "-Werror", "-fno-stack-protector", "-Wdeclaration-after-statement", "-std=gnu90"}
				pkg.LDFlags = []string{"-lpthread"}
				file := filepath.Join(pkg.Directory, "cgo.go")
				if e := os.WriteFile(file, []byte("package cgo\nimport \"C\"\n"), 0600); e != nil {
					t.Fatal(e)
				}
				if _, e := pinned(file, p.Inputs); e != nil {
					t.Fatal(e)
				}
			}
			p.Packages = []Package{pkg}
			obj := lease.ObjectDir + string(filepath.Separator)
			args := []string{"-objdir", obj, "-importpath", family}
			if family == "runtime/cgo" {
				args = append(args, "-import_runtime_cgo=false", "-import_syscall=false")
			}
			args = append(args, "-ldflags="+strconv.Quote("-O2")+" "+strconv.Quote("-g")+" "+strconv.Quote(pkg.LDFlags[0]), "--", "-I", obj, "-O2", "-g")
			args = append(args, pkg.CFlags...)
			source := filepath.Join(pkg.Directory, pkg.Cgo[0])
			args = append(args, source)
			if _, e := cgoLease(p, args, pkg.Directory, nil); e != nil {
				t.Fatalf("source-grounded absolute Cgo request: %v", e)
			}
			for _, kind := range []string{"basename", "foreign source", "source alias", "extra source", "changed flags"} {
				t.Run(kind, func(t *testing.T) {
					copy := slices.Clone(args)
					switch kind {
					case "basename":
						copy[len(copy)-1] = filepath.Base(source)
					case "foreign source":
						copy[len(copy)-1] = "/foreign/" + filepath.Base(source)
					case "source alias":
						alias := filepath.Join(pkg.Directory, "alias.go")
						if e := os.Symlink(source, alias); e != nil {
							t.Fatal(e)
						}
						copy[len(copy)-1] = alias
					case "extra source":
						copy = append(copy, source)
					case "changed flags":
						copy = append(copy[:len(copy)-1], "-O0", source)
					}
					if _, e := cgoLease(p, copy, pkg.Directory, nil); e == nil || spawns(log) != 0 {
						t.Fatalf("foreign Cgo recipe reached child: %v", e)
					}
				})
			}
		})
	}
}

func TestSourceToolEnvironmentsReachFakeChildAndRejectOverrides(t *testing.T) {
	p, s, _, log := fakeFixture(t)
	p.Environment["GOROOT"], p.Environment["CGO_LDFLAGS"] = "/selected/go", "-O2 -g"
	path := filepath.Join(p.Options.Root, "environment.log")
	p.Environment["FAKE_ENV_LOG"] = path
	for _, kind := range []string{"translation", "dynimport", "link", "metadata"} {
		t.Run(kind, func(t *testing.T) {
			name, metadata := "cgo", false
			lease := &Lease{Kind: "cgo"}
			switch kind {
			case "dynimport":
				lease.Kind = "dynimport"
			case "link":
				name, lease = "link", &Lease{Kind: "link"}
			case "metadata":
				metadata, lease = true, nil
			}
			incoming := sourceToolOverrides(p, name, metadata, lease)
			env, e := toolEnvironment(p, s, Pin{}, name, metadata, lease, incoming)
			if e != nil {
				t.Fatal(e)
			}
			dir := t.TempDir()
			if _, _, e = runProgram(context.Background(), dir, p.Options.Root, []string{p.CC.Path}, env, nil, false, true); e != nil {
				t.Fatal(e)
			}
			body, e := os.ReadFile(path)
			if e != nil || string(body) != incoming["CGO_LDFLAGS"]+"\n"+incoming["GOROOT"]+"\n" {
				t.Fatalf("actual child environment differs: %v %q", e, body)
			}
			for _, key := range []string{"GOROOT", "CGO_LDFLAGS", "FOREIGN"} {
				bad := maps.Clone(incoming)
				bad[key] = "/foreign"
				before := spawns(log)
				if _, e := toolEnvironment(p, s, Pin{}, name, metadata, lease, bad); e == nil || spawns(log) != before {
					t.Fatal("arbitrary incoming tool override accepted")
				}
			}
		})
	}
}

func selectionFixture(t *testing.T, role BuildRole) (Plan, Session, Pin, runner.CommandReceipt, Pin) {
	t.Helper()
	root := t.TempDir()
	p := Plan{Role: role, Schema: "nanopore-h08-cgo-build-plan/v2", Options: Options{Role: role, Root: root, GoRoot: "/selected/go", WorkRoot: filepath.Join(root, "work")}, Environment: map[string]string{}, Inputs: map[string]string{}}
	s := Session{Phase: "build", Directory: filepath.Join(root, "session"), Driver: Pin{Path: "/sealed/interposer", SHA256: strings.Repeat("a", 64)}}
	dir := filepath.Join(s.Directory, "commands", "go-list-fixture")
	if e := os.MkdirAll(dir, 0700); e != nil {
		t.Fatal(e)
	}
	var descriptions []map[string]any
	if role == H08Command {
		pkg := Package{ImportPath: selectionTarget(p), Directory: root, Go: []string{"main.go"}}
		p.Packages = []Package{pkg}
		descriptions = append(descriptions, map[string]any{"ImportPath": pkg.ImportPath, "Dir": pkg.Directory, "GoFiles": pkg.Go, "DefaultGODEBUG": "unitdefault=1"})
	} else {
		pkg := Package{ImportPath: TensorPackage, Directory: filepath.Join(root, "internal/tensor"), Go: []string{"tensor.go"}, TestGo: []string{"tensor_test.go"}}
		p.Packages = []Package{pkg}
		p.CPU13 = &CPU13Profile{TestCopies: []string{TensorPackage}, MainSource: "package main\n"}
		p.CPU13.MainSHA256 = digest([]byte(p.CPU13.MainSource))
		cache := filepath.Join(p.Options.WorkRoot, "cache", "testmain-d")
		if e := os.MkdirAll(filepath.Dir(cache), 0700); e != nil {
			t.Fatal(e)
		}
		if e := os.WriteFile(cache, []byte(p.CPU13.MainSource), 0600); e != nil {
			t.Fatal(e)
		}
		p.CPU13.OverlayPath = filepath.Join(p.Options.WorkRoot, "cpu13-overlay.json")
		descriptions = append(descriptions,
			map[string]any{"ImportPath": TensorPackage, "Dir": pkg.Directory, "GoFiles": pkg.Go},
			map[string]any{"ImportPath": TensorPackage + testSuffix, "Dir": pkg.Directory, "ForTest": TensorPackage, "GoFiles": append(slices.Clone(pkg.Go), pkg.TestGo...)},
			map[string]any{"ImportPath": TensorPackage + "_test" + testSuffix, "Dir": pkg.Directory, "ForTest": TensorPackage, "Name": "tensor_test", "GoFiles": []string{OverlayName}, "Imports": []string{"os", "testing", Module + "/verification/hacgpu/h08"}},
			map[string]any{"ImportPath": TensorPackage + ".test", "Dir": pkg.Directory, "Name": "main", "GoFiles": []string{cache}, "DefaultGODEBUG": "unitdefault=1"})
	}
	var body []byte
	for _, q := range descriptions {
		b, _ := json.Marshal(q)
		body = append(body, append(b, '\n')...)
	}
	stdout, stderr := filepath.Join(dir, "stdout"), filepath.Join(dir, "stderr")
	if e := os.WriteFile(stdout, body, 0600); e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(stderr, nil, 0600); e != nil {
		t.Fatal(e)
	}
	out, _ := closedPin(stdout)
	errout, _ := closedPin(stderr)
	sp := Pin{Path: filepath.Join(s.Directory, "session.json"), SHA256: strings.Repeat("b", 64)}
	c := runner.CommandReceipt{Args: listArgs(p), Environment: phaseEnvironment(p, s, sp), Started: "2026-10-25T00:00:00Z", Finished: "2026-10-25T00:00:01Z", Exit: 0, Stdout: stdout, Stderr: stderr, StdoutSHA: out.SHA256, StderrSHA: errout.SHA256}
	if e := write(filepath.Join(dir, "command.json"), c); e != nil {
		t.Fatal(e)
	}
	q, e := captureGoSelection(p, s, sp, c)
	if e != nil {
		t.Fatal(e)
	}
	return p, s, sp, c, q
}

func TestGoLinkAssignmentsBindFixedRoleAndSelectionProducer(t *testing.T) {
	for _, role := range []BuildRole{H08Command, CPU13Tests} {
		t.Run(string(role), func(t *testing.T) {
			p, s, sp, command, q := selectionFixture(t, role)
			g, inputs, e := readGoSelection(p, s, sp, q)
			if e != nil || len(inputs) != 4 || selectionProducerMatches(g, command) != nil {
				t.Fatalf("fixed producer proof incomplete: %v", e)
			}
			args := []string{"-X=runtime.godebugDefault=" + g.DefaultGODEBUG}
			if role == CPU13Tests {
				args = append(args, "-X", "testing.testBinary=1")
			}
			if e = validateLinkAssignments(p, "link", args, g.DefaultGODEBUG); e != nil {
				t.Fatal(e)
			}
			other := p
			other.Role = H08Command
			if p.Role == H08Command {
				other.Role = CPU13Tests
			}
			if validateLinkAssignments(other, "link", args, g.DefaultGODEBUG) == nil {
				t.Fatal("another role accepted the linker assignments")
			}
			for _, copy := range [][]string{nil, {"-X=foreign.symbol=1"}, {"-X=runtime.godebugDefault=changed"}, append(slices.Clone(args), args[0]), append(slices.Clone(args), "-X", "foreign.symbol=1")} {
				if validateLinkAssignments(p, "link", copy, g.DefaultGODEBUG) == nil {
					t.Fatal("foreign, missing or duplicate assignment accepted")
				}
			}
			if validateLinkAssignments(p, "compile", args, g.DefaultGODEBUG) == nil {
				t.Fatal("linker assignment accepted for compiler")
			}
			changedCommand := command
			changedCommand.Args = append(slices.Clone(command.Args), "-foreign")
			if selectionProducerMatches(g, changedCommand) == nil {
				t.Fatal("selection accepted a different phase producer")
			}
			g.DefaultGODEBUG = "forged"
			if e = write(filepath.Join(s.Directory, "changed-selection.json"), g); e != nil {
				t.Fatal(e)
			}
			// Rehash at the fixed record path to reach the producer-value guard.
			body, _ := json.Marshal(g)
			if e = os.WriteFile(q.Path, body, 0600); e != nil {
				t.Fatal(e)
			}
			changed, _ := closedPin(q.Path)
			if _, _, e = readGoSelection(p, s, sp, changed); e == nil || !strings.Contains(e.Error(), "differs from actual producer") {
				t.Fatalf("rehashed selection value reached no producer guard: %v", e)
			}
		})
	}
}

func TestGoSelectionRejectsRehashedRoleAndProducerChanges(t *testing.T) {
	for _, kind := range []string{"role", "producer command", "producer environment", "missing producer"} {
		t.Run(kind, func(t *testing.T) {
			p, s, sp, c, q := selectionFixture(t, H08Command)
			g, _, e := readGoSelection(p, s, sp, q)
			if e != nil {
				t.Fatal(e)
			}
			switch kind {
			case "role":
				g.Role = CPU13Tests
			case "producer command", "producer environment":
				if kind == "producer command" {
					c.Args = append(slices.Clone(c.Args), "-foreign")
				} else {
					c.Environment = append(slices.Clone(c.Environment), "FOREIGN=1")
				}
				body, _ := json.Marshal(c)
				if e = os.WriteFile(g.Producer.Path, body, 0600); e != nil {
					t.Fatal(e)
				}
				g.Producer, _ = closedPin(g.Producer.Path)
			case "missing producer":
				g.Producer = Pin{}
			}
			body, _ := json.Marshal(g)
			if e = os.WriteFile(q.Path, body, 0600); e != nil {
				t.Fatal(e)
			}
			q, _ = closedPin(q.Path)
			if _, _, e = readGoSelection(p, s, sp, q); e == nil {
				t.Fatal("rehashed foreign producer/role accepted")
			}
		})
	}
}

func TestGoToolReaderBindsSourceEnvironmentProfile(t *testing.T) {
	p, s, _, _ := fakeFixture(t)
	s.Phase, s.Driver = "build", p.CC
	p.Environment["GOROOT"], p.Environment["CGO_LDFLAGS"] = p.Options.GoRoot, "-O2 -g"
	tool := filepath.Join(t.TempDir(), "compile")
	body, e := os.ReadFile(p.CC.Path)
	if e != nil {
		t.Fatal(e)
	}
	if e = os.WriteFile(tool, body, 0700); e != nil {
		t.Fatal(e)
	}
	q, e := pinned(tool, p.Inputs)
	if e != nil {
		t.Fatal(e)
	}
	p.GoTools = map[string]string{q.Path: q.SHA256}
	args := []string{q.Path, "-p", p.Packages[0].ImportPath}
	dir, e := opDir(s, "tool")
	if e != nil {
		t.Fatal(e)
	}
	op := operation{Schema: "nanopore-h08-cgo-operation/v1", Role: "go-tool-compile", Status: "accepted", Directory: p.Options.Root, Package: p.Packages[0].ImportPath, OriginalArgs: args, Expected: []int{0}, Inputs: operationInputs(p, s), ToolOverrides: sourceToolOverrides(p, "compile", false, nil)}
	sp := Pin{Path: filepath.Join(s.Directory, "session.json"), SHA256: strings.Repeat("c", 64)}
	env, e := toolEnvironment(p, s, sp, "compile", false, nil, op.ToolOverrides)
	if e != nil {
		t.Fatal(e)
	}
	op.Snapshots, e = copySnapshots(filepath.Join(s.Directory, "inputs"), op.Inputs)
	if e != nil {
		t.Fatal(e)
	}
	if e = write(filepath.Join(dir, "before.json"), op.Inputs); e != nil {
		t.Fatal(e)
	}
	op.Before, _ = closedPin(filepath.Join(dir, "before.json"))
	c, proc, e := runProgram(context.Background(), dir, p.Options.Root, args, env, nil, false, false)
	if e != nil {
		t.Fatal(e)
	}
	op.Commands, op.Processes = []runner.CommandReceipt{c}, []Process{proc}
	if e = verifyToolOperation(p, s, sp, op, dir, nil); e != nil {
		t.Fatal(e)
	}
	for _, key := range []string{"GOROOT", "CGO_LDFLAGS", "missing"} {
		copy := op
		copy.ToolOverrides = maps.Clone(op.ToolOverrides)
		if key == "missing" {
			copy.ToolOverrides = nil
		} else {
			copy.ToolOverrides[key] = "/foreign"
		}
		if verifyToolOperation(p, s, sp, copy, dir, nil) == nil {
			t.Fatal("reader accepted changed or omitted environment profile")
		}
	}
}

func TestSelectedExternalLinkerToolAdmission(t *testing.T) {
	p, _, lease, log := fakeFixture(t)
	driver := p.CC
	cwd := p.Packages[0].Directory
	archive := filepath.Join(lease.ObjectDir, "selected.a")
	if e := os.WriteFile(archive, []byte("archive"), 0600); e != nil {
		t.Fatal(e)
	}
	q, e := closedPin(archive)
	if e != nil {
		t.Fatal(e)
	}
	snaps, e := copySnapshots(filepath.Join(t.TempDir(), "snapshots"), map[string]string{q.Path: q.SHA256})
	if e != nil {
		t.Fatal(e)
	}
	recorded := operation{Inputs: map[string]string{q.Path: q.SHA256}, Bindings: map[string]string{archive: q.Path}, Snapshots: snaps, Outputs: map[string]string{}}
	for _, test := range []struct {
		name, value string
		valid       bool
	}{
		{"selected bare", driver.Path, true},
		{"selected generated quote", strconv.Quote(driver.Path), true},
		{"foreign bare", "/foreign/driver", false},
		{"foreign quoted", strconv.Quote("/foreign/driver"), false},
		{"bare extra argument", driver.Path + " -foreign", false},
		{"quoted extra argument", strconv.Quote(driver.Path) + " -foreign", false},
		{"quoted extra command", strconv.Quote(driver.Path) + "; /foreign/driver", false},
		{"missing close quote", `"` + driver.Path, false},
		{"missing open quote", driver.Path + `"`, false},
		{"mismatched quote", `"` + driver.Path + `'`, false},
		{"single quotes", `'` + driver.Path + `'`, false},
		{"escaped quotes", `\"` + driver.Path + `\"`, false},
		{"escaped path", strconv.Quote(strings.Replace(driver.Path, "/", `\x2f`, 1)), false},
		{"leading space", " " + strconv.Quote(driver.Path), false},
		{"trailing space", strconv.Quote(driver.Path) + " ", false},
		{"adjacent empty field", strconv.Quote(driver.Path) + `""`, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			args := []string{"-extld=" + test.value, archive}
			for _, mode := range []string{"producer", "recorded"} {
				var op *operation
				if mode == "recorded" {
					op = &recorded
				}
				inputs, e := toolInputsEvidence(p, driver, "link", args, cwd, op, nil, nil)
				if test.valid {
					if e != nil || inputs[q.Path] != q.SHA256 {
						t.Fatalf("%s selected driver/input rejected: %v", mode, e)
					}
				} else if e == nil || !strings.Contains(e.Error(), "foreign external linker before Go tool child") {
					t.Fatalf("%s driver control missed selection gate: %v", mode, e)
				}
			}
			if spawns(log) != 0 {
				t.Fatal("driver admission forked a fake compiler")
			}
		})
	}
	changed := snaps[q.Path]
	changed.Path = filepath.Join(t.TempDir(), "changed-snapshot")
	if e := os.WriteFile(changed.Path, []byte("changed"), 0600); e != nil {
		t.Fatal(e)
	}
	recorded.Snapshots[q.Path] = changed
	if _, e := toolInputsEvidence(p, driver, "link", []string{"-extld=" + strconv.Quote(driver.Path), archive}, cwd, &recorded, nil, nil); e == nil {
		t.Fatal("quoted driver bypassed recorded input authentication")
	}
}
