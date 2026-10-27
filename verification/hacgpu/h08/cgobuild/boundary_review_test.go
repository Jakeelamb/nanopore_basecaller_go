package cgobuild

import (
	"maps"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func TestCanonicalDerivedAndExistingOutputBoundsBeforeFakeFork(t *testing.T) {
	for _, kind := range []string{"CC output symlink", "Go output symlink", "transient C symlink", "link object symlink", "transient directory alias escape", "link directory alias escape"} {
		t.Run(kind, func(t *testing.T) {
			p, s, l, log := fakeFixture(t)
			t.Chdir(p.Packages[0].Directory)
			foreign := filepath.Join(t.TempDir(), "foreign.c")
			if e := os.WriteFile(foreign, []byte("foreign untouched"), 0600); e != nil {
				t.Fatal(e)
			}
			args, source := diagnostics(p, l)
			if e := os.WriteFile(source, []byte("int x;"), 0600); e != nil {
				t.Fatal(e)
			}
			linkLease := *l
			linkLease.Kind = "link"
			switch kind {
			case "CC output symlink", "Go output symlink":
				out := filepath.Join(l.ObjectDir, "_cgo_1.o")
				if e := os.Symlink(foreign, out); e != nil {
					t.Fatal(e)
				}
				if kind == "Go output symlink" {
					if _, e := toolInputs(p, p.CC, "link", []string{"-o", out}, p.Options.Root); e == nil {
						t.Fatal("existing output symlink accepted")
					}
					break
				}
				if _, e := interposeCC(s, p, args, nil, l, false, fakeBoundary); e == nil {
					t.Fatal("existing compiler output symlink accepted")
				}
			case "transient C symlink":
				os.Remove(source)
				if e := os.Symlink(foreign, source); e != nil {
					t.Fatal(e)
				}
				if _, e := interposeCC(s, p, args, nil, l, false, fakeBoundary); e == nil {
					t.Fatal("foreign transient canonical target accepted")
				}
			case "link object symlink":
				obj := filepath.Join(l.ObjectDir, "foreign.o")
				if e := os.Symlink(foreign, obj); e != nil {
					t.Fatal(e)
				}
				if _, e := interposeCC(s, p, []string{"-o", filepath.Join(l.ObjectDir, "a.out"), obj}, nil, &linkLease, false, fakeBoundary); e == nil {
					t.Fatal("foreign link canonical target accepted")
				}
			case "transient directory alias escape":
				alias := filepath.Join(p.Options.WorkRoot, "cgo-tmp", "alias")
				if e := os.Symlink(filepath.Dir(foreign), alias); e != nil {
					t.Fatal(e)
				}
				args[len(args)-1] = filepath.Join(alias, "cgo-gcc-input-foreign.c")
				if e := os.Rename(foreign, filepath.Join(filepath.Dir(foreign), "cgo-gcc-input-foreign.c")); e != nil {
					t.Fatal(e)
				}
				foreign = filepath.Join(filepath.Dir(foreign), "cgo-gcc-input-foreign.c")
				if _, e := interposeCC(s, p, args, nil, l, false, fakeBoundary); e == nil {
					t.Fatal("foreign transient directory alias accepted")
				}
			case "link directory alias escape":
				if e := os.Symlink(foreign, filepath.Join(filepath.Dir(foreign), "foreign.o")); e != nil {
					t.Fatal(e)
				}
				alias := filepath.Join(p.Options.WorkRoot, "alias")
				if e := os.Symlink(filepath.Dir(foreign), alias); e != nil {
					t.Fatal(e)
				}
				if _, e := interposeCC(s, p, []string{"-o", filepath.Join(l.ObjectDir, "a.out"), filepath.Join(alias, "foreign.o")}, nil, &linkLease, false, fakeBoundary); e == nil {
					t.Fatal("foreign link directory alias accepted")
				}
			}
			if spawns(log) != 0 {
				t.Fatal("invalid path reached a compiler child")
			}
			body, e := os.ReadFile(foreign)
			if e != nil || string(body) != "foreign untouched" {
				t.Fatal("foreign file changed")
			}
		})
	}
}

func TestExistingPrivateOutputAndDirectoryAliasRemainAccepted(t *testing.T) {
	p, s, l, log := fakeFixture(t)
	t.Chdir(p.Packages[0].Directory)
	args, source := diagnostics(p, l)
	if e := os.WriteFile(source, []byte("int x;"), 0600); e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(filepath.Join(l.ObjectDir, "_cgo_1.o"), []byte("existing private regular"), 0600); e != nil {
		t.Fatal(e)
	}
	alias := filepath.Join(p.Options.WorkRoot, "cgo-tmp", "private-alias")
	if e := os.Symlink(filepath.Dir(source), alias); e != nil {
		t.Fatal(e)
	}
	args[len(args)-1] = filepath.Join(alias, filepath.Base(source))
	if _, e := interposeCC(s, p, args, nil, l, false, fakeBoundary); e != nil {
		t.Fatal(e)
	}
	if spawns(log) != 2 {
		t.Fatal("private alias/regular output did not reach exact M+actual fake steps")
	}
}

func TestOutputReplacementBeforeChildRejectsWithoutFakeFork(t *testing.T) {
	p, s, l, log := fakeFixture(t)
	t.Chdir(p.Packages[0].Directory)
	args, source := diagnostics(p, l)
	if e := os.WriteFile(source, []byte("int x;"), 0600); e != nil {
		t.Fatal(e)
	}
	foreign := filepath.Join(t.TempDir(), "untouched")
	if e := os.WriteFile(foreign, []byte("foreign untouched"), 0600); e != nil {
		t.Fatal(e)
	}
	calls := 0
	boundary := func(p Plan) error {
		calls++
		if calls == 2 {
			if e := os.Symlink(foreign, filepath.Join(l.ObjectDir, "_cgo_1.o")); e != nil {
				return e
			}
		}
		return fakeBoundary(p)
	}
	if _, e := interposeCC(s, p, args, nil, l, false, boundary); e == nil {
		t.Fatal("output replacement before exec accepted")
	}
	if spawns(log) != 0 {
		t.Fatal("output replacement reached fake child")
	}
}

func TestGoPackImportcfgAndEmbedcfgBeforeForkBoundaries(t *testing.T) {
	p, _, l, log := fakeFixture(t)
	obj := filepath.Join(l.ObjectDir, "input.o")
	if e := os.WriteFile(obj, []byte("object"), 0600); e != nil {
		t.Fatal(e)
	}
	foreign := filepath.Join(t.TempDir(), "foreign.a")
	if e := os.WriteFile(foreign, []byte("foreign"), 0600); e != nil {
		t.Fatal(e)
	}
	cfg := filepath.Join(l.ObjectDir, "any-config-name")
	if e := write(cfg, struct {
		Patterns map[string][]string
		Files    map[string]string
	}{map[string][]string{"x": {"x"}}, map[string]string{"x": foreign}}); e != nil {
		t.Fatal(e)
	}
	for _, test := range []struct {
		name string
		args []string
	}{{"pack", []string{"r", filepath.Join(filepath.Dir(foreign), "new.a"), obj}}, {"inline importcfg", []string{"-importcfg=" + foreign}}, {"embed", []string{"-embedcfg", cfg}}} {
		t.Run(test.name, func(t *testing.T) {
			if _, e := toolInputs(p, p.CC, "pack", test.args, p.Options.Root); test.name == "pack" && e == nil {
				t.Fatal("foreign pack output accepted")
			}
			if test.name != "pack" {
				if _, e := toolInputs(p, p.CC, "compile", test.args, p.Options.Root); e == nil {
					t.Fatal("unclosed config/payload accepted")
				}
			}
		})
	}
	if _, e := toolInputs(p, p.CC, "pack", []string{"r", filepath.Join(l.ObjectDir, "new.a"), obj}, p.Options.Root); e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(cfg, []byte("packagefile x="+foreign+"\n"), 0600); e != nil {
		t.Fatal(e)
	}
	if _, e := toolInputs(p, p.CC, "compile", []string{"-importcfg", cfg}, p.Options.Root); e == nil {
		t.Fatal("renamed importcfg bypassed archive closure")
	}
	if spawns(log) != 0 {
		t.Fatal("tool parser tests forked compiler")
	}
}

func TestMetadataReaderRejectsRehashedCommandInputsAndQueryAnswers(t *testing.T) {
	for _, mutation := range []string{"none", "command", "environment", "omitted plugin", "query answer"} {
		t.Run(mutation, func(t *testing.T) {
			p, s, _, _ := fakeFixture(t)
			t.Chdir(p.Options.Root)
			if e := os.Remove(filepath.Join(s.Directory, "derived", "tensor.json")); e != nil {
				t.Fatal(e)
			}
			resource := p.Resources[0].Input
			p.Environment["FAKE_QUERY_PATH"] = resource.Path
			p.Queries = []Query{{Name: "resource", Argv: []string{p.CC.Path, "-print-file-name=resource"}, Expected: resource}}
			planPath := filepath.Join(s.Directory, "plan.json")
			if e := write(planPath, p); e != nil {
				t.Fatal(e)
			}
			pp, _ := closedPin(planPath)
			s.Schema, s.Phase, s.Plan = "nanopore-h08-cgo-session/v1", "queries", pp
			if e := write(filepath.Join(s.Directory, "session.json"), s); e != nil {
				t.Fatal(e)
			}
			sp, _ := closedPin(filepath.Join(s.Directory, "session.json"))
			op, e := interposeCC(s, p, p.Queries[0].Argv[1:], nil, nil, false, fakeBoundary)
			if e != nil {
				t.Fatal(e)
			}
			r := Receipt{Schema: "nanopore-h08-cgo-execution/v1", Status: "metadata_queries_passed", Session: sp, Plan: pp, RuntimeHeld: true}
			if e = collect(s, &r); e != nil {
				t.Fatal(e)
			}
			if e = ValidateReceipt(p, r); e != nil {
				t.Fatalf("genuine fake query evidence rejected: %v", e)
			}
			if mutation == "none" {
				return
			}
			c := &op.Commands[0]
			switch mutation {
			case "command":
				c.Args = []string{"/bin/true"}
			case "environment":
				c.Environment = []string{"FOREIGN=1"}
			case "omitted plugin":
				delete(op.Inputs, p.Compiler.Plugins[0].Path)
				delete(op.Snapshots, p.Compiler.Plugins[0].Path)
				before := maps.Clone(op.Inputs)
				os.Remove(filepath.Join(filepath.Dir(c.Stdout), "before.json"))
				if e = write(filepath.Join(filepath.Dir(c.Stdout), "before.json"), before); e != nil {
					t.Fatal(e)
				}
			case "query answer":
				if e = os.WriteFile(c.Stdout, nil, 0600); e != nil {
					t.Fatal(e)
				}
				q, _ := closedPin(c.Stdout)
				c.StdoutSHA = q.SHA256
			}
			os.Remove(filepath.Join(filepath.Dir(c.Stdout), "command.json"))
			if e = write(filepath.Join(filepath.Dir(c.Stdout), "command.json"), *c); e != nil {
				t.Fatal(e)
			}
			os.Remove(r.Operations[0].Path)
			if e = write(r.Operations[0].Path, op); e != nil {
				t.Fatal(e)
			}
			r.Operations[0], _ = closedPin(r.Operations[0].Path)
			if e = ValidateReceipt(p, r); e == nil {
				t.Fatalf("self-rehashed %s proof accepted", mutation)
			}
		})
	}
}

func TestNativeReaderReparsesDepAndExactOutputTarget(t *testing.T) {
	p, s, l, _ := fakeFixture(t)
	t.Chdir(p.Packages[0].Directory)
	args := []string{"-I", p.Packages[0].Directory, "-fPIC", "-m64", "-pthread", "-O2", "-g", "-I", l.ObjectDir, "-o", filepath.Join(l.ObjectDir, "_x001.o"), "-c", "loader.c"}
	op, e := interposeCC(s, p, args, nil, nil, false, fakeBoundary)
	if e != nil {
		t.Fatal(e)
	}
	derived, e := derivedRead(s)
	if e != nil {
		t.Fatal(e)
	}
	dir := filepath.Dir(filepath.Dir(op.Commands[0].Stdout))
	if e = verifyNativeOperation(p, s, op, dir, derived); e != nil {
		t.Fatal(e)
	}
	bad := op
	bad.Output = p.CC
	if verifyNativeOperation(p, s, bad, dir, derived) == nil {
		t.Fatal("unrelated output accepted")
	}
	bad = op
	bad.Commands = slices.Clone(op.Commands)
	bad.Commands[1].Args = []string{"/bin/true"}
	if verifyNativeOperation(p, s, bad, dir, derived) == nil {
		t.Fatal("unrelated compile command accepted")
	}
	if e = os.WriteFile(op.ActualDep.Path, []byte("native: /not/the/frozen/source.c\n"), 0600); e != nil {
		t.Fatal(e)
	}
	op.ActualDep, _ = closedPin(op.ActualDep.Path)
	if verifyNativeOperation(p, s, op, dir, derived) == nil {
		t.Fatal("rehashed mismatching MD accepted")
	}
}

func TestFinalOutputAndGeneratedProducerCannotBeRebound(t *testing.T) {
	p, s, l, _ := fakeFixture(t)
	final := filepath.Join(p.Options.WorkRoot, "h08")
	if e := os.WriteFile(final, []byte("final link bytes"), 0600); e != nil {
		t.Fatal(e)
	}
	q, _ := closedPin(final)
	r := Receipt{Output: q}
	ops := map[string]operation{"native": {Role: "go-link", Output: q}, "go": {Role: "go-tool-link", Output: q}}
	if e := finalOutputBindings(p, r, ops); e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(final, []byte("mutated final bytes"), 0600); e != nil {
		t.Fatal(e)
	}
	r.Output, _ = closedPin(final)
	if finalOutputBindings(p, r, ops) == nil {
		t.Fatal("rehashed final output unrelated to link bytes accepted")
	}
	d := filepath.Join(s.Directory, "producer")
	if e := os.Mkdir(d, 0700); e != nil {
		t.Fatal(e)
	}
	command := runner.CommandReceipt{Args: []string{"selected-cgo"}, Exit: 0}
	if e := write(filepath.Join(d, "command.json"), command); e != nil {
		t.Fatal(e)
	}
	producer, _ := closedPin(filepath.Join(d, "command.json"))
	g := GeneratedSet{Package: l.Package, Stage: "translation", Producer: producer, Files: map[string]Pin{"x": {Path: filepath.Join(l.ObjectDir, "x")}}}
	parents := map[string]operation{d: {Role: "go-tool-cgo", Lease: l, Commands: []runner.CommandReceipt{command}}}
	if e := generatedProducer(g, parents); e != nil {
		t.Fatal(e)
	}
	g.Producer = q
	if generatedProducer(g, parents) == nil {
		t.Fatal("arbitrary generated producer pin accepted")
	}
}

func TestRecordedLinkMapAndTraceMustMatchBeforeChildObjects(t *testing.T) {
	p, s, l, _ := fakeFixture(t)
	object := filepath.Join(l.ObjectDir, "object.o")
	if e := os.WriteFile(object, []byte("object"), 0600); e != nil {
		t.Fatal(e)
	}
	q, _ := closedPin(object)
	inputs := map[string]string{q.Path: q.SHA256}
	snapshots, e := copySnapshots(filepath.Join(s.Directory, "inputs"), inputs)
	if e != nil {
		t.Fatal(e)
	}
	op := operation{Directory: p.Options.Root, Request: &ccRequest{Inputs: inputs}, Inputs: inputs, Snapshots: snapshots}
	if e = recordedLinkEvidence(object+"\n", "LOAD "+object+"\n", op); e != nil {
		t.Fatal(e)
	}
	if recordedLinkEvidence(object+"\n", "LOAD /foreign/new.o\n", op) == nil {
		t.Fatal("foreign rehashed map accepted")
	}
	if recordedLinkEvidence(object+"\n", "\n", op) == nil {
		t.Fatal("empty rehashed map accepted")
	}
}
