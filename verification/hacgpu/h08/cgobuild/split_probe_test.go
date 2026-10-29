package cgobuild

import (
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func splitProbeFixture(t *testing.T, flag string, stdin bool, compileExit, linkExit int) (Plan, Session, operation, string) {
	t.Helper()
	p, s, l, _ := fakeFixture(t)
	source := filepath.Join(l.ObjectDir, "trivial.c")
	put := func(path, body string) Pin {
		t.Helper()
		if e := os.WriteFile(path, []byte(body), 0600); e != nil {
			t.Fatal(e)
		}
		q, e := closedPin(path)
		if e != nil {
			t.Fatal(e)
		}
		return q
	}
	put(source, "int main() { return 0; }")
	tool := put(filepath.Join(p.Options.Root, "link"), "selected Go linker")
	p.GoTools = map[string]string{tool.Path: tool.SHA256}
	l.Kind, l.Tool = "link", tool
	l.Directory = p.Options.Root
	args := []string{flag, "-o", filepath.Join(l.ObjectDir, "probe"), source}
	var lease *Lease = l
	if stdin {
		args = []string{"-Wl,--no-gc-sections", "-O2", "-g", "-x", "c", "-", "-o", "/dev/null"}
		lease = nil
	}
	r, e := classifyRequest(p, p.Options.Root, args, nil, lease, nil, nil)
	if e != nil {
		t.Fatal(e)
	}
	dir := filepath.Join(s.Directory, "commands", "split")
	if e = os.MkdirAll(dir, 0700); e != nil {
		t.Fatal(e)
	}
	inputs := operationInputs(p, s)
	maps.Copy(inputs, r.Inputs)
	if lease != nil {
		maps.Copy(inputs, lease.Inputs)
	}
	op := operation{Schema: "nanopore-h08-cgo-operation/v1", Role: r.Role, Status: "accepted", Directory: p.Options.Root, OriginalArgs: args, Request: &r, Lease: lease, Inputs: inputs, Snapshots: map[string]Pin{}, Expected: r.Expected, DependencyBindings: maps.Clone(r.Bindings)}
	if stdin {
		op.Stdin = put(filepath.Join(dir, "stdin.c"), "")
		inputs[op.Stdin.Path] = op.Stdin.SHA256
	}
	command := func(label string, args []string, exit int, stdout string) {
		t.Helper()
		cd := filepath.Join(dir, label)
		if e := os.MkdirAll(cd, 0700); e != nil {
			t.Fatal(e)
		}
		out := put(filepath.Join(cd, "stdout"), stdout)
		err := put(filepath.Join(cd, "stderr"), "")
		bp := saveQueryFixture(t, filepath.Join(cd, "before.json"), inputs)
		snaps, e := copySnapshots(filepath.Join(s.Directory, "inputs"), inputs)
		if e != nil {
			t.Fatal(e)
		}
		maps.Copy(op.Snapshots, snaps)
		c := runner.CommandReceipt{Args: append([]string{p.CC.Path}, args...), Environment: append(environment(p), "TERM=dumb"), Exit: exit, Started: "2026-10-26T01:00:01Z", Finished: "2026-10-26T01:00:02Z", Stdout: out.Path, StdoutSHA: out.SHA256, Stderr: err.Path, StderrSHA: err.SHA256}
		saveQueryFixture(t, filepath.Join(cd, "command.json"), c)
		op.Commands = append(op.Commands, c)
		op.Before = bp
	}
	preExit := 0
	if compileExit != 0 {
		preExit = 1
		op.DepUnavailable = "source-defined unsupported Go flag probe; retained preflight/actual diagnostics"
	}
	command("preflight", preprocessorArgs(r.Args, filepath.Join(dir, "before.d")), preExit, "")
	dep := "native:\n"
	if !stdin {
		dep = "native: " + source + "\n"
	}
	if preExit == 0 {
		if !stdin {
			op.Dependencies = []string{source}
		}
		op.PreflightDep = put(filepath.Join(dir, "before.d"), dep)
	}
	object := filepath.Join(dir, "probe.o")
	command("probe-compile", compileProbeArgs(r.Args, object, filepath.Join(dir, "actual.d")), compileExit, "")
	if compileExit != 0 {
		op.Status = "accepted_expected_exit"
		return p, s, op, dir
	}
	obj := put(object, "object produced from selected compile")
	inputs[obj.Path] = obj.SHA256
	op.ActualDep = put(filepath.Join(dir, "actual.d"), dep)
	actual := append(linkProbeArgs(r.Args, object), "-Wl,-t", "-Wl,-Map="+filepath.Join(dir, "link.map"))
	command("actual", actual, linkExit, object+"\n")
	op.LinkMap = put(filepath.Join(dir, "link.map"), "LOAD "+object+"\n")
	if linkExit == 0 && r.Output != "" {
		op.Output = put(r.Output, "linked output")
		op.OutputSnapshot = op.Output
	}
	if linkExit != 0 {
		op.Status = "accepted_expected_exit"
	}
	return p, s, op, dir
}

func readSplitProbe(t *testing.T, p Plan, s Session, op operation, dir string) error {
	t.Helper()
	for _, c := range op.Commands {
		saveQueryFixture(t, filepath.Join(filepath.Dir(c.Stdout), "command.json"), c)
	}
	pin := saveQueryFixture(t, filepath.Join(dir, "operation.json"), op)
	receipt := saveQueryFixture(t, filepath.Join(dir, "receipt.json"), Receipt{Operations: []Pin{pin}})
	var r Receipt
	if e := read(receipt, &r); e != nil {
		t.Fatal(e)
	}
	var actual operation
	if e := read(r.Operations[0], &actual); e != nil {
		t.Fatal(e)
	}
	return verifyNativeOperation(p, s, actual, dir, nil, attestedQueryResources{})
}

func TestSplitProbeActualLinkInputs(t *testing.T) {
	for _, flag := range []string{"-Wl,--build-id=0x1234567890abcdef", "-Wl,--export-dynamic-symbol=main", "-no-pie"} {
		t.Run(flag, func(t *testing.T) {
			p, s, op, dir := splitProbeFixture(t, flag, false, 0, 0)
			if e := readSplitProbe(t, p, s, op, dir); e != nil {
				t.Fatal(e)
			}
		})
	}
	t.Run("stdin", func(t *testing.T) {
		p, s, op, dir := splitProbeFixture(t, "", true, 0, 0)
		if e := readSplitProbe(t, p, s, op, dir); e != nil {
			t.Fatal(e)
		}
	})
	t.Run("expected failed link with map", func(t *testing.T) {
		p, s, op, dir := splitProbeFixture(t, "-nopie", false, 0, 1)
		if e := readSplitProbe(t, p, s, op, dir); e != nil {
			t.Fatal(e)
		}
	})
	t.Run("unsupported before linking", func(t *testing.T) {
		p, s, op, dir := splitProbeFixture(t, "-Qunused-arguments", false, 1, 1)
		if len(op.Commands) != 2 || op.LinkMap.Path != "" {
			t.Fatal("unsupported fixture linked")
		}
		if e := readSplitProbe(t, p, s, op, dir); e != nil {
			t.Fatal(e)
		}
	})
}

func TestSplitProbeRehashedRejections(t *testing.T) {
	for _, test := range []struct{ name, want string }{
		{"wrong source", "Go linker probe source differs"},
		{"wrong compile object", "actual selected command/environment differs"},
		{"wrong link object", "actual selected command/environment differs"},
		{"wrong compile source", "actual selected command/environment differs"},
		{"wrong compile environment", "actual selected command/environment differs"},
		{"missing object snapshot", "probe object before-link pin omitted"},
		{"mismatched object snapshot", "probe object before-link pin omitted"},
		{"missing before object", "complete before-child input set differs"},
		{"mismatched before object", "complete before-child input set differs"},
		{"omitted actual object", "link evidence omitted requested object"},
		{"extra foreign input", "outside before-child snapshots"},
		{"source in trace", "split link used compile source instead of object"},
		{"changed compile exit", "failed probe operation incomplete"},
		{"missing preflight", "successful probe compile lacks preflight equality proof"},
	} {
		t.Run(test.name, func(t *testing.T) {
			p, s, op, dir := splitProbeFixture(t, "-no-pie", false, 0, 0)
			object := filepath.Join(dir, "probe.o")
			switch test.name {
			case "wrong source":
				path := filepath.Join(dir, "wrong-source.c")
				if e := os.WriteFile(path, []byte("int main() { return 1; }"), 0600); e != nil {
					t.Fatal(e)
				}
				q, e := closedPin(path)
				if e != nil {
					t.Fatal(e)
				}
				op.Snapshots[op.Request.Source.Path] = q
				op.Inputs[op.Request.Source.Path] = q.SHA256
				op.Request.Source.SHA256 = q.SHA256
				op.Request.Inputs[op.Request.Source.Path] = q.SHA256
				for _, c := range op.Commands {
					bp := filepath.Join(filepath.Dir(c.Stdout), "before.json")
					var before map[string]string
					pin, e := closedPin(bp)
					if e != nil {
						t.Fatal(e)
					}
					if e = read(pin, &before); e != nil {
						t.Fatal(e)
					}
					before[op.Request.Source.Path] = q.SHA256
					updated := saveQueryFixture(t, bp, before)
					if bp == op.Before.Path {
						op.Before = updated
					}
				}
			case "wrong compile object":
				c := &op.Commands[1]
				c.Args = slices.Clone(c.Args)
				for i, a := range c.Args {
					if a == object {
						c.Args[i] = filepath.Join(dir, "other.o")
					}
				}
			case "wrong link object":
				c := &op.Commands[2]
				c.Args = slices.Clone(c.Args)
				for i, a := range c.Args {
					if a == object {
						c.Args[i] = filepath.Join(dir, "other.o")
					}
				}
			case "wrong compile source":
				c := &op.Commands[1]
				c.Args = slices.Clone(c.Args)
				for i, a := range c.Args {
					if a == op.Request.Source.Path {
						c.Args[i] = "/foreign/trivial.c"
					}
				}
			case "wrong compile environment":
				op.Commands[1].Environment = append(slices.Clone(op.Commands[1].Environment), "FOREIGN=1")
			case "missing object snapshot":
				delete(op.Snapshots, object)
			case "mismatched object snapshot":
				q := op.Snapshots[object]
				q.SHA256 = strings.Repeat("0", 64)
				op.Snapshots[object] = q
			case "missing before object", "mismatched before object":
				before := maps.Clone(op.Inputs)
				if test.name == "missing before object" {
					delete(before, object)
				} else {
					before[object] = strings.Repeat("0", 64)
				}
				op.Before = saveQueryFixture(t, op.Before.Path, before)
			case "omitted actual object", "extra foreign input", "source in trace":
				trace := ""
				for _, resource := range p.Resources {
					if trace == "" {
						trace = resource.Input.Path + "\n"
					}
				}
				if test.name == "extra foreign input" {
					trace = object + "\n/foreign/object.o\n"
				}
				if test.name == "source in trace" {
					trace = object + "\n" + op.Request.Source.Path + "\n"
				}
				c := &op.Commands[2]
				if e := os.WriteFile(c.Stdout, []byte(trace), 0600); e != nil {
					t.Fatal(e)
				}
				q, e := closedPin(c.Stdout)
				if e != nil {
					t.Fatal(e)
				}
				c.StdoutSHA = q.SHA256
				body := ""
				for _, path := range strings.Fields(trace) {
					body += "LOAD " + path + "\n"
				}
				if e := os.WriteFile(op.LinkMap.Path, []byte(body), 0600); e != nil {
					t.Fatal(e)
				}
				op.LinkMap, e = closedPin(op.LinkMap.Path)
				if e != nil {
					t.Fatal(e)
				}
			case "changed compile exit":
				op.Commands[1].Exit = 1
			case "missing preflight":
				op.PreflightDep = Pin{}
				op.DepUnavailable = "source-defined unsupported Go flag probe; retained preflight/actual diagnostics"
			}
			if e := readSplitProbe(t, p, s, op, dir); e == nil || !strings.Contains(e.Error(), test.want) {
				t.Fatalf("want semantic rejection %q, got %v", test.want, e)
			}
		})
	}
}
