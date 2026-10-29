package cgobuild

import (
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func queryResourceFixture(t *testing.T, name string) (Plan, Receipt, operation, Receipt, operation) {
	t.Helper()
	p, s, _, _ := fakeFixture(t)
	_, file, _, _ := runtime.Caller(0)
	sourceBody, e := os.ReadFile(filepath.Join(filepath.Dir(file), "native.go"))
	if e != nil {
		t.Fatal(e)
	}
	source := filepath.Join(p.Options.Root, "verification/hacgpu/h08/cgobuild/native.go")
	if e := os.MkdirAll(filepath.Dir(source), 0700); e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(source, sourceBody, 0600); e != nil {
		t.Fatal(e)
	}
	src, e := pinned(source, p.Inputs)
	if e != nil {
		t.Fatal(e)
	}
	if src.SHA256 != queryProducerNativeSHA {
		t.Fatal("producer profile fixture changed; review resource-check control flow")
	}
	resource := p.Resources[0].Input
	p.Resources = []Resource{{Name: name, Input: resource, Query: []string{p.CC.Path, "-print-file-name=" + name}}}
	p.Queries = []Query{{Name: name, Argv: p.Resources[0].Query, Expected: resource}}
	root := t.TempDir()
	put := func(path string, body []byte) Pin {
		t.Helper()
		if e := os.MkdirAll(filepath.Dir(path), 0700); e != nil {
			t.Fatal(e)
		}
		if e := os.WriteFile(path, body, 0600); e != nil {
			t.Fatal(e)
		}
		q, e := closedPin(path)
		if e != nil {
			t.Fatal(e)
		}
		return q
	}
	command := func(dir string, args []string, stdout string) runner.CommandReceipt {
		out := put(filepath.Join(dir, "stdout"), []byte(stdout))
		err := put(filepath.Join(dir, "stderr"), nil)
		return runner.CommandReceipt{Args: args, Exit: 0, Started: "2026-10-26T01:00:01Z", Finished: "2026-10-26T01:00:02Z", Stdout: out.Path, StdoutSHA: out.SHA256, Stderr: err.Path, StderrSHA: err.SHA256}
	}
	tools := map[string]string{}
	for _, path := range slices.Sorted(maps.Keys(p.Inputs)) {
		if path != source && len(tools) < 9 {
			tools[path] = p.Inputs[path]
		}
	}
	tools[p.CC.Path] = p.CC.SHA256
	inputs := maps.Clone(tools)
	inputs[source] = src.SHA256
	snaps, e := copySnapshots(filepath.Join(root, "controller-inputs"), inputs)
	if e != nil {
		t.Fatal(e)
	}
	listPath := filepath.Join(root, "list.json")
	if e := write(listPath, struct {
		Dir     string
		GoFiles []string
	}{filepath.Dir(source), []string{"native.go"}}); e != nil {
		t.Fatal(e)
	}
	lp, e := closedPin(listPath)
	if e != nil {
		t.Fatal(e)
	}
	args := []string{p.CC.Path, "build", "-a", "-x", "-work", "-p=2", "-pgo=off", "-trimpath", "-buildvcs=false", "-o", p.CC.Path, "./verification/hacgpu/h08/cgobuild/cmd"}
	c := command(filepath.Join(root, "controller-build"), args, "")
	c.Environment = []string{"CGO_ENABLED=0", "GOTOOLCHAIN=local", "GOPROXY=off", "GOSUMDB=off", "GOENV=off", "GOWORK=off"}
	cp := saveQueryFixture(t, filepath.Join(root, "controller.json"), Controller{Schema: "nanopore-h08-cgo-interposer/v1", Executable: p.CC, Inputs: inputs, Snapshots: snaps, Tools: tools, Lists: []Pin{lp, lp}, Build: c})
	pp := saveQueryFixture(t, filepath.Join(root, "plan.json"), p)
	s.Schema, s.Phase, s.Plan, s.Controller, s.Driver = "nanopore-h08-cgo-session/v1", "queries", pp, cp, p.CC
	sp := saveQueryFixture(t, filepath.Join(s.Directory, "session.json"), s)
	dir := filepath.Join(s.Directory, "commands", "query")
	answer := "/lib/../lib/" + name
	if slices.Contains([]string{"crtbegin.o", "crtbeginS.o", "crtend.o", "crtendS.o", "libgcc.a", "libgcc_s.so", "libgcc_s.so.1", "liblto_plugin.so"}, name) {
		answer = p.Options.Root + "/toolchain/usr/bin/../lib/gcc/x86_64-pc-linux-gnu/15.3.0/" + name
	}
	c = command(filepath.Join(dir, "actual"), p.Queries[0].Argv, answer+"\n")
	c.Environment = append(environment(p), "TERM=dumb")
	saveQueryFixture(t, filepath.Join(dir, "actual/command.json"), c)
	bp := saveQueryFixture(t, filepath.Join(dir, "actual/before.json"), p.Inputs)
	snapshots, e := copySnapshots(filepath.Join(root, "query-inputs"), p.Inputs)
	if e != nil {
		t.Fatal(e)
	}
	req, e := classifyRequest(p, p.Options.Root, p.Queries[0].Argv[1:], nil, nil, nil, nil)
	if e != nil {
		t.Fatal(e)
	}
	op := operation{Schema: "nanopore-h08-cgo-operation/v1", Role: "metadata", Status: "accepted", Directory: p.Options.Root, Request: &req, OriginalArgs: p.Queries[0].Argv[1:], Inputs: maps.Clone(p.Inputs), Snapshots: snapshots, Commands: []runner.CommandReceipt{c}, Before: bp, Expected: []int{0}}
	opPin := saveQueryFixture(t, filepath.Join(dir, "operation.json"), op)
	qr := Receipt{Schema: "nanopore-h08-cgo-execution/v1", Status: "metadata_queries_passed", Role: p.Role, Plan: pp, Controller: cp, Session: sp, RuntimeHeld: true, Started: "2026-10-26T01:00:00Z", Finished: "2026-10-26T01:00:03Z", Operations: []Pin{opPin}}
	build := Receipt{Plan: pp, Controller: cp, Started: "2026-10-26T02:00:00Z", Queries: saveQueryFixture(t, filepath.Join(root, "queries.json"), qr)}
	link := operation{Directory: p.Options.Root, Request: &ccRequest{Inputs: map[string]string{resource.Path: resource.SHA256}}, Inputs: map[string]string{resource.Path: resource.SHA256}, Snapshots: map[string]Pin{resource.Path: snapshots[resource.Path]}}
	return p, build, link, qr, op
}

func saveQueryFixture(t *testing.T, path string, v any) Pin {
	t.Helper()
	if e := os.MkdirAll(filepath.Dir(path), 0700); e != nil {
		t.Fatal(e)
	}
	if e := os.Remove(path); e != nil && !os.IsNotExist(e) {
		t.Fatal(e)
	}
	if e := write(path, v); e != nil {
		t.Fatal(e)
	}
	q, e := closedPin(path)
	if e != nil {
		t.Fatal(e)
	}
	return q
}

func TestQueryResourceAliasesBindExactRawAnswers(t *testing.T) {
	for _, name := range []string{"crt1.o", "Scrt1.o", "crti.o", "crtn.o", "libc.so", "libdl.a", "libpthread.a", "libc_nonshared.a", "crtbegin.o", "crtbeginS.o", "crtend.o", "crtendS.o", "libgcc.a", "libgcc_s.so", "libgcc_s.so.1", "liblto_plugin.so"} {
		t.Run("selected "+name, func(t *testing.T) {
			p, build, link, _, op := queryResourceFixture(t, name)
			resources, e := deriveQueryResources(p, build)
			if e != nil {
				t.Fatal(e)
			}
			body, err := os.ReadFile(op.Commands[0].Stdout)
			if err != nil {
				t.Fatal(err)
			}
			raw := strings.TrimSpace(string(body))
			if e := recordedLinkEvidence(raw+"\n", "LOAD "+raw+"\n", link, resources); e != nil {
				t.Fatal(e)
			}
		})
	}

	for _, test := range []struct{ name, want string }{
		{"exact", ""}, {"alternative", "unselected raw link spelling"}, {"unknown", "outside before-child snapshots"}, {"omitted", "omitted requested object"},
		{"missing snapshot", "before-child snapshot differs"}, {"missing link snapshot", "outside before-child snapshots"}, {"wrong query", "wrong or duplicate selected prior query"},
		{"wrong command", "actual selected command/environment differs"}, {"query clock", "prior query command clock differs"}, {"receipt clock", "prior query clock is not before build"},
		{"wrong source", "prior query producer source/driver differs"}, {"wrong resource", "prospective query resource differs"},
	} {
		t.Run(test.name, func(t *testing.T) {
			p, build, link, qr, op := queryResourceFixture(t, "crt1.o")
			trace := "/lib/../lib/crt1.o\n"
			body := "LOAD /lib/../lib/crt1.o\n"
			switch test.name {
			case "alternative":
				trace = "/lib/../lib/./crt1.o\n"
				body = "LOAD /lib/../lib/./crt1.o\n"
			case "unknown":
				trace = "/foreign/object.o\n"
				body = "LOAD /foreign/object.o\n"
			case "omitted":
				extra := filepath.Join(p.Options.Root, "required.o")
				link.Request.Inputs[extra] = "required"
			case "missing link snapshot":
				delete(link.Snapshots, p.Queries[0].Expected.Path)
			case "missing snapshot":
				delete(op.Snapshots, p.Queries[0].Expected.Path)
			case "wrong query":
				op.OriginalArgs = []string{"-print-file-name=other.o"}
			case "wrong command":
				op.Commands[0].Args = []string{p.CC.Path, "-print-file-name=other.o"}
			case "query clock":
				op.Commands[0].Started = "2026-10-26T01:30:00Z"
				op.Commands[0].Finished = "2026-10-26T01:30:01Z"
			case "receipt clock":
				qr.Finished = build.Started
			case "wrong source":
				var controller Controller
				if e := read(build.Controller, &controller); e != nil {
					t.Fatal(e)
				}
				source := filepath.Join(p.Options.Root, "verification/hacgpu/h08/cgobuild/native.go")
				delete(controller.Inputs, source)
				delete(controller.Snapshots, source)
				replacement := filepath.Join(p.Options.Root, "other.go")
				data, e := os.ReadFile(source)
				if e != nil {
					t.Fatal(e)
				}
				if e = os.WriteFile(replacement, data, 0600); e != nil {
					t.Fatal(e)
				}
				controller.Inputs[replacement] = queryProducerNativeSHA
				controller.Snapshots[replacement] = op.Snapshots[source]
				lp := saveQueryFixture(t, controller.Lists[0].Path, struct {
					Dir     string
					GoFiles []string
				}{p.Options.Root, []string{"other.go"}})
				controller.Lists = []Pin{lp, lp}
				build.Controller = saveQueryFixture(t, build.Controller.Path, controller)
				qr.Controller = build.Controller
				var session Session
				if e := read(qr.Session, &session); e != nil {
					t.Fatal(e)
				}
				session.Controller = build.Controller
				qr.Session = saveQueryFixture(t, qr.Session.Path, session)
			case "wrong resource":
				p.Resources[0].Input = p.CC
			}
			if test.name == "wrong resource" {
				build.Plan = saveQueryFixture(t, build.Plan.Path, p)
				qr.Plan = build.Plan
				var session Session
				if e := read(qr.Session, &session); e != nil {
					t.Fatal(e)
				}
				session.Plan = build.Plan
				qr.Session = saveQueryFixture(t, qr.Session.Path, session)
			}
			saveQueryFixture(t, filepath.Join(filepath.Dir(op.Commands[0].Stdout), "command.json"), op.Commands[0])
			qr.Operations[0] = saveQueryFixture(t, qr.Operations[0].Path, op)
			build.Queries = saveQueryFixture(t, build.Queries.Path, qr)
			resources, e := deriveQueryResources(p, build)
			if e == nil {
				e = recordedLinkEvidence(trace, body, link, resources)
			}
			if test.want == "" {
				if e != nil {
					t.Fatal(e)
				}
			} else if e == nil || !strings.Contains(e.Error(), test.want) {
				t.Fatalf("want semantic rejection %q, got %v", test.want, e)
			}
		})
	}
}

func TestFailedReceiptRemainsStrict(t *testing.T) {
	if e := ValidateReceipt(Plan{}, Receipt{Schema: "nanopore-h08-cgo-execution/v1", Status: "failed", RuntimeHeld: true}); e == nil || !strings.Contains(e.Error(), "native phase receipt incomplete") {
		t.Fatalf("failed receipt without error bypassed envelope: %v", e)
	}

	for _, status := range []string{"failed", "metadata_queries_passed", "compiled_unexecuted"} {
		r := Receipt{Schema: "nanopore-h08-cgo-execution/v1", Status: status, RuntimeHeld: true, Error: "link used input outside before-child snapshots"}
		if e := ValidateReceipt(Plan{}, r); e == nil || !strings.Contains(e.Error(), "native phase receipt incomplete") {
			t.Fatalf("status %s bypassed error envelope: %v", status, e)
		}
	}
	for _, status := range []string{"metadata_queries_passed", "compiled_unexecuted"} {
		p, build, _, qr, _ := queryResourceFixture(t, "crt1.o")
		qr.Status = status
		if status == "metadata_queries_passed" {
			var op operation
			if e := read(qr.Operations[0], &op); e != nil {
				t.Fatal(e)
			}
			if e := os.WriteFile(op.Commands[0].Stdout, []byte(p.Queries[0].Expected.Path+"\n"), 0600); e != nil {
				t.Fatal(e)
			}
			q, e := closedPin(op.Commands[0].Stdout)
			if e != nil {
				t.Fatal(e)
			}
			op.Commands[0].StdoutSHA = q.SHA256
			saveQueryFixture(t, filepath.Join(filepath.Dir(q.Path), "command.json"), op.Commands[0])
			qr.Operations[0] = saveQueryFixture(t, qr.Operations[0].Path, op)
			if e := ValidateReceipt(p, qr); e != nil {
				t.Fatal(e)
			}
			pin := saveQueryFixture(t, build.Queries.Path, qr)
			if e := VerifyQueries(pin, build.Plan); e != nil {
				t.Fatal(e)
			}
			qr.Status = "failed"
			qr.Error = "original producer failure"
			pin = saveQueryFixture(t, pin.Path, qr)
			if e := VerifyQueries(pin, build.Plan); e == nil || !strings.Contains(e.Error(), "accepted query phase required") {
				t.Fatal(e)
			}
		} else if e := ValidateReceipt(p, qr); e == nil || !strings.Contains(e.Error(), "actual Go selection producer omitted") {
			t.Fatalf("compiled phase skipped selection: %v", e)
		}
	}
}

func TestQualifiedQueryResourceRejections(t *testing.T) {
	for _, name := range []string{"crtbegin.o", "crtbeginS.o", "crtend.o", "crtendS.o", "libgcc.a", "libgcc_s.so", "libgcc_s.so.1", "liblto_plugin.so", "libc_nonshared.a"} {
		for _, test := range []struct{ name, want string }{
			{"alternate", "unselected raw link spelling"},
			{"wrong query", "wrong or duplicate selected prior query"},
			{"wrong expected", "prospective query resource differs"},
			{"wrong before", "before-child snapshot differs"},
		} {
			t.Run(name+"/"+test.name, func(t *testing.T) {
				p, build, link, qr, op := queryResourceFixture(t, name)
				body, e := os.ReadFile(op.Commands[0].Stdout)
				if e != nil {
					t.Fatal(e)
				}
				raw := strings.TrimSpace(string(body))
				switch test.name {
				case "alternate":
					raw = strings.TrimSuffix(raw, name) + "./" + name
				case "wrong query":
					op.OriginalArgs = []string{"-print-file-name=unrecorded.o"}
				case "wrong expected":
					p.Queries[0].Expected = p.CC
				case "wrong before":
					snap := op.Snapshots[p.Queries[0].Expected.Path]
					snap.SHA256 = p.CC.SHA256
					op.Snapshots[p.Queries[0].Expected.Path] = snap
				}
				build.Plan = saveQueryFixture(t, build.Plan.Path, p)
				qr.Plan = build.Plan
				var session Session
				if e = read(qr.Session, &session); e != nil {
					t.Fatal(e)
				}
				session.Plan = build.Plan
				qr.Session = saveQueryFixture(t, qr.Session.Path, session)
				qr.Operations[0] = saveQueryFixture(t, qr.Operations[0].Path, op)
				build.Queries = saveQueryFixture(t, build.Queries.Path, qr)
				resources, e := deriveQueryResources(p, build)
				if e == nil {
					e = recordedLinkEvidence(raw+"\n", "LOAD "+raw+"\n", link, resources)
				}
				if e == nil || !strings.Contains(e.Error(), test.want) {
					t.Fatalf("want semantic rejection %q, got %v", test.want, e)
				}
			})
		}
	}
}
