package fakecapture

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08/cgobuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
)

// These fixtures are inert nonexecutable files. Separate lifecycle tests use
// only a Go self-child; neither represents selected/native execution evidence.
func inert(t *testing.T, path, body string) Pin {
	t.Helper()
	if e := os.WriteFile(path, []byte(body), 0600); e != nil {
		t.Fatal(e)
	}
	p, e := closedbuild.PinFile(path)
	if e != nil {
		t.Fatal(e)
	}
	return p
}

func requestFixture(t *testing.T) (Request, Pin) {
	t.Helper()
	root := t.TempDir()
	if e := os.MkdirAll(filepath.Join(root, "artifacts", "tensor"), 0700); e != nil {
		t.Fatal(e)
	}
	q := Request{Schema: RequestSchema, Root: root, Out: filepath.Join(root, "artifacts", "tensor", "capture")}
	for name, dest := range map[string]*Pin{"build.json": &q.ControllerBuild, "selection.json": &q.ControllerBuildSelection, "target": &q.Target, "logger": &q.Logger, "compiler": &q.Compiler} {
		*dest = inert(t, filepath.Join(root, name), "INERT "+name)
	}
	return q, requestPin(t, q, "request.json")
}

func requestPin(t *testing.T, q Request, name string) Pin {
	t.Helper()
	b, e := json.Marshal(q)
	if e != nil {
		t.Fatal(e)
	}
	return inert(t, filepath.Join(q.Root, name), string(b))
}

func TestStrictRequestBoundary(t *testing.T) {
	q, p := requestFixture(t)
	if _, e := readRequest(p, true); e != nil {
		t.Fatal(e)
	}
	body, e := os.ReadFile(p.Path)
	if e != nil {
		t.Fatal(e)
	}
	for name, raw := range map[string]string{
		"unknown":   strings.TrimSuffix(string(body), "}") + `,"extra":1}`,
		"duplicate": strings.TrimSuffix(string(body), "}") + `,"Root":"` + q.Root + `"}`,
		"case":      strings.Replace(string(body), `"Root":`, `"root":`, 1),
		"omitted":   strings.Replace(string(body), `"Schema":"`+RequestSchema+`",`, "", 1),
		"null":      strings.Replace(string(body), `"Root":"`+q.Root+`"`, `"Root":null`, 1),
		"trailing":  string(body) + ` {}`,
	} {
		t.Run(name, func(t *testing.T) {
			bad := inert(t, filepath.Join(q.Root, name+".json"), raw)
			if _, e := readRequest(bad, true); e == nil {
				t.Fatal("ambiguous request accepted")
			}
		})
	}
	for name, mutate := range map[string]func(*Request){
		"schema":           func(q *Request) { q.Schema = "other" },
		"partial":          func(q *Request) { q.Logger.SHA256 = "" },
		"changed compiler": func(q *Request) { q.Compiler.SHA256 = strings.Repeat("0", 64) },
		"changed self":     func(q *Request) { q.Logger.SHA256 = strings.Repeat("0", 64) },
		"relative":         func(q *Request) { q.Target.Path = "target" },
		"unclean":          func(q *Request) { q.Out = q.Root + "/artifacts/tensor/../capture" },
		"escape":           func(q *Request) { q.Out = filepath.Join(q.Root, "escape") },
		"role alias":       func(q *Request) { q.Logger = q.Compiler },
	} {
		t.Run(name, func(t *testing.T) {
			bad := q
			mutate(&bad)
			if _, e := readRequest(requestPin(t, bad, name+".json"), true); e == nil {
				t.Fatal("invalid request accepted")
			}
		})
	}
	// The independent request pin itself must not be inferred from new bytes.
	if e := os.WriteFile(p.Path, append(body, ' '), 0600); e != nil {
		t.Fatal(e)
	}
	if _, e := readRequest(p, true); e == nil {
		t.Fatal("changed request accepted")
	}
}

func TestOutputAndSymlinkCollisions(t *testing.T) {
	q, p := requestFixture(t)
	if e := os.Mkdir(q.Out, 0700); e != nil {
		t.Fatal(e)
	}
	if _, e := readRequest(p, true); e == nil {
		t.Fatal("existing output accepted")
	}
	q.Out += "-other"
	link := filepath.Join(q.Root, "compiler-link")
	if e := os.Symlink(q.Compiler.Path, link); e != nil {
		t.Fatal(e)
	}
	q.Compiler.Path = link
	if _, e := readRequest(requestPin(t, q, "link.json"), true); e == nil {
		t.Fatal("noncanonical compiler accepted")
	}
}

func TestUnselectedAndWrongRoleMetadata(t *testing.T) {
	q, p := requestFixture(t)
	if _, e := Preflight(p); e == nil {
		t.Fatal("unselected inert artifact accepted")
	}
	// Minimal role-rejection metadata is deliberately not a successful native
	// artifact. sessionPlan must reject before any command can be formed.
	for _, role := range []cgobuild.BuildRole{cgobuild.H08Command, "", "unknown"} {
		b, e := json.Marshal(cgobuild.Receipt{Role: role})
		if e != nil {
			t.Fatal(e)
		}
		q.ControllerBuild = inert(t, filepath.Join(q.Root, "role.json"), string(b))
		if _, e = sessionPlan(q, Session{}); e == nil {
			t.Fatal("wrong role accepted")
		}
	}
	if _, e := os.Lstat(q.Out); !os.IsNotExist(e) {
		t.Fatal("metadata preflight created output")
	}
}

func TestCRecipeAndSourceBoundary(t *testing.T) {
	q, _ := requestFixture(t)
	dir := filepath.Join(q.Out, "tmp", "test-case")
	if e := os.MkdirAll(dir, 0700); e != nil {
		t.Fatal(e)
	}
	source := inert(t, filepath.Join(dir, "backend.c"), "unsigned int nt_version(void){return 5;}\n")
	args := []string{"-shared", "-fPIC", source.Path, "-o", filepath.Join(dir, "backend.so")}
	if _, e := compilerArgs(q, args); e != nil {
		t.Fatal(e)
	}
	if _, _, e := sourceBytes(source.Path); e != nil {
		t.Fatal(e)
	}
	for name, change := range map[string]func([]string) []string{
		"extra":          func(a []string) []string { return append(a, "-lcuda") },
		"C++":            func(a []string) []string { a[2] += "pp"; return a },
		"relative":       func(a []string) []string { a[2] = "backend.c"; return a },
		"escape":         func(a []string) []string { a[4] = filepath.Join(q.Root, "escape.so"); return a },
		"alternate flag": func(a []string) []string { a[1] = "-fpic"; return a },
	} {
		t.Run(name, func(t *testing.T) {
			if _, e := compilerArgs(q, change(append([]string{}, args...))); e == nil {
				t.Fatal("recipe accepted")
			}
		})
	}
	for _, body := range []string{"#include <torch/torch.h>\n", "# include <cuda.h>\n", "#include \"x.h\"\n", "namespace x {}", "extern \"C\" void x();", "#define H <ATen/ATen.h>\n#include H\n"} {
		inert(t, source.Path, body)
		if _, _, e := sourceBytes(source.Path); e == nil {
			t.Fatalf("framework/C++ source accepted: %q", body)
		}
	}
	inert(t, args[4], "existing")
	if _, e := compilerArgs(q, args); e == nil {
		t.Fatal("output overwrite accepted")
	}
	if e := os.Remove(args[4]); e != nil {
		t.Fatal(e)
	}
	outside := t.TempDir()
	if e := os.Symlink(outside, filepath.Join(dir, "outside")); e != nil {
		t.Fatal(e)
	}
	args[4] = filepath.Join(dir, "outside", "escape.so")
	if _, e := compilerArgs(q, args); e == nil {
		t.Fatal("symlink escape accepted")
	}
}

func TestPrivateSessionBoundary(t *testing.T) {
	q, request := requestFixture(t)
	path := Pin{Path: filepath.Join(q.Out, "session.json"), SHA256: strings.Repeat("1", 64)}
	s := Session{Schema: sessionSchema, Request: request, Logger: q.Logger, Compiler: q.Compiler, Target: q.Target, Root: q.Root, Out: q.Out, Parent: Process{PID: 123, Start: "456", Cgroup: "0::/inert\n"}, Deadline: time.Now().Add(time.Minute).UTC().Format(time.RFC3339Nano)}
	if e := sessionBinding(s, path, q); e != nil {
		t.Fatal(e)
	}
	for name, mutate := range map[string]func(*Session){
		"schema":        func(s *Session) { s.Schema = "other" },
		"self":          func(s *Session) { s.Logger = q.Compiler },
		"compiler":      func(s *Session) { s.Compiler = q.Target },
		"target":        func(s *Session) { s.Target = q.Logger },
		"parent start":  func(s *Session) { s.Parent.Start = "" },
		"parent cgroup": func(s *Session) { s.Parent.Cgroup = "" },
		"expired":       func(s *Session) { s.Deadline = time.Now().Add(-time.Second).UTC().Format(time.RFC3339Nano) },
	} {
		t.Run(name, func(t *testing.T) {
			bad := s
			mutate(&bad)
			if e := sessionBinding(bad, path, q); e == nil {
				t.Fatal("session mismatch accepted")
			}
		})
	}
	if e := sessionBinding(s, Pin{Path: filepath.Join(q.Root, "session.json")}, q); e == nil {
		t.Fatal("public session path accepted")
	}
	parent, e := identity(os.Getpid())
	if e != nil {
		t.Fatal(e)
	}
	for _, field := range []string{"start", "cgroup"} {
		bad := s
		bad.Parent = parent
		if field == "start" {
			bad.Parent.Start += "-changed"
		} else {
			bad.Parent.Cgroup += "changed"
		}
		if _, _, e := liveSession(bad, q); e == nil || !strings.Contains(e.Error(), "parent PID/start/cgroup changed") {
			t.Fatalf("changed live parent %s was not rejected: %v", field, e)
		}
	}
}

func TestBoundaryRejectionStartsNoCommand(t *testing.T) {
	q, request := requestFixture(t)
	if e := checkSelf(q.Logger); e == nil {
		t.Fatal("inert selected logger matched actual running test executable")
	}
	if _, e := Capture(context.Background(), request, false); e == nil {
		t.Fatal("admission omitted")
	}
	for _, kind := range []string{"changed pin", "cancelled", "argv"} {
		t.Run(kind, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			p := q.Compiler
			args := []string{p.Path}
			switch kind {
			case "changed pin":
				p.SHA256 = strings.Repeat("0", 64)
			case "cancelled":
				cancel()
			case "argv":
				args[0] = q.Target.Path
			}
			dir := t.TempDir()
			r, e := runLogged(ctx, dir, q.Root, p, args, []string{"PATH=/usr/bin:/bin"})
			if e == nil || r.Process.PID != 0 || r.Command.Exit != -1 || r.Observation.Born.PID != 0 {
				t.Fatal("boundary reached child execution")
			}
			for _, name := range []string{"launch.json", "born.json", "process.json"} {
				if _, e = os.Lstat(filepath.Join(dir, name)); !os.IsNotExist(e) {
					t.Fatalf("unexpected launch evidence: %s", name)
				}
			}
			if _, e = os.Stat(filepath.Join(dir, "command.json")); e != nil {
				t.Fatal("failure receipt lost", e)
			}
		})
	}
}

func TestMissingSnapshotsCannotComplete(t *testing.T) {
	q, request := requestFixture(t)
	target := Process{PID: 10, PPID: 9, Start: "inert", Cgroup: "0::/inert\n"}
	for _, missing := range []string{"source", "output"} {
		t.Run(missing, func(t *testing.T) {
			dir := t.TempDir()
			s, e := writeNew(filepath.Join(dir, "session.json"), Session{Request: request})
			if e != nil {
				t.Fatal(e)
			}
			r := CompileRecord{Schema: "nanopore-h08-fake-compile/v1", Status: "compiled", Session: s, Request: request, Cwd: q.Root, Entry: "cc", Target: target,
				Helper: Process{PID: 11, PPID: 10, Start: "helper", Cgroup: target.Cgroup}, Compiler: Process{PID: 12, PPID: 11, Start: "compiler", Cgroup: target.Cgroup}}
			r.Compile.Source = Pin{Path: filepath.Join(q.Out, "tmp", "case", "backend.c"), SHA256: hash([]byte("source"))}
			r.Compile.Output = Pin{Path: filepath.Join(q.Out, "tmp", "case", "backend.so"), SHA256: hash([]byte("output"))}
			r.SuppliedArgs = []string{"-shared", "-fPIC", r.Compile.Source.Path, "-o", r.Compile.Output.Path}
			r.Compile.Command.Args = append([]string{q.Compiler.Path}, r.SuppliedArgs...)
			r.Compile.SourceBefore = Pin{Path: filepath.Join(dir, "source.c"), SHA256: r.Compile.Source.SHA256}
			r.Compile.OutputSnapshot = Pin{Path: filepath.Join(dir, "output.so"), SHA256: r.Compile.Output.SHA256}
			if missing == "source" {
				inert(t, r.Compile.OutputSnapshot.Path, "output")
			} else {
				inert(t, r.Compile.SourceBefore.Path, "source")
			}
			if e = completeCompile(q, s, dir, target, r); e == nil || !strings.Contains(e.Error(), missing+map[string]string{"source": ".c", "output": ".so"}[missing]) {
				t.Fatalf("missing snapshot was not the rejecting boundary: %v", e)
			}
		})
	}
}

func TestClosedCompilerEnvironment(t *testing.T) {
	p := cgobuild.Plan{Environment: map[string]string{"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "TZ": "UTC", "LD_PRELOAD": "/inert.so", "NANOPORE_TENSOR_BACKEND": "/inert-backend.so"}}
	env, e := compilerEnvironment(p, "/private/tmp")
	if e != nil {
		t.Fatal(e)
	}
	for _, kv := range env {
		if strings.Contains(kv, "LD_") || strings.Contains(kv, "NANOPORE") {
			t.Fatal("unrelated plan environment copied")
		}
	}
	p.Environment["PATH"] = "/unselected"
	if _, e = compilerEnvironment(p, "/private/tmp"); e == nil {
		t.Fatal("changed plan environment accepted")
	}
}

// The signature is intentionally compiled against the public dependency; no
// private import, reflection shim or copied fake-proof reader substitutes it.
var _ func(Pin, Pin, Pin, Pin) (h08.FakeProof, error) = h08.ReadFakeLoaderProof
