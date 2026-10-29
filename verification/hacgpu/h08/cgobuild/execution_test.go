package cgobuild

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

const fakeSource = `package main
import("encoding/binary";"fmt";"os";"path/filepath";"slices";"strings")
func value(args []string,name string)string{for i,a:=range args{if a==name&&i+1<len(args){return args[i+1]}};return ""}
func main(){a:=os.Args[1:];f,e:=os.OpenFile(os.Getenv("FAKE_CC_LOG"),os.O_CREATE|os.O_APPEND|os.O_WRONLY,0600);if e!=nil{panic(e)};fmt.Fprintln(f,strings.Join(a," "));f.Close()
if p:=os.Getenv("FAKE_ENV_LOG");p!=""{os.WriteFile(p,[]byte(os.Getenv("CGO_LDFLAGS")+"\n"+os.Getenv("GOROOT")+"\n"),0600);return}
if p:=os.Getenv("FAKE_QUERY_PATH");p!=""&&len(a)==1&&strings.HasPrefix(a[0],"-print-file-name="){fmt.Println(p);return}
if slices.Contains(a,"-Qunused-arguments"){fmt.Fprintln(os.Stderr,"unsupported option -Qunused-arguments");os.Exit(1)}
source:="";for _,s:=range a{if strings.HasSuffix(s,".c")||strings.HasSuffix(s,".S"){source=s}};if source!=""{source,_=filepath.Abs(source)}
if dep:=value(a,"-MF");dep!=""{body:="native:";if source!=""{body+=" "+source};body+="\n";if e=os.WriteFile(dep,[]byte(body),0600);e!=nil{panic(e)}}
if slices.Contains(a,"-M"){if path:=os.Getenv("FAKE_EDIT_AFTER_M");path!=""{os.WriteFile(path,[]byte("changed by fake after preflight"),0600)};return}
if slices.Contains(a,"-O0"){fmt.Fprintln(os.Stderr,"source-defined intentional Cgo diagnostic");os.Exit(1)}
if out:=value(a,"-o");out!=""&&out!="/dev/null"{body:=make([]byte,64);copy(body,[]byte{0x7f,'E','L','F',2,1,1});binary.LittleEndian.PutUint16(body[16:],1);binary.LittleEndian.PutUint16(body[18:],62);binary.LittleEndian.PutUint32(body[20:],1);binary.LittleEndian.PutUint16(body[52:],64);if e=os.WriteFile(out,body,0600);e!=nil{panic(e)}}
}`

var fakeOnce sync.Once
var fakeBinary string
var fakeBuildError error

func fakeCompiler(t *testing.T) string {
	t.Helper()
	fakeOnce.Do(func() {
		dir, e := os.MkdirTemp("", "h08-sealed-go-fake-")
		if e != nil {
			fakeBuildError = e
			return
		}
		source := filepath.Join(dir, "main.go")
		fakeBinary = filepath.Join(dir, "gcc-15")
		if e = os.WriteFile(source, []byte(fakeSource), 0600); e != nil {
			fakeBuildError = e
			return
		}
		c := exec.Command(filepath.Join(runtime.GOROOT(), "bin/go"), "build", "-p=1", "-pgo=off", "-trimpath", "-buildvcs=false", "-o", fakeBinary, source)
		c.Env = append(os.Environ(), "CGO_ENABLED=0", "GOTOOLCHAIN=local", "GOPROXY=off", "GOSUMDB=off", "GOENV=off", "GOWORK=off")
		output, e := c.CombinedOutput()
		if e != nil {
			fakeBuildError = fmt.Errorf("Go-only fake build: %w: %s", e, output)
		}
	})
	if fakeBuildError != nil {
		t.Fatal(fakeBuildError)
	}
	return fakeBinary
}
func fakeFixture(t *testing.T) (Plan, Session, *Lease, string) {
	t.Helper()
	root := t.TempDir()
	work := filepath.Join(root, "work")
	pkgDir := filepath.Join(root, "pkg")
	obj := filepath.Join(work, "go-work", "b001")
	for _, dir := range []string{pkgDir, obj, filepath.Join(work, "cgo-tmp")} {
		if e := os.MkdirAll(dir, 0700); e != nil {
			t.Fatal(e)
		}
	}
	p := Plan{Options: Options{Root: root, GoRoot: runtime.GOROOT(), WorkRoot: work}, Inputs: map[string]string{}, Environment: map[string]string{"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "TZ": "UTC", "FAKE_CC_LOG": filepath.Join(root, "spawn.log")}}
	pkg := Package{ImportPath: Module + "/internal/tensor", Directory: pkgDir, Cgo: []string{"native_cgo.go"}, C: []string{"loader.c"}, Headers: []string{"loader.h", "abi.h"}}
	p.Packages = []Package{pkg}
	for _, name := range []string{"native_cgo.go", "loader.c", "loader.h", "abi.h"} {
		path := filepath.Join(pkgDir, name)
		if e := os.WriteFile(path, []byte("fixture "+name), 0600); e != nil {
			t.Fatal(e)
		}
		q, e := pinned(path, p.Inputs)
		if e != nil {
			t.Fatal(e)
		}
		_ = q
	}
	cc, e := pinned(fakeCompiler(t), p.Inputs)
	if e != nil {
		t.Fatal(e)
	}
	p.CC = cc
	p.CXX = Pin{Path: filepath.Join(root, "g++-15"), SHA256: cc.SHA256}
	p.Compiler.Tools = []closedbuild.Tool{{Name: "gcc-driver", Executable: cc}, {Name: "driver", Executable: p.CXX}}
	plugin := filepath.Join(root, "liblto_plugin.so")
	if e = os.WriteFile(plugin, []byte("fake plugin"), 0600); e != nil {
		t.Fatal(e)
	}
	q, e := pinned(plugin, p.Inputs)
	if e != nil {
		t.Fatal(e)
	}
	p.Compiler.Plugins = []Pin{q}
	resources := filepath.Join(root, "resources")
	if e = os.Mkdir(resources, 0700); e != nil {
		t.Fatal(e)
	}
	for _, name := range resourceNames {
		path := filepath.Join(resources, name)
		if e = os.WriteFile(path, []byte(name), 0600); e != nil {
			t.Fatal(e)
		}
		q, e := pinned(path, p.Inputs)
		if e != nil {
			t.Fatal(e)
		}
		p.Resources = append(p.Resources, Resource{Name: name, Input: q})
	}
	s := Session{Directory: filepath.Join(root, "session")}
	for _, name := range []string{"commands", "inputs", "derived", "rejected"} {
		if e = os.MkdirAll(filepath.Join(s.Directory, name), 0700); e != nil {
			t.Fatal(e)
		}
	}
	for _, name := range RequiredGenerated(pkg) {
		if e = os.WriteFile(filepath.Join(obj, name), []byte(name), 0600); e != nil {
			t.Fatal(e)
		}
	}
	producer := filepath.Join(root, "producer.json")
	if e = os.WriteFile(producer, []byte("sealed fake producer receipt"), 0600); e != nil {
		t.Fatal(e)
	}
	pr, e := closedPin(producer)
	if e != nil {
		t.Fatal(e)
	}
	g, e := CaptureGenerated(pkg, "translation", work, obj, filepath.Join(s.Directory, "inputs"), pr)
	if e != nil {
		t.Fatal(e)
	}
	if e = write(filepath.Join(s.Directory, "derived", "tensor.json"), g); e != nil {
		t.Fatal(e)
	}
	lease := &Lease{Schema: "nanopore-h08-go-tool-lease/v1", Kind: "cgo", Package: pkg.ImportPath, Directory: pkgDir, ObjectDir: obj, Tool: cc, Inputs: map[string]string{filepath.Join(pkgDir, "native_cgo.go"): p.Inputs[filepath.Join(pkgDir, "native_cgo.go")]}}
	return p, s, lease, p.Environment["FAKE_CC_LOG"]
}
func fakeBoundary(p Plan) error { return errors.Join(ValidateSpecification(p), checkPins(p.Inputs)) }
func diagnostics(p Plan, l *Lease) ([]string, string) {
	source := filepath.Join(p.Options.WorkRoot, "cgo-tmp", "cgo-gcc-input-fixture.c")
	args := []string{"-w", "-Wno-error", "-o" + filepath.Join(l.ObjectDir, "_cgo_1.o"), "-gdwarf-2", "-c", "-I", l.ObjectDir, "-O0", "-m64", "-fno-lto", "-fdiagnostics-color=never", "-I", ".", source}
	return args, source
}
func spawns(path string) int { b, _ := os.ReadFile(path); return strings.Count(string(b), "\n") }

func TestExpectedNonzeroCgoProbeRetainsMMDAndDoesNotBypassUnknownInputs(t *testing.T) {
	p, s, l, log := fakeFixture(t)
	t.Chdir(p.Packages[0].Directory)
	args, source := diagnostics(p, l)
	if e := os.WriteFile(source, []byte("int type_probe;"), 0600); e != nil {
		t.Fatal(e)
	}
	op, e := interposeCC(s, p, args, nil, l, false, fakeBoundary)
	if e != nil {
		t.Fatal(e)
	}
	if op.Status != "accepted_expected_exit" || op.Commands[len(op.Commands)-1].Exit != 1 || op.ActualDep.Path == "" || spawns(log) != 2 {
		t.Fatalf("expected failure proof: %+v", op)
	}
	bad := append(slices.Clone(args), "-include", "/foreign/header.h")
	before := spawns(log)
	if _, e = interposeCC(s, p, bad, nil, l, false, fakeBoundary); e == nil {
		t.Fatal("unknown input accepted through diagnostic profile")
	}
	if spawns(log) != before {
		t.Fatal("unknown probe input forked fake compiler")
	}
	if noRejections(s) == nil {
		t.Fatal("Go-tolerated rejection could qualify whole build")
	}
}
func TestUnsupportedGoProbeHasExplicitNonzeroStatusAndEmptyStdinGuard(t *testing.T) {
	p, s, _, log := fakeFixture(t)
	t.Chdir(p.Packages[0].Directory)
	args := []string{"-Qunused-arguments", "-O2", "-g", "-c", "-x", "c", "-", "-o", "/dev/null"}
	op, e := interposeCC(s, p, args, nil, nil, false, fakeBoundary)
	if e != nil {
		t.Fatal(e)
	}
	if op.Status != "accepted_expected_exit" || op.DepUnavailable == "" || spawns(log) != 2 {
		t.Fatalf("unsupported flag proof: %+v", op)
	}
	before := spawns(log)
	if _, e = interposeCC(s, p, args, []byte("#include \"foreign.h\""), nil, false, fakeBoundary); e == nil {
		t.Fatal("feature probe accepted unplanned stdin")
	}
	if spawns(log) != before {
		t.Fatal("stdin rejection occurred after fork")
	}
}
func TestNativeTUAndPreflightSourceEditGuardUseActualFakeProcess(t *testing.T) {
	p, s, l, log := fakeFixture(t)
	t.Chdir(p.Packages[0].Directory)
	args := []string{"-I", p.Packages[0].Directory, "-fPIC", "-m64", "-pthread", "-O2", "-g", "-I", l.ObjectDir, "-o", filepath.Join(l.ObjectDir, "_x001.o"), "-c", "loader.c"}
	op, e := interposeCC(s, p, args, nil, nil, false, fakeBoundary)
	if e != nil {
		t.Fatal(e)
	}
	if op.Output.Path == "" || op.ActualDep.Path == "" || spawns(log) != 2 {
		t.Fatal("fake TU output/dep evidence omitted")
	}
	args, source := diagnostics(p, l)
	if e = os.WriteFile(source, []byte("int x;"), 0600); e != nil {
		t.Fatal(e)
	}
	p.Environment["FAKE_EDIT_AFTER_M"] = filepath.Join(p.Packages[0].Directory, "native_cgo.go")
	before := spawns(log)
	if _, e = interposeCC(s, p, args, nil, l, false, fakeBoundary); e == nil {
		t.Fatal("source changed during preflight accepted")
	}
	if spawns(log) != before+1 {
		t.Fatal("actual compiler forked after preflight source edit")
	}
}
func TestMissingPluginCRTCCAndForeignRootRejectBeforeFakeFork(t *testing.T) {
	for _, kind := range []string{"plugin", "CRT", "CC", "foreign root", "generated bytes"} {
		t.Run(kind, func(t *testing.T) {
			p, s, l, log := fakeFixture(t)
			t.Chdir(p.Packages[0].Directory)
			args := []string{"-I", p.Packages[0].Directory, "-fPIC", "-m64", "-pthread", "-O2", "-g", "-I", l.ObjectDir, "-o", filepath.Join(l.ObjectDir, "_x001.o"), "-c", "loader.c"}
			switch kind {
			case "plugin":
				p.Compiler.Plugins = nil
			case "CRT":
				p.Resources = slices.DeleteFunc(p.Resources, func(r Resource) bool { return r.Name == "crt1.o" })
			case "CC":
				p.CC.Path = "/usr/bin/gcc"
			case "foreign root":
				foreign := filepath.Join(p.Options.Root, "foreign.c")
				if e := os.WriteFile(foreign, []byte("int x;"), 0600); e != nil {
					t.Fatal(e)
				}
				args[len(args)-1] = foreign
			case "generated bytes":
				if e := os.WriteFile(filepath.Join(l.ObjectDir, "native_cgo.cgo2.c"), []byte("changed"), 0600); e != nil {
					t.Fatal(e)
				}
			}
			if _, e := interposeCC(s, p, args, nil, nil, false, fakeBoundary); e == nil {
				t.Fatal("unclosed native request accepted")
			}
			if spawns(log) != 0 {
				t.Fatal("invalid request forked fake compiler")
			}
		})
	}
}
func TestIncompleteFinalManifestAndUnadmittedExecutionCannotQualify(t *testing.T) {
	_, s, _, _ := fakeFixture(t)
	s.Schema = "nanopore-h08-cgo-session/v1"
	s.Phase = "build"
	path := filepath.Join(s.Directory, "session.json")
	if e := write(path, s); e != nil {
		t.Fatal(e)
	}
	sp, e := closedPin(path)
	if e != nil {
		t.Fatal(e)
	}
	r := Receipt{Schema: "nanopore-h08-cgo-execution/v1", Status: "compiled_unexecuted", Session: sp, RuntimeHeld: true, Commands: []runner.CommandReceipt{{Exit: 0}}}
	if ValidateReceipt(Plan{}, r) == nil {
		t.Fatal("outer Go exit zero qualified incomplete final proof")
	}
	out := filepath.Join(t.TempDir(), "absent")
	if _, e = Execute(DriverOptions{Plan: Pin{Path: "/must/not/open"}, Out: out, Phase: "build"}); !errors.Is(e, runner.ErrAdmissionBlocked) {
		t.Fatalf("native admission gate: %v", e)
	}
	if _, e = os.Stat(out); !os.IsNotExist(e) {
		t.Fatal("unadmitted output created")
	}
}

func TestChildRecorderRetainsExitAndGroupCleanup(t *testing.T) {
	d := t.TempDir()
	log := filepath.Join(d, "calls")
	args := []string{fakeCompiler(t), "-Qunused-arguments"}
	r, p, e := runProgram(context.Background(), d, d, args, []string{"FAKE_CC_LOG=" + log}, nil, false, true)
	if r.Exit != 1 || expectedError(e, r.Exit, []int{0, 1}) != nil || p.PID == 0 || !p.Group {
		t.Fatalf("actual fake child receipt: %+v %v", r, e)
	}
	if _, e = processIdentity(p.PID); e == nil {
		t.Fatal("completed child still alive")
	}
}

func TestGoToolOutputAndExternalLinkerPathsRejectBeforeFork(t *testing.T) {
	p, _, l, log := fakeFixture(t)
	driver := p.CC
	cwd := p.Packages[0].Directory
	outside := filepath.Join(t.TempDir(), "new-output")
	parent := filepath.Join(p.Options.WorkRoot, "symlink-parent")
	if e := os.Symlink(filepath.Dir(outside), parent); e != nil {
		t.Fatal(e)
	}
	for _, args := range [][]string{{"-o", outside}, {"-o", filepath.Join(parent, "new-output")}, {"-o", filepath.Join(p.Options.WorkRoot, "missing-parent", "out")}, {"-o=" + outside}, {"-extld=/usr/bin/gcc"}} {
		if _, e := toolInputs(p, driver, "link", args, cwd); e == nil {
			t.Fatalf("unplanned Go tool path accepted: %v", args)
		}
	}
	if _, e := toolInputs(p, driver, "link", []string{"-o", filepath.Join(l.ObjectDir, "new-output"), "-extld=" + driver.Path}, cwd); e != nil {
		t.Fatalf("private absent output/selected extld rejected: %v", e)
	}
	if spawns(log) != 0 {
		t.Fatal("Go-tool path control forked a fake compiler")
	}
}

func TestExactMetadataCCIdentityKeepsCompileStdinRequirementSeparate(t *testing.T) {
	p, s, _, log := fakeFixture(t)
	t.Chdir(p.Options.Root)
	args := []string{"-###", "-x", "c", "-c", "-"}
	p.Queries = []Query{{Name: "cc-tool-id", Argv: append([]string{p.CC.Path}, args...)}}
	op, e := interposeCC(s, p, args, nil, nil, false, fakeBoundary)
	if e != nil {
		t.Fatal(e)
	}
	dir := filepath.Dir(filepath.Dir(op.Commands[0].Stdout))
	if len(op.Commands) != 1 || op.Stdin.Path != "" || spawns(log) != 1 {
		t.Fatal("metadata identity executed a compile profile")
	}
	if e = verifyNativeOperation(p, s, op, dir, nil, attestedQueryResources{}); e != nil {
		t.Fatal(e)
	}
	path := filepath.Join(dir, "stdin.c")
	if e = os.WriteFile(path, nil, 0400); e != nil {
		t.Fatal(e)
	}
	q, e := closedPin(path)
	if e != nil {
		t.Fatal(e)
	}
	bad := op
	bad.Stdin = q
	if e = verifyNativeOperation(p, s, bad, dir, nil, attestedQueryResources{}); e == nil || !strings.Contains(e.Error(), "unexpected retained metadata stdin") {
		t.Fatalf("metadata stdin evidence accepted: %v", e)
	}
	probe := []string{"-Qunused-arguments", "-O2", "-g", "-c", "-x", "c", "-", "-o", "/dev/null"}
	compiled, e := interposeCC(s, p, probe, nil, nil, false, fakeBoundary)
	if e != nil {
		t.Fatal(e)
	}
	if compiled.Stdin.Path == "" {
		t.Fatal("nonmetadata stdin was not captured")
	}
	compiled.Stdin = Pin{}
	if e = verifyNativeOperation(p, s, compiled, filepath.Dir(filepath.Dir(compiled.Commands[0].Stdout)), nil, attestedQueryResources{}); e == nil || !strings.Contains(e.Error(), "compiler stdin not retained") {
		t.Fatalf("nonmetadata missing stdin accepted: %v", e)
	}
	if _, e = classifyCC(p, p.Options.Root, append(slices.Clone(args), "-unknown"), nil, nil, nil); e == nil {
		t.Fatal("nonexact metadata request admitted")
	}
}
