package decodergpu

import (
	"encoding/json"
	"errors"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func testFile(t *testing.T, dir, name, body string) File {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := artifactPin(path)
	if err != nil {
		t.Fatal(err)
	}
	return p
}
func TestNativePhaseAdmissionCreatesNoChildOrAttempt(t *testing.T) {
	out := filepath.Join(t.TempDir(), "not-created")
	for _, phase := range []string{"query", "headers", "build"} {
		if _, err := RunCompilerPhase(CompilerPhaseOptions{Phase: phase, Out: out}); err == nil || !strings.Contains(err.Error(), "admission") {
			t.Fatalf("phase admitted: %v", err)
		}
	}
	if err := RunCapture(CaptureOptions{Out: out}); err == nil || !strings.Contains(err.Error(), "admission") {
		t.Fatal(err)
	}
	if _, err := os.Stat(out); !os.IsNotExist(err) {
		t.Fatalf("unadmitted attempt was created: %v", err)
	}
}
func TestCompilerProofBoundaryCoversFailure(t *testing.T) {
	missing := errors.New("missing compiler-only DSO")
	childFailure := errors.New("compiler failed")
	calls := 0
	_, err := checkedCompilerCommand(func() error { return missing }, func() (runner.CommandReceipt, error) { calls++; return runner.CommandReceipt{}, nil })
	if !errors.Is(err, missing) || calls != 0 {
		t.Fatalf("missing DSO reached child: %d %v", calls, err)
	}
	checks := 0
	_, err = checkedCompilerCommand(func() error {
		checks++
		if checks == 2 {
			return missing
		}
		return nil
	}, func() (runner.CommandReceipt, error) { calls++; return runner.CommandReceipt{Exit: 1}, childFailure })
	if calls != 1 || checks != 2 || !errors.Is(err, missing) || !errors.Is(err, childFailure) {
		t.Fatalf("failure lost after-command proof: %d %d %v", calls, checks, err)
	}
}
func TestCompilerSelectionDependenciesAndLinkMapRejectDrift(t *testing.T) {
	dir := t.TempDir()
	tool := testFile(t, dir, "cc1plus", "fake selected tool")
	other := testFile(t, dir, "other", "different selection")
	q := ResourceQuery{"-print-prog-name=cc1plus", tool}
	static := testFile(t, dir, "libgcc.a", "fake static fallback")
	if got, err := resolveLibrary("-lgcc", []string{dir}); err != nil || got != static {
		t.Fatalf("linker-script static fallback: %v %v", got, err)
	}
	if err := verifyQuery(tool.Path+"\n", q, dir); err != nil {
		t.Fatal(err)
	}
	for _, raw := range []string{other.Path, tool.Path + "\n" + other.Path, "", "cc1plus"} {
		if err := verifyQuery(raw, q, dir); err == nil {
			t.Fatalf("unselected query output accepted: %q", raw)
		}
	}
	if err := verifyDependencies([]string{"a", "b"}, []string{"a", "b"}); err != nil {
		t.Fatal(err)
	}
	if err := verifyDependencies([]string{"a", "b"}, []string{"a", "c"}); err == nil {
		t.Fatal("changed actual header set accepted")
	}
	koi := testFile(t, dir, "libkoi.a", "fake original archive")
	obj := testFile(t, dir, "oracle.o", "fake object")
	inputs := map[string]string{koi.Path: koi.SHA256}
	objects := map[string]string{obj.Path: obj.SHA256}
	good := "LOAD " + koi.Path + "\nLOAD " + obj.Path + "\n" + koi.Path + "(decoder_lib.cu.o)\n"
	if err := verifyDecoderLinkMap(good, dir, inputs, objects, koi.Path); err != nil {
		t.Fatal(err)
	}
	for _, bad := range []string{good + "LOAD " + other.Path + "\n", strings.ReplaceAll(good, "(decoder_lib.cu.o)", "(unrelated.o)"), ""} {
		if err := verifyDecoderLinkMap(bad, dir, inputs, objects, koi.Path); err == nil {
			t.Fatal("unknown map input/decoder selection accepted")
		}
	}
}
func TestImmutableSnapshotsAndNativeLimits(t *testing.T) {
	dir := t.TempDir()
	source := testFile(t, dir, "source.go", "original source")
	bindings, copies, err := snapshotPins(map[string]string{source.Path: source.SHA256}, filepath.Join(dir, "copies"))
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(source.Path, []byte("later consumer"), 0600); err != nil {
		t.Fatal(err)
	}
	if err = checkPins(copies); err != nil {
		t.Fatal("historical snapshot lost", err)
	}
	if err = source.verify(); err == nil {
		t.Fatal("current source drift accepted")
	}
	copyPath := bindings[source.Path]
	copyPin, err := artifactPin(copyPath)
	if err != nil {
		t.Fatal(err)
	}
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	relative, err := filepath.Rel(cwd, copyPath)
	if err != nil {
		t.Fatal(err)
	}
	if got, e := normalizeAnchor(File{relative, copyPin.SHA256}); e != nil || got != copyPin {
		t.Fatalf("relative external anchor: %v %v", got, e)
	}
	if _, e := normalizeAnchor(File{relative, ""}); e == nil {
		t.Fatal("unanchored external receipt accepted")
	}
	if err = os.Chmod(bindings[source.Path], 0600); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(bindings[source.Path], []byte("tampered copy"), 0600); err != nil {
		t.Fatal(err)
	}
	if err = checkPins(copies); err == nil {
		t.Fatal("tampered historical snapshot accepted")
	}
	if err = checkNativeLimits("200000 100000", "3221225472", "0"); err != nil {
		t.Fatal(err)
	}
	for _, v := range [][3]string{{"max 100000", "3221225472", "0"}, {"300000 100000", "3221225472", "0"}, {"200000 100000", "max", "0"}, {"200000 100000", "4294967296", "0"}, {"200000 100000", "3221225472", "1"}} {
		if err = checkNativeLimits(v[0], v[1], v[2]); err == nil {
			t.Fatalf("unbounded native scope accepted: %v", v)
		}
	}
}
func TestGoCustodyRequiresActualSourcesAndEmbeds(t *testing.T) {
	dir := t.TempDir()
	g := GoClosure{Pins: map[string]string{}, Selection: map[string]string{}, Controller: filepath.Join(dir, "controller")}
	for _, name := range []string{"go", "compile", "asm", "link", "source.go", "embed.bin", "go.mod"} {
		p := testFile(t, dir, name, name)
		g.Pins[p.Path] = p.SHA256
		g.Selection[p.Path] = p.Path
	}
	deps, _ := json.Marshal(goPackage{Dir: dir, ImportPath: "fake", GoFiles: []string{"source.go"}, EmbedFiles: []string{"embed.bin"}})
	dep := testFile(t, dir, "deps.json", string(deps))
	env, _ := json.Marshal(map[string]string{"GOTOOLDIR": dir, "GOMOD": filepath.Join(dir, "go.mod"), "CGO_ENABLED": "0", "GOOS": "linux", "GOARCH": "amd64", "GOAMD64": "v1", "GOWORK": "off", "GOENV": "", "GOFLAGS": ""})
	environment := testFile(t, dir, "env.json", string(env))
	goBin := filepath.Join(dir, "go")
	g.Commands = []runner.CommandReceipt{{Args: []string{goBin, "list", "-deps", "-json", "-pgo=off", "./verification/decodergpu/cmd"}, Stdout: dep.Path}, {Args: []string{goBin, "env", "-json"}, Stdout: environment.Path}, {Args: []string{goBin, "build", "-x", "-pgo=off", "-trimpath", "-buildvcs=true", "-o", g.Controller, "./verification/decodergpu/cmd"}}}
	if err := verifyGoSelection(g, false); err != nil {
		t.Fatal(err)
	}
	changed := g
	changed.Pins = maps.Clone(g.Pins)
	delete(changed.Pins, filepath.Join(dir, "embed.bin"))
	if err := verifyGoSelection(changed, false); err == nil {
		t.Fatal("omitted actual embed accepted")
	}
	changed = g
	changed.Selection = maps.Clone(g.Selection)
	delete(changed.Selection, filepath.Join(dir, "source.go"))
	if err := verifyGoSelection(changed, false); err == nil {
		t.Fatal("omitted selected source binding accepted")
	}
	changed = g
	changed.Pins = maps.Clone(g.Pins)
	delete(changed.Pins, filepath.Join(dir, "go.mod"))
	if err := verifyGoSelection(changed, false); err == nil {
		t.Fatal("omitted actual module input accepted")
	}
}
func TestRuntimeMapAndSelectedImageGates(t *testing.T) {
	dir := t.TempDir()
	known := testFile(t, dir, "libknown.so", "fake mapped object")
	unknown := testFile(t, dir, "libunknown.so", "other mapped object")
	line := func(path string) string { return "1000-2000 r--p 00000000 00:01 1 " + path + "\n" }
	allowed := map[string]string{known.Path: known.SHA256}
	if err := CheckRuntimeMaps(line(known.Path), allowed); err != nil {
		t.Fatal(err)
	}
	for _, body := range []string{line(known.Path) + line(unknown.Path), line(known.Path + " (deleted)"), ""} {
		if err := CheckRuntimeMaps(body, allowed); err == nil {
			t.Fatal("unknown runtime maps accepted")
		}
	}
	if _, err := bindSelectedImages(nil, nil, nil, ImagePolicy{}); err == nil {
		t.Fatal("missing actual module/function records released selected-image gate")
	}
}
