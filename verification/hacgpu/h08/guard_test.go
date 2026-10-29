package h08

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
)

func TestH08CPU13OverlayOnlyWrapsMaintainedTestMain(t *testing.T) {
	f, e := parser.ParseFile(token.NewFileSet(), "testmain.go", CPU13OverlaySource, 0)
	if e != nil {
		t.Fatal(e)
	}
	if f.Name.Name != "tensor_test" {
		t.Fatal("overlay imports tested package internally")
	}
	functions := []string{}
	for _, d := range f.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok {
			functions = append(functions, fn.Name.Name)
		}
	}
	if !slices.Equal(functions, []string{"TestMain"}) || !strings.Contains(CPU13OverlaySource, "h08.CPU13TestMain(m.Run)") {
		t.Fatal("overlay copies or replaces maintained tests")
	}
}

func hostJSON(t *testing.T, path string, value any) Pin {
	t.Helper()
	if e := writeJSONFile(path, value); e != nil {
		t.Fatal(e)
	}
	p, e := closedbuild.PinFile(path)
	if e != nil {
		t.Fatal(e)
	}
	return p
}

func hostGuardPin(t *testing.T, p Pin) supgpu.Pin {
	t.Helper()
	v, err := guardPin(p)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func hostGuardPolicy(t *testing.T, target Pin, dir string) supgpu.RuntimePolicy {
	p := hostGuardPin(t, target)
	return supgpu.RuntimePolicy{Schema: "nanopore-supgpu-selected-runtime/v1", Lazy: "audit-every-loader-object-before-initializers/v1", Reference: p, Observer: p, Audit: p, Allowed: []supgpu.RuntimeCode{}, Alternates: []supgpu.RuntimeCode{}, Directories: []string{dir}}
}

func TestH08GuardContractRejectsRehashedPhaseAndCustodyChanges(t *testing.T) {
	dir := t.TempDir()
	request := hostJSON(t, filepath.Join(dir, "request"), map[string]string{"fixed": "input-only"})
	build := hostJSON(t, filepath.Join(dir, "build"), map[string]string{"test": "guard membership only"})
	selection := hostJSON(t, filepath.Join(dir, "selection"), map[string]string{"test": "guard membership only"})
	target := request
	c := supgpu.PhaseContract{Schema: "nanopore-audited-phase-contract/v1", Role: "observer", Action: "hac-all-active", Names: []string{"pre-model", "pre-forward", "post-forward"}}
	cp := hostJSON(t, filepath.Join(dir, "contract"), c)
	p := supgpu.GuardPlan{Schema: "nanopore-audited-guard-plan/v1", Contract: hostGuardPin(t, cp), Policy: hostGuardPolicy(t, target, dir), Inputs: []supgpu.Pin{hostGuardPin(t, request), hostGuardPin(t, build), hostGuardPin(t, selection)}}
	pp := hostJSON(t, filepath.Join(dir, "plan"), p)
	a := RuntimeAdmission{Schema: "nanopore-h08-runtime-admission/v2", Request: request, ParentBuild: build, ParentBuildSelection: selection, Plan: pp, Contract: cp, Command: supgpu.GuardCommand{Directory: dir, Output: filepath.Join(dir, "fresh")}}
	check := func(a RuntimeAdmission) error { return validateGuardInputs(a, target, request, nil, "hac-all-active") }
	if e := check(a); e != nil {
		t.Fatal(e)
	}
	for _, names := range [][]string{{"pre-model", "post-forward"}, {"pre-model", "post-forward", "pre-forward"}, {"pre-model", "pre-forward", "pre-forward"}} {
		changed := c
		changed.Names = names
		changedPin := hostJSON(t, filepath.Join(dir, "changed-contract"), changed)
		changedPlan := p
		changedPlan.Contract = hostGuardPin(t, changedPin)
		changedA := a
		changedA.Contract = changedPin
		changedA.Plan = hostJSON(t, filepath.Join(dir, "changed-plan"), changedPlan)
		if e := check(changedA); e == nil {
			t.Fatal("rehashed shortened/reordered phase contract accepted")
		}
	}
	p.Inputs = nil
	a.Plan = hostJSON(t, filepath.Join(dir, "missing-input"), p)
	if e := check(a); e == nil {
		t.Fatal("request custody removed")
	}
}

func TestH08StatusesRequireExactNamedTestsAndNoSkips(t *testing.T) {
	names := genuineNames()
	if len(names) != 11 || len(fakeNames) != 2 || slices.Contains(names, fakeNames[0]) || slices.Contains(names, fakeNames[1]) {
		t.Fatal("genuine/fake split changed")
	}
	var raw strings.Builder
	for _, name := range names {
		fmt.Fprintf(&raw, "=== RUN   %s\n--- PASS: %s (0.00s)\n", name, name)
	}
	if _, e := testStatuses([]byte(raw.String()), names); e != nil {
		t.Fatal(e)
	}
	for _, body := range []string{strings.Replace(raw.String(), "--- PASS:", "--- SKIP:", 1), strings.Replace(raw.String(), "--- PASS:", "--- FAIL:", 1), raw.String() + "=== RUN   Extra\n--- PASS: Extra (0s)\n", raw.String() + "--- PASS: " + names[0] + " (0s)\n", strings.Replace(raw.String(), "=== RUN   "+names[0]+"\n", "", 1), raw.String() + "    --- SKIP: nested (0s)\n"} {
		if _, e := testStatuses([]byte(body), names); e == nil {
			t.Fatal("incomplete/repeated/skipped result accepted")
		}
	}
}

func TestH08RuntimeCannotArmFromReceiptOrFlag(t *testing.T) {
	dir := t.TempDir()
	for _, value := range []any{map[string]any{"Admitted": true, "RuntimeHeld": false, "Status": "passed"}, RuntimeAdmission{Schema: "nanopore-h08-runtime-admission/v1"}, RuntimeAdmission{Schema: "nanopore-h08-runtime-admission/v2"}} {
		pin := hostJSON(t, filepath.Join(dir, "unclosed.json"), value)
		if _, e := Consume(context.Background(), ConsumerOptions{Admission: pin}); e == nil {
			t.Fatal("self-declared authority accepted")
		}
	}
	if _, e := os.Stat(filepath.Join(dir, "guard-output")); !os.IsNotExist(e) {
		t.Fatal("invalid admission created an attempt")
	}
}

func TestH08ChildRequiresPrivateParentBeforeRequestRead(t *testing.T) {
	if os.Getenv("H08_HOST_PIPE_CHILD") == "1" {
		_, e := RunGuardedChild(context.Background(), Pin{Path: "/does-not-exist", SHA256: strings.Repeat("a", 64)})
		if e == nil || !strings.Contains(e.Error(), "phase pipes") {
			os.Exit(2)
		}
		return
	}
	executable, e := os.Executable()
	if e != nil {
		t.Fatal(e)
	}
	command := exec.Command(executable, "-test.run=^TestH08ChildRequiresPrivateParentBeforeRequestRead$")
	command.Env = append(os.Environ(), "H08_HOST_PIPE_CHILD=1")
	if b, e := command.CombinedOutput(); e != nil {
		t.Fatalf("child private gate: %s: %v", b, e)
	}
}

func TestH08CPUFixtureGateRejectsWrongDistributionBeforeLoading(t *testing.T) {
	dir := t.TempDir()
	view := FixtureView{Schema: "nanopore-h08-cpu-fixture-view/v1", Directory: dir, Archive: Pin{Path: filepath.Join(dir, "archive"), SHA256: strings.Repeat("0", 64)}}
	p := hostJSON(t, filepath.Join(dir, "view.json"), view)
	if _, e := ReadFixtureView(p); e == nil || !strings.Contains(e.Error(), "distribution") {
		t.Fatalf("wrong distribution accepted: %v", e)
	}
	if _, e := ReadCPU13(hostJSON(t, filepath.Join(dir, "old.json"), CPU13Receipt{Schema: "nanopore-h08-cpu13/v1", Status: "passed"}), closedbuild.ArtifactOptions{}); e == nil {
		t.Fatal("unqualified CPU receipt accepted")
	}
}
