package h08

import (
	"context"
	"fmt"
	"io"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08/cgobuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func rejectedBuildPair(t *testing.T, role cgobuild.BuildRole) (Pin, Pin) {
	t.Helper()
	dir := t.TempDir()
	r := cgobuild.Receipt{Role: role, Status: "failed", Error: "host rejection fixture"}
	evidence := hostJSON(t, filepath.Join(dir, "receipt.json"), r)
	selection := hostJSON(t, filepath.Join(dir, "selection.json"), map[string]any{
		"Schema": "nanopore-h08-successful-selection/v1", "Role": role,
		"Producer": map[string]Pin{"Receipt": evidence, "Plan": r.Plan, "Controller": r.Controller, "Queries": r.Queries, "Output": r.Output},
	})
	return evidence, selection
}

func selectionRejections(t *testing.T, role cgobuild.BuildRole) (Pin, map[string]struct {
	pin  Pin
	want string
}) {
	t.Helper()
	evidence, selected := rejectedBuildPair(t, role)
	otherRole := cgobuild.CPU13Tests
	if role == otherRole {
		otherRole = cgobuild.H08Command
	}
	_, swapped := rejectedBuildPair(t, role)
	wrongRole := hostJSON(t, filepath.Join(t.TempDir(), "wrong-role.json"), map[string]any{
		"Schema": "nanopore-h08-successful-selection/v1", "Role": otherRole,
		"Producer": map[string]Pin{"Receipt": evidence, "Plan": {}, "Controller": {}, "Queries": {}, "Output": {}},
	})
	wrongVersion := hostJSON(t, filepath.Join(t.TempDir(), "version.json"), map[string]string{"Schema": "nanopore-h08-successful-selection/v0"})
	changed := selected
	changed.SHA256 = strings.Repeat("0", 64)
	body, err := readPin(selected)
	if err != nil {
		t.Fatal(err)
	}
	changedBytes := Pin{Path: filepath.Join(t.TempDir(), "changed-bytes.json"), SHA256: selected.SHA256}
	if err = os.WriteFile(changedBytes.Path, append(body, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	return evidence, map[string]struct {
		pin  Pin
		want string
	}{
		"missing":       {Pin{}, "pin differs"},
		"path-only":     {Pin{Path: selected.Path}, "pin differs"},
		"hash-only":     {Pin{SHA256: selected.SHA256}, "pin differs"},
		"changed":       {changed, "pin differs"},
		"changed-bytes": {changedBytes, "pin differs"},
		"swapped":       {swapped, "role/receipt differs"},
		"wrong-role":    {wrongRole, "role/receipt differs"},
		"version":       {wrongVersion, "unknown artifact selection schema"},
	}
}

func requireSelectionError(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("want selection rejection %q, got %v", want, err)
	}
}

func TestH08SelectedAdmissionRejectsBeforeNativeWork(t *testing.T) {
	evidence, cases := selectionRejections(t, cgobuild.H08Command)
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			q := FixedRequest{Schema: "nanopore-h08-fixed-request/v1", Phase: "all-active", ControllerBuild: evidence, ControllerBuildSelection: c.pin, Out: filepath.Join(dir, "numerical")}
			request := hostJSON(t, filepath.Join(dir, "request.json"), q)
			a := RuntimeAdmission{Schema: "nanopore-h08-runtime-admission/v2", Request: request, ParentBuild: evidence, ParentBuildSelection: c.pin, Command: supgpu.GuardCommand{Output: filepath.Join(dir, "guard")}}
			admission := hostJSON(t, filepath.Join(dir, "admission.json"), a)
			_, _, err := ReadAdmission(admission)
			requireSelectionError(t, err, c.want)
			_, err = Consume(context.Background(), ConsumerOptions{Admission: admission})
			requireSelectionError(t, err, c.want)
			_, err = consumeGuarded(context.Background(), q, nil)
			requireSelectionError(t, err, c.want)
			requireSelectionError(t, actualController(evidence, c.pin), c.want)
			for _, path := range []string{q.Out, a.Command.Output} {
				if _, err = os.Stat(path); !os.IsNotExist(err) {
					t.Fatalf("rejected selection created output %s: %v", path, err)
				}
			}
			cpu := CPU13Request{Schema: "nanopore-h08-cpu13-request/v1"}
			a.Request = hostJSON(t, filepath.Join(dir, "cpu-request.json"), cpu)
			admission = hostJSON(t, filepath.Join(dir, "cpu-admission.json"), a)
			_, _, err = readCPU13Admission(admission)
			requireSelectionError(t, err, c.want)
			_, err = RunCPU13(admission)
			requireSelectionError(t, err, c.want)
		})
	}
}

func TestH08RequestVersionsAndSameParentPair(t *testing.T) {
	build, selection := rejectedBuildPair(t, cgobuild.H08Command)
	_, different := rejectedBuildPair(t, cgobuild.H08Command)
	for _, phase := range []string{"all-active", "padding"} {
		for _, name := range []string{"admission-version", "request-version", "parent-build", "parent-selection", "omitted-selection"} {
			t.Run(phase+"/"+name, func(t *testing.T) {
				dir := t.TempDir()
				q := FixedRequest{Schema: "nanopore-h08-fixed-request/v1", Phase: phase, ControllerBuild: build, ControllerBuildSelection: selection, AllActiveReceipt: build}
				a := RuntimeAdmission{Schema: "nanopore-h08-runtime-admission/v2", ParentBuild: build, ParentBuildSelection: selection}
				want := "custody differs"
				switch name {
				case "admission-version":
					a.Schema = "nanopore-h08-runtime-admission/v1"
					want = "admission schema differs"
				case "request-version":
					q.Schema = ""
					want = "request schema differs"
				case "parent-build":
					a.ParentBuild = different
				case "parent-selection":
					a.ParentBuildSelection = different
				case "omitted-selection":
					q.ControllerBuildSelection = Pin{}
					a.ParentBuildSelection = Pin{}
					want = "pin differs"
				}
				var wire any = q
				if name == "omitted-selection" {
					wire = map[string]any{"Schema": q.Schema, "Phase": q.Phase, "ControllerBuild": build, "AllActiveReceipt": build}
				}
				a.Request = hostJSON(t, filepath.Join(dir, "request.json"), wire)
				_, _, err := ReadAdmission(hostJSON(t, filepath.Join(dir, "admission.json"), a))
				requireSelectionError(t, err, want)
			})
		}
	}
	for _, schema := range []string{"", "nanopore-h08-cpu13-request/v0"} {
		dir := t.TempDir()
		q := CPU13Request{Schema: schema, ControllerBuild: build, ControllerBuildSelection: selection}
		a := RuntimeAdmission{Schema: "nanopore-h08-runtime-admission/v2", Request: hostJSON(t, filepath.Join(dir, "request"), q)}
		_, _, err := readCPU13Admission(hostJSON(t, filepath.Join(dir, "admission"), a))
		requireSelectionError(t, err, "CPU13 request schema differs")
	}
	dir := t.TempDir()
	_, _, err := readCPU13Admission(hostJSON(t, filepath.Join(dir, "old-admission"), RuntimeAdmission{Schema: "nanopore-h08-runtime-admission/v1"}))
	requireSelectionError(t, err, "admission schema differs")
}

func TestH08ChildrenRejectSelectionBeforeNativeWork(t *testing.T) {
	if profile := os.Getenv("H08_SELECTION_CHILD"); profile != "" {
		request := Pin{Path: os.Getenv("H08_SELECTION_REQUEST"), SHA256: os.Getenv("H08_SELECTION_SHA")}
		if profile == "fixed" {
			_, err := RunGuardedChild(context.Background(), request)
			if err == nil {
				t.Fatal("child admitted invalid selection")
			}
			fmt.Println(err)
			return
		}
		before := runtimeEnvironment()
		os.Args = []string{os.Args[0], "-h08-request", request.Path, "-h08-request-sha", request.SHA256, "-test.run", "^(" + strings.Join(genuineNames(), "|") + ")$"}
		ran := false
		code := CPU13TestMain(func() int { ran = true; return 0 })
		if code != 1 || ran || !maps.Equal(before, runtimeEnvironment()) || os.Getenv("NANOPORE_TENSOR_BACKEND") != "selection-sentinel" {
			t.Fatal("invalid selection reached tests or changed backend environment")
		}
		return
	}
	for _, profile := range []string{"fixed", "cpu13"} {
		role := cgobuild.H08Command
		if profile == "cpu13" {
			role = cgobuild.CPU13Tests
		}
		build, cases := selectionRejections(t, role)
		oldRequest := cases["missing"]
		oldRequest.want = "request schema differs"
		cases["request-version"] = oldRequest
		for name, c := range cases {
			t.Run(profile+"/"+name, func(t *testing.T) {
				dir := t.TempDir()
				var q any = FixedRequest{Schema: "nanopore-h08-fixed-request/v1", Phase: "all-active", ControllerBuild: build, ControllerBuildSelection: c.pin, Out: filepath.Join(dir, "numerical")}
				if profile == "cpu13" {
					q = CPU13Request{Schema: "nanopore-h08-cpu13-request/v1", ControllerBuild: build, ControllerBuildSelection: c.pin}
				}
				if name == "missing" {
					schema := "nanopore-h08-fixed-request/v1"
					if profile == "cpu13" {
						schema = "nanopore-h08-cpu13-request/v1"
					}
					q = map[string]any{"Schema": schema, "Phase": "all-active", "ControllerBuild": build}
				}
				if name == "request-version" {
					q = map[string]any{"Schema": "old-unversioned-request", "Phase": "all-active", "ControllerBuild": build}
				}
				request := hostJSON(t, filepath.Join(dir, "request"), q)
				executable, err := os.Executable()
				if err != nil {
					t.Fatal(err)
				}
				out, writer, err := os.Pipe()
				if err != nil {
					t.Fatal(err)
				}
				defer out.Close()
				defer writer.Close()
				reader, in, err := os.Pipe()
				if err != nil {
					t.Fatal(err)
				}
				defer reader.Close()
				defer in.Close()
				ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
				defer cancel()
				command := exec.CommandContext(ctx, executable, "-test.run=^TestH08ChildrenRejectSelectionBeforeNativeWork$")
				command.Env = append(os.Environ(), "H08_SELECTION_CHILD="+profile, "H08_SELECTION_REQUEST="+request.Path, "H08_SELECTION_SHA="+request.SHA256, "NANOPORE_TENSOR_BACKEND=selection-sentinel")
				command.ExtraFiles = []*os.File{out, in, writer, reader} // FD5/6 are phase pipes; no approval is sent.
				body, err := command.CombinedOutput()
				if err != nil || !strings.Contains(string(body), c.want) {
					t.Fatalf("child selector boundary: %s: %v", body, err)
				}
				writer.Close()
				if body, err = io.ReadAll(out); err != nil || len(body) != 0 {
					t.Fatalf("rejected child requested a native phase: %s: %v", body, err)
				}
				if body, err = os.ReadFile(filepath.Join(dir, "numerical", "consumer.json")); !os.IsNotExist(err) {
					t.Fatalf("rejected child wrote numerical evidence: %s: %v", body, err)
				}
			})
		}
	}
}

func TestH08GuardRequiresBothRolePairs(t *testing.T) {
	dir := t.TempDir()
	parent, parentSelection := rejectedBuildPair(t, cgobuild.H08Command)
	target, targetSelection := rejectedBuildPair(t, cgobuild.CPU13Tests)
	q := CPU13Request{Schema: "nanopore-h08-cpu13-request/v1", ControllerBuild: target, ControllerBuildSelection: targetSelection}
	request := hostJSON(t, filepath.Join(dir, "request"), q)
	c := supgpu.PhaseContract{Schema: "nanopore-audited-phase-contract/v1", Role: "observer", Action: "hac-cpu13", Names: []string{"pre-model", "pre-forward", "post-forward"}}
	cp := hostJSON(t, filepath.Join(dir, "contract"), c)
	required := []Pin{parent, parentSelection, target, targetSelection}
	p := supgpu.GuardPlan{Schema: "nanopore-audited-guard-plan/v1", Contract: guardPin(cp), Policy: supgpu.RuntimePolicy{Reference: guardPin(target), Observer: guardPin(target)}, Inputs: []supgpu.Pin{guardPin(request)}}
	for _, pin := range required {
		p.Inputs = append(p.Inputs, guardPin(pin))
	}
	a := RuntimeAdmission{Schema: "nanopore-h08-runtime-admission/v2", Request: request, ParentBuild: parent, ParentBuildSelection: parentSelection, Contract: cp, Command: supgpu.GuardCommand{Directory: dir, Output: filepath.Join(dir, "guard")}}
	check := func(plan supgpu.GuardPlan) error {
		a.Plan = hostJSON(t, filepath.Join(dir, "plan"), plan)
		var decoded RuntimeAdmission
		if err := pinnedJSON(hostJSON(t, filepath.Join(dir, "admission"), a), &decoded); err != nil {
			return err
		}
		return validateGuardInputs(decoded, target, request, []Pin{target, targetSelection}, "hac-cpu13")
	}
	if err := check(p); err != nil {
		t.Fatal(err)
	}
	for _, omitted := range required {
		mutant := p
		mutant.Inputs = slices.DeleteFunc(slices.Clone(p.Inputs), func(pin supgpu.Pin) bool { return pin == guardPin(omitted) })
		requireSelectionError(t, check(mutant), "guard omits required custody input")
	}
}

func TestH08NumericalReceiptBindsSelectedPair(t *testing.T) {
	build, selection := rejectedBuildPair(t, cgobuild.H08Command)
	_, other := rejectedBuildPair(t, cgobuild.H08Command)
	_, wrongRole := rejectedBuildPair(t, cgobuild.CPU13Tests)
	q := FixedRequest{ControllerBuild: build, ControllerBuildSelection: selection}
	r := ConsumerReceipt{Schema: "nanopore-hac-fixed-consumer/v4", ControllerBuild: build, ControllerBuildSelection: selection, Inputs: map[string]string{build.Path: build.SHA256, selection.Path: selection.SHA256}}
	if err := validateConsumerBuild(r, q); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"old-version", "missing", "path-only", "hash-only", "changed", "swapped", "wrong-role", "build-membership", "selection-membership"} {
		t.Run(name, func(t *testing.T) {
			mutant := r
			mutant.Inputs = maps.Clone(r.Inputs)
			switch name {
			case "old-version":
				mutant.Schema = "nanopore-hac-fixed-consumer/v3"
			case "missing":
				mutant.ControllerBuildSelection = Pin{}
			case "path-only":
				mutant.ControllerBuildSelection.SHA256 = ""
			case "hash-only":
				mutant.ControllerBuildSelection.Path = ""
			case "changed":
				mutant.ControllerBuildSelection.SHA256 = strings.Repeat("0", 64)
			case "swapped":
				mutant.ControllerBuildSelection = other
			case "wrong-role":
				mutant.ControllerBuildSelection = wrongRole
			case "build-membership":
				delete(mutant.Inputs, build.Path)
			case "selection-membership":
				delete(mutant.Inputs, selection.Path)
			}
			var decoded ConsumerReceipt
			if err := pinnedJSON(hostJSON(t, filepath.Join(t.TempDir(), "consumer.json"), mutant), &decoded); err != nil {
				t.Fatal(err)
			}
			requireSelectionError(t, validateConsumerBuild(decoded, q), "numerical receipt")
		})
	}
	r.ControllerBuildSelection = Pin{}
	q.ControllerBuildSelection = Pin{}
	requireSelectionError(t, validateConsumerBuild(r, q), "omits build/selection input")
}

func TestH08FakeProofRequiresSelectedPair(t *testing.T) {
	dir := t.TempDir()
	build, selection := rejectedBuildPair(t, cgobuild.CPU13Tests)
	_, other := rejectedBuildPair(t, cgobuild.CPU13Tests)
	_, wrongRole := rejectedBuildPair(t, cgobuild.H08Command)
	file := func(name, body string) Pin {
		t.Helper()
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
		return Pin{Path: path, SHA256: digest([]byte(body))}
	}
	target, logger, compiler := file("target", "target fixture"), file("logger", "logger fixture"), file("compiler", "compiler fixture")
	var raw strings.Builder
	for _, name := range fakeNames {
		fmt.Fprintf(&raw, "=== RUN   %s\n--- PASS: %s (0.00s)\n", name, name)
	}
	stdout, stderr := file("stdout", raw.String()), file("stderr", "")
	command := runner.CommandReceipt{Started: "2026-10-28T00:00:00Z", Finished: "2026-10-28T00:00:01Z", Stdout: stdout.Path, StdoutSHA: stdout.SHA256, Stderr: stderr.Path, StderrSHA: stderr.SHA256}
	r := FakeProof{Schema: "nanopore-h08-fake-loader-proof/v2", Status: "two_fake_loader_tests_passed", ControllerBuild: build, ControllerBuildSelection: selection, Target: target, Logger: logger, Compiler: compiler, Command: command, Inputs: map[string]string{}, Tests: map[string]string{}}
	r.Command.Args = fakeCommand(target)
	for _, pin := range []Pin{build, selection, target, logger, compiler} {
		r.Inputs[pin.Path] = pin.SHA256
	}
	for _, name := range fakeNames {
		r.Tests[name] = "pass"
	}
	source := file("source.c", "int host_fixture;\n")
	for i := 0; i < 34; i++ {
		output := file(fmt.Sprintf("output-%d", i), "host fixture bytes")
		c := FakeCompile{Source: source, SourceBefore: source, Output: output, OutputSnapshot: output, Command: command}
		c.Command.Args = []string{compiler.Path, "-shared", "-fPIC", source.Path, "-o", output.Path}
		r.Compiles = append(r.Compiles, c)
	}
	check := func(proof FakeProof) error {
		_, err := readFakeProof(hostJSON(t, filepath.Join(dir, "proof.json"), proof), target, build, selection)
		return err
	}
	if err := check(r); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"old-version", "missing", "path-only", "hash-only", "changed", "swapped", "wrong-role", "build-membership", "selection-membership", "compile-count", "test-skip"} {
		t.Run(name, func(t *testing.T) {
			mutant := r
			mutant.Inputs = maps.Clone(r.Inputs)
			want := "fake-only target/selection/custody differs"
			switch name {
			case "old-version":
				mutant.Schema = "nanopore-h08-fake-loader-proof/v1"
			case "missing":
				mutant.ControllerBuildSelection = Pin{}
			case "path-only":
				mutant.ControllerBuildSelection.SHA256 = ""
			case "hash-only":
				mutant.ControllerBuildSelection.Path = ""
			case "changed":
				mutant.ControllerBuildSelection.SHA256 = strings.Repeat("0", 64)
			case "swapped":
				mutant.ControllerBuildSelection = other
			case "wrong-role":
				mutant.ControllerBuildSelection = wrongRole
			case "build-membership":
				delete(mutant.Inputs, build.Path)
				want = "fake input custody omitted"
			case "selection-membership":
				delete(mutant.Inputs, selection.Path)
				want = "fake input custody omitted"
			case "compile-count":
				mutant.Compiles = r.Compiles[:33]
			case "test-skip":
				mutant.Command.Stdout = file("skip-stdout", strings.Replace(raw.String(), "--- PASS:", "--- SKIP:", 1)).Path
				mutant.Command.StdoutSHA = digest([]byte(strings.Replace(raw.String(), "--- PASS:", "--- SKIP:", 1)))
				want = "fake-only raw test result differs"
			}
			requireSelectionError(t, check(mutant), want)
		})
	}
	r.ControllerBuildSelection = Pin{}
	_, err := readFakeProof(hostJSON(t, filepath.Join(dir, "missing-both.json"), r), target, build, Pin{})
	requireSelectionError(t, err, "fake input custody omitted")
}
