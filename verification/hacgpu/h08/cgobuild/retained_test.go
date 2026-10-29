package cgobuild

import (
	"encoding/hex"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func retainedTupleFixture(t *testing.T) (retainedSelection, retainedHandoff, Plan, Receipt) {
	t.Helper()
	root := t.TempDir()
	pin := func(name string) Pin { return saveQueryFixture(t, filepath.Join(root, name), name) }
	producer := retainedProducerPins{pin("receipt"), pin("plan"), pin("controller"), pin("queries"), pin("output")}
	s := retainedSelection{Producer: producer, Profile: retainedProfile, Outer: pin("outer"), Cleanup: pin("cleanup"), FailureStderr: pin("stderr")}
	h := retainedHandoff{Status: "APPLICATION_COMPILED_BUT_FAILED_RECEIPT_VALIDATION_UNQUALIFIED_UNEXECUTED", Role: string(H08Command), Error: retainedFailure, Receipt: producer.Receipt, Plan: producer.Plan, Controller: producer.Controller, Queries: producer.Queries, Output: producer.Output, Outer: s.Outer, Cleanup: s.Cleanup, RawError: s.FailureStderr, Operations: 297, Statuses: map[string]int{"accepted": 289, "accepted_expected_exit": 8}}
	s.Handoff = saveQueryFixture(t, filepath.Join(root, "handoff.json"), h)
	p := Plan{Role: H08Command}
	r := Receipt{Schema: "nanopore-h08-cgo-execution/v1", Role: H08Command, RuntimeHeld: true, Status: "failed", Error: retainedFailure, Plan: producer.Plan, Controller: producer.Controller, Queries: producer.Queries, Output: producer.Output, Commands: []runner.CommandReceipt{{Exit: 0}, {Exit: 0}}, Generated: make([]Pin, 4), Selection: pin("selection")}
	return s, h, p, r
}

func TestRetainedProducerTupleControls(t *testing.T) {
	cases := []struct {
		name, want string
		change     func(*retainedSelection, *retainedHandoff, *Plan, *Receipt)
	}{
		{"profile", "unknown retained producer profile", func(s *retainedSelection, _ *retainedHandoff, _ *Plan, _ *Receipt) { s.Profile = "other" }},
		{"error", "error/status", func(_ *retainedSelection, _ *retainedHandoff, _ *Plan, r *Receipt) { r.Error = "another error" }},
		{"status", "error/status", func(_ *retainedSelection, _ *retainedHandoff, _ *Plan, r *Receipt) { r.Status = "compiled_unexecuted" }},
		{"role", "role differs", func(_ *retainedSelection, _ *retainedHandoff, _ *Plan, r *Receipt) { r.Role = CPU13Tests }},
		{"tuple", "independent retained producer tuple", func(_ *retainedSelection, h *retainedHandoff, _ *Plan, _ *Receipt) {
			h.Plan.SHA256 = strings.Repeat("a", 64)
		}},
		{"output", "independent retained producer tuple", func(_ *retainedSelection, _ *retainedHandoff, _ *Plan, r *Receipt) {
			r.Output.SHA256 = strings.Repeat("b", 64)
		}},
		{"omitted", "pin omitted", func(s *retainedSelection, _ *retainedHandoff, _ *Plan, _ *Receipt) { s.Handoff = Pin{} }},
		{"stage", "final-reader stage", func(_ *retainedSelection, _ *retainedHandoff, _ *Plan, r *Receipt) { r.Commands[1].Exit = 1 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s, h, p, r := retainedTupleFixture(t)
			if e := retainedTuple(s, h, p, r); e != nil {
				t.Fatal(e)
			}
			tc.change(&s, &h, &p, &r)
			if s.Handoff.Path != "" {
				s.Handoff = saveQueryFixture(t, s.Handoff.Path, h)
			}
			if e := retainedTuple(s, h, p, r); e == nil || !strings.Contains(e.Error(), tc.want) {
				t.Fatalf("want %s: %v", tc.want, e)
			}
		})
	}
}

func TestRetainedFrozenSourceControl(t *testing.T) {
	p := Plan{Options: Options{Root: t.TempDir()}, Inputs: map[string]string{}}
	c := Controller{Inputs: map[string]string{}, Snapshots: map[string]Pin{}}
	source := filepath.Join(p.Options.Root, "verification/hacgpu/h08/cgobuild/execution.go")
	q := saveQueryFixture(t, filepath.Join(t.TempDir(), "source"), "changed control flow")
	p.Inputs[source] = q.SHA256
	c.Inputs[source] = q.SHA256
	c.Snapshots[source] = q
	if e := retainedSources(p, c); e == nil || !strings.Contains(e.Error(), "source/control flow") {
		t.Fatalf("rehash changed source: %v", e)
	}
}

func TestCompleteOperationInventory(t *testing.T) {
	for _, name := range []string{"omitted", "extra", "duplicate"} {
		t.Run(name, func(t *testing.T) {
			s := Session{Directory: t.TempDir()}
			a := saveQueryFixture(t, filepath.Join(s.Directory, "commands/a/operation.json"), operation{})
			b := saveQueryFixture(t, filepath.Join(s.Directory, "commands/b/operation.json"), operation{})
			pins := []Pin{a, b}
			if e := completeOperationInventory(s, pins); e != nil {
				t.Fatal(e)
			}
			switch name {
			case "omitted":
				pins = pins[:1]
			case "extra":
				saveQueryFixture(t, filepath.Join(s.Directory, "commands/c/operation.json"), operation{})
			case "duplicate":
				pins = append(pins, a)
			}
			rp := saveQueryFixture(t, filepath.Join(s.Directory, "receipt.json"), Receipt{Operations: pins})
			var r Receipt
			if e := read(rp, &r); e != nil {
				t.Fatal(e)
			}
			want := "complete operation inventory"
			if name == "duplicate" {
				want = "duplicate operation inventory"
			}
			if e := completeOperationInventory(s, r.Operations); e == nil || !strings.Contains(e.Error(), want) {
				t.Fatalf("want %s: %v", want, e)
			}
		})
	}
}

func TestRetainedOuterSemanticRejection(t *testing.T) {
	s, h, p, r := retainedTupleFixture(t)
	root := filepath.Dir(s.Outer.Path)
	log := func(name string) Pin {
		path := filepath.Join(root, name)
		if e := os.WriteFile(path, nil, 0600); e != nil {
			t.Fatal(e)
		}
		q, e := closedPin(path)
		if e != nil {
			t.Fatal(e)
		}
		return q
	}
	raw := retainedCommand{Argv: []string{"foreign"}, Started: "2026-10-26T06:00:00Z", Finished: "2026-10-26T07:00:00Z", Exit: 0, Stdout: log("stdout"), Stderr: log("stderr")}
	s.Outer = saveQueryFixture(t, s.Outer.Path, raw)
	h.Outer = s.Outer
	h.RawController = saveQueryFixture(t, filepath.Join(root, "build-receipt.json"), raw)
	s.Handoff = saveQueryFixture(t, s.Handoff.Path, h)
	if e := retainedOuter(s, h, p, r, Session{}, Controller{}); e == nil || !strings.Contains(e.Error(), "outer command/exit/environment differs") {
		t.Fatalf("rehashed foreign outer command: %v", e)
	}
}

func TestRetainedCleanupSemanticControls(t *testing.T) {
	root := t.TempDir()
	session := Session{Directory: filepath.Join(root, "session"), ParentPID: 100, ParentStart: "start"}
	s := retainedSelection{Producer: retainedProducerPins{Receipt: Pin{Path: "/receipt", SHA256: strings.Repeat("a", 64)}, Output: Pin{Path: "/output", SHA256: strings.Repeat("b", 64)}}}
	s.Outer = saveQueryFixture(t, filepath.Join(root, "outer-receipt.json"), retainedCommand{Started: "2026-10-26T06:00:00Z", Finished: "2026-10-26T07:00:00Z"})
	controller := Process{PID: 100, Start: "start", Cgroup: "0::/synthetic-record-only\n", Group: true}
	saveQueryFixture(t, filepath.Join(root, "controller-process.json"), controller)
	c := retainedCleanup{UTC: "2026-10-26T07:00:01Z", Status: "FAILED_BUILD_RECEIPT_VALIDATION_RELEASED_NO_RETRY", SystemdProperties: map[string]string{"ActiveState": "failed", "SubState": "failed", "MainPID": "0", "Result": "exit-code", "ExecMainStatus": "1", "ControlGroup": ""}, CgroupAbsent: true, ControllerPIDAndGroupAbsent: true, BeforeAfterPinsEqual: true, CanonicalReadonlyNonblockingAvailableAtObservation: true, Controller: controller, ActualProcessCount: 346, Receipt: s.Producer.Receipt, Output: s.Producer.Output}
	for i := 0; i < 346; i++ {
		c.ActualProcesses = append(c.ActualProcesses, struct {
			Record                                                 Pin
			Identity                                               Process
			OriginalPIDAbsent, CurrentPIDAbsent, OwnedGroupChecked bool
			GroupMembers                                           []int
		}{})
		entry := &c.ActualProcesses[i]
		entry.Identity = Process{PID: 200 + i, Start: fmt.Sprint(i), Cgroup: controller.Cgroup, Group: i < 97}
		entry.Record = saveQueryFixture(t, filepath.Join(session.Directory, "commands", fmt.Sprint(i), "process.json"), entry.Identity)
		entry.OriginalPIDAbsent = true
		entry.CurrentPIDAbsent = true
		entry.OwnedGroupChecked = entry.Identity.Group
	}
	c.RawCommand.Argv = []string{"systemctl", "--user", "show", filepath.Base(root) + ".service", "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "Result", "-p", "ExecMainStatus", "-p", "ControlGroup"}
	c.RawCommand.StartedUTC = "2026-10-26T07:00:01Z"
	c.RawCommand.FinishedUTC = "2026-10-26T07:00:02Z"
	put := func(name, body string) Pin {
		path := filepath.Join(root, name)
		if e := os.WriteFile(path, []byte(body), 0600); e != nil {
			t.Fatal(e)
		}
		q, e := closedPin(path)
		if e != nil {
			t.Fatal(e)
		}
		return q
	}
	c.RawCommand.Stdout = put("cleanup.stdout", "ActiveState=failed\nSubState=failed\nMainPID=0\nResult=exit-code\nExecMainStatus=1\nControlGroup=\n")
	c.RawCommand.Stderr = put("cleanup.stderr", "")
	original := saveQueryFixture(t, filepath.Join(root, "baseline-cleanup.json"), c)
	s.Cleanup = original
	if e := retainedRelease(s, session); e != nil {
		t.Fatalf("metadata helper baseline: %v", e)
	}
	cases := []struct {
		name, want string
		mutate     func(*retainedCleanup)
	}{
		{"omitted", "cleanup release incomplete", func(c *retainedCleanup) { c.ActualProcesses = c.ActualProcesses[:345] }},
		{"duplicate", "process membership differs", func(c *retainedCleanup) { c.ActualProcesses[1] = c.ActualProcesses[0] }},
		{"identity", "process/group differs", func(c *retainedCleanup) { c.ActualProcesses[0].Identity.Start = "changed" }},
		{"group", "process/group differs", func(c *retainedCleanup) { c.ActualProcesses[0].OwnedGroupChecked = false }},
		{"exit", "raw command differs", func(c *retainedCleanup) { c.RawCommand.Exit = 1 }},
		{"clock", "release clock differs", func(c *retainedCleanup) { c.UTC = "2026-10-26T05:00:00Z" }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var mutant retainedCleanup
			if e := read(original, &mutant); e != nil {
				t.Fatal(e)
			}
			tc.mutate(&mutant)
			selected := s
			selected.Cleanup = saveQueryFixture(t, filepath.Join(root, tc.name+"-cleanup.json"), mutant)
			selected.Handoff = saveQueryFixture(t, filepath.Join(root, tc.name+"-handoff.json"), retainedHandoff{Cleanup: selected.Cleanup})
			if e := retainedRelease(selected, session); e == nil || !strings.Contains(e.Error(), tc.want) {
				t.Fatalf("want %s: %v", tc.want, e)
			}
		})
	}
}

func TestRetainedRawFailureStrict(t *testing.T) {
	_, _, p, r := retainedTupleFixture(t)
	p.Schema = "nanopore-h08-cgo-build-plan/v2"
	p.Options.Role = H08Command
	p.BuildArgv = plannedBuildArgs(p)
	r.Plan = saveQueryFixture(t, filepath.Join(t.TempDir(), "plan.json"), p)
	pin := saveQueryFixture(t, filepath.Join(t.TempDir(), "receipt.json"), r)
	if e := ValidateReceipt(p, r); e == nil || !strings.Contains(e.Error(), "native phase receipt incomplete") {
		t.Fatalf("strict validation: %v", e)
	}
	if _, e := ReadH08Build(pin); e == nil || !strings.Contains(e.Error(), "native phase receipt incomplete") {
		t.Fatalf("H08 strict envelope rejection: %v", e)
	}
	if _, e := ReadCPU13Build(pin); e == nil || !strings.Contains(e.Error(), "wrong fixed build role") {
		t.Fatalf("CPU13 fixed-role rejection: %v", e)
	}
}

func retainedEnvPin(t *testing.T, name string) Pin {
	t.Helper()
	path, sha := os.Getenv(name), os.Getenv(name+"_SHA256")
	hash, e := hex.DecodeString(sha)
	if !filepath.IsAbs(path) || e != nil || len(hash) != 32 || strings.ToLower(sha) != sha {
		t.Fatalf("%s requires explicit absolute path and independent SHA256", name)
	}
	return Pin{Path: path, SHA256: sha}
}

func TestGenuineRetainedFullBody(t *testing.T) {
	if os.Getenv("H08_RETAINED_FULL_RECEIPT") == "" && os.Getenv("H08_RETAINED_FULL_RECEIPT_SHA256") == "" && os.Getenv("H08_RETAINED_FULL_HANDOFF") == "" && os.Getenv("H08_RETAINED_FULL_HANDOFF_SHA256") == "" {
		t.Skip("explicit independent receipt and handoff selections required")
	}
	receipt := retainedEnvPin(t, "H08_RETAINED_FULL_RECEIPT")
	handoff := retainedEnvPin(t, "H08_RETAINED_FULL_HANDOFF")
	root := "/home/jake/Projects/nanopore_basecaller_go/artifacts"
	pin := func(path, sha string) Pin { return Pin{Path: filepath.Join(root, path), SHA256: sha} }
	s := retainedSelection{Profile: retainedProfile, Handoff: handoff, Producer: retainedProducerPins{
		Receipt:    receipt,
		Plan:       pin("tensor/h08-cgo-command-plan-006/plan.json", "67e67736e47d7b365dcccc5f1cb94b18720100f7beffd40bc9f9b1353a65090c"),
		Controller: pin("h08-cgo-final-link-host-20261025-004/controller.json", "91fcc7edf9c6771ecd31258ec4919ae0da6556cf9736bd0224719aa1c0276da7"),
		Queries:    pin("tensor/h08-cgo-command-queries-006/receipt.json", "8f4e70aa825fbb30683ee046751819877a0326b19a47bb0c590d6e911d339a12"),
		Output:     pin("tensor/h08-cgo-command-plan-006/work/h08", "52ae02a2370d718123bc7088d3de1d5d483122323eaa5c7e794a854df1917c69")},
		Outer:         pin("h08-cgo-build-host-20261025-008/outer-receipt.json", "48fa1f105d5e142f6c0edb241e524cef7277371f36cdf84059f2424cb5d0e1c4"),
		Cleanup:       pin("h08-cgo-build-host-20261025-008/cleanup.json", "5e9008dc3d83a916b0003e1087f3bf33e3c56ca65dbc071f35b1192dbfa03b4a"),
		FailureStderr: pin("h08-cgo-build-host-20261025-008/commands/build.stderr", "72c0f7bcef3ee1cac4064c048a04e90fb656dbf4be444a8ba9127f3a539fd309")}
	var original Receipt
	if e := read(receipt, &original); e != nil {
		t.Fatal(e)
	}
	r, e := readRetainedCompiled(s)
	if e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(r, original) || r.Status != "failed" || r.Error != retainedFailure {
		t.Fatal("original failed receipt changed")
	}
	var p Plan
	if e = read(s.Producer.Plan, &p); e != nil {
		t.Fatal(e)
	}
	if e = ValidateReceipt(p, r); e == nil {
		t.Fatal("strict raw validator accepted failed root afterward")
	}
	if _, e = ReadH08Build(receipt); e == nil {
		t.Fatal("strict H08 reader accepted failed root afterward")
	}
	if _, e = ReadCPU13Build(receipt); e == nil {
		t.Fatal("strict CPU13 reader accepted failed root afterward")
	}
	if e = verifyPin(receipt); e != nil {
		t.Fatal(e)
	}
	counts := map[string]int{}
	for _, q := range r.Operations {
		var op operation
		if e = read(q, &op); e != nil {
			t.Fatal(e)
		}
		counts[op.Status]++
	}
	if len(r.Operations) != 297 || !maps.Equal(counts, map[string]int{"accepted": 289, "accepted_expected_exit": 8}) {
		t.Fatal("full operation counts differ")
	}
	t.Logf("bounded retained-data common-body proof: 297 operations / 289 accepted / 8 expected-exit; original root %s unchanged and failed; no qualified observation/runtime", receipt.SHA256)
}
