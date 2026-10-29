package cgobuild

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func observationTestPin(t *testing.T, path string, value any) Pin {
	t.Helper()
	body, e := json.Marshal(value)
	if e != nil {
		t.Fatal(e)
	}
	return observationTestBytes(t, path, body)
}
func observationTestBytes(t *testing.T, path string, body []byte) Pin {
	t.Helper()
	if e := os.MkdirAll(filepath.Dir(path), 0700); e != nil {
		t.Fatal(e)
	}
	if e := os.WriteFile(path, body, 0600); e != nil {
		t.Fatal(e)
	}
	p, e := closedPin(path)
	if e != nil {
		t.Fatal(e)
	}
	return p
}
func observationSyntheticExecution(t *testing.T) (observerExecutionWire, compiledObservationWire, observedSelectionWire) {
	t.Helper()
	dir := t.TempDir()
	request := observationTestPin(t, filepath.Join(dir, "request.json"), "synthetic request, not an observer seal")
	o := compiledObservationWire{Schema: observationSchema, Request: request, RuntimeHeld: true,
		Proofs: observationProofWire{Aliases: map[string]Pin{}, Operations: []observedOperationWire{}}}
	evidence := observationTestPin(t, filepath.Join(dir, "observation.json"), o)
	stdout := observationTestPin(t, filepath.Join(dir, "stdout"), evidence)
	stderr := observationTestBytes(t, filepath.Join(dir, "stderr"), []byte{})
	process := Process{PID: 123, Start: "123456", Cgroup: "0::/synthetic/observer.service\n", Group: true}
	rawOut := observationTestBytes(t, filepath.Join(dir, "cleanup.stdout"), []byte("ActiveState=inactive\nSubState=dead\nMainPID=0\nControlGroup=\n"))
	c := observerCleanupWire{Schema: cleanupSchema, UTC: "2026-10-28T00:00:04Z", Unit: "observer.service", Process: process, PIDAbsent: true, GroupAbsent: true, CgroupAbsent: true,
		RawCommand: runner.CommandReceipt{Args: []string{"systemctl", "--user", "show", "observer.service", "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "ControlGroup"}, Environment: []string{}, Started: "2026-10-28T00:00:02Z", Finished: "2026-10-28T00:00:03Z", Exit: 0, Stdout: rawOut.Path, StdoutSHA: rawOut.SHA256, Stderr: stderr.Path, StderrSHA: stderr.SHA256}}
	x := observerExecutionWire{Schema: executionSchema, Request: request, Observation: evidence, Cwd: dir, Process: process,
		Command: runner.CommandReceipt{Args: []string{"/synthetic/observer"}, Environment: observerEnvironment(), Started: "2026-10-28T00:00:00Z", Finished: "2026-10-28T00:00:01Z", Exit: 0, Stdout: stdout.Path, StdoutSHA: stdout.SHA256, Stderr: stderr.Path, StderrSHA: stderr.SHA256},
		Cleanup: observationTestPin(t, filepath.Join(dir, "cleanup.json"), c)}
	s := observedSelectionWire{Schema: observedSelectionSchema, Request: request, Execution: observationTestPin(t, filepath.Join(dir, "execution.json"), x)}
	return x, o, s
}

func TestObservationExactWireControls(t *testing.T) {
	x, _, _ := observationSyntheticExecution(t)
	valid := observationTestPin(t, filepath.Join(t.TempDir(), "valid.json"), x)
	var got observerExecutionWire
	if e := exactRead(valid, &got); e != nil {
		t.Fatal(e)
	}
	body, e := os.ReadFile(valid.Path)
	if e != nil {
		t.Fatal(e)
	}
	var fields map[string]json.RawMessage
	if e = json.Unmarshal(body, &fields); e != nil {
		t.Fatal(e)
	}
	for _, name := range []string{"missing-exit", "unknown", "duplicate", "wrong-case", "null", "trailing"} {
		t.Run(name, func(t *testing.T) {
			b := append([]byte{}, body...)
			switch name {
			case "missing-exit":
				b = []byte(strings.Replace(string(b), `"exit_code":0,`, "", 1))
			case "unknown":
				b = append([]byte(`{"Foreign":1,`), b[1:]...)
			case "duplicate":
				b = append([]byte(`{"Schema":"`+executionSchema+`",`), b[1:]...)
			case "wrong-case":
				b = []byte(strings.Replace(string(b), `"Schema":`, `"schema":`, 1))
			case "null":
				b = []byte(strings.Replace(string(b), `"PIDAbsent":true`, `"PIDAbsent":null`, 1))
				b = []byte(strings.Replace(string(b), `"Cwd":"`+x.Cwd+`"`, `"Cwd":null`, 1))
			case "trailing":
				b = append(b, []byte(" {}")...)
			}
			pin := observationTestBytes(t, filepath.Join(t.TempDir(), "mutant.json"), b)
			if e := exactRead(pin, &observerExecutionWire{}); e == nil {
				t.Fatal("accepted rehashed malformed wire")
			}
		})
	}
}

func TestObservationExecutionRehashedControls(t *testing.T) {
	cases := []struct {
		name, want string
		mutate     func(*testing.T, *observerExecutionWire)
	}{
		{"schema", "identity/exit", func(_ *testing.T, x *observerExecutionWire) { x.Schema = "other" }},
		{"request", "identity/exit", func(_ *testing.T, x *observerExecutionWire) { x.Request.SHA256 = strings.Repeat("a", 64) }},
		{"observation", "identity/exit", func(_ *testing.T, x *observerExecutionWire) { x.Observation.SHA256 = strings.Repeat("a", 64) }},
		{"exit", "identity/exit", func(_ *testing.T, x *observerExecutionWire) { x.Command.Exit = 1 }},
		{"process", "identity/exit", func(_ *testing.T, x *observerExecutionWire) { x.Process.PID = 0 }},
		{"clock", "clocks", func(_ *testing.T, x *observerExecutionWire) { x.Command.Finished = "2026-10-28T00:10:01Z" }},
		{"raw-output", "raw output pin", func(t *testing.T, x *observerExecutionWire) {
			p := observationTestPin(t, x.Command.Stdout, Pin{})
			x.Command.StdoutSHA = p.SHA256
		}},
		{"cleanup-pid", "cleanup identity", func(t *testing.T, x *observerExecutionWire) {
			var c observerCleanupWire
			if e := exactRead(x.Cleanup, &c); e != nil {
				t.Fatal(e)
			}
			c.Process.Start = "other"
			x.Cleanup = observationTestPin(t, x.Cleanup.Path, c)
		}},
		{"cleanup-group", "cleanup identity", func(t *testing.T, x *observerExecutionWire) {
			var c observerCleanupWire
			if e := exactRead(x.Cleanup, &c); e != nil {
				t.Fatal(e)
			}
			c.GroupAbsent = false
			x.Cleanup = observationTestPin(t, x.Cleanup.Path, c)
		}},
		{"cleanup-cgroup", "cleanup identity", func(t *testing.T, x *observerExecutionWire) {
			var c observerCleanupWire
			if e := exactRead(x.Cleanup, &c); e != nil {
				t.Fatal(e)
			}
			c.CgroupAbsent = false
			x.Cleanup = observationTestPin(t, x.Cleanup.Path, c)
		}},
		{"cleanup-unit", "cleanup identity", func(t *testing.T, x *observerExecutionWire) {
			var c observerCleanupWire
			if e := exactRead(x.Cleanup, &c); e != nil {
				t.Fatal(e)
			}
			c.Unit = "foreign.service"
			x.Cleanup = observationTestPin(t, x.Cleanup.Path, c)
		}},
		{"cleanup-raw", "raw properties", func(t *testing.T, x *observerExecutionWire) {
			var c observerCleanupWire
			if e := exactRead(x.Cleanup, &c); e != nil {
				t.Fatal(e)
			}
			p := observationTestBytes(t, c.RawCommand.Stdout, []byte("ActiveState=active\nSubState=running\nMainPID=123\nControlGroup=/synthetic\n"))
			c.RawCommand.StdoutSHA = p.SHA256
			x.Cleanup = observationTestPin(t, x.Cleanup.Path, c)
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x, _, s := observationSyntheticExecution(t)
			wantObservation, wantRequest := x.Observation, x.Request
			if e := checkObserverExecution(x, wantObservation, wantRequest); e != nil {
				t.Fatalf("synthetic helper positive: %v", e)
			}
			tc.mutate(t, &x)
			s.Execution = observationTestPin(t, s.Execution.Path, x)
			var reloaded observerExecutionWire
			if e := exactRead(s.Execution, &reloaded); e != nil {
				t.Fatal(e)
			}
			if e := checkObserverExecution(reloaded, wantObservation, wantRequest); e == nil || !strings.Contains(e.Error(), tc.want) {
				t.Fatalf("want %s: %v", tc.want, e)
			}
		})
	}
}

func TestArtifactSelectionCannotFallback(t *testing.T) {
	x, o, s := observationSyntheticExecution(t)
	for _, reader := range []func(Pin, Pin) (Artifact, error){ReadH08Artifact, ReadCPU13Artifact} {
		if a, e := reader(x.Observation, Pin{}); e == nil || a.Output() != (Pin{}) {
			t.Fatal("missing selection admitted")
		}
	}
	for _, name := range []string{"schema", "request", "hold", "failed-exit"} {
		t.Run(name, func(t *testing.T) {
			localO, localS, localX := o, s, x
			switch name {
			case "schema":
				localO.Schema = "other"
			case "request":
				localS.Request.SHA256 = strings.Repeat("b", 64)
			case "hold":
				localO.RuntimeHeld = false
			case "failed-exit":
				localX.Command.Exit = 1
				localS.Execution = observationTestPin(t, filepath.Join(t.TempDir(), "execution.json"), localX)
			}
			evidence := observationTestPin(t, filepath.Join(t.TempDir(), "observation.json"), localO)
			selection := observationTestPin(t, filepath.Join(t.TempDir(), "selection.json"), localS)
			if a, e := ReadH08Artifact(evidence, selection); e == nil || a.Output() != (Pin{}) {
				t.Fatal("rehashed invalid selection admitted")
			}
		})
	}
	if a, e := ReadH08Artifact(x.Observation, observationTestPin(t, filepath.Join(t.TempDir(), "selection.json"), s)); e == nil || a.Output() != (Pin{}) {
		t.Fatal("synthetic record mistaken for witnessed artifact")
	}
}

func TestSuccessfulArtifactSelectedTupleControls(t *testing.T) {
	dir := t.TempDir()
	r := Receipt{Role: H08Command, Status: "failed", Error: "unknown historical failure"}
	evidence := observationTestPin(t, filepath.Join(dir, "receipt.json"), r)
	s := successfulSelectionWire{Schema: successfulSelectionSchema, Role: H08Command, Producer: retainedProducerPins{Receipt: evidence}}
	for _, name := range []string{"role", "receipt", "plan", "controller", "queries", "output", "failed"} {
		t.Run(name, func(t *testing.T) {
			mutant := s
			want := "tuple differs"
			switch name {
			case "role":
				mutant.Role = CPU13Tests
				want = "role/receipt"
			case "receipt":
				mutant.Producer.Receipt.SHA256 = strings.Repeat("c", 64)
				want = "role/receipt"
			case "plan":
				mutant.Producer.Plan = evidence
			case "controller":
				mutant.Producer.Controller = evidence
			case "queries":
				mutant.Producer.Queries = evidence
			case "output":
				mutant.Producer.Output = evidence
			case "failed":
				want = ""
			}
			selection := observationTestPin(t, filepath.Join(t.TempDir(), "selection.json"), mutant)
			a, e := ReadH08Artifact(evidence, selection)
			if e == nil || a.Output() != (Pin{}) || !strings.Contains(e.Error(), want) {
				t.Fatalf("want %s: %v", want, e)
			}
		})
	}
}

func TestObservationSelectedBindings(t *testing.T) {
	x, _, _ := observationSyntheticExecution(t)
	q := observationRequestWire{Role: H08Command, ObserverRoot: x.Cwd, OutputDirectory: filepath.Dir(x.Observation.Path), ObserverController: x.ObserverController}
	if e := selectedObservation(q, x, x.Observation, H08Command); e != nil {
		t.Fatal(e)
	}
	for _, name := range []string{"role", "observer", "cwd", "output"} {
		t.Run(name, func(t *testing.T) {
			mutant := q
			switch name {
			case "role":
				mutant.Role = CPU13Tests
			case "observer":
				mutant.ObserverController = x.Observation
			case "cwd":
				mutant.ObserverRoot += "/foreign"
			case "output":
				mutant.OutputDirectory += "/foreign"
			}
			request := observationTestPin(t, filepath.Join(t.TempDir(), "request.json"), mutant)
			var reloaded observationRequestWire
			if e := exactRead(request, &reloaded); e != nil {
				t.Fatal(e)
			}
			if e := selectedObservation(reloaded, x, x.Observation, H08Command); e == nil {
				t.Fatal("selected binding substitution accepted")
			}
		})
	}
	for _, refs := range [][2]Pin{{}, {x.Observation, x.Request}} {
		if e := observationAlgorithms(refs); e == nil || !strings.Contains(e.Error(), "references differ") {
			t.Fatal(e)
		}
	}
}

func TestObservationUnknownHistoricalFailureRemainsRejected(t *testing.T) {
	s, h, p, r := retainedTupleFixture(t)
	r.Error = "unknown historical failure"
	h.Error = r.Error
	s.Handoff = observationTestPin(t, s.Handoff.Path, h)
	if e := retainedTuple(s, h, p, r); e == nil || !strings.Contains(e.Error(), "error/status") {
		t.Fatalf("unknown retained error: %v", e)
	}
	if e := ValidateReceipt(p, r); e == nil || !strings.Contains(e.Error(), "receipt incomplete") {
		t.Fatalf("strict raw unknown error: %v", e)
	}
}

func TestObservationRequestRehashedControls(t *testing.T) {
	dir := t.TempDir()
	q := observationRequestWire{Schema: requestSchema, Role: H08Command, Retained: retainedSelection{Profile: retainedProfile}, ObserverRoot: dir, OutputDirectory: filepath.Join(dir, "artifacts/tensor/fresh")}
	for _, name := range []string{"schema", "role", "profile", "root", "output", "algorithm"} {
		t.Run(name, func(t *testing.T) {
			mutant := q
			want := "schema/role/profile"
			switch name {
			case "schema":
				mutant.Schema = "foreign"
			case "role":
				mutant.Role = CPU13Tests
			case "profile":
				mutant.Retained.Profile = "another failure profile"
			case "root":
				mutant.ObserverRoot = "relative"
				want = "root/output"
			case "output":
				mutant.OutputDirectory = dir
				want = "root/output"
			case "algorithm":
				want = "algorithm references"
			}
			pin := observationTestPin(t, filepath.Join(t.TempDir(), "request.json"), mutant)
			if _, e := readObservationRequest(pin); e == nil || !strings.Contains(e.Error(), want) {
				t.Fatalf("want %s: %v", want, e)
			}
		})
	}
}

func TestObservationInvocationRehashedControls(t *testing.T) {
	x, _, _ := observationSyntheticExecution(t)
	exe := Pin{Path: "/synthetic/observer", SHA256: strings.Repeat("a", 64)}
	out := filepath.Dir(x.Observation.Path)
	x.Executable = exe
	x.Command.Args = observerArgs(exe, x.Request, out)
	if e := checkObserverInvocation(x, exe, x.Request, out); e != nil {
		t.Fatal(e)
	}
	for _, name := range []string{"executable", "argv-request", "argv-out", "environment"} {
		t.Run(name, func(t *testing.T) {
			mutant := x
			mutant.Command.Args = append([]string{}, x.Command.Args...)
			mutant.Command.Environment = append([]string{}, x.Command.Environment...)
			switch name {
			case "executable":
				mutant.Executable.SHA256 = strings.Repeat("b", 64)
			case "argv-request":
				mutant.Command.Args[6] = strings.Repeat("c", 64)
			case "argv-out":
				mutant.Command.Args[8] += "/foreign"
			case "environment":
				mutant.Command.Environment[0] = "CGO_ENABLED=1"
			}
			pin := observationTestPin(t, filepath.Join(t.TempDir(), "execution.json"), mutant)
			var actual observerExecutionWire
			if e := exactRead(pin, &actual); e != nil {
				t.Fatal(e)
			}
			if e := checkObserverInvocation(actual, exe, x.Request, out); e == nil {
				t.Fatal("rehash invocation substitution accepted")
			}
		})
	}
}

func TestGenuineSelectedObservation(t *testing.T) {
	names := []string{"H08_OBSERVATION", "H08_OBSERVATION_SHA256", "H08_OBSERVATION_SELECTION", "H08_OBSERVATION_SELECTION_SHA256"}
	values := make([]string, len(names))
	present := 0
	for i, n := range names {
		values[i] = os.Getenv(n)
		if values[i] != "" {
			present++
		}
	}
	if present == 0 {
		t.Skip("independent genuine Observation and Selection pins not selected")
	}
	if present != len(names) {
		t.Fatal("partial genuine observation selection")
	}
	a, e := ReadH08Artifact(Pin{Path: values[0], SHA256: values[1]}, Pin{Path: values[2], SHA256: values[3]})
	if e != nil {
		t.Fatal(e)
	}
	if a.Output() == (Pin{}) {
		t.Fatal("missing selected output")
	}
	t.Logf("selected historical compiled output: %+v; runtime held", a.Output())
}

func TestGenuineSelectedObservationRehashedNegative(t *testing.T) {
	kind := os.Getenv("H08_OBSERVATION_NEGATIVE")
	if kind == "" {
		t.Skip("genuine semantic negative family not selected")
	}
	if kind != "output" && kind != "native" && kind != "summary" {
		t.Fatal("unknown negative family")
	}
	evidence := Pin{Path: os.Getenv("H08_OBSERVATION"), SHA256: os.Getenv("H08_OBSERVATION_SHA256")}
	selection := Pin{Path: os.Getenv("H08_OBSERVATION_SELECTION"), SHA256: os.Getenv("H08_OBSERVATION_SELECTION_SHA256")}
	var s observedSelectionWire
	if e := exactRead(selection, &s); e != nil {
		t.Fatal(e)
	}
	var o compiledObservationWire
	if e := exactRead(evidence, &o); e != nil {
		t.Fatal(e)
	}
	var q observationRequestWire
	if e := exactRead(s.Request, &q); e != nil {
		t.Fatal(e)
	}
	var x observerExecutionWire
	if e := exactRead(s.Execution, &x); e != nil {
		t.Fatal(e)
	}
	dir, e := os.MkdirTemp(filepath.Join(q.ObserverRoot, "artifacts/tensor"), "observation-negative-")
	if e != nil {
		t.Fatal(e)
	}
	t.Logf("preserved adversarial fixture: %s", dir)
	q.OutputDirectory = dir
	request := observationTestPin(t, filepath.Join(dir, "request.json"), q)
	o.Request = request
	switch kind {
	case "output":
		o.Output.SHA256 = strings.Repeat("a", 64)
	case "native":
		o.NativePreimage.SHA256 = strings.Repeat("b", 64)
	case "summary":
		o.Proofs.ChecksumOffset++
	}
	mutant := observationTestPin(t, filepath.Join(dir, "observation.json"), o)
	x.Request, x.Observation = request, mutant
	x.Command.Args = observerArgs(x.Executable, request, dir)
	stdout := observationTestPin(t, filepath.Join(dir, "stdout"), mutant)
	x.Command.Stdout, x.Command.StdoutSHA = stdout.Path, stdout.SHA256
	s.Request = request
	s.Execution = observationTestPin(t, filepath.Join(dir, "execution.json"), x)
	selected := observationTestPin(t, filepath.Join(dir, "selection.json"), s)
	if a, e := ReadH08Artifact(mutant, selected); e == nil || !strings.Contains(e.Error(), "serialized proofs/output differ") || a.Output() != (Pin{}) {
		t.Fatalf("semantic %s gate: %v", kind, e)
	}
}
