package decodergpu

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func lineageReplayWatchFixture(t *testing.T) (File, LineageRuntime, CompilerPhase, LineagePlan, LineageLoadReceipt, watchObservation) {
	t.Helper()
	c, _ := lineageFixture()
	runtime := File{"/runtime/runtime.json", "runtime-pin"}
	r := LineageRuntime{WorkingDirectory: "/repo", Output: "/output", Contract: c}
	b := CompilerPhase{Output: File{"/probe", "native-pin"}}
	p := LineagePlan{Go: GoClosure{Controller: "/producer-controller", ControllerSHA: "producer-pin"}}
	saved := LineageLoadReceipt{Schema: "nanopore-decoder-load-execution/v1", Runtime: runtime, Status: "failed", Error: lineageEnvironmentError, ControllerPID: 10, Observation: File{"/output/native/observation.json", "observation-pin"}}
	saved.Guarded.Command.Exit = 0
	expected := lineageWatchPlan(r, runtime, b, p)
	raw, err := json.Marshal(map[string]any{
		"schema": "nanopore-native-role-watch/v1", "plan": expected, "controller_pid": 10, "controller_pgid": 10, "controller_exit": 1,
		"own_cgroup": "0::/fixture\n", "signal_policy": "current-identity-revalidated-pidfd-members-only",
		"controller_identity":   map[string]any{"pid": 10, "pgid": 10, "sid": 10, "start_ticks": 77, "image": p.Go.Controller, "image_sha256": p.Go.ControllerSHA, "argv": expected.Controller, "cgroup": "0::/fixture\n"},
		"roles":                 []map[string]any{{"name": "lineage", "pid": 20, "pgid": 20, "sid": 10, "start_ticks": 88, "image_sha256": b.Output.SHA256, "argv": lineageCommand(r, b).Args, "limit_seconds": 180, "cgroup": "0::/fixture\n"}},
		"members_after_cleanup": map[string][]any{"10": {}, "20": {}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var w watchObservation
	if err = json.Unmarshal(raw, &w); err != nil {
		t.Fatal(err)
	}
	return runtime, r, b, p, saved, w
}

func TestLineageReplayRequiresExactObservationFailure(t *testing.T) {
	runtime, r, b, p, saved, w := lineageReplayWatchFixture(t)
	if err := validateLineageSavedLoad(saved, runtime, lineageEnvironmentReaderFailed); err != nil {
		t.Fatal(err)
	}
	expected := lineageWatchPlan(r, runtime, b, p)
	if err := validateLineageWatch(w, expected, r, b, p, saved, 20, lineageEnvironmentReaderFailed); err != nil {
		t.Fatal(err)
	}
	if err := validateLineageSavedLoad(saved, runtime, lineageLoadCompleted); err == nil {
		t.Fatal("failed receipt entered successful CheckLineageLoad path")
	}
	for _, kind := range []string{"status", "error", "native-exit", "saved-binding", "observation", "runtime", "unknown-outcome"} {
		t.Run(kind, func(t *testing.T) {
			_, _, _, _, bad, _ := lineageReplayWatchFixture(t)
			outcome := lineageEnvironmentReaderFailed
			switch kind {
			case "status":
				bad.Status = "load_lineage_pending_supervisor"
			case "error":
				bad.Error = "another native or reader failure"
			case "native-exit":
				bad.Guarded.Command.Exit = 1
			case "saved-binding":
				bad.Binding.ModuleID = 23
			case "observation":
				bad.Observation = File{}
			case "runtime":
				bad.Runtime.SHA256 = "other"
			case "unknown-outcome":
				outcome = lineageLoadOutcome(255)
			}
			if err := validateLineageSavedLoad(bad, runtime, outcome); err == nil {
				t.Fatal("unrecognized historical failure entered replay")
			}
		})
	}
	for _, kind := range []string{"controller-exit", "timeout", "signal", "ownership", "group", "controller-birth", "controller-image", "native-pid", "native-birth", "native-session", "native-argv", "extra-role", "changed-plan"} {
		t.Run(kind, func(t *testing.T) {
			_, r, b, p, saved, bad := lineageReplayWatchFixture(t)
			switch kind {
			case "controller-exit":
				*bad.ControllerExit = 0
			case "timeout":
				bad.Timeout = true
			case "signal":
				bad.Signals = []json.RawMessage{json.RawMessage(`{}`)}
			case "ownership":
				bad.OwnershipErrors = []json.RawMessage{json.RawMessage(`{}`)}
			case "group":
				bad.Members["20"] = []json.RawMessage{json.RawMessage(`{"pid":20}`)}
			case "controller-birth":
				bad.ControllerIdentity.Start = 0
			case "controller-image":
				bad.ControllerIdentity.Image = "/current-replay-controller"
			case "native-pid":
				bad.Roles[0].PID = 21
			case "native-birth":
				bad.Roles[0].Start = 0
			case "native-session":
				bad.Roles[0].SID = 20
			case "native-argv":
				bad.Roles[0].Args[4] = "/other-original"
			case "extra-role":
				bad.Roles = append(bad.Roles, bad.Roles[0])
			case "changed-plan":
				bad.Plan.Limit = 241
			}
			if err := validateLineageWatch(bad, expected, r, b, p, saved, 20, lineageEnvironmentReaderFailed); err == nil {
				t.Fatal("failed supervision or historical identity entered replay")
			}
		})
	}
	dir := t.TempDir()
	stdout, stderr := testFile(t, dir, "stdout", "saved raw output"), testFile(t, dir, "stderr", "saved consumer error")
	invocation := SupervisorInvocation{Args: []string{"/python", "-B", "/watcher", "/plan", "/watch", "pin"}, Environment: []string{"LANG=C"}, Stdout: stdout.Path, Stderr: stderr.Path}
	command := runner.CommandReceipt{Args: invocation.Args, Environment: invocation.Environment, Started: "2026-10-26T00:00:00Z", Finished: "2026-10-26T00:00:01Z", Exit: 1, Stdout: stdout.Path, StdoutSHA: stdout.SHA256, Stderr: stderr.Path, StderrSHA: stderr.SHA256}
	if err := lineageSavedSupervisorCommand(command, invocation, lineageEnvironmentReaderFailed); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"exit", "clock", "argv", "environment", "raw-hash"} {
		t.Run("supervisor-"+kind, func(t *testing.T) {
			bad := command
			switch kind {
			case "exit":
				bad.Exit = 0
			case "clock":
				bad.Finished = "2026-10-25T00:00:00Z"
			case "argv":
				bad.Args = []string{"/different"}
			case "environment":
				bad.Environment = []string{"LANG=other"}
			case "raw-hash":
				bad.StdoutSHA = strings.Repeat("0", 64)
			}
			if err := lineageSavedSupervisorCommand(bad, invocation, lineageEnvironmentReaderFailed); err == nil {
				t.Fatal("altered supervisor command entered replay")
			}
		})
	}
	command.Exit = 0
	if err := lineageSavedSupervisorCommand(command, invocation, lineageLoadCompleted); err != nil {
		t.Fatal("successful supervisor path changed", err)
	}
	saved.Status, saved.Error = "load_lineage_pending_supervisor", ""
	*w.ControllerExit = 0
	if err := validateLineageSavedLoad(saved, runtime, lineageLoadCompleted); err != nil {
		t.Fatal("successful load path changed", err)
	}
	if err := validateLineageWatch(w, expected, r, b, p, saved, 20, lineageLoadCompleted); err != nil {
		t.Fatal("successful watcher path changed", err)
	}
}

func TestLineageReplayKeepsProducerAndConsumerCustodySeparate(t *testing.T) {
	c, _ := lineageFixture()
	producer := LineagePlan{WorkingDirectory: "/repo", Producer: File{"/decoder-build", "producer"}, ProducerRuntime: File{"/decoder-runtime", "runtime"}, Capture: File{"/capture002", lineageCaptureSHA}, Cases: []LineageContract{c}, SourceBindings: map[string]string{}, SnapshotPins: map[string]string{}, Inputs: map[string]string{}, Go: GoClosure{Controller: "/old-controller", ControllerSHA: "old-go"}}
	consumer := producer
	consumer.Go = GoClosure{Controller: "/current-controller", ControllerSHA: "new-go"}
	consumer.SourceBindings, consumer.SnapshotPins, consumer.Inputs = map[string]string{}, map[string]string{}, map[string]string{}
	for i, source := range lineageSourcePaths {
		original := filepath.Join("/repo", source)
		oldCopy, newCopy := filepath.Join("/old", filepath.Base(source)), filepath.Join("/new", filepath.Base(source))
		want := strings.Repeat(string(rune('a'+i)), 64)
		producer.SourceBindings[original], consumer.SourceBindings[original] = oldCopy, newCopy
		producer.SnapshotPins[oldCopy], consumer.SnapshotPins[newCopy] = want, want
		producer.Inputs[original], consumer.Inputs[original] = want, want
	}
	if err := verifyLineageReplayConsumer(producer, consumer); err != nil {
		t.Fatal("new Go consumer was confused with old native producer", err)
	}
	raw, err := json.Marshal(consumer)
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"producer", "capture", "working-directory", "case-environment", "recipe", "compiler", "native-membership", "native-bytes"} {
		t.Run(kind, func(t *testing.T) {
			var bad LineagePlan
			if err := json.Unmarshal(raw, &bad); err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "producer":
				bad.Producer.SHA256 = "other"
			case "capture":
				bad.Capture.SHA256 = "other"
			case "working-directory":
				bad.WorkingDirectory = "/other"
			case "case-environment":
				bad.Cases[0].Environment = append(bad.Cases[0].Environment, "OTHER=1")
			case "recipe":
				bad.Recipe.Flags = []string{"-changed"}
			case "compiler":
				bad.Compiler.Inputs = map[string]string{"other-tool": "other"}
			case "native-membership":
				for key := range bad.SourceBindings {
					delete(bad.SourceBindings, key)
					break
				}
			case "native-bytes":
				for original, copy := range bad.SourceBindings {
					bad.Inputs[original] = strings.Repeat("f", 64)
					bad.SnapshotPins[copy] = bad.Inputs[original]
					break
				}
			}
			if err := verifyLineageReplayConsumer(producer, bad); err == nil {
				t.Fatal("changed producer evidence entered current-consumer replay")
			}
		})
	}
}
