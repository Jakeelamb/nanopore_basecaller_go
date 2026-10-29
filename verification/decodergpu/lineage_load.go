package decodergpu

import (
	"context"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type LineageLoadOptions struct {
	Runtime  File
	Out      string
	Admitted bool
}
type LineageLoadReceipt struct {
	Schema, Status, Error, Started, Finished string
	Runtime                                  File
	ControllerPID                            int
	Guarded                                  supgpu.GuardedRun
	Code                                     []supgpu.Pin
	Observation                              File
	Binding                                  LineageBinding
}

type lineageLoadOutcome uint8

const (
	lineageLoadCompleted lineageLoadOutcome = iota
	lineageEnvironmentReaderFailed
)

func lineageControllerExit(outcome lineageLoadOutcome) (int, error) {
	switch outcome {
	case lineageLoadCompleted:
		return 0, nil
	case lineageEnvironmentReaderFailed:
		return 1, nil
	default:
		return 0, fmt.Errorf("unknown lineage saved execution outcome")
	}
}

func validateLineageSavedLoad(receipt LineageLoadReceipt, runtime File, outcome lineageLoadOutcome) error {
	if receipt.Schema != "nanopore-decoder-load-execution/v1" || receipt.Runtime != runtime || receipt.Guarded.Command.Exit != 0 {
		return fmt.Errorf("lineage execution status/binding differs")
	}
	switch outcome {
	case lineageLoadCompleted:
		if receipt.Status == "load_lineage_pending_supervisor" && receipt.Error == "" {
			return nil
		}
	case lineageEnvironmentReaderFailed:
		if receipt.Status == "failed" && receipt.Error == lineageEnvironmentError && receipt.Binding == (LineageBinding{}) && receipt.Observation.Path != "" && receipt.Observation.SHA256 != "" {
			return nil
		}
	}
	return fmt.Errorf("lineage execution status/binding differs")
}

func RunLineageLoad(o LineageLoadOptions) (receipt LineageLoadReceipt, err error) {
	if !o.Admitted {
		return receipt, fmt.Errorf("lineage load-only phase requires separate coordinator admission")
	}
	if err = requireNativeLimits(); err != nil {
		return receipt, err
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return receipt, err
	}
	defer lock.Close()
	o.Runtime, err = normalizeAnchor(o.Runtime)
	if err != nil {
		return receipt, err
	}
	r, b, _, _, err := readLineageRuntime(o.Runtime, true)
	if err != nil {
		return receipt, err
	}
	out, err := filepath.Abs(o.Out)
	if err != nil {
		return receipt, err
	}
	if out != r.Output {
		return receipt, fmt.Errorf("lineage prospective output differs")
	}
	out, err = runner.NewAttempt(out)
	if err != nil {
		return receipt, err
	}
	receipt = LineageLoadReceipt{Schema: "nanopore-decoder-load-execution/v1", Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Runtime: o.Runtime, ControllerPID: os.Getpid()}
	defer func() {
		receipt.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			receipt.Error = err.Error()
		}
		if e := writeJSON(filepath.Join(out, "load.json"), receipt); err == nil {
			err = e
		}
	}()
	guard, err := guardPin(r.Guard)
	if err != nil {
		return receipt, err
	}
	phases, err := guardPin(r.Phases)
	if err != nil {
		return receipt, err
	}
	command := lineageCommand(r, b)
	receipt.Guarded, err = supgpu.ExecuteGuarded(context.Background(), guard, phases, command)
	if err != nil {
		return receipt, err
	}
	receipt.Code, err = supgpu.ReplayGuarded(guard, phases, command, receipt.Guarded)
	if err != nil {
		return receipt, err
	}
	receipt.Observation, err = artifactPin(filepath.Join(out, "native/observation.json"))
	if err != nil {
		return receipt, err
	}
	observation, binding, err := ReadLineageObservation(receipt.Observation, r.Contract)
	if err != nil {
		return receipt, err
	}
	var transcript supgpu.GuardReceipt
	if err = readJSON(File{receipt.Guarded.Transcript.Path, receipt.Guarded.Transcript.SHA256}, &transcript); err != nil {
		return receipt, err
	}
	if observation.PID != transcript.PID {
		return receipt, fmt.Errorf("lineage observer PID differs from actual guarded child")
	}
	receipt.Binding = binding
	if _, _, _, _, err = readLineageRuntime(o.Runtime, true); err != nil {
		return receipt, err
	}
	receipt.Status = "load_lineage_pending_supervisor"
	return receipt, nil
}

func CheckLineageLoad(runtime, load, commandFile, watcherFile File) (receipt LineageLoadReceipt, err error) {
	for _, file := range []*File{&runtime, &load, &commandFile, &watcherFile} {
		*file, err = normalizeAnchor(*file)
		if err != nil {
			return
		}
	}
	r, b, _, consumer, err := readLineageRuntime(runtime, false)
	if err != nil {
		return receipt, err
	}
	if load.Path != filepath.Join(r.Output, "load.json") {
		return receipt, fmt.Errorf("lineage load receipt identity differs")
	}
	if err = readJSON(load, &receipt); err != nil {
		return receipt, err
	}
	if err = validateLineageSavedLoad(receipt, runtime, lineageLoadCompleted); err != nil {
		return receipt, err
	}
	transcript, _, err := lineageSavedGuard(r, b, receipt)
	if err != nil {
		return receipt, err
	}
	observation, binding, err := ReadLineageObservation(receipt.Observation, r.Contract)
	if err != nil {
		return receipt, err
	}
	if !reflect.DeepEqual(binding, receipt.Binding) {
		return receipt, fmt.Errorf("lineage raw input/output/operation binding differs")
	}
	if observation.PID != transcript.PID {
		return receipt, fmt.Errorf("lineage observer PID differs")
	}
	if err = checkLineageSavedSupervisor(runtime, r, b, consumer, receipt, transcript.PID, commandFile, watcherFile, lineageLoadCompleted); err != nil {
		return receipt, err
	}
	receipt.Status = "guarded_load_lineage_reproduced_only"
	return receipt, nil
}

func lineageSavedGuard(r LineageRuntime, b CompilerPhase, receipt LineageLoadReceipt) (transcript supgpu.GuardReceipt, code []supgpu.Pin, err error) {
	if receipt.Observation.Path != filepath.Join(r.Output, "native/observation.json") || len(r.Contract.Environment) < 2 || r.Contract.Environment[0] != "SUPGPU_COMMAND_CWD="+r.WorkingDirectory || r.Contract.Environment[1] != "SUPGPU_GATE_FDS=3,4,5,6" {
		return transcript, nil, fmt.Errorf("lineage observation or receipt annotation identity differs")
	}
	guard, err := guardPin(r.Guard)
	if err != nil {
		return transcript, nil, err
	}
	phases, err := guardPin(r.Phases)
	if err != nil {
		return transcript, nil, err
	}
	code, err = supgpu.ReplayGuarded(guard, phases, lineageCommand(r, b), receipt.Guarded)
	if err != nil {
		return transcript, nil, err
	}
	if !reflect.DeepEqual(code, receipt.Code) {
		return transcript, nil, fmt.Errorf("lineage replayed selected code differs")
	}
	if err = readJSON(File{receipt.Guarded.Transcript.Path, receipt.Guarded.Transcript.SHA256}, &transcript); err != nil {
		return transcript, nil, err
	}
	return transcript, code, nil
}

func checkLineageSavedSupervisor(runtime File, r LineageRuntime, b CompilerPhase, consumer LineagePlan, receipt LineageLoadReceipt, nativePID int, commandFile, watcherFile File, outcome lineageLoadOutcome) error {
	watchPath := filepath.Join(r.Directory, "watch-plan.json")
	watchPin, err := artifactPin(watchPath)
	if err != nil {
		return err
	}
	expected := lineageWatchPlan(r, runtime, b, consumer)
	var saved WatchPlan
	if err = readJSON(watchPin, &saved); err != nil {
		return err
	}
	if !reflect.DeepEqual(saved, expected) {
		return fmt.Errorf("lineage prospective watcher role plan differs")
	}
	invocation := lineageWatchInvocation(r, watchPin)
	if commandFile.Path != invocation.Receipt || watcherFile.Path != filepath.Join(r.Output+"-supervisor", "watcher.json") {
		return fmt.Errorf("lineage supervisor paths differ")
	}
	var actual runner.CommandReceipt
	if err = readJSON(commandFile, &actual); err != nil {
		return err
	}
	if err = lineageSavedSupervisorCommand(actual, invocation, outcome); err != nil {
		return err
	}
	var watch watchObservation
	if err = readJSON(watcherFile, &watch); err != nil {
		return err
	}
	return validateLineageWatch(watch, expected, r, b, consumer, receipt, nativePID, outcome)
}

func lineageSavedSupervisorCommand(actual runner.CommandReceipt, invocation SupervisorInvocation, outcome lineageLoadOutcome) error {
	exit, err := lineageControllerExit(outcome)
	if err != nil {
		return err
	}
	started, startErr := time.Parse(time.RFC3339Nano, actual.Started)
	finished, finishErr := time.Parse(time.RFC3339Nano, actual.Finished)
	if actual.Exit != exit || startErr != nil || finishErr != nil || finished.Before(started) || !slices.Equal(actual.Args, invocation.Args) || !slices.Equal(actual.Environment, invocation.Environment) || actual.Stdout != invocation.Stdout || actual.Stderr != invocation.Stderr {
		return fmt.Errorf("lineage actual supervisor command differs")
	}
	if err = (File{actual.Stdout, actual.StdoutSHA}).verify(); err != nil {
		return err
	}
	return (File{actual.Stderr, actual.StderrSHA}).verify()
}

func validateLineageWatch(w watchObservation, expected WatchPlan, r LineageRuntime, b CompilerPhase, consumer LineagePlan, receipt LineageLoadReceipt, nativePID int, outcome lineageLoadOutcome) error {
	exit, err := lineageControllerExit(outcome)
	if err != nil {
		return err
	}
	if w.Schema != "nanopore-native-role-watch/v1" || !reflect.DeepEqual(w.Plan, expected) || w.ControllerExit == nil || *w.ControllerExit != exit || w.Timeout || w.Error != "" || w.CleanupError != "" || len(w.OwnershipErrors) != 0 || len(w.Signals) != 0 || len(w.Roles) != 1 || w.SignalPolicy != "current-identity-revalidated-pidfd-members-only" {
		return fmt.Errorf("lineage supervisor execution/cleanup failed")
	}
	id := w.ControllerIdentity
	if w.ControllerPID != receipt.ControllerPID || w.ControllerPGID != id.PID || id.PID != receipt.ControllerPID || id.PGID != id.PID || id.SID != id.PID || id.Start == 0 || id.Image != consumer.Go.Controller || id.SHA256 != consumer.Go.ControllerSHA || !slices.Equal(id.Args, expected.Controller) || id.Cgroup == "" || id.Cgroup != w.OwnCgroup {
		return fmt.Errorf("lineage supervised consumer identity differs")
	}
	role := w.Roles[0]
	if role.Name != "lineage" || role.PID != nativePID || role.PGID != nativePID || role.SID != id.SID || role.Start == 0 || role.Limit != 180 || role.SHA256 != b.Output.SHA256 || role.Cgroup != id.Cgroup || !slices.Equal(role.Args, lineageCommand(r, b).Args) {
		return fmt.Errorf("lineage supervised native identity differs")
	}
	want := map[string][]any{fmt.Sprint(id.PID): {}, fmt.Sprint(nativePID): {}}
	if len(w.Members) != 2 {
		return fmt.Errorf("lineage supervisor group observations incomplete")
	}
	for pid := range want {
		members, ok := w.Members[pid]
		if !ok || len(members) != 0 {
			return fmt.Errorf("lineage owned process group remains")
		}
	}
	return nil
}

// Combining two reproductions preserves ambiguity. No historical capture field
// or numerical qualification status is changed by this report.
func CombineLineage(a, b LineageLoadReceipt) (string, error) {
	if a.Status != "guarded_load_lineage_reproduced_only" || b.Status != a.Status || a.Binding.Architecture == b.Binding.Architecture || a.Binding.Output.SHA256 != lineageOutputSHA || b.Binding.Output.SHA256 != lineageOutputSHA || !maps.Equal(map[string]bool{a.Binding.Architecture: true, b.Binding.Architecture: true}, map[string]bool{"sm_120": true, "sm_120a": true}) {
		return "", fmt.Errorf("two independently qualified original cases required")
	}
	return "both_original_inputs_reproduce_output_source_architecture_ambiguous", nil
}
