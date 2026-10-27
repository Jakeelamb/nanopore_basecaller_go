package decodergpu

import (
	"fmt"
	"path/filepath"
	"reflect"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type LineageReplayOptions struct {
	ConsumerPlan, Runtime, Load, SupervisorCommand, SupervisorWatch File
	Out                                                             string
}

// This report cannot replace a producer receipt or a numerical capture.
type LineageReplayReport struct {
	Schema, Status, Error, Scope, Started, Finished                    string
	ConsumerPlan, ConsumerController, ProducerPlan, ProducerController File
	Runtime, NativeBuild, NativeExecutable, SourceLoad                 File
	SupervisorCommand, SupervisorWatch, Observation                    File
	SourceStatus, SourceError                                          string
	NativeExit, SourceControllerExit, NativePID                        int
	Binding                                                            LineageBinding
	Code                                                               []supgpu.Pin
}

func verifyLineageReplayConsumer(producer, consumer LineagePlan) error {
	if producer.Producer != consumer.Producer || producer.ProducerRuntime != consumer.ProducerRuntime || producer.Capture != consumer.Capture || producer.WorkingDirectory != consumer.WorkingDirectory || !reflect.DeepEqual(producer.Cases, consumer.Cases) || !reflect.DeepEqual(producer.Recipe, consumer.Recipe) || !reflect.DeepEqual(producer.Compiler, consumer.Compiler) {
		return fmt.Errorf("lineage replay consumer changed producer inputs")
	}
	if len(producer.SourceBindings) != len(lineageSourcePaths) || len(consumer.SourceBindings) != len(producer.SourceBindings) {
		return fmt.Errorf("lineage replay native source membership differs")
	}
	for original, copy := range producer.SourceBindings {
		currentCopy, ok := consumer.SourceBindings[original]
		want := producer.Inputs[original]
		if !ok || want == "" || consumer.Inputs[original] != want || producer.SnapshotPins[copy] != want || consumer.SnapshotPins[currentCopy] != want {
			return fmt.Errorf("lineage replay native source bytes differ")
		}
	}
	return nil
}

// ReplayLineageLoad reads existing bytes only. The old producer, guard and
// watcher identities remain separate from this currently executing consumer.
func ReplayLineageLoad(o LineageReplayOptions) (report LineageReplayReport, err error) {
	for _, file := range []*File{&o.ConsumerPlan, &o.Runtime, &o.Load, &o.SupervisorCommand, &o.SupervisorWatch} {
		*file, err = normalizeAnchor(*file)
		if err != nil {
			return report, err
		}
	}
	var consumer LineagePlan
	if err = readJSON(o.ConsumerPlan, &consumer); err != nil {
		return report, err
	}
	if err = verifyLineagePlan(consumer, true, true); err != nil {
		return report, err
	}
	r, b, producer, producerConsumer, err := readLineageRuntime(o.Runtime, false)
	if err != nil {
		return report, err
	}
	if err = verifyLineageReplayConsumer(producer, consumer); err != nil {
		return report, err
	}
	if o.Load.Path != filepath.Join(r.Output, "load.json") {
		return report, fmt.Errorf("lineage load receipt identity differs")
	}
	var saved LineageLoadReceipt
	if err = readJSON(o.Load, &saved); err != nil {
		return report, err
	}
	if err = validateLineageSavedLoad(saved, o.Runtime, lineageEnvironmentReaderFailed); err != nil {
		return report, err
	}
	if o.Out == "" {
		return report, fmt.Errorf("lineage replay requires a fresh report directory")
	}
	out, err := runner.NewAttempt(o.Out)
	if err != nil {
		return report, err
	}
	report = LineageReplayReport{
		Schema: "nanopore-decoder-load-replay/v1", Status: "rejected", Scope: "contemporaneous-load-only", Started: time.Now().UTC().Format(time.RFC3339Nano),
		ConsumerPlan: o.ConsumerPlan, ConsumerController: File{consumer.Go.Controller, consumer.Go.ControllerSHA}, ProducerPlan: b.Plan, ProducerController: File{producerConsumer.Go.Controller, producerConsumer.Go.ControllerSHA},
		Runtime: o.Runtime, NativeBuild: r.Build, NativeExecutable: b.Output, SourceLoad: o.Load, SupervisorCommand: o.SupervisorCommand, SupervisorWatch: o.SupervisorWatch,
		SourceStatus: saved.Status, SourceError: saved.Error, NativeExit: saved.Guarded.Command.Exit, SourceControllerExit: 1, Observation: saved.Observation,
	}
	defer func() {
		report.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			report.Error = err.Error()
		}
		if e := writeJSON(filepath.Join(out, "replay.json"), report); e != nil {
			if err == nil {
				err = e
			}
			report.Status, report.Error = "rejected", err.Error()
		}
	}()
	transcript, code, err := lineageSavedGuard(r, b, saved)
	if err != nil {
		return report, err
	}
	report.NativePID = transcript.PID
	if err = checkLineageSavedSupervisor(o.Runtime, r, b, producerConsumer, saved, transcript.PID, o.SupervisorCommand, o.SupervisorWatch, lineageEnvironmentReaderFailed); err != nil {
		return report, err
	}
	observation, binding, err := ReadLineageObservation(saved.Observation, r.Contract)
	if observation.PID > 0 && observation.PID != transcript.PID {
		return report, fmt.Errorf("lineage observer PID differs")
	}
	if err != nil {
		return report, err
	}
	for _, file := range []File{o.ConsumerPlan, o.Runtime, o.Load, o.SupervisorCommand, o.SupervisorWatch, saved.Observation, observation.InputCopy, binding.Output} {
		if err = file.verify(); err != nil {
			return report, err
		}
	}
	if _, _, _, _, err = readLineageRuntime(o.Runtime, false); err != nil {
		return report, err
	}
	if err = verifyLineagePlan(consumer, true, true); err != nil {
		return report, err
	}
	if _, _, err = lineageSavedGuard(r, b, saved); err != nil {
		return report, err
	}
	report.Binding, report.Code = binding, code
	report.Status = "load_only_observation_qualified"
	return report, nil
}
