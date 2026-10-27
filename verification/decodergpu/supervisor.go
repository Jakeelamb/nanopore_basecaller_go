package decodergpu

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"reflect"
	"strconv"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type WatchRole struct {
	Name   string   `json:"name"`
	Args   []string `json:"argv"`
	SHA256 string   `json:"sha256"`
	Limit  int      `json:"limit_seconds"`
}
type WatchPlan struct {
	Controller []string    `json:"controller_argv"`
	Limit      int         `json:"controller_limit_seconds"`
	Roles      []WatchRole `json:"roles"`
}
type SupervisorInvocation struct {
	Args, Environment                  []string
	Directory, Stdout, Stderr, Receipt string
}

func expectedWatchPlan(r RuntimePlan, runtime File, b CompilerPhase, p ClosedBuildPlan) WatchPlan {
	args := []string{r.Consumer.Path, "-unit", "capture", "-build", r.Build.Path, "-build-sha256", r.Build.SHA256, "-preparation", r.ConsumerPreparation.Path, "-preparation-sha256", r.ConsumerPreparation.SHA256, "-runtime-plan", runtime.Path, "-runtime-plan-sha256", runtime.SHA256, "-admitted", "-out", r.CaptureDirectory}
	limit := 240
	if numericBuild(p) {
		limit = 540
	}
	return WatchPlan{args, limit, []WatchRole{{"decoder", decoderCommand(r, b, p).Args, b.Output.SHA256, 180}}}
}
func expectedSupervisor(r RuntimePlan, watch File) SupervisorInvocation {
	dir := r.CaptureDirectory + "-supervisor"
	return SupervisorInvocation{Args: []string{r.WatcherInterpreter.Path, "-B", r.Watcher.Path, watch.Path, filepath.Join(dir, "watcher.json"), watch.SHA256}, Environment: []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "TZ=UTC", "GOMAXPROCS=2", "GOMEMLIMIT=3GiB", "PYTHONDONTWRITEBYTECODE=1"}, Directory: r.WorkingDirectory, Stdout: filepath.Join(dir, "stdout"), Stderr: filepath.Join(dir, "stderr"), Receipt: filepath.Join(dir, "command.json")}
}

type watchObservation struct {
	Schema             string    `json:"schema"`
	Plan               WatchPlan `json:"plan"`
	ControllerPID      int       `json:"controller_pid"`
	ControllerPGID     int       `json:"controller_pgid"`
	ControllerExit     *int      `json:"controller_exit"`
	ControllerIdentity struct {
		PID, PGID, SID int
		Image          string
		Args           []string `json:"argv"`
		SHA256         string   `json:"image_sha256"`
		Cgroup         string
		Start          uint64 `json:"start_ticks"`
	} `json:"controller_identity"`
	Roles []struct {
		Name           string `json:"name"`
		PID, PGID, SID int
		Args           []string `json:"argv"`
		SHA256         string   `json:"image_sha256"`
		Limit          int      `json:"limit_seconds"`
		Start          uint64   `json:"start_ticks"`
		Cgroup         string
	} `json:"roles"`
	Timeout         bool                         `json:"timeout"`
	Error           string                       `json:"error"`
	CleanupError    string                       `json:"cleanup_error"`
	OwnershipErrors []json.RawMessage            `json:"ownership_errors"`
	Signals         []json.RawMessage            `json:"signals"`
	Members         map[string][]json.RawMessage `json:"members_after_cleanup"`
	OwnCgroup       string                       `json:"own_cgroup"`
	SignalPolicy    string                       `json:"signal_policy"`
}

func checkSupervisor(r RuntimePlan, runtime, commandFile, watchFile File, b CompilerPhase, p ClosedBuildPlan, c CaptureReceipt) error {
	watchPlanPath := filepath.Join(filepath.Dir(runtime.Path), "watch-plan.json")
	watchPlanPin, err := artifactPin(watchPlanPath)
	if err != nil {
		return err
	}
	var saved WatchPlan
	if err = readJSON(watchPlanPin, &saved); err != nil {
		return err
	}
	expected := expectedWatchPlan(r, runtime, b, p)
	if !reflect.DeepEqual(saved, expected) {
		return fmt.Errorf("independent exact watcher role plan changed")
	}
	invocation := expectedSupervisor(r, watchPlanPin)
	if commandFile.Path != invocation.Receipt || watchFile.Path != filepath.Join(r.CaptureDirectory+"-supervisor", "watcher.json") {
		return fmt.Errorf("supervisor output identity differs")
	}
	var command runner.CommandReceipt
	if err = readJSON(commandFile, &command); err != nil {
		return err
	}
	if err = commandEvidence(command); err != nil {
		return err
	}
	if !reflect.DeepEqual(command.Args, invocation.Args) || !reflect.DeepEqual(command.Environment, invocation.Environment) || command.Stdout != invocation.Stdout || command.Stderr != invocation.Stderr {
		return fmt.Errorf("actual supervisor command differs from prospective invocation")
	}
	var observation watchObservation
	if err = readJSON(watchFile, &observation); err != nil {
		return err
	}
	var transcript supgpu.GuardReceipt
	if err = readJSON(File{c.Guarded.Transcript.Path, c.Guarded.Transcript.SHA256}, &transcript); err != nil {
		return err
	}
	return validateWatchObservation(observation, expected, r, b, p, c, transcript.PID)
}

func validateWatchObservation(observation watchObservation, expected WatchPlan, r RuntimePlan, b CompilerPhase, p ClosedBuildPlan, c CaptureReceipt, rolePID int) error {
	if observation.Schema != "nanopore-native-role-watch/v1" || !reflect.DeepEqual(observation.Plan, expected) || observation.ControllerExit == nil || *observation.ControllerExit != 0 || observation.Timeout || observation.Error != "" || observation.CleanupError != "" || len(observation.OwnershipErrors) != 0 || len(observation.Signals) != 0 || len(observation.Roles) != 1 || observation.SignalPolicy != "current-identity-revalidated-pidfd-members-only" {
		return fmt.Errorf("supervisor failed or lacked exact role execution/cleanup")
	}
	identity := observation.ControllerIdentity
	if observation.ControllerPID != c.ControllerPID || identity.PID != c.ControllerPID || observation.ControllerPGID != c.ControllerPID || identity.PGID != c.ControllerPID || identity.SID != c.ControllerPID || identity.Image != r.Consumer.Path || identity.SHA256 != r.Consumer.SHA256 || identity.Start == 0 || !reflect.DeepEqual(identity.Args, expected.Controller) || identity.Cgroup == "" || identity.Cgroup != observation.OwnCgroup {
		return fmt.Errorf("supervised current consumer identity differs")
	}
	role := observation.Roles[0]
	if role.Name != "decoder" || role.PID != rolePID || role.PGID != role.PID || role.SID != identity.SID || role.Start == 0 || role.Limit != 180 || role.SHA256 != b.Output.SHA256 || role.Cgroup != identity.Cgroup || !reflect.DeepEqual(role.Args, decoderCommand(r, b, p).Args) {
		return fmt.Errorf("supervised native decoder identity differs from guarded execution")
	}
	if len(observation.Members) != 2 {
		return fmt.Errorf("supervisor cleanup observations incomplete")
	}
	for _, pid := range []int{identity.PID, role.PID} {
		members, ok := observation.Members[strconv.Itoa(pid)]
		if !ok || len(members) != 0 {
			return fmt.Errorf("supervised process group remains")
		}
	}
	return nil
}

func sameCapturedBuffers(a, b []CapturedBuffer) bool { return reflect.DeepEqual(a, b) }

// CheckCapture is an import, never a producer replay. Its independent runtime,
// capture and supervision pins must come from the accepted execution envelope.
func CheckCapture(o CaptureOptions, capture, command, watch File) (c CaptureReceipt, err error) {
	proofs := newNumericProofs()
	r, b, p, err := readGuardedCaptureScoped(o, false, proofs)
	if err != nil {
		return c, err
	}
	for _, file := range []*File{&capture, &command, &watch} {
		*file, err = normalizeAnchor(*file)
		if err != nil {
			return c, err
		}
	}
	if capture.Path != filepath.Join(r.CaptureDirectory, "capture.json") {
		return c, fmt.Errorf("capture receipt path differs")
	}
	if err = readJSON(capture, &c); err != nil {
		return c, err
	}
	runtime, err := normalizeAnchor(o.Runtime)
	if err != nil {
		return c, err
	}
	if !captureSchemaMatches(c, p) || c.Status != "native_captured_pending_supervisor" || c.Error != "" || c.Build != r.Build || c.Consumer != r.ConsumerPreparation || c.Runtime != runtime || c.Contract.Validate() != nil {
		return c, fmt.Errorf("capture identity/status/contract differs")
	}
	guard, err := guardPin(r.Guard)
	if err != nil {
		return c, err
	}
	phases, err := guardPin(r.PhaseContract)
	if err != nil {
		return c, err
	}
	code, err := supgpu.ReplayGuarded(guard, phases, decoderCommand(r, b, p), c.Guarded)
	if err != nil {
		return c, err
	}
	if !reflect.DeepEqual(code, c.Code) {
		return c, fmt.Errorf("replayed selected code differs")
	}
	old, err := readPreparation(p.Preparation, false, proofs)
	if err != nil {
		return c, err
	}
	images, err := readCaptureImagesScoped(r, filepath.Join(r.CaptureDirectory, "native/images-decode"), old.Images, proofs)
	if err != nil {
		return c, err
	}
	if !reflect.DeepEqual(images, c.Images) {
		return c, fmt.Errorf("actual selected image evidence differs")
	}
	buffers, files, err := readNativeBuffers(filepath.Join(r.CaptureDirectory, "native"), c.Contract)
	if err != nil {
		return c, err
	}
	if numericBuild(p) {
		identity, e := readNumericIdentityScoped(filepath.Join(r.CaptureDirectory, "native"), r, images, c.Guarded, proofs)
		if e != nil {
			return c, e
		}
		if !reflect.DeepEqual(identity, c.Identity) {
			return c, fmt.Errorf("numeric native identity changed")
		}
		for _, row := range identity {
			for path, sha := range row.Files {
				files[path] = sha
			}
		}
	} else if len(c.Identity) != 0 {
		return c, fmt.Errorf("legacy capture has numeric identity")
	}
	if !sameCapturedBuffers(buffers, c.Buffers) || !reflect.DeepEqual(files, c.Files) {
		return c, fmt.Errorf("retained stage buffer pins/prospective masks differ")
	}
	if err = checkSupervisor(r, runtime, command, watch, b, p, c); err != nil {
		return c, err
	}
	if err = proofs.finish(); err != nil {
		return c, err
	}
	c.Status = "native_oracle_qualified"
	c.SourceCapture, c.SupervisorCommand, c.SupervisorReceipt = capture, command, watch
	c.QualifiedUTC = time.Now().UTC().Format(time.RFC3339Nano)
	return c, nil
}
