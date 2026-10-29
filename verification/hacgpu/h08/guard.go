package h08

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08/cgobuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

// Admission pins come from the coordinator's separately reviewed launch, not
// from a child receipt or a local permission flag.
type RuntimeAdmission struct {
	Schema                  string
	Request, Plan, Contract Pin
	Identity                Pin
	ParentBuild             Pin
	ParentBuildSelection    Pin
	Command                 supgpu.GuardCommand
}

type GuardedConsumer struct {
	Schema, Status            string
	Admission                 Pin
	Run                       supgpu.GuardedRun
	Consumer                  Pin
	BeforeDevice, AfterDevice string
	Metadata                  []runner.CommandReceipt
}

func fixedCommand(executable Pin, request Pin) []string {
	return []string{executable.Path, "-mode", "guarded-child", "-request", request.Path, "-request-sha", request.SHA256}
}

func guardPin(p Pin) supgpu.Pin { return supgpu.Pin{Path: p.Path, SHA256: p.SHA256} }

func pinnedJSON(p Pin, v any) error {
	actual, e := closedbuild.PinFile(p.Path)
	if e != nil || actual != p {
		return fmt.Errorf("canonical admission pin differs: %s: %v", p.Path, e)
	}
	return readJSON(p, v)
}

func validateGuardInputs(a RuntimeAdmission, target Pin, request Pin, required []Pin, action string) error {
	var p supgpu.GuardPlan
	var c supgpu.PhaseContract
	if e := pinnedJSON(a.Plan, &p); e != nil {
		return e
	}
	if e := pinnedJSON(a.Contract, &c); e != nil {
		return e
	}
	if a.Schema != "nanopore-h08-runtime-admission/v2" || p.Schema != "nanopore-audited-guard-plan/v1" || p.Contract != guardPin(a.Contract) || c.Schema != "nanopore-audited-phase-contract/v1" || c.Role != "observer" || c.Action != action || !slices.Equal(c.Names, []string{"pre-model", "pre-forward", "post-forward"}) {
		return fmt.Errorf("H08 exact phase/admission contract differs")
	}
	if p.Policy.Observer != guardPin(target) || p.Policy.Reference != guardPin(target) || !filepath.IsAbs(a.Command.Directory) || !filepath.IsAbs(a.Command.Output) {
		return fmt.Errorf("H08 guard selected target/directories differ")
	}
	selected := map[string]string{}
	for _, v := range p.Inputs {
		if _, exists := selected[v.Path]; exists {
			return fmt.Errorf("duplicate guard input")
		}
		selected[v.Path] = v.SHA256
	}
	for _, v := range append(slices.Clone(required), request, a.ParentBuild, a.ParentBuildSelection) {
		if !filepath.IsAbs(v.Path) || len(v.SHA256) != 64 || selected[v.Path] != v.SHA256 {
			return fmt.Errorf("guard omits required custody input: %s", v.Path)
		}
	}
	return checkPins(selected)
}

// ReadAdmission does only authenticated file reads. It cannot load the selected
// backend, execute a tool, acquire a lock or create an attempt.
func ReadAdmission(pin Pin) (a RuntimeAdmission, o FixedRequest, err error) {
	if pin.Path == "" || pin.SHA256 == "" {
		return a, o, ErrRuntimeHeld
	}
	if err = pinnedJSON(pin, &a); err != nil {
		return
	}
	if a.Schema != "nanopore-h08-runtime-admission/v2" {
		return a, o, fmt.Errorf("H08 runtime admission schema differs")
	}
	if err = pinnedJSON(a.Request, &o); err != nil {
		return
	}
	if err = validateFixedRequest(context.Background(), o); err != nil {
		return
	}
	if a.ParentBuild != o.ControllerBuild || a.ParentBuildSelection != o.ControllerBuildSelection {
		return a, o, fmt.Errorf("H08 parent/child current build custody differs")
	}
	build, e := cgobuild.ReadH08Artifact(o.ControllerBuild, o.ControllerBuildSelection)
	if e != nil {
		return a, o, e
	}
	backend, e := AuthenticateBackend(o.BackendArtifact, o.Backend)
	if e != nil {
		return a, o, e
	}
	if e = validateBackendPolicy(a, backend); e != nil {
		return a, o, e
	}
	if _, e = ReadCPU13(o.CPU13, o.BackendArtifact); e != nil {
		return a, o, e
	}
	if _, e = readIdentityPlan(a.Identity); e != nil {
		return a, o, e
	}
	if o.Phase == "padding" {
		previous, e := ReadConsumer(o.AllActiveReceipt)
		if e != nil || previous.Status != "fixed_all_active_passed" {
			return a, o, fmt.Errorf("padding requires audited all-active acceptance: %v", e)
		}
	}
	if !slices.Equal(a.Command.Args, fixedCommand(build.Output(), a.Request)) || a.Command.Output == o.Out || !filepath.IsAbs(o.Out) {
		return a, o, fmt.Errorf("H08 child argv/output differs from independent request")
	}
	required := []Pin{a.ParentBuild, a.ParentBuildSelection, a.Identity, o.ControllerBuild, o.ControllerBuildSelection, o.CPU13, o.BackendArtifact.Build, o.BackendArtifact.Plan, o.BackendArtifact.Producer, o.BackendArtifact.Output}
	if o.Phase == "padding" {
		required = append(required, o.AllActiveReceipt)
	}
	err = validateGuardInputs(a, build.Output(), a.Request, required, "hac-"+o.Phase)
	return
}

func validateBackendPolicy(a RuntimeAdmission, b closedbuild.BackendArtifact) error {
	var plan supgpu.GuardPlan
	if e := pinnedJSON(a.Plan, &plan); e != nil {
		return e
	}
	allowed := map[string]string{}
	for _, code := range plan.Policy.Allowed {
		allowed[code.Object.Path] = code.Object.SHA256
	}
	if allowed[b.Output.Path] != b.Output.SHA256 {
		return fmt.Errorf("audited runtime omits selected genuine backend")
	}
	for _, code := range b.Runtime {
		if allowed[code.Pin.Path] != code.Pin.SHA256 {
			return fmt.Errorf("audited runtime differs from qualified backend provider: %s", code.Pin.Path)
		}
	}
	return nil
}

func Consume(ctx context.Context, options ConsumerOptions) (r GuardedConsumer, err error) {
	if ctx == nil {
		return r, fmt.Errorf("nil context")
	}
	if err = ctx.Err(); err != nil {
		return r, err
	}
	a, o, err := ReadAdmission(options.Admission)
	if err != nil {
		return r, err
	}
	if err = actualController(a.ParentBuild, a.ParentBuildSelection); err != nil {
		return r, err
	}
	if err = runner.CheckRoot(); err != nil {
		return r, err
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return r, err
	}
	defer lock.Close()
	r = GuardedConsumer{Schema: "nanopore-h08-guarded-consumer/v1", Status: "failed", Admission: options.Admission}
	witness, err := ReadWitness(o.Sources)
	if err != nil {
		return r, err
	}
	metadata, err := readIdentityPlan(a.Identity)
	if err != nil {
		return r, err
	}
	if metadata.Executable != witness.IdentityTool {
		return r, fmt.Errorf("metadata tool differs from retained source identity tool")
	}
	metadataOut := a.Command.Output + "-identity"
	if err = os.Mkdir(metadataOut, 0755); err != nil {
		return r, err
	}
	defer func() {
		if err != nil {
			r.Status = "failed"
		}
		if e := writeJSONFile(filepath.Join(metadataOut, "parent.json"), r); err == nil {
			err = e
		}
	}()
	before, value, err := executeIdentity(ctx, metadataOut, "before", metadata, o.Ordinal)
	r.Metadata = append(r.Metadata, before)
	r.BeforeDevice = value
	if err != nil || value != witness.Device {
		return r, fmt.Errorf("source UUID metadata bookend differs: %v", err)
	}
	r.Run, err = supgpu.ExecuteGuarded(guardPin(a.Plan), guardPin(a.Contract), a.Command)
	if err != nil {
		return r, err
	}
	after, value, err := executeIdentity(ctx, metadataOut, "after", metadata, o.Ordinal)
	r.Metadata = append(r.Metadata, after)
	r.AfterDevice = value
	if err != nil || value != r.BeforeDevice {
		return r, fmt.Errorf("UUID metadata changed across guarded graph: %v", err)
	}
	if _, err = supgpu.ReplayGuarded(guardPin(a.Plan), guardPin(a.Contract), a.Command, r.Run); err != nil {
		return r, err
	}
	r.Consumer, err = closedbuild.PinFile(filepath.Join(o.Out, "consumer.json"))
	if err != nil {
		return r, err
	}
	var numerical ConsumerReceipt
	if err = pinnedJSON(r.Consumer, &numerical); err != nil {
		return r, err
	}
	if err = validateConsumerBuild(numerical, o); err != nil {
		return r, err
	}
	want := "fixed_all_active_passed"
	if o.Phase == "padding" {
		want = "one_slot_independence_probe_passed_production_policy_held"
	}
	if numerical.Status != want || numerical.CPU13 != o.CPU13 || numerical.Backend.Output != o.BackendArtifact.Output || numerical.Phase != o.Phase {
		return r, fmt.Errorf("guarded numerical receipt differs from selected phase")
	}
	if err = checkPins(numerical.Inputs); err != nil {
		return r, err
	}
	if err = checkPins(numerical.Outputs); err != nil {
		return r, err
	}
	r.Status = want
	err = closedbuild.Write(filepath.Join(a.Command.Output, "h08.json"), r)
	return
}

func ReadConsumer(pin Pin) (r GuardedConsumer, err error) {
	if err = pinnedJSON(pin, &r); err != nil {
		return
	}
	if r.Schema != "nanopore-h08-guarded-consumer/v1" || r.Status != "fixed_all_active_passed" {
		return r, fmt.Errorf("audited all-active receipt absent")
	}
	var pre RuntimeAdmission
	var preRequest FixedRequest
	if err = pinnedJSON(r.Admission, &pre); err != nil {
		return
	}
	if err = pinnedJSON(pre.Request, &preRequest); err != nil {
		return
	}
	if preRequest.Phase != "all-active" {
		return r, fmt.Errorf("prior phase is not all-active")
	}
	a, o, e := ReadAdmission(r.Admission)
	if e != nil {
		return r, e
	}
	if o.Phase != "all-active" {
		return r, fmt.Errorf("prior phase is not all-active")
	}
	if _, err = supgpu.ReplayGuarded(guardPin(a.Plan), guardPin(a.Contract), a.Command, r.Run); err != nil {
		return
	}
	w, e := ReadWitness(o.Sources)
	if e != nil {
		return r, e
	}
	if len(r.Metadata) != 2 || r.BeforeDevice != w.Device || r.AfterDevice != w.Device {
		return r, fmt.Errorf("audited source UUID bookends absent")
	}
	p, e := readIdentityPlan(a.Identity)
	if e != nil {
		return r, e
	}
	for i, c := range r.Metadata {
		label := "before"
		if i == 1 {
			label = "after"
		}
		if err = validateIdentityCommand(c, a.Command.Output+"-identity", label, p, o.Ordinal, w.Device); err != nil {
			return
		}
	}
	before, e := time.Parse(time.RFC3339Nano, r.Metadata[0].Finished)
	start, f := time.Parse(time.RFC3339Nano, r.Run.Command.Started)
	end, g := time.Parse(time.RFC3339Nano, r.Run.Command.Finished)
	after, h := time.Parse(time.RFC3339Nano, r.Metadata[1].Started)
	if e != nil || f != nil || g != nil || h != nil || before.After(start) || end.After(after) {
		return r, fmt.Errorf("UUID metadata does not surround actual guarded child")
	}
	var numerical ConsumerReceipt
	if err = pinnedJSON(r.Consumer, &numerical); err != nil {
		return
	}
	if err = validateConsumerBuild(numerical, o); err != nil {
		return
	}
	if r.Consumer.Path != filepath.Join(o.Out, "consumer.json") || numerical.Status != r.Status || numerical.Executable != aPolicyTarget(a) || numerical.CPU13 != o.CPU13 || numerical.Backend.Output != o.BackendArtifact.Output || numerical.Sources != o.Sources || len(numerical.Stages) != 13 || len(numerical.Caches) != 58 || len(numerical.Scratch) != 8 || numerical.ProductionForwardCalls != 1 || numerical.InactiveProbeCalls != 0 {
		return r, fmt.Errorf("audited all-active numerical evidence differs")
	}
	if err = checkPins(numerical.Inputs); err != nil {
		return
	}
	err = checkPins(numerical.Outputs)
	return
}

func validateConsumerBuild(r ConsumerReceipt, o FixedRequest) error {
	if r.Schema != "nanopore-hac-fixed-consumer/v4" || r.ControllerBuild != o.ControllerBuild || r.ControllerBuildSelection != o.ControllerBuildSelection {
		return fmt.Errorf("numerical receipt build/selection/schema differs")
	}
	for _, p := range []Pin{o.ControllerBuild, o.ControllerBuildSelection} {
		if !filepath.IsAbs(p.Path) || len(p.SHA256) != 64 || r.Inputs[p.Path] != p.SHA256 {
			return fmt.Errorf("numerical receipt omits build/selection input")
		}
	}
	return nil
}

func actualController(build, selection Pin) error {
	b, e := cgobuild.ReadH08Artifact(build, selection)
	if e != nil {
		return e
	}
	path, e := os.Executable()
	if e != nil {
		return e
	}
	p, e := closedbuild.PinFile(path)
	if e != nil || p != b.Output() {
		return fmt.Errorf("executing parent differs from current CGO1 controller: %v", e)
	}
	return nil
}

func aPolicyTarget(a RuntimeAdmission) Pin {
	var p supgpu.GuardPlan
	if pinnedJSON(a.Plan, &p) != nil {
		return Pin{}
	}
	return Pin{Path: p.Policy.Observer.Path, SHA256: p.Policy.Observer.SHA256}
}

// RunGuardedChild is entered only by the exact externally bound child argv.
// An ordinary invocation lacks the private phase pipes and stops before model
// deserialization, backend loading, device construction or Forward.
func RunGuardedChild(ctx context.Context, request Pin) (r ConsumerReceipt, err error) {
	if ctx == nil {
		return r, fmt.Errorf("nil context")
	}
	if err = ctx.Err(); err != nil {
		return r, err
	}
	phases, err := supgpu.OpenPhaseClient()
	if err != nil {
		return r, errors.Join(ErrRuntimeHeld, err)
	}
	defer func() { err = errors.Join(err, phases.Close()) }()
	var o FixedRequest
	if err = pinnedJSON(request, &o); err != nil {
		return r, err
	}
	return consumeGuarded(ctx, o, phases)
}

func writeJSONFile(path string, value any) error {
	b, e := json.MarshalIndent(value, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(path, append(b, '\n'), 0600)
}

type IdentityPlan struct {
	Schema     string
	Executable Pin
	Seeds      []Pin
	SearchDirs []string
	Closure    nativeproof.ELFClosure
}
