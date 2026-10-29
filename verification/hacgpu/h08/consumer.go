package h08

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08/cgobuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type FixedRequest struct {
	Schema                                   string
	Sources                                  Sources
	Backend, Model, Out, Phase               string
	ControllerBuild, CPU13, AllActiveReceipt Pin
	ControllerBuildSelection                 Pin
	BackendArtifact                          closedbuild.ArtifactOptions
	Ordinal                                  int
}
type ConsumerOptions struct{ Admission Pin }
type StageComparison struct {
	Name, File, SHA256, DefinedSHA, NativeDefinedSHA, WorkspaceFile, WorkspaceSHA string
	DefinedBytes                                                                  int
	Equal                                                                         bool
	Contract                                                                      StageContract
}
type CacheComparison struct {
	Role, File, SHA256, NativeSHA string
	Bytes                         int
	Equal                         bool
}
type ScratchComparison struct {
	Name, GoSHA, NativeSHA string
	Bytes                  int64
	Equal                  bool
}
type Control struct{ Name, Status, File, SHA256, Error string }
type ConsumerReceipt struct {
	Schema, Status, Error, Started, Finished, Attempt, Phase string
	Invocation                                               []string
	Sources                                                  Sources
	Executable, ControllerBuild, CPU13, AllActiveReceipt     Pin
	ControllerBuildSelection                                 Pin
	Backend                                                  closedbuild.BackendArtifact
	Inputs, Outputs, Runtime                                 map[string]string
	BeforeDevice, AfterDevice                                string
	Environment                                              map[string]h07.RuntimeEnvValue
	Commands                                                 []runner.CommandReceipt
	Stages                                                   []StageComparison
	Caches                                                   []CacheComparison
	Scratch                                                  []ScratchComparison
	Controls                                                 []Control
	ActiveSlots, PhysicalSlots                               int
	Metadata                                                 model.HACGPUScoreMetadata
	WorkspacePattern                                         byte
	ProductionForwardCalls, InactiveProbeCalls               int
	Limits                                                   []string
}

var ErrRuntimeHeld = errors.New("H08 runtime HOLD: independently pinned audited admission, current CGO1 build and genuine CPU13 qualification required")

// AuthenticateBackend is the host-only boundary. It reads no retained NN
// witness and deserializes no model weights. It confers no runtime authority.
func AuthenticateBackend(o closedbuild.ArtifactOptions, selected string) (closedbuild.BackendArtifact, error) {
	p, err := closedbuild.PinFile(selected)
	if err != nil {
		return closedbuild.BackendArtifact{}, err
	}
	if p != o.Output {
		return closedbuild.BackendArtifact{}, fmt.Errorf("H08 selected backend differs from admitted output")
	}
	return closedbuild.ReadArtifact(o)
}

func verifyCache(c model.HACGPUCacheTensor, w CacheContract) error {
	if c.Role != w.Role || c.DType != w.DType || !slices.Equal(c.Shape, w.Shape) || len(c.Data) != w.Bytes || digest(c.Data) != w.SHA256 {
		return fmt.Errorf("production cache differs from retained native role: %s", w.Role)
	}
	return nil
}

func stageSelection(s StageContract, b []byte, firstSlot bool) ([]byte, error) {
	if !firstSlot {
		return selectedBytes(s, b)
	}
	if int64(len(b)) != s.Bytes {
		return nil, fmt.Errorf("stage extent differs")
	}
	copy := s
	copy.Shape = slices.Clone(s.Shape)
	elem := int64(2)
	if s.DType == "Char" {
		elem = 1
	}
	if s.Layout == "NTC" {
		copy.Shape[0] = 1
		copy.Bytes = copy.Shape[1] * copy.Shape[2] * elem
		copy.Defined = nil
		var err error
		copy.Defined, err = definedRanges(copy)
		if err != nil {
			return nil, err
		}
		return selectedBytes(copy, b[:copy.Bytes])
	}
	t, n, c := s.Shape[0], s.Shape[1], s.Shape[2]
	out := []byte{}
	for row := int64(0); row < t; row++ {
		start := row * n * c * elem
		end := start + c*elem
		for _, r := range s.Defined {
			if start >= r.Offset && end <= r.Offset+r.Bytes {
				out = append(out, b[start:end]...)
				break
			}
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("first-slot comparison is empty")
	}
	return out, nil
}

func runtimeEnvironment() map[string]h07.RuntimeEnvValue {
	out := map[string]h07.RuntimeEnvValue{}
	for _, key := range []string{"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG", "DORADO_MATMUL_FP16_MODE", "LD_LIBRARY_PATH", "LD_PRELOAD"} {
		v, present := os.LookupEnv(key)
		out[key] = h07.RuntimeEnvValue{Present: present, Value: v}
	}
	return out
}

// Consume is the only native entry. The caller must independently admit this
// exact phase and pin the complete current Go build, backend and prospective
// runtime policy. No source oracle or stock model executable is invoked.
func validateFixedRequest(ctx context.Context, o FixedRequest) error {
	if o.Schema != "nanopore-h08-fixed-request/v1" {
		return fmt.Errorf("H08 fixed request schema differs")
	}
	if o.Phase != "all-active" && o.Phase != "padding" {
		return fmt.Errorf("unknown fixed graph phase")
	}
	if o.Phase == "padding" && len(o.AllActiveReceipt.SHA256) != 64 {
		return fmt.Errorf("accepted all-active receipt must precede the padding probe")
	}
	if ctx == nil {
		return fmt.Errorf("nil context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if o.Ordinal < 0 || int64(o.Ordinal) > 2147483647 {
		return fmt.Errorf("invalid CUDA ordinal")
	}
	return nil
}

func consumeGuarded(ctx context.Context, o FixedRequest, phases *supgpu.PhaseClient) (r ConsumerReceipt, err error) {
	if err = validateFixedRequest(ctx, o); err != nil {
		return r, err
	}
	controller, err := cgobuild.ReadH08Artifact(o.ControllerBuild, o.ControllerBuildSelection)
	if err != nil {
		return r, err
	}
	r = ConsumerReceipt{Schema: "nanopore-hac-fixed-consumer/v4", Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Phase: o.Phase, Sources: o.Sources, Invocation: slices.Clone(os.Args), Inputs: map[string]string{}, Outputs: map[string]string{}, Runtime: map[string]string{}, ControllerBuild: o.ControllerBuild, ControllerBuildSelection: o.ControllerBuildSelection, CPU13: o.CPU13, AllActiveReceipt: o.AllActiveReceipt, Limits: []string{"No source producer, full fallback, VCS, other geometry, biological or performance qualification.", "One-slot ProbeInactiveSlots never changes production admission; production remains held for that policy until a separately reviewed integration.", "Reusing retained unit observations does not repair their historical compiler/link/runtime custody limits."}}
	r.Attempt, err = runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
			r.Status = "failed"
		}
		raw, e := json.MarshalIndent(r, "", "  ")
		if e == nil {
			e = os.WriteFile(filepath.Join(r.Attempt, "consumer.json"), append(raw, '\n'), 0644)
		}
		if err == nil {
			err = e
		}
	}()
	executable, err := os.Executable()
	if err != nil {
		return r, err
	}
	pin, err := runner.SHA(executable)
	if err != nil {
		return r, err
	}
	r.Executable = Pin{Path: executable, SHA256: pin}
	if controller.Output() != r.Executable {
		return r, fmt.Errorf("current executable differs from CGO1 output")
	}
	if _, err = ReadCPU13(o.CPU13, o.BackendArtifact); err != nil {
		return r, err
	}
	r.Inputs[o.CPU13.Path] = o.CPU13.SHA256
	r.Inputs[o.ControllerBuild.Path] = o.ControllerBuild.SHA256
	r.Inputs[o.ControllerBuildSelection.Path] = o.ControllerBuildSelection.SHA256
	r.Inputs[executable] = pin
	w, err := ReadWitness(o.Sources)
	if err != nil {
		return r, err
	}
	maps.Copy(r.Inputs, w.Inputs)
	r.Backend, err = AuthenticateBackend(o.BackendArtifact, o.Backend)
	if err != nil {
		return r, err
	}
	maps.Copy(r.Inputs, r.Backend.Inputs)
	r.Environment = runtimeEnvironment()
	// Loader settings are selected by the audited parent. Restore only source
	// numerical choices; never replace its prospective library policy.
	for _, key := range []string{"DORADO_MATMUL_FP16_MODE", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG"} {
		v := w.Environment[key]
		if v.Present {
			err = os.Setenv(key, v.Value)
		} else {
			err = os.Unsetenv(key)
		}
		if err != nil {
			return r, err
		}
	}
	r.Environment = runtimeEnvironment()
	if o.Phase == "padding" {
		guarded, e := ReadConsumer(o.AllActiveReceipt)
		if e != nil {
			return r, e
		}
		var before ConsumerReceipt
		if err = pinnedJSON(guarded.Consumer, &before); err != nil {
			return r, err
		}
		if err = validateConsumerBuild(before, o); err != nil {
			return r, err
		}
		if before.Status != "fixed_all_active_passed" || before.Executable != r.Executable || before.Backend.Output != r.Backend.Output || before.Sources != r.Sources || len(before.Stages) != 13 || len(before.Caches) != 58 || len(before.Scratch) != 8 || before.ProductionForwardCalls != 1 || before.InactiveProbeCalls != 0 {
			return r, fmt.Errorf("padding probe lacks same-build all-active acceptance")
		}
		if err = checkPins(before.Inputs); err != nil {
			return r, err
		}
		if err = checkPins(before.Outputs); err != nil {
			return r, err
		}
		r.Inputs[o.AllActiveReceipt.Path] = o.AllActiveReceipt.SHA256
	}
	observeMaps := func(label string) error {
		body, e := os.ReadFile("/proc/self/maps")
		if e != nil {
			return e
		}
		path := filepath.Join(r.Attempt, label+".maps")
		if e = os.WriteFile(path, body, 0644); e != nil {
			return e
		}
		r.Outputs[path] = digest(body)
		return nil
	}
	// Metadata and all prospective pins precede even the CPU loader.
	if err = phases.Checkpoint("pre-model"); err != nil {
		return r, err
	}
	if err = observeMaps("before-loader"); err != nil {
		return r, err
	}
	engine, err := tensor.Open(o.Backend)
	if err != nil {
		return r, err
	}
	defer func() { err = errors.Join(err, engine.Close()) }()
	if err = observeMaps("after-loader"); err != nil {
		return r, err
	}
	weights, err := model.LoadHACGPUWeights(ctx, engine, o.Model)
	if err != nil {
		return r, err
	}
	for _, p := range weights.Receipts() {
		path := filepath.Join(o.Model, p.Name)
		if _, err = readPin(Pin{Path: path, SHA256: p.SHA256}); err != nil {
			return r, err
		}
		r.Inputs[path] = p.SHA256
	}
	r.Inputs[filepath.Join(o.Model, "config.toml")] = model.HACConfigSHA256
	if err = observeMaps("after-weights"); err != nil {
		return r, err
	}
	device, err := engine.NewCUDADevice(o.Ordinal)
	if err != nil {
		return r, err
	}
	defer func() { err = errors.Join(err, device.Close()) }()
	stream, err := device.NewStream()
	if err != nil {
		return r, err
	}
	defer func() { err = errors.Join(err, stream.Close()) }()
	if err = observeMaps("before-cache-upload"); err != nil {
		return r, err
	}
	plan, err := model.PlanHACGPU(model.FixedHACGPUProfile())
	if err != nil {
		return r, err
	}
	m, err := model.NewHACGPU(device, stream, weights)
	if m != nil {
		defer func() { err = errors.Join(err, m.Close()) }()
	}
	if err != nil {
		return r, err
	}
	caches, err := m.CacheTensors(stream)
	if err != nil {
		return r, err
	}
	expected := map[string]CacheContract{}
	for _, c := range w.Caches {
		if _, ok := expected[c.Role]; ok {
			return r, fmt.Errorf("duplicate cache witness role")
		}
		expected[c.Role] = c
	}
	if len(caches) != 58 {
		return r, fmt.Errorf("missing uploaded cache owner")
	}
	actual := map[string]model.HACGPUCacheTensor{}
	for _, c := range caches {
		want, ok := expected[c.Role]
		if !ok {
			return r, fmt.Errorf("unexpected cache owner")
		}
		if _, ok = actual[c.Role]; ok {
			return r, fmt.Errorf("repeated production cache")
		}
		if err = verifyCache(c, want); err != nil {
			return r, err
		}
		actual[c.Role] = c
		path := filepath.Join(r.Attempt, "cache-"+c.Role+".bin")
		if err = os.WriteFile(path, c.Data, 0644); err != nil {
			return r, err
		}
		r.Outputs[path] = digest(c.Data)
		r.Caches = append(r.Caches, CacheComparison{c.Role, path, digest(c.Data), want.SHA256, len(c.Data), true})
	}
	// A same-shaped foreign-layer substitution must fail the exact production
	// cache comparator without changing any uploaded owner or launching a kernel.
	foreign := actual["flstm0.dn-ih-cache"]
	foreign.Data = bytes.Clone(actual["flstm1.dn-ih-cache"].Data)
	controlErr := verifyCache(foreign, expected[foreign.Role])
	if controlErr == nil {
		return r, fmt.Errorf("foreign layer control ineffective")
	}
	controlPath := filepath.Join(r.Attempt, "control-foreign-layer.bin")
	if err = os.WriteFile(controlPath, foreign.Data, 0644); err != nil {
		return r, err
	}
	r.Outputs[controlPath] = digest(foreign.Data)
	r.Controls = append(r.Controls, Control{"same-shaped-foreign-layer", "rejected", controlPath, digest(foreign.Data), controlErr.Error()})
	if err = observeMaps("before-forward"); err != nil {
		return r, err
	}
	input, err := w.Capture.RawPayload(o.Sources.Capture, "input")
	if err != nil {
		return r, err
	}
	active := 64
	r.WorkspacePattern = 0xa5
	if o.Phase == "padding" {
		active = 1
		input = bytes.Clone(input[:192*2])
		r.WorkspacePattern = 0
	}
	deviceInput, err := device.NewStorage(stream, tensor.DTypeF16, []int64{int64(active), 1, 192}, []tensor.KoiTag{tensor.TagN, tensor.TagC, tensor.TagT}, input)
	if err != nil {
		return r, err
	}
	defer func() { err = errors.Join(err, deviceInput.Close()) }()
	index := 0
	allocation := uint64(0)
	observe := func(event model.HACGPUObservation) error {
		if index >= len(w.Stages) {
			return fmt.Errorf("unexpected production observation")
		}
		contract := w.Stages[index]
		if event.Name != contract.Name || event.View.Offset != contract.Offset || event.View.Geometry.LiveBytes != contract.Bytes || event.View.Orientation != contract.Orientation {
			return fmt.Errorf("wrong production stage/order/view: %s", event.Name)
		}
		meta, e := event.Storage.Meta()
		if e != nil {
			return e
		}
		parent, e := event.Workspace.Meta()
		if e != nil {
			return e
		}
		if allocation == 0 {
			allocation = parent.Allocation
		}
		if parent.Allocation != allocation || meta.Allocation != allocation || meta.OffsetElements*uint64(meta.DType.Size()) != uint64(contract.Offset) || !slices.Equal(meta.Shape, contract.Shape) {
			return fmt.Errorf("stage escaped its owned invocation workspace")
		}
		body, e := event.Storage.Download(stream)
		if e != nil {
			return e
		}
		whole, e := event.Workspace.Download(stream)
		if e != nil {
			return e
		}
		if int64(len(whole)) != w.Capture.Reservation || !bytes.Equal(body, whole[contract.Offset:contract.Offset+contract.Bytes]) {
			return fmt.Errorf("stage storage/backing alias differs")
		}
		if active == 1 && event.Name == "input" && !bytes.Equal(body[192*2:], make([]byte, len(body)-192*2)) {
			return fmt.Errorf("inactive input slots were not zero padded")
		}
		native, e := w.Capture.RawPayload(o.Sources.Capture, contract.Name)
		if e != nil {
			return e
		}
		got, e := stageSelection(contract, body, active == 1)
		if e != nil {
			return e
		}
		want, e := stageSelection(contract, native, active == 1)
		if e != nil {
			return e
		}
		path := filepath.Join(r.Attempt, "stage-"+event.Name+".bin")
		parentPath := filepath.Join(r.Attempt, "workspace-"+event.Name+".bin")
		for p, b := range map[string][]byte{path: body, parentPath: whole} {
			if e = os.WriteFile(p, b, 0644); e != nil {
				return e
			}
			r.Outputs[p] = digest(b)
		}
		comparison := StageComparison{event.Name, path, digest(body), digest(got), digest(want), parentPath, digest(whole), len(got), bytes.Equal(got, want), contract}
		r.Stages = append(r.Stages, comparison)
		if !comparison.Equal {
			return fmt.Errorf("source-initialized stage bytes differ: %s", event.Name)
		}
		if active == 64 && (strings.HasPrefix(event.Name, "flstm") || event.Name == "linear1") {
			size := int64(524288)
			if strings.HasPrefix(event.Name, "flstm") {
				size, e = model.HACFusedScratchBytes(64, 32)
				if e != nil {
					return e
				}
			}
			stage, _ := w.Capture.Stage(event.Name)
			nativeParent, e := readPin(Pin{Path: filepath.Join(o.Sources.Capture, stage.BackingFile), SHA256: stage.BackingSHA})
			if e != nil {
				return e
			}
			if size > int64(len(whole)) || size > int64(len(nativeParent)) {
				return fmt.Errorf("declared initialized scratch escapes backing")
			}
			comparison := ScratchComparison{event.Name, digest(whole[:size]), digest(nativeParent[:size]), size, bytes.Equal(whole[:size], nativeParent[:size])}
			r.Scratch = append(r.Scratch, comparison)
			if !comparison.Equal {
				return fmt.Errorf("source-initialized scratch differs: %s", event.Name)
			}
		}
		index++
		return observeMaps("after-" + event.Name)
	}
	var result *model.HACGPUResult
	if err = phases.Checkpoint("pre-forward"); err != nil {
		return r, err
	}
	if active == 64 {
		r.ProductionForwardCalls++
		result, err = m.ForwardObserved(ctx, stream, deviceInput, plan, model.HACGPUBatch{ActiveSlots: 64}, r.WorkspacePattern, observe)
	} else {
		r.InactiveProbeCalls++
		result, err = m.ProbeInactiveSlots(ctx, stream, deviceInput, plan, r.WorkspacePattern, observe)
	}
	forwardErr := err
	if result != nil {
		defer func() { err = errors.Join(err, result.Close()) }()
	}
	if err = phases.Checkpoint("post-forward"); err != nil {
		return r, errors.Join(forwardErr, err)
	}
	if forwardErr != nil {
		return r, forwardErr
	}
	if result == nil || index != 13 || len(r.Stages) != 13 {
		return r, fmt.Errorf("incomplete graph result/stages/device bookends")
	}
	if active == 64 && len(r.Scratch) != 8 {
		return r, fmt.Errorf("missing required initialized scratch comparisons")
	}
	r.ActiveSlots, r.PhysicalSlots, r.Metadata = result.ActiveSlots, result.PhysicalSlots, result.Metadata
	reference, _ := model.Pinned(model.HAC)
	if r.ActiveSlots != active || r.PhysicalSlots != 64 || r.Metadata.DecoderClampValue != 0 || r.Metadata.ScoreMin != -5 || r.Metadata.ScoreMax != 5 {
		return r, fmt.Errorf("score metadata/accounting differs")
	}
	if r.Metadata.QScale != reference.QScale || r.Metadata.QBias != reference.QBias || r.Metadata.BlankScore != reference.BlankScore || r.Metadata.StateLen != reference.StateLen || r.Metadata.MeanQStart != reference.MeanQStart {
		return r, fmt.Errorf("pinned HAC Q/CRF metadata differs")
	}
	// The result remains usable after Forward closed its invocation workspace.
	retained, err := result.Scores.Download(stream)
	if err != nil {
		return r, err
	}
	if digest(retained) != r.Stages[12].SHA256 {
		return r, fmt.Errorf("returned scores changed after workspace release")
	}
	path := filepath.Join(r.Attempt, "retained-owned-scores.bin")
	if err = os.WriteFile(path, retained, 0644); err != nil {
		return r, err
	}
	r.Outputs[path] = digest(retained)
	changed := bytes.Clone(retained)
	changed[0] ^= 1
	contract := w.Stages[12]
	native, err := w.Capture.RawPayload(o.Sources.Capture, "scores")
	if err != nil {
		return r, err
	}
	a, err := stageSelection(contract, changed, active == 1)
	if err != nil {
		return r, err
	}
	b, err := stageSelection(contract, native, active == 1)
	if err != nil {
		return r, err
	}
	if bytes.Equal(a, b) {
		return r, fmt.Errorf("changed output control ineffective")
	}
	path = filepath.Join(r.Attempt, "control-changed-scores.bin")
	if err = os.WriteFile(path, changed, 0644); err != nil {
		return r, err
	}
	r.Outputs[path] = digest(changed)
	r.Controls = append(r.Controls, Control{"changed-defined-score", "rejected", path, digest(changed), "source-defined score comparator rejected rehashed changed output"})
	if err = observeMaps("after-owned-result"); err != nil {
		return r, err
	}
	if err = checkPins(r.Inputs); err != nil {
		return r, err
	}
	if err = checkPins(r.Outputs); err != nil {
		return r, err
	}
	if _, err = AuthenticateBackend(o.BackendArtifact, o.Backend); err != nil {
		return r, err
	}
	r.Status = "fixed_all_active_passed"
	if active == 1 {
		r.Status = "one_slot_independence_probe_passed_production_policy_held"
	}
	return r, nil
}
