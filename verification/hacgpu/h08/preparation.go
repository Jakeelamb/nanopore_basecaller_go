package h08

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"syscall"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08/cgobuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
)

// ReadCPU13Admission authenticates metadata only, using the runtime reader.
func ReadCPU13Admission(pin Pin) (RuntimeAdmission, CPU13Request, error) {
	return readCPU13Admission(pin)
}

// ReadFakeLoaderProof authenticates an existing proof against independent pins.
func ReadFakeLoaderProof(pin, target, build, selection Pin) (FakeProof, error) {
	return readFakeProof(pin, target, build, selection)
}

// CPU13PreparationOptions selects existing artifacts, never future build pins.
// All outputs are fresh strict descendants of Directory/artifacts. Runtime
// outputs remain absent for the guarded runtime to create.
type CPU13PreparationOptions struct {
	Schema                                                   string
	ParentBuild, ParentBuildSelection                        Pin
	ControllerBuild, ControllerBuildSelection                Pin
	Fixtures, FakeProof, Policy                              Pin
	Backend                                                  closedbuild.ArtifactOptions
	Directory, MetadataDirectory, GuardOutput, RequestOutput string
}

type CPU13Preparation struct {
	Admission, Request, Plan, Contract, Policy Pin
	RuntimeHeld                                bool
}

func preparationPin(p Pin) error {
	b, err := hex.DecodeString(p.SHA256)
	if !filepath.IsAbs(p.Path) || filepath.Clean(p.Path) != p.Path || err != nil || len(b) != 32 {
		return fmt.Errorf("absolute clean path and full independent SHA256 required: %s", p.Path)
	}
	return nil
}

// ReadPreparationJSON authenticates a regular metadata file of at most 64 MiB
// and decodes one typed JSON object. The checked descriptor supplies the exact
// bytes hashed and decoded; a FIFO without a writer rejects without blocking.
func ReadPreparationJSON(p Pin, value any) error {
	if err := preparationPin(p); err != nil {
		return err
	}
	f, err := os.OpenFile(p.Path, os.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("regular preparation input required: %s", p.Path)
	}
	const maxBytes = 64 << 20
	if info.Size() > maxBytes {
		return fmt.Errorf("preparation JSON exceeds 64 MiB: %s", p.Path)
	}
	b, err := io.ReadAll(io.LimitReader(f, maxBytes+1))
	if err != nil {
		return err
	}
	if len(b) > maxBytes {
		return fmt.Errorf("preparation JSON exceeds 64 MiB: %s", p.Path)
	}
	if int64(len(b)) != info.Size() {
		return fmt.Errorf("preparation JSON size changed: %s", p.Path)
	}
	if digest(b) != p.SHA256 {
		return fmt.Errorf("preparation request SHA256 differs: %s", p.Path)
	}
	trimmed := bytes.TrimSpace(b)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return fmt.Errorf("typed JSON object required")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err = d.Decode(value); err != nil {
		return err
	}
	var extra any
	if err = d.Decode(&extra); !errors.Is(err, io.EOF) {
		return fmt.Errorf("trailing preparation JSON: %v", err)
	}
	return nil
}

func preparationInputs(o CPU13PreparationOptions) []Pin {
	return []Pin{o.ParentBuild, o.ParentBuildSelection, o.ControllerBuild, o.ControllerBuildSelection, o.Fixtures, o.FakeProof, o.Policy, o.Backend.Build, o.Backend.Plan, o.Backend.Producer, o.Backend.Output}
}

func pathWithin(parent, child string) bool {
	rel, err := filepath.Rel(parent, child)
	return err == nil && rel != ".." && !filepath.IsAbs(rel) && (rel == "." || len(rel) < 3 || rel[:3] != "../")
}

func validatePreparation(o CPU13PreparationOptions, selected Pin) error {
	if o.Schema != "nanopore-h08-cpu13-preparation/v1" {
		return fmt.Errorf("unsupported CPU13 preparation schema")
	}
	seen := map[string]bool{}
	for _, p := range append(preparationInputs(o), selected) {
		if err := preparationPin(p); err != nil {
			return err
		}
		if seen[p.Path] {
			return fmt.Errorf("duplicate selected preparation path: %s", p.Path)
		}
		seen[p.Path] = true
	}
	if !filepath.IsAbs(o.Directory) || filepath.Clean(o.Directory) != o.Directory {
		return fmt.Errorf("absolute clean command directory required")
	}
	outputs := []string{o.MetadataDirectory, o.GuardOutput, o.RequestOutput}
	artifacts := filepath.Join(o.Directory, "artifacts")
	for i, path := range outputs {
		if !filepath.IsAbs(path) || filepath.Clean(path) != path || pathWithin(path, o.Directory) {
			return fmt.Errorf("invalid preparation output: %s", path)
		}
		if path == artifacts || !pathWithin(artifacts, path) {
			return fmt.Errorf("preparation output must be a strict descendant of command directory artifacts: %s", path)
		}
		parent := filepath.Dir(path)
		canonical, err := filepath.EvalSymlinks(parent)
		if err != nil || canonical != parent {
			return fmt.Errorf("output parent must exist without symlink aliases: %s: %v", parent, err)
		}
		if _, err := os.Lstat(path); !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("preparation output must be absent: %s: %v", path, err)
		}
		for _, other := range outputs[:i] {
			if pathWithin(path, other) || pathWithin(other, path) {
				return fmt.Errorf("preparation outputs collide")
			}
		}
		for input := range seen {
			if pathWithin(path, input) || pathWithin(input, path) {
				return fmt.Errorf("preparation output collides with selected input")
			}
		}
	}
	info, err := os.Stat(o.Directory)
	if err != nil {
		return err
	}
	if !info.IsDir() {
		return fmt.Errorf("command directory must exist")
	}
	canonical, err := filepath.EvalSymlinks(o.Directory)
	if err != nil || canonical != o.Directory {
		return fmt.Errorf("command directory must have no symlink aliases: %v", err)
	}
	return nil
}

func writePreparationJSON(path string, value any) (Pin, error) {
	b, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return Pin{}, err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return Pin{}, err
	}
	_, err = f.Write(append(b, '\n'))
	err = errors.Join(err, f.Close())
	if err != nil {
		return Pin{}, err
	}
	return closedbuild.PinFile(path)
}

func preparationPolicyTarget(policy supgpu.RuntimePolicy, target Pin) error {
	p, err := guardPin(target)
	if err != nil {
		return err
	}
	if policy.Reference != p || policy.Observer != p {
		return fmt.Errorf("selected policy target differs; policy cannot be retargeted")
	}
	return nil
}

// PrepareCPU13Admission reads the independently pinned typed options and writes
// fresh metadata. Partial outputs are retained on any failure. No target runs.
func PrepareCPU13Admission(selected Pin) (result CPU13Preparation, err error) {
	var o CPU13PreparationOptions
	if err = ReadPreparationJSON(selected, &o); err != nil {
		return result, err
	}
	if err = validatePreparation(o, selected); err != nil {
		return result, err
	}
	policyPin, err := guardPin(o.Policy)
	if err != nil {
		return result, err
	}
	policy, err := supgpu.ReadRuntimePolicy(policyPin)
	if err != nil {
		return result, err
	}
	if _, err = cgobuild.ReadH08Artifact(o.ParentBuild, o.ParentBuildSelection); err != nil {
		return result, err
	}
	target, err := cgobuild.ReadCPU13Artifact(o.ControllerBuild, o.ControllerBuildSelection)
	if err != nil {
		return result, err
	}
	view, err := ReadFixtureView(o.Fixtures)
	if err != nil {
		return result, err
	}
	if view.Backend != o.Backend.Output {
		return result, fmt.Errorf("fixture backend differs from selected backend")
	}
	if _, err = AuthenticateBackend(o.Backend, view.Backend.Path); err != nil {
		return result, err
	}
	if _, err = readFakeProof(o.FakeProof, target.Output(), o.ControllerBuild, o.ControllerBuildSelection); err != nil {
		return result, err
	}
	if err = preparationPolicyTarget(policy, target.Output()); err != nil {
		return result, err
	}
	if err = os.Mkdir(o.MetadataDirectory, 0700); err != nil {
		return result, err
	}
	request := CPU13Request{Schema: "nanopore-h08-cpu13-request/v1", Fixtures: o.Fixtures, ControllerBuild: o.ControllerBuild, ControllerBuildSelection: o.ControllerBuildSelection, FakeProof: o.FakeProof, Backend: o.Backend, Out: o.RequestOutput}
	if result.Request, err = writePreparationJSON(filepath.Join(o.MetadataDirectory, "request.json"), request); err != nil {
		return result, err
	}
	contract := supgpu.PhaseContract{Schema: "nanopore-audited-phase-contract/v1", Role: "observer", Action: "hac-cpu13", Names: []string{"pre-model", "pre-forward", "post-forward"}}
	if result.Contract, err = writePreparationJSON(filepath.Join(o.MetadataDirectory, "contract.json"), contract); err != nil {
		return result, err
	}
	cp, err := guardPin(result.Contract)
	if err != nil {
		return result, err
	}
	plan := supgpu.GuardPlan{Schema: "nanopore-audited-guard-plan/v1", Policy: policy, Contract: cp, Inputs: []supgpu.Pin{}}
	for _, p := range append(preparationInputs(o), selected, result.Request) {
		v, e := guardPin(p)
		if e != nil {
			return result, e
		}
		plan.Inputs = append(plan.Inputs, v)
	}
	if result.Plan, err = writePreparationJSON(filepath.Join(o.MetadataDirectory, "plan.json"), plan); err != nil {
		return result, err
	}
	a := RuntimeAdmission{Schema: "nanopore-h08-runtime-admission/v2", Request: result.Request, Plan: result.Plan, Contract: result.Contract, ParentBuild: o.ParentBuild, ParentBuildSelection: o.ParentBuildSelection, Command: supgpu.GuardCommand{Directory: o.Directory, Output: o.GuardOutput, Args: cpu13Command(target.Output(), result.Request)}}
	if result.Admission, err = writePreparationJSON(filepath.Join(o.MetadataDirectory, "admission.json"), a); err != nil {
		return result, err
	}
	if _, _, err = readCPU13Admission(result.Admission); err != nil {
		return result, err
	}
	result.Policy = o.Policy
	result.RuntimeHeld = true
	return result, nil
}
