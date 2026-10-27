package decodergpu

import (
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type LineageRecipe struct {
	Compiler, Driver, CUPTI      File
	Flags, Libraries, SearchDirs []string
	Resources                    []ResourceQuery
	Inputs                       map[string]string
}
type LineagePlan struct {
	Schema, Directory, Source, WorkingDirectory string
	Producer, ProducerRuntime, Capture          File
	Go                                          GoClosure
	Cases                                       []LineageContract
	Recipe                                      LineageRecipe
	Compiler                                    CompilerClosure
	SourceBindings, SnapshotPins, Inputs        map[string]string
}
type LineagePrepareOptions struct {
	Producer, Runtime, Capture File
	Out                        string
}

var lineageSourcePaths = []string{"verification/decodergpu/native/lineage_probe.cpp", "verification/decodergpu/native/lineage_observer.inc", "verification/decodergpu/native/sha256.inc", "verification/decodergpu/native/runtime_gate.inc"}

func lineageInputs(p LineagePlan, old ClosedBuildPlan) (map[string]string, error) {
	inputs := map[string]string{}
	for _, set := range []map[string]string{p.Recipe.Inputs, p.Compiler.Inputs, p.SnapshotPins, p.Go.SnapshotPins} {
		maps.Copy(inputs, set)
	}
	for original, copy := range p.SourceBindings {
		want, ok := p.SnapshotPins[copy]
		if !ok {
			return nil, fmt.Errorf("lineage source binding has no immutable payload")
		}
		inputs[original] = want
	}
	for _, file := range []File{p.Producer, p.ProducerRuntime, p.Capture, old.Preparation, {p.Go.Controller, p.Go.ControllerSHA}} {
		if err := addPin(inputs, file); err != nil {
			return nil, err
		}
	}
	for _, contract := range p.Cases {
		for _, file := range []File{contract.Archive, contract.Member, contract.ExpectedOutput, contract.Input.Image} {
			if err := addPin(inputs, file); err != nil {
				return nil, err
			}
		}
	}
	for _, command := range p.Go.Commands {
		if err := commandEvidence(command); err != nil {
			return nil, err
		}
		inputs[command.Stdout] = command.StdoutSHA
		inputs[command.Stderr] = command.StderrSHA
	}
	return inputs, nil
}

func lineageEnvironment(c CaptureReceipt) []string {
	env := []string{}
	for _, entry := range c.Guarded.Command.Environment {
		if !strings.HasPrefix(entry, "SUPGPU_TARGET_PID=") {
			env = append(env, entry)
		}
	}
	return env
}

func lineageRecipe(old ClosedBuildPlan, r RuntimePlan) (recipe LineageRecipe, err error) {
	recipe = LineageRecipe{Compiler: old.Recipe.Compiler, CUPTI: old.Recipe.CUPTI, Flags: []string{"-O2", "-DNDEBUG", "-std=c++23", "-fPIC", "-fno-lto", "-isystem", "/opt/cuda/targets/x86_64-linux/include"}, Resources: slices.Clone(old.Recipe.Resources), SearchDirs: slices.Clone(old.Recipe.SearchDirs), Inputs: map[string]string{}}
	for _, code := range r.Policy.Allowed {
		if code.SONAME == "libcuda.so.1" {
			if recipe.Driver.Path != "" {
				return recipe, fmt.Errorf("lineage selected driver ambiguous")
			}
			recipe.Driver = File{code.Object.Path, code.Object.SHA256}
		}
	}
	if recipe.Driver.Path == "" || recipe.CUPTI.Path == "" {
		return recipe, fmt.Errorf("lineage pinned driver/CUPTI absent")
	}
	recipe.Libraries = []string{"-pie", "-fno-lto", recipe.Driver.Path, recipe.CUPTI.Path, "-Wl,-rpath," + filepath.Dir(recipe.Driver.Path), "-Wl,-rpath," + filepath.Dir(recipe.CUPTI.Path)}
	for _, file := range []File{recipe.Compiler, recipe.Driver, recipe.CUPTI} {
		if err = addPin(recipe.Inputs, file); err != nil {
			return recipe, err
		}
	}
	for _, resource := range recipe.Resources {
		if err = addPin(recipe.Inputs, resource.Expected); err != nil {
			return recipe, err
		}
	}
	for _, name := range []string{"cuda.h", "cupti.h", "cupti_callbacks.h", "cupti_driver_cbid.h", "generated_cuda_meta.h"} {
		if err = filePin(recipe.Inputs, filepath.Join("/opt/cuda/targets/x86_64-linux/include", name)); err != nil {
			return recipe, err
		}
	}
	resolve := func(name, from string) (string, error) {
		if filepath.IsAbs(name) {
			return name, nil
		}
		if !strings.HasPrefix(name, "-l") {
			p := filepath.Join(filepath.Dir(from), name)
			if info, e := os.Stat(p); e == nil && info.Mode().IsRegular() {
				return p, nil
			}
		}
		p, e := resolveLibrary(name, recipe.SearchDirs)
		return p.Path, e
	}
	if err = ExpandLinkerScripts(recipe.Inputs, resolve); err != nil {
		return recipe, err
	}
	for _, root := range []File{recipe.Driver, recipe.CUPTI} {
		closure, e := nativeproof.CloseELF([]nativeproof.Pin{{Path: root.Path, SHA256: root.SHA256}}, []string{filepath.Dir(recipe.Driver.Path), filepath.Dir(recipe.CUPTI.Path), "/usr/lib"})
		if e != nil {
			return recipe, e
		}
		if e = checkHostELF(closure); e != nil {
			return recipe, e
		}
		for _, pin := range closure.Inputs {
			if err = addPin(recipe.Inputs, File{pin.Path, pin.SHA256}); err != nil {
				return recipe, err
			}
		}
	}
	return recipe, nil
}

func lineageAnchors(producer, runtime, capture File) (CompilerPhase, ClosedBuildPlan, RuntimePlan, CaptureReceipt, error) {
	var c CaptureReceipt
	b, p, err := readBuiltPhaseMode(producer, false)
	if err != nil {
		return b, p, RuntimePlan{}, c, err
	}
	var r RuntimePlan
	if err = readJSON(runtime, &r); err != nil {
		return b, p, r, c, err
	}
	if r.Build != producer {
		return b, p, r, c, fmt.Errorf("lineage selected runtime producer differs")
	}
	if err = verifyGuardRuntime(r, b, p, false); err != nil {
		return b, p, r, c, err
	}
	if capture.SHA256 != lineageCaptureSHA {
		return b, p, r, c, fmt.Errorf("lineage requires unchanged failed capture002")
	}
	if err = readJSON(capture, &c); err != nil {
		return b, p, r, c, err
	}
	if c.Status != "failed" || c.Error != "observed module differs from original selected member images" || c.Runtime != runtime || c.Build != producer || c.Guarded.Command.Exit != 0 || len(c.Images.Modules) != 1 || c.Images.Modules[0].Payload.SHA256 != lineageOutputSHA {
		return b, p, r, c, fmt.Errorf("lineage historical failure/output binding differs")
	}
	if err = c.Images.Modules[0].Payload.verify(); err != nil {
		return b, p, r, c, err
	}
	return b, p, r, c, nil
}

func PrepareLineage(o LineagePrepareOptions) (p LineagePlan, err error) {
	for _, anchor := range []*File{&o.Producer, &o.Runtime, &o.Capture} {
		*anchor, err = normalizeAnchor(*anchor)
		if err != nil {
			return p, err
		}
	}
	_, old, r, c, err := lineageAnchors(o.Producer, o.Runtime, o.Capture)
	if err != nil {
		return p, err
	}
	if err = requireNativeLimits(); err != nil {
		return p, err
	}
	p = LineagePlan{Schema: "nanopore-decoder-load-lineage-plan/v1", Producer: o.Producer, ProducerRuntime: o.Runtime, Capture: o.Capture, Inputs: map[string]string{}}
	p.WorkingDirectory, err = os.Getwd()
	if err != nil {
		return p, err
	}
	p.WorkingDirectory, err = filepath.EvalSymlinks(p.WorkingDirectory)
	if err != nil {
		return p, err
	}
	p.Directory, err = runner.NewAttempt(o.Out)
	if err != nil {
		return p, err
	}
	p.Go, err = freezeLineageGo(p.Directory)
	if err != nil {
		return p, err
	}
	p.Compiler = old.Compiler
	p.Recipe, err = lineageRecipe(old, r)
	if err != nil {
		return p, err
	}
	prepared, err := ReadPreparation(old.Preparation, false)
	if err != nil {
		return p, err
	}
	for _, image := range prepared.Images.Images {
		contract := LineageContract{Schema: "nanopore-decoder-load-lineage/v1", Architecture: image.Architecture, Archive: old.Recipe.Koi, Member: prepared.Images.Member, Capture: o.Capture, ExpectedOutput: c.Images.Modules[0].Payload, Policy: prepared.Images, Input: image, Lookup: LineageFourLookups, Environment: lineageEnvironment(c)}
		if err = contract.Validate(); err != nil {
			return p, err
		}
		p.Cases = append(p.Cases, contract)
	}
	native := map[string]string{}
	for _, source := range lineageSourcePaths {
		if err = filePin(native, source); err != nil {
			return p, err
		}
	}
	p.SourceBindings, p.SnapshotPins, err = snapshotPins(native, filepath.Join(p.Directory, "native-inputs"))
	if err != nil {
		return p, err
	}
	for original, copy := range p.SourceBindings {
		body, e := os.ReadFile(copy)
		if e != nil {
			return p, e
		}
		alias := filepath.Join(p.Directory, "native-inputs", filepath.Base(original))
		if e = os.WriteFile(alias, body, 0400); e != nil {
			return p, e
		}
		if e = filePin(p.SnapshotPins, alias); e != nil {
			return p, e
		}
		if filepath.Base(original) == "lineage_probe.cpp" {
			p.Source = alias
		}
	}
	p.Inputs, err = lineageInputs(p, old)
	if err != nil {
		return p, err
	}
	if err = checkPins(p.Inputs); err != nil {
		return p, err
	}
	err = writeJSON(filepath.Join(p.Directory, "plan.json"), p)
	return p, err
}

func verifyLineagePlan(p LineagePlan, current, executing bool) error {
	if p.Schema != "nanopore-decoder-load-lineage-plan/v1" || p.Source != filepath.Join(p.Directory, "native-inputs/lineage_probe.cpp") || len(p.Cases) != 2 || len(p.SourceBindings) != len(lineageSourcePaths) {
		return fmt.Errorf("finite lineage plan/source/case shape differs")
	}
	_, old, r, c, err := lineageAnchors(p.Producer, p.ProducerRuntime, p.Capture)
	if err != nil {
		return err
	}
	recipe, err := lineageRecipe(old, r)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(recipe, p.Recipe) || !reflect.DeepEqual(old.Compiler, p.Compiler) {
		return fmt.Errorf("lineage selected compiler/driver-only recipe differs")
	}
	if err = checkLineageGo(p.Go, current, executing); err != nil {
		return err
	}
	prepared, err := ReadPreparation(old.Preparation, false)
	if err != nil {
		return err
	}
	for i, image := range prepared.Images.Images {
		want := LineageContract{Schema: "nanopore-decoder-load-lineage/v1", Architecture: image.Architecture, Archive: old.Recipe.Koi, Member: prepared.Images.Member, Capture: p.Capture, ExpectedOutput: c.Images.Modules[0].Payload, Policy: prepared.Images, Input: image, Lookup: LineageFourLookups, Environment: lineageEnvironment(c)}
		if !reflect.DeepEqual(p.Cases[i], want) {
			return fmt.Errorf("lineage prospective case changed")
		}
	}
	if err = verifyLineageSourceSnapshots(p, current); err != nil {
		return err
	}

	expected, err := lineageInputs(p, old)
	if err != nil {
		return err
	}
	if !maps.Equal(expected, p.Inputs) {
		return fmt.Errorf("lineage complete before-query input membership differs")
	}
	return checkProducerInputs(p.Inputs, ClosedBuildPlan{Directory: p.Directory, SourceBindings: p.SourceBindings, SnapshotPins: p.SnapshotPins}, current)
}

func verifyLineageSourceSnapshots(p LineagePlan, current bool) error {
	root := ""
	if !current {
		var err error
		root, err = historicalProducerRoot(p.Go)
		if err != nil {
			return err
		}
	}
	for _, source := range lineageSourcePaths {
		original := filepath.Join(root, source)
		if current {
			var err error
			original, err = canonical(source)
			if err != nil {
				return err
			}
		}
		copy, ok := p.SourceBindings[original]
		if !ok || p.SnapshotPins[copy] != p.Inputs[original] || p.SnapshotPins[filepath.Join(p.Directory, "native-inputs", filepath.Base(original))] != p.Inputs[original] {
			return fmt.Errorf("lineage native source snapshot membership differs")
		}
	}
	return nil
}
