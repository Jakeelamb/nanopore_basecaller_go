package decodergpu

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type PrepareOptions struct {
	Capture, Model, BuildDir, Dorado, Static, SM120a, Out string
	NumericLineage                                        File
}
type Preparation struct {
	Schema, Status, Started, Finished, Error string
	Contract                                 Contract
	Images                                   ImagePolicy
	NumericLineage                           File `json:",omitempty"`
	ScoreStage                               any
	Inputs                                   map[string]string
	Build                                    BuildPlan
	Go                                       GoClosure
	Bootstrap, BootstrapSHA                  string
	NativeHeld                               []string
}

func Prepare(o PrepareOptions) (r Preparation, err error) {
	r = Preparation{Schema: "nanopore-decoder-fixed-hac-preparation/v2", Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Inputs: map[string]string{}, Contract: FixedContract(), NativeHeld: []string{"all new C++ compiler/tool queries/preprocess/compile/link", "native ELF library load and device/GPU execution", "runtime-selected cubin/PTX qualification", "production decoder port and numerical parity"}}
	var out string
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
		}
		if out != "" {
			b, e := json.MarshalIndent(r, "", "  ")
			if e == nil {
				e = os.WriteFile(filepath.Join(out, "preparation.json"), append(b, '\n'), 0600)
			}
			if err == nil {
				err = e
			}
		}
	}()
	if err = r.Contract.Validate(); err != nil {
		return r, err
	}
	if o.SM120a == "" {
		o.SM120a = "artifacts/decoder-static-20261025-sm120a-001/decoder_lib.sm_120a.cubin"
	}
	r.Images, err = PlanDeviceImages(o.Static, o.SM120a)
	if err != nil {
		return r, err
	}
	for _, image := range r.Images.Images {
		r.Inputs[image.Image.Path] = image.Image.SHA256
	}
	w, err := ReadWitness(o.Capture, o.Model)
	if err != nil {
		return r, err
	}
	r.ScoreStage = w.ScoreStage
	for p, h := range w.Pins {
		r.Inputs[p] = h
	}
	sourcePins := map[string]string{
		filepath.Join(o.Dorado, "dorado/basecall/decode/CUDADecoder.cpp"): "e913013043ad0e2d352551b21adc74a7dcca7417e4b89a791978ce36a985e297",
		filepath.Join(o.Static, "decoder_lib.cu.o"):                       "930b35afa9e352b412b58e07a43d693630280ed9f422dc324106b2b8e700758f",
		filepath.Join(o.Static, "decoder_lib.sm_120.ptx"):                 "7e46f59645025bca2cf7a519c9f0334b5435942ca9006d0dec5faa6d3e63b01a",
		filepath.Join(o.Static, "decoder_lib.sm_120.cubin"):               "292621a0efb7da3a9345797b469d73fb04ee04a88ad1aa0e52c94944214da938",
	}
	for path, want := range sourcePins {
		if err = filePin(r.Inputs, path); err != nil {
			return r, err
		}
		actual, e := canonical(path)
		if e != nil {
			return r, e
		}
		if r.Inputs[actual] != want {
			return r, fmt.Errorf("pinned source/static evidence changed: %s", path)
		}
	}
	for _, name := range []string{"dorado/basecall/include/basecall/DecodedChunk.h", "dorado/basecall/decode/Decoder.cpp", "dorado/basecall/CudaCaller.cpp", "dorado/config/BasecallModelConfig.cpp"} {
		if err = filePin(r.Inputs, filepath.Join(o.Dorado, name)); err != nil {
			return r, err
		}
	}
	out, err = runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	for name, b := range map[string][]byte{"scores-original.f16": w.Scores, "params.bin": r.Contract.Parameters(), "chunks.bin": r.Contract.DescriptorBytes()} {
		path := filepath.Join(out, name)
		if err = os.WriteFile(path, b, 0600); err != nil {
			return r, err
		}
		if err = filePin(r.Inputs, path); err != nil {
			return r, err
		}
	}
	if err = ValidateWire(r.Contract.Parameters(), r.Contract.DescriptorBytes()); err != nil {
		return r, err
	}
	if o.NumericLineage.Path != "" {
		proof, e := readNumericLineage(o.NumericLineage)
		if e != nil {
			return r, e
		}
		r.Schema = "nanopore-decoder-fixed-hac-preparation/v3"
		r.NumericLineage = o.NumericLineage
		for path, sha := range proof.Inputs {
			r.Inputs[path] = sha
		}
	}
	r.Build, err = PlanBuild(o.BuildDir, filepath.Join(out, "held-build"))
	if err != nil {
		return r, err
	}
	for path, pin := range r.Build.Inputs {
		r.Inputs[path] = pin
	}
	if r.Schema == "nanopore-decoder-fixed-hac-preparation/v3" {
		sources, e := nativePreparationSources("nanopore-decoder-closed-build/v3")
		if e != nil {
			return r, e
		}
		for path, sha := range sources {
			r.Inputs[path] = sha
		}
	}
	// The four linked host symbols must come from the originally selected archive.
	archiveFound := false
	for path, pin := range r.Build.Inputs {
		if filepath.Base(path) == "libkoi.a" {
			if pin != "17cf3083c5db538f1e275b0a41ac19560b5986c143634e8a25540b9735f5805f" {
				return r, fmt.Errorf("selected Koi archive changed")
			}
			archiveFound = true
		}
	}
	if !archiveFound {
		return r, fmt.Errorf("prospective link does not include selected Koi archive")
	}
	r.Bootstrap, err = os.Executable()
	if err != nil {
		return r, err
	}
	b, err := os.ReadFile(r.Bootstrap)
	if err != nil {
		return r, err
	}
	r.Bootstrap = filepath.Join(out, "preparation-bootstrap")
	r.BootstrapSHA = digest(b)
	if err = os.WriteFile(r.Bootstrap, b, 0700); err != nil {
		return r, err
	}
	dir, err := os.Getwd()
	if err != nil {
		return r, err
	}
	r.Go, err = FreezeGoController(dir, out)
	if err != nil {
		return r, err
	}
	if err = checkPins(r.Inputs); err != nil {
		return r, err
	}
	if err = checkPins(r.Go.Pins); err != nil {
		return r, err
	}
	r.Status = "host_prepared_native_held"
	return r, nil
}
