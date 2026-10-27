package h06

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

const (
	Batch      = 64
	Frames     = 32
	Channels   = 1024
	InputBytes = (Frames + 3) * Batch * Channels * 2
	TempBytes  = Batch * (256 + 4096) * 2
	ArenaBytes = InputBytes + TempBytes
)

type Prepared struct {
	Schema, Transform, Capture, Model string
	Input, WeightList, Config         nativeproof.Pin
	Main, Helper                      nativeproof.Pin
	Weights                           []model.WeightReceipt
	Generated                         Generated
	Inputs                            []nativeproof.Pin
	Limits                            []string
}

func artifactPin(path string) (nativeproof.Pin, error) {
	path, err := filepath.Abs(path)
	if err != nil {
		return nativeproof.Pin{}, err
	}
	sha, err := runner.SHA(path)
	return nativeproof.Pin{Path: path, SHA256: sha}, err
}

// halfFixture is a deterministic, bounded standalone layer input. It is not
// claimed to be a half-convolution result or a supported full fallback graph.
func halfFixture(source []byte) ([]byte, error) {
	if len(source) != 64*192*2 {
		return nil, fmt.Errorf("fixed H01 normalized input extent required")
	}
	for i := 0; i < len(source); i += 2 {
		if binary.LittleEndian.Uint16(source[i:])&0x7c00 == 0x7c00 {
			return nil, fmt.Errorf("finite H01 input required")
		}
	}
	out := make([]byte, InputBytes)
	for i := 0; i < len(out); i += 2 {
		bits := binary.LittleEndian.Uint16(source[i%len(source):])
		if bits&0x7fff > 0x3c00 {
			bits = bits&0x8000 | 0x3c00
		}
		binary.LittleEndian.PutUint16(out[i:], bits)
	}
	for _, row := range []int{0, Frames + 2} {
		for j := 0; j < Batch*Channels; j++ {
			binary.LittleEndian.PutUint16(out[2*(row*Batch*Channels+j):], 0x3c00)
		}
	}
	return out, nil
}

func Prepare(capture, doradoSource, out string) (p Prepared, err error) {
	w, err := h07.ReadWitness(capture)
	if err != nil {
		return p, err
	}
	stage, ok := w.Capture.Stage("input")
	if !ok || stage.DType != "Half" || stage.Layout != "NTC" || !slices.Equal(stage.Shape, []int64{64, 192, 1}) || stage.Bytes != 64*192*2 || len(stage.UnwrittenT) != 0 {
		return p, fmt.Errorf("normalized H01 half input contract differs")
	}
	sourcePath := filepath.Join(capture, stage.DataFile)
	raw, err := os.ReadFile(sourcePath)
	if err != nil {
		return p, err
	}
	if digest(raw) != stage.DataSHA {
		return p, fmt.Errorf("normalized input digest differs")
	}
	input, err := halfFixture(raw)
	if err != nil {
		return p, err
	}
	spec, err := model.Inspect(w.Capture.ModelDir)
	if err != nil {
		return p, err
	}
	if spec.Architecture != model.HAC {
		return p, fmt.Errorf("pinned HAC weights required")
	}
	manifest, err := model.WeightManifest(model.HAC)
	if err != nil {
		return p, err
	}
	byName := map[string]model.WeightReceipt{}
	for _, v := range manifest {
		byName[v.Name] = v
	}
	weightText := "nanopore-hac-half-weights/v1\n"
	pins := []nativeproof.Pin{}
	suffix := []string{"dn_weight_ih", "dn_weight_hh", "up_weight_ih", "up_weight_hh", "up_bias_ih", "up_bias_hh"}
	shapes := [][]int64{{128, 1024}, {128, 1024}, {4096, 128}, {4096, 128}, {4096}, {4096}}
	for layer := 0; layer < 7; layer++ {
		for i, part := range suffix {
			name := fmt.Sprintf("%d.rnn.%s.tensor", 4+layer, part)
			pin, ok := byName[name]
			if !ok || !slices.Equal(pin.Shape, shapes[i]) {
				return p, fmt.Errorf("named half weight geometry differs: %s", name)
			}
			actual, e := artifactPin(filepath.Join(w.Capture.ModelDir, name))
			if e != nil {
				return p, e
			}
			if actual.SHA256 != pin.SHA256 {
				return p, fmt.Errorf("weight digest differs: %s", name)
			}
			if strings.ContainsAny(actual.Path, "\t\r\n") {
				return p, fmt.Errorf("weight path cannot be represented")
			}
			weightText += name + "\t" + actual.SHA256 + "\t" + actual.Path + "\n"
			pins = append(pins, actual)
			p.Weights = append(p.Weights, pin)
		}
	}
	for path, pin := range w.Pins {
		pins = append(pins, nativeproof.Pin{Path: path, SHA256: pin})
	}
	pins = append(pins, nativeproof.Pin{Path: sourcePath, SHA256: stage.DataSHA})
	for _, path := range []string{filepath.Join(w.Capture.ModelDir, "config.toml"), "verification/hacgpu/h06/native/oracle.cpp", "verification/nativeproof/sha256.inc"} {
		pin, e := artifactPin(path)
		if e != nil {
			return p, e
		}
		pins = append(pins, pin)
	}
	if out, err = runner.NewAttempt(out); err != nil {
		return p, err
	}
	p.Schema = "nanopore-hac-half-prepared/v2"
	p.Transform = "cycle-h01-half-clamp-unit-interval-and-poison-sentinels/v1"
	p.Capture = capture
	p.Model = w.Capture.ModelDir
	p.Inputs = pins
	p.Limits = []string{"Standalone forced CUBLAS half FLSTM only; transformed HG002 fixture is not a source half-convolution output.", "No source/model/GPU execution in preparation. Reference plus observed first-pass stack calls need separate admission."}
	if err = os.WriteFile(filepath.Join(out, "input.f16"), input, 0644); err != nil {
		return p, err
	}
	if p.Input, err = artifactPin(filepath.Join(out, "input.f16")); err != nil {
		return p, err
	}
	if err = os.WriteFile(filepath.Join(out, "weights.tsv"), []byte(weightText), 0644); err != nil {
		return p, err
	}
	if p.WeightList, err = artifactPin(filepath.Join(out, "weights.tsv")); err != nil {
		return p, err
	}
	if p.Config, err = artifactPin(filepath.Join(p.Model, "config.toml")); err != nil {
		return p, err
	}
	p.Generated, err = GenerateObserver(filepath.Join(doradoSource, "dorado/nn/FLSTMStack.cpp"), filepath.Join(out, "generated"))
	if err != nil {
		return p, err
	}
	for _, item := range []struct {
		path string
		dst  *nativeproof.Pin
	}{
		{"verification/hacgpu/h06/native/oracle.cpp", &p.Main},
		{"verification/nativeproof/sha256.inc", &p.Helper},
	} {
		data, e := os.ReadFile(item.path)
		if e != nil {
			return p, e
		}
		copyPath := filepath.Join(out, "sources", item.path)
		if e = os.MkdirAll(filepath.Dir(copyPath), 0755); e != nil {
			return p, e
		}
		if e = os.WriteFile(copyPath, data, 0444); e != nil {
			return p, e
		}
		if *item.dst, e = artifactPin(copyPath); e != nil {
			return p, e
		}
		p.Inputs = append(p.Inputs, *item.dst)
	}
	for _, path := range []string{p.Generated.Source, p.Generated.Observed, p.Generated.Header, p.Generated.HeaderSource, filepath.Join(out, "generated/generated.json")} {
		pin, e := artifactPin(path)
		if e != nil {
			return p, e
		}
		p.Inputs = append(p.Inputs, pin)
	}
	p.Inputs = append(p.Inputs, p.Input, p.WeightList, p.Config)
	canonical := map[string]string{}
	for _, pin := range p.Inputs {
		actual, e := filepath.EvalSymlinks(pin.Path)
		if e != nil {
			return p, e
		}
		actual, e = filepath.Abs(actual)
		if e != nil {
			return p, e
		}
		if old, ok := canonical[actual]; ok && old != pin.SHA256 {
			return p, fmt.Errorf("conflicting preparation pin: %s", actual)
		}
		canonical[actual] = pin.SHA256
	}
	p.Inputs = nil
	for path, sha := range canonical {
		p.Inputs = append(p.Inputs, nativeproof.Pin{Path: path, SHA256: sha})
	}
	slices.SortFunc(p.Inputs, func(a, b nativeproof.Pin) int { return strings.Compare(a.Path, b.Path) })
	for _, pin := range p.Inputs {
		sha, e := runner.SHA(pin.Path)
		if e != nil {
			return p, e
		}
		if sha != pin.SHA256 {
			return p, fmt.Errorf("preparation input changed: %s", pin.Path)
		}
	}
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return p, err
	}
	err = os.WriteFile(filepath.Join(out, "prepared.json"), append(data, '\n'), 0644)
	return p, err
}
