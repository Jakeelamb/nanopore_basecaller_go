package crforacle

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
)

const retainedBuildSHA = "a394f8cdbe256cefbe3e904808c6e5e46ebd52e0d9191961689994394f20f7b9"
const retainedRunSHA = "1975918d15e4b60e75371d5fee580c0e74acd991b2fb17e96d6cf1ac67dfb3dc"
const backendManifestSHA = "578be0cfb718a9c46b9c290299c5268c437217fd2150938fa6e77977f25c112c"
const retainedOriginal = "/home/jake/Projects/nanopore_basecaller_go/artifacts/crf/check-002/native-attempts/attempt-20261023T023747.033251747Z"

var retainedStages = [...]string{"scores", "forward", "backward", "posts"}

type RetainedFixture struct {
	Fixture
	Expected map[string][]float32
}

// Replay verifies the accepted receipts and reads their pinned bytes. It never
// builds or executes an oracle, including when a pin or dependency is missing.
func Replay(source, library, root string) ([]RetainedFixture, map[string]string, error) {
	pins := map[string]string{}
	read := func(path, want string) ([]byte, error) {
		data, err := readPinned(path, want)
		if err == nil {
			pins[path] = want
		}
		return data, err
	}
	buildData, err := read(filepath.Join(root, "oracle", "build.json"), retainedBuildSHA)
	if err != nil {
		return nil, nil, err
	}
	runData, err := read(filepath.Join(root, "native", "run.json"), retainedRunSHA)
	if err != nil {
		return nil, nil, err
	}
	var build, run Receipt
	if err := json.Unmarshal(buildData, &build); err != nil {
		return nil, nil, err
	}
	if err := json.Unmarshal(runData, &run); err != nil {
		return nil, nil, err
	}
	body, slices, err := Extract(source)
	if err != nil {
		return nil, nil, err
	}
	if build.Status != "built" || run.Status != "produced" || !reflect.DeepEqual(slices, build.Sources) {
		return nil, nil, fmt.Errorf("retained source or status mismatch")
	}
	for _, s := range slices {
		pins[filepath.Join(source, s.Path)] = s.SourceSHA256
	}
	local := func(path string) string {
		if strings.HasPrefix(path, retainedOriginal+string(filepath.Separator)) {
			return filepath.Join(root, strings.TrimPrefix(path, retainedOriginal+string(filepath.Separator)))
		}
		return path
	}
	cpp := retainedOriginal + "/oracle/reference.cpp"
	if build.Inputs[cpp] != digest([]byte(body)) || build.Inputs["adapter.cpp"] != digest([]byte(adapter)) {
		return nil, nil, fmt.Errorf("retained adapter mismatch")
	}
	for path, want := range build.Inputs {
		if path == "adapter.cpp" {
			continue
		}
		if err := verifyPin(local(path), want, pins); err != nil {
			return nil, nil, err
		}
	}
	for path, want := range build.Outputs {
		if err := verifyPin(local(path), want, pins); err != nil {
			return nil, nil, err
		}
	}
	for path, want := range run.Inputs {
		if err := verifyPin(local(path), want, pins); err != nil {
			return nil, nil, err
		}
	}
	for _, receipt := range []Receipt{build, run} {
		for _, c := range receipt.Commands {
			if c.Exit != 0 || c.Started == "" || c.Finished == "" {
				return nil, nil, fmt.Errorf("unsuccessful retained command")
			}
			if err := verifyPin(local(c.Log), c.LogSHA, pins); err != nil {
				return nil, nil, err
			}
		}
	}
	inputPath := retainedOriginal + "/native/inputs.bin"
	input, err := read(local(inputPath), run.Inputs[inputPath])
	if err != nil {
		return nil, nil, err
	}
	fixtures, err := parseRetainedInputs(input)
	if err != nil {
		return nil, nil, err
	}
	out := make([]RetainedFixture, len(fixtures))
	for i, f := range fixtures {
		out[i] = RetainedFixture{Fixture: f, Expected: map[string][]float32{}}
		for _, stage := range retainedStages {
			path := fmt.Sprintf("%s/native/%d/%s.bin", retainedOriginal, i, stage)
			data, err := read(local(path), run.Outputs[path])
			if err != nil {
				return nil, nil, err
			}
			n := (f.Blocks + 1) * f.States
			if stage == "scores" {
				n = len(f.Scores)
			}
			if len(data) != 4*n {
				return nil, nil, fmt.Errorf("retained stage length mismatch: %s", path)
			}
			values := make([]float32, n)
			for j := range values {
				values[j] = math.Float32frombits(binary.LittleEndian.Uint32(data[4*j:]))
			}
			out[i].Expected[stage] = values
		}
	}
	backendPins, err := VerifyBackend(library)
	if err != nil {
		return nil, nil, err
	}
	for path, want := range backendPins {
		pins[path] = want
	}
	return out, pins, nil
}

func readPinned(path, want string) ([]byte, error) {
	if len(want) != 64 {
		return nil, fmt.Errorf("missing SHA256 pin: %s", path)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if digest(data) != want {
		return nil, fmt.Errorf("hash pin mismatch: %s", path)
	}
	return data, nil
}
func verifyPin(path, want string, pins map[string]string) error {
	if len(want) != 64 {
		return fmt.Errorf("missing SHA256 pin: %s", path)
	}
	got, err := Hash(path)
	if err != nil {
		return err
	}
	if got != want {
		return fmt.Errorf("hash pin mismatch: %s", path)
	}
	pins[path] = got
	return nil
}

// VerifyBackend checks the accepted ABI2 binary and every declared runtime
// dependency before loading. It does not compile or execute native software.
func VerifyBackend(library string) (map[string]string, error) {
	path := filepath.Join(filepath.Dir(library), "manifest.json")
	data, err := readPinned(path, backendManifestSHA)
	if err != nil {
		return nil, err
	}
	var backend struct {
		OutputSHA string            `json:"output_sha256"`
		Runtime   map[string]string `json:"runtime_sha256"`
	}
	if err := json.Unmarshal(data, &backend); err != nil {
		return nil, err
	}
	pins := map[string]string{path: backendManifestSHA}
	if err := verifyPin(library, backend.OutputSHA, pins); err != nil {
		return nil, err
	}
	for path, want := range backend.Runtime {
		if err := verifyPin(path, want, pins); err != nil {
			return nil, err
		}
	}
	return pins, nil
}

func parseRetainedInputs(data []byte) ([]Fixture, error) {
	r := bytes.NewReader(data)
	var header [2]uint32
	if err := binary.Read(r, binary.LittleEndian, &header); err != nil {
		return nil, err
	}
	fixtures := Synthetic()
	var captured struct {
		Name           string
		Blocks, States int
		Blank          float32
		Scores         string
		Pins           map[string]string
	}
	if err := json.Unmarshal(capturedManifest, &captured); err != nil {
		return nil, err
	}
	fixtures = append(fixtures, Fixture{Name: captured.Name, Blocks: captured.Blocks, States: captured.States, Blank: captured.Blank})
	if header != [2]uint32{0x43524631, uint32(len(fixtures))} {
		return nil, fmt.Errorf("retained input header mismatch")
	}
	for i := range fixtures {
		f := &fixtures[i]
		var shape [3]uint32
		if err := binary.Read(r, binary.LittleEndian, &shape); err != nil {
			return nil, err
		}
		if shape != [3]uint32{uint32(f.Blocks), uint32(f.States), math.Float32bits(f.Blank)} {
			return nil, fmt.Errorf("retained fixture %d metadata mismatch", i)
		}
		raw := make([]byte, f.Blocks*4*f.States*4)
		if _, err := io.ReadFull(r, raw); err != nil {
			return nil, err
		}
		if i == len(fixtures)-1 {
			if digest(raw) != captured.Pins[captured.Scores] {
				return nil, fmt.Errorf("captured HAC score pin mismatch")
			}
			f.Scores = make([]float32, len(raw)/4)
		}
		for j := range f.Scores {
			v := binary.LittleEndian.Uint32(raw[4*j:])
			if i != len(fixtures)-1 && v != math.Float32bits(f.Scores[j]) {
				return nil, fmt.Errorf("retained fixture %d score mismatch", i)
			}
			f.Scores[j] = math.Float32frombits(v)
		}
	}
	if r.Len() != 0 {
		return nil, fmt.Errorf("trailing retained input bytes")
	}
	return fixtures, nil
}
