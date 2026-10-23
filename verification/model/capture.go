package modeloracle

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

const TorchNNLinearHeaderSHA256 = "fdb854ec80107b177287e63be43a9b12923cec01a62f2b96b05cbc9d1743fd87"

func TorchNNLinearHeader(library string) (string, string, error) {
	data, err := os.ReadFile(filepath.Join(filepath.Dir(library), "manifest.json"))
	if err != nil {
		return "", "", err
	}
	var backend struct {
		Torch string `json:"torch_root"`
	}
	if err = json.Unmarshal(data, &backend); err != nil {
		return "", "", err
	}
	path := filepath.Join(backend.Torch, "include/torch/csrc/api/include/torch/nn/functional/linear.h")
	h, err := Hash(path)
	if err != nil {
		return "", "", err
	}
	if h != TorchNNLinearHeaderSHA256 {
		return "", "", fmt.Errorf("C++ nn linear header pin differs")
	}
	return path, h, nil
}

// GoSourceHashes identifies the graph, tensor boundary and verification source
// used by the current replay. It does not promote an older Go replay to current.
func GoSourceHashes() (map[string]string, error) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return nil, fmt.Errorf("cannot locate verifier source")
	}
	root := filepath.Dir(filepath.Dir(filepath.Dir(file)))
	out := map[string]string{}
	for _, rel := range []string{"internal/model", "internal/tensor", "verification/model"} {
		err := filepath.WalkDir(filepath.Join(root, rel), func(path string, entry os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				return nil
			}
			ext := filepath.Ext(path)
			if ext != ".go" && ext != ".c" && ext != ".cpp" && ext != ".h" && ext != ".json" {
				return nil
			}
			h, err := Hash(path)
			if err != nil {
				return err
			}
			name, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			out[filepath.ToSlash(name)] = h
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// VerifySUPCapture permits reuse of native captures only when every native
// source, adapter, library, compiler and generated-file input still matches.
func VerifySUPCapture(sourceRoot, library, directory string) (string, error) {
	data, err := os.ReadFile(filepath.Join(directory, "manifest.json"))
	if err != nil {
		return "", err
	}
	var m Manifest
	if err = json.Unmarshal(data, &m); err != nil {
		return "", err
	}
	if m.Status != "built" {
		return "", fmt.Errorf("native capture status %q", m.Status)
	}
	if m.Inputs["adapter.cpp"] != digest([]byte(supAdapter)) {
		return "", fmt.Errorf("SUP adapter changed")
	}
	sources, err := ExtractSUP(sourceRoot)
	if err != nil {
		return "", err
	}
	if len(m.Sources) != len(sources) {
		return "", fmt.Errorf("native source count differs")
	}
	for i, s := range sources {
		old := m.Sources[i]
		if old.Path != s.Path || old.SHA256 != s.SHA256 || len(old.Slices) != len(s.Slices) {
			return "", fmt.Errorf("native source differs %s", s.Path)
		}
		for j, part := range s.Slices {
			if old.Slices[j].SHA256 != part.SHA256 {
				return "", fmt.Errorf("native slice differs %s", s.Path)
			}
		}
	}
	if m.Inputs["dev_options"] != "mha_num_splits=12;mha_use_mask=true;use_uswiglu=false;koi_use_i8=true CPU branch inactive" {
		return "", fmt.Errorf("native default options differ")
	}
	for path, want := range m.Inputs {
		if !filepath.IsAbs(path) {
			continue
		}
		got, err := Hash(path)
		if err != nil {
			return "", err
		}
		if got != want {
			return "", fmt.Errorf("capture dependency changed %s", path)
		}
	}
	got, err := Hash(library)
	if err != nil {
		return "", err
	}
	if m.Inputs[library] != got {
		return "", fmt.Errorf("tensor backend differs")
	}
	backend, err := os.ReadFile(filepath.Join(filepath.Dir(library), "manifest.json"))
	if err != nil {
		return "", err
	}
	var currentBackend, capturedBackend bytes.Buffer
	if err = json.Compact(&currentBackend, backend); err != nil {
		return "", err
	}
	if err = json.Compact(&capturedBackend, m.Backend); err != nil {
		return "", err
	}
	if !bytes.Equal(currentBackend.Bytes(), capturedBackend.Bytes()) {
		return "", fmt.Errorf("backend receipt differs")
	}
	generated, err := TreeHashes(filepath.Join(directory, "synthetic"))
	if err != nil {
		return "", err
	}
	if len(generated) != len(m.Generated) {
		return "", fmt.Errorf("capture generated file count differs")
	}
	for name, h := range generated {
		if m.Generated[name] != h {
			return "", fmt.Errorf("capture payload differs %s", name)
		}
	}
	exe := filepath.Join(directory, "model-reference")
	if !strings.HasPrefix(m.Qualification, "CPU13 F32 selected original Dorado function algorithm parity") {
		return "", fmt.Errorf("capture qualification differs")
	}
	if m.Inputs[exe] == "" {
		return "", fmt.Errorf("capture lacks executable hash")
	}
	return exe, nil
}

// VerifyStockCapture checks an already completed bounded oracle run. It never
// invokes the producer or rewrites its receipt.
func VerifyStockCapture(exe, modelDirectory, directory string) error {
	data, err := os.ReadFile(filepath.Join(directory, "receipt.json"))
	if err != nil {
		return err
	}
	var command Command
	if err = json.Unmarshal(data, &command); err != nil {
		return err
	}
	if command.Exit != 0 || len(command.Args) != 3 || command.Args[0] != exe || command.Args[1] != directory || command.Args[2] != modelDirectory {
		return fmt.Errorf("stock capture command differs")
	}
	data, err = os.ReadFile(filepath.Join(directory, "generated-sha256.json"))
	if err != nil {
		return err
	}
	var pins map[string]string
	if err = json.Unmarshal(data, &pins); err != nil {
		return err
	}
	for name, want := range pins {
		if filepath.IsAbs(name) || filepath.Clean(name) == ".." || strings.HasPrefix(filepath.Clean(name), ".."+string(filepath.Separator)) {
			return fmt.Errorf("invalid stock capture path")
		}
		got, err := Hash(filepath.Join(directory, name))
		if err != nil {
			return err
		}
		if got != want {
			return fmt.Errorf("stock capture payload changed %s", name)
		}
	}
	return nil
}
