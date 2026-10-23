// Package modeloracle builds a CPU algorithm oracle from verbatim, hash-pinned
// Dorado functions. Module storage declarations are verification adapters, not ABI replicas.
package modeloracle

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

//go:embed native/oracle.cpp
var adapter string

type Slice struct {
	Path         string
	Start, End   int
	SHA256, Text string
}
type Source struct {
	Path, SHA256 string
	Slices       []Slice
}

var pins = []struct {
	path, hash string
	ranges     [][2]int
}{
	{"dorado/nn/FLSTMStack.cpp", "97c2df7d81b7e8bb7c1d3c807ff4e80f39973df0bbc63cd2e2aa38d203a20160", [][2]int{{19, 93}}},
	{"dorado/nn/ConvStack.cpp", "d71cbdb1100010991753e4b3b80b8866265d97a08855f0b8ff1e41c00c91f6b2", [][2]int{{110, 128}, {169, 188}}},
	{"dorado/nn/CRFModules.cpp", "b74290106409135e9eeafc97f031f996a63898b356af194ed75a944f2dffed09", [][2]int{{15, 34}, {125, 134}}},
	{"dorado/config/include/config/common.h", "4d5011d1a9502c18530b48cda4ccc4db2696778bfa0efdcb4758fbbafebb0340", nil},
	{"dorado/torch_utils/torch_utils.cpp", "01a711350bea009e9c0b5d0a3578327c10ac65fa773c1cec1e1cafafd8b19482", [][2]int{{15, 34}}},
	{"dorado/basecall/model/CRFModel.cpp", "37f1f9712c3e61957ff17099d4cffb606c6cc784d2973c040bcc77129b5f9964", [][2]int{{66, 66}, {135, 146}}},
}

func Hash(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err = io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
func digest(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }

func Extract(root string) ([]Source, error) {
	var out []Source
	for _, pin := range pins {
		data, err := os.ReadFile(filepath.Join(root, pin.path))
		if err != nil {
			return nil, err
		}
		h := digest(data)
		if pin.hash != "" && h != pin.hash {
			return nil, fmt.Errorf("source pin mismatch %s: %s", pin.path, h)
		}
		s := Source{Path: pin.path, SHA256: h}
		lines := strings.Split(string(data), "\n")
		for _, r := range pin.ranges {
			txt := strings.Join(lines[r[0]-1:r[1]], "\n") + "\n"
			s.Slices = append(s.Slices, Slice{pin.path, r[0], r[1], digest([]byte(txt)), txt})
		}
		out = append(out, s)
	}
	return out, nil
}

type Command struct {
	Args              []string
	Started, Finished string
	Exit              int
	Log               string
}
type Manifest struct {
	Qualification, Status, Error string
	Sources                      []Source
	Inputs                       map[string]string
	Commands                     []Command
	Backend                      json.RawMessage
	Generated                    map[string]string
}

// Build requires a completed qualified backend receipt beside the chosen library.
// Every attempt gets a fresh output directory; failed build logs are preserved.
func Build(sourceRoot, library, output string) (exe string, ret error) {
	if err := os.Mkdir(output, 0755); err != nil {
		return "", err
	}
	m := Manifest{Qualification: "CPU13 F32 selected original Dorado function algorithm parity; synthetic and short stock-weight synthetic-signal inference; not original module ABI, GPU12.8, real-read, biological or performance evidence", Status: "failed", Inputs: map[string]string{}}
	defer func() {
		if ret != nil {
			m.Error = ret.Error()
		}
		data, err := json.MarshalIndent(m, "", "  ")
		if err == nil {
			err = os.WriteFile(filepath.Join(output, "manifest.json"), append(data, '\n'), 0644)
		}
		if ret == nil && err != nil {
			ret = err
		}
	}()
	sources, err := Extract(sourceRoot)
	if err != nil {
		return "", err
	}
	m.Sources = sources
	backendData, err := os.ReadFile(filepath.Join(filepath.Dir(library), "manifest.json"))
	if err != nil {
		return "", err
	}
	m.Backend = backendData
	var backend struct {
		Torch      string `json:"torch_root"`
		ArchiveSHA string `json:"archive_sha256"`
		Status     string `json:"status"`
		OutputSHA  string `json:"output_sha256"`
	}
	if err = json.Unmarshal(backendData, &backend); err != nil {
		return "", err
	}
	if backend.Status != "built" || backend.ArchiveSHA != "8680441dbe3b0990fa7a99a70319affe25bc51f0cd6e105c3e07d522d7355929" {
		return "", fmt.Errorf("backend lacks qualified CPU13 receipt")
	}
	h, err := Hash(library)
	if err != nil {
		return "", err
	}
	if h != backend.OutputSHA {
		return "", fmt.Errorf("backend hash differs from receipt")
	}
	m.Inputs[library] = h
	for _, name := range []string{"libtorch_cpu.so", "libc10.so"} {
		p := filepath.Join(backend.Torch, "lib", name)
		h, err := Hash(p)
		if err != nil {
			return "", err
		}
		m.Inputs[p] = h
	}
	m.Inputs["adapter.cpp"] = digest([]byte(adapter))
	var defs, initialization, modelForward, modelSequence strings.Builder
	for _, s := range sources {
		for _, slice := range s.Slices {
			dst := &defs
			if s.Path == "dorado/torch_utils/torch_utils.cpp" {
				dst = &initialization
			}
			if s.Path == "dorado/basecall/model/CRFModel.cpp" {
				if slice.Start == 66 {
					dst = &modelSequence
				} else {
					dst = &modelForward
				}
			}
			fmt.Fprintf(dst, "\n#line %d %q\n%s", slice.Start, slice.Path, slice.Text)
		}
	}
	body := strings.Replace(adapter, "// SOURCE_FUNCTIONS", defs.String(), 1)
	body = strings.Replace(body, "// INITIALIZATION_FUNCTIONS", initialization.String(), 1)
	body = strings.Replace(body, "// MODEL_FORWARD", modelForward.String(), 1)
	body = strings.Replace(body, "// MODEL_SEQUENCE", modelSequence.String(), 1)
	cpp := filepath.Join(output, "reference.cpp")
	if err = os.WriteFile(cpp, []byte(body), 0644); err != nil {
		return "", err
	}
	m.Inputs[cpp] = digest([]byte(body))
	compiler, err := exec.LookPath("g++")
	if err != nil {
		return "", err
	}
	compiler, err = filepath.EvalSymlinks(compiler)
	if err != nil {
		return "", err
	}
	m.Inputs[compiler], err = Hash(compiler)
	if err != nil {
		return "", err
	}
	run := func(label string, args []string) error {
		c := Command{Args: args, Started: time.Now().UTC().Format(time.RFC3339Nano), Log: filepath.Join(output, label+".log"), Exit: -1}
		cmd := exec.Command(args[0], args[1:]...)
		data, err := cmd.CombinedOutput()
		c.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if cmd.ProcessState != nil {
			c.Exit = cmd.ProcessState.ExitCode()
		}
		m.Commands = append(m.Commands, c)
		if er := os.WriteFile(c.Log, data, 0644); er != nil {
			return er
		}
		if err != nil {
			return fmt.Errorf("%s: %w; see %s", label, err, c.Log)
		}
		return nil
	}
	if err = run("compiler", []string{compiler, "--version"}); err != nil {
		return "", err
	}
	exe = filepath.Join(output, "model-reference")
	lib := filepath.Join(backend.Torch, "lib")
	args := []string{compiler, "-std=c++20", "-O1", "-g0", "-pthread", "-D_GLIBCXX_USE_CXX11_ABI=1", "-DDORADO_CUDA_BUILD=0", "-DDORADO_METAL_BUILD=0", "-I" + filepath.Join(sourceRoot, "dorado/config/include"), "-I" + filepath.Join(backend.Torch, "include"), "-I" + filepath.Join(backend.Torch, "include/torch/csrc/api/include"), "-MMD", "-MF", filepath.Join(output, "reference.d"), cpp, "-o", exe, "-L" + lib, "-Wl,-rpath," + lib, "-ltorch_cpu", "-lc10"}
	if err = run("compile", args); err != nil {
		return "", err
	}
	m.Inputs[exe], err = Hash(exe)
	if err != nil {
		return "", err
	}
	if err = run("synthetic", []string{exe, filepath.Join(output, "synthetic")}); err != nil {
		return "", err
	}
	m.Generated, err = TreeHashes(filepath.Join(output, "synthetic"))
	if err != nil {
		return "", err
	}
	m.Status = "built"
	return exe, nil
}

// RunStock records a separate receipt and never changes the original model files.
func RunStock(exe, modelDirectory, output string) error {
	if err := os.Mkdir(output, 0755); err != nil {
		return err
	}
	r := Command{Args: []string{exe, output, modelDirectory}, Started: time.Now().UTC().Format(time.RFC3339Nano), Exit: -1, Log: filepath.Join(output, "run.log")}
	cmd := exec.Command(exe, output, modelDirectory)
	data, err := cmd.CombinedOutput()
	r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
	if cmd.ProcessState != nil {
		r.Exit = cmd.ProcessState.ExitCode()
	}
	if er := os.WriteFile(r.Log, data, 0644); er != nil {
		return er
	}
	b, er := json.MarshalIndent(r, "", "  ")
	if er != nil {
		return er
	}
	if er = os.WriteFile(filepath.Join(output, "receipt.json"), append(b, '\n'), 0644); er != nil {
		return er
	}
	if err != nil {
		return err
	}
	hashes, err := TreeHashes(output)
	if err != nil {
		return err
	}
	b, err = json.MarshalIndent(hashes, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(output, "generated-sha256.json"), append(b, '\n'), 0644)
}

func TreeHashes(root string) (map[string]string, error) {
	out := map[string]string{}
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		h, err := Hash(path)
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		out[filepath.ToSlash(rel)] = h
		return nil
	})
	return out, err
}

type Ref struct {
	Name, Path string
	Shape      []int64
}
type Fixture struct {
	Name, Kind              string
	Channels, Inner, Layers int
	Reverse                 bool
	Input                   Ref
	Weights, Stages         []Ref
}

func Fixtures(directory string) ([]Fixture, error) {
	data, err := os.ReadFile(filepath.Join(directory, "fixtures.json"))
	if err != nil {
		return nil, err
	}
	var fs []Fixture
	err = json.Unmarshal(data, &fs)
	return fs, err
}
