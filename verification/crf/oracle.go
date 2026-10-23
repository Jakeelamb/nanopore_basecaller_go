// Package crforacle extracts and builds original Dorado CRF scan functions.
package crforacle

import (
	"crypto/sha256"
	_ "embed"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const SourcePath = "dorado/basecall/decode/CPUDecoder.cpp"
const SourceSHA256 = "b8e147e771108134860a9eb291b12ec1b9c08b7e8f684ade6620844372249235"
const initPath = "dorado/torch_utils/torch_utils.cpp"
const initSHA = "01a711350bea009e9c0b5d0a3578327c10ac65fa773c1cec1e1cafafd8b19482"

//go:embed native/oracle.cpp
var adapter string

//go:embed model-score-fixture.json
var capturedManifest []byte

type SourceSlice struct {
	Path                      string
	Start, End                int
	SourceSHA256, SliceSHA256 string
}
type Command struct {
	Args     []string `json:"command"`
	Started  string   `json:"started_utc"`
	Finished string   `json:"finished_utc"`
	Exit     int      `json:"exit_code"`
	Log      string   `json:"log"`
	LogSHA   string   `json:"log_sha256"`
}
type Receipt struct {
	Qualification string            `json:"qualification"`
	Status        string            `json:"status"`
	Error         string            `json:"error,omitempty"`
	Sources       []SourceSlice     `json:"sources"`
	Inputs        map[string]string `json:"input_sha256"`
	Backend       json.RawMessage   `json:"backend"`
	Commands      []Command         `json:"commands"`
	Outputs       map[string]string `json:"output_sha256"`
}

func Hash(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
func digest(data []byte) string { sum := sha256.Sum256(data); return hex.EncodeToString(sum[:]) }
func writeJSON(path string, v any) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0644)
}
func execute(root, label string, args ...string) (Command, error) {
	r := Command{Args: args, Started: time.Now().UTC().Format(time.RFC3339Nano), Exit: -1, Log: filepath.Join(root, label+".log")}
	cmd := exec.Command(args[0], args[1:]...)
	data, err := cmd.CombinedOutput()
	r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
	if cmd.ProcessState != nil {
		r.Exit = cmd.ProcessState.ExitCode()
	}
	r.LogSHA = digest(data)
	if w := os.WriteFile(r.Log, data, 0644); w != nil {
		return r, w
	}
	if err != nil {
		return r, fmt.Errorf("%s: %w; see %s", label, err, r.Log)
	}
	return r, nil
}
func VerifySources(root string) error {
	for path, want := range map[string]string{SourcePath: SourceSHA256, initPath: initSHA} {
		got, err := Hash(filepath.Join(root, path))
		if err != nil {
			return err
		}
		if got != want {
			return fmt.Errorf("source pin mismatch: %s", path)
		}
	}
	return nil
}
func Extract(root string) (string, []SourceSlice, error) {
	if err := VerifySources(root); err != nil {
		return "", nil, err
	}
	body := adapter
	var slices []SourceSlice
	for _, part := range []struct {
		path, hash, marker string
		start, end         int
	}{
		{SourcePath, SourceSHA256, "// SCAN_SOURCE", 16, 37},
		{SourcePath, SourceSHA256, "// SCORE_SOURCE", 42, 91},
		{SourcePath, SourceSHA256, "// POSTERIOR_SOURCE", 126, 129},
		{initPath, initSHA, "// INITIALIZATION_SOURCE", 15, 34},
	} {
		data, err := os.ReadFile(filepath.Join(root, part.path))
		if err != nil {
			return "", nil, err
		}
		lines := strings.SplitAfter(string(data), "\n")
		text := strings.Join(lines[part.start-1:part.end], "")
		slices = append(slices, SourceSlice{part.path, part.start, part.end, part.hash, digest([]byte(text))})
		body = strings.Replace(body, part.marker, fmt.Sprintf("#line %d %q\n%s", part.start, part.path, text), 1)
	}
	return body, slices, nil
}
func Build(sourceRoot, library, output string) (exe string, ret error) {
	if err := os.Mkdir(output, 0755); err != nil {
		return "", err
	}
	m := Receipt{Qualification: "CPU13 float32 original CRF scan function parity, Go-owned production schedule; not GPU12.8, full basecalling, biological or performance evidence", Status: "failed", Inputs: map[string]string{}}
	defer func() {
		if ret != nil {
			m.Error = ret.Error()
		}
		if err := writeJSON(filepath.Join(output, "build.json"), m); ret == nil {
			ret = err
		}
	}()
	body, slices, err := Extract(sourceRoot)
	if err != nil {
		return "", err
	}
	m.Sources = slices
	raw, err := os.ReadFile(filepath.Join(filepath.Dir(library), "manifest.json"))
	if err != nil {
		return "", err
	}
	m.Backend = raw
	var backend struct {
		Torch      string            `json:"torch_root"`
		ArchiveSHA string            `json:"archive_sha256"`
		Status     string            `json:"status"`
		ABI        int               `json:"abi_version"`
		OutputSHA  string            `json:"output_sha256"`
		Inputs     map[string]string `json:"input_sha256"`
	}
	if err := json.Unmarshal(raw, &backend); err != nil {
		return "", err
	}
	if backend.ABI != 2 || backend.Status != "built" || backend.ArchiveSHA != "8680441dbe3b0990fa7a99a70319affe25bc51f0cd6e105c3e07d522d7355929" {
		return "", fmt.Errorf("qualified CPU13 ABI2 backend required")
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
		path := filepath.Join(backend.Torch, "lib", name)
		h, err := Hash(path)
		if err != nil {
			return "", err
		}
		if backend.Inputs[path] != h {
			return "", fmt.Errorf("CPU library pin mismatch: %s", path)
		}
		m.Inputs[path] = h
	}
	cpp := filepath.Join(output, "reference.cpp")
	if err := os.WriteFile(cpp, []byte(body), 0644); err != nil {
		return "", err
	}
	m.Inputs[cpp] = digest([]byte(body))
	m.Inputs["adapter.cpp"] = digest([]byte(adapter))
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
	command, err := execute(output, "compiler", compiler, "--version")
	m.Commands = append(m.Commands, command)
	if err != nil {
		return "", err
	}
	exe = filepath.Join(output, "crf-reference")
	lib := filepath.Join(backend.Torch, "lib")
	args := []string{compiler, "-std=c++17", "-O1", "-g0", "-pthread", "-D_GLIBCXX_USE_CXX11_ABI=1", "-DDORADO_CUDA_BUILD=0", "-I" + filepath.Join(backend.Torch, "include"), "-I" + filepath.Join(backend.Torch, "include/torch/csrc/api/include"), "-MMD", "-MF", filepath.Join(output, "reference.d"), cpp, "-o", exe, "-L" + lib, "-Wl,-rpath," + lib, "-ltorch_cpu", "-lc10"}
	command, err = execute(output, "compile", args...)
	m.Commands = append(m.Commands, command)
	if err != nil {
		return "", err
	}
	deps, err := os.ReadFile(filepath.Join(output, "reference.d"))
	if err != nil {
		return "", err
	}
	for _, path := range strings.Fields(strings.ReplaceAll(string(deps), "\\\n", "")) {
		if !strings.HasPrefix(path, backend.Torch+string(filepath.Separator)) {
			continue
		}
		h, err := Hash(path)
		if err != nil {
			return "", err
		}
		if backend.Inputs[path] != h {
			return "", fmt.Errorf("oracle header lacks matching backend pin: %s", path)
		}
		m.Inputs[path] = h
	}
	m.Outputs = map[string]string{}
	m.Outputs[exe], err = Hash(exe)
	if err != nil {
		return "", err
	}
	m.Status = "built"
	return exe, nil
}

type Fixture struct {
	Name   string    `json:"name"`
	Blocks int       `json:"blocks"`
	States int       `json:"states"`
	Blank  float32   `json:"blank"`
	Scores []float32 `json:"-"`
}

func Synthetic() []Fixture {
	var cases []Fixture
	for _, states := range []int{4, 16, 256, 1024} {
		for _, blocks := range []int{1, 2, 9, 32} {
			for pattern := 0; pattern < 4; pattern++ {
				f := Fixture{Name: fmt.Sprintf("s%d_t%d_p%d", states, blocks, pattern), Blocks: blocks, States: states, Blank: []float32{2, 0, -1.5, .25}[pattern], Scores: make([]float32, blocks*4*states)}
				for i := range f.Scores {
					switch pattern {
					case 0:
						f.Scores[i] = 0
					case 1:
						f.Scores[i] = float32((i*73+i/17)%257-128) / 16
					case 2:
						f.Scores[i] = float32((i*13+i/7)%5-2) * 1000
					case 3:
						f.Scores[i] = float32((i*11+i/3)%33-16) / 32
					}
				}
				cases = append(cases, f)
			}
		}
	}
	cases = append(cases, Fixture{Name: "maximum_states", Blocks: 1, States: 16384, Blank: 2, Scores: make([]float32, 4*16384)})
	return cases
}
func Captured(repoRoot string) (Fixture, map[string]string, error) {
	var pin struct {
		Name           string `json:"name"`
		Blocks, States int
		Blank          float32
		Pins           map[string]string
		Scores         string
	}
	if err := json.Unmarshal(capturedManifest, &pin); err != nil {
		return Fixture{}, nil, err
	}
	actualPins := map[string]string{}
	for relative, want := range pin.Pins {
		path := filepath.Join(repoRoot, relative)
		got, err := Hash(path)
		if err != nil {
			return Fixture{}, nil, err
		}
		if got != want {
			return Fixture{}, nil, fmt.Errorf("captured model score provenance pin mismatch: %s", path)
		}
		actualPins[path] = got
	}
	raw, err := os.ReadFile(filepath.Join(repoRoot, pin.Scores))
	if err != nil {
		return Fixture{}, nil, err
	}
	if len(raw) != pin.Blocks*pin.States*4*4 {
		return Fixture{}, nil, fmt.Errorf("captured score byte count mismatch")
	}
	scores := make([]float32, len(raw)/4)
	for i := range scores {
		scores[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return Fixture{Name: pin.Name, Blocks: pin.Blocks, States: pin.States, Blank: pin.Blank, Scores: scores}, actualPins, nil
}
func Run(exe, output string, fixtures []Fixture) (ret error) {
	if err := os.Mkdir(output, 0755); err != nil {
		return err
	}
	m := Receipt{Status: "failed", Inputs: map[string]string{}}
	defer func() {
		if ret != nil {
			m.Error = ret.Error()
		}
		if err := writeJSON(filepath.Join(output, "run.json"), m); ret == nil {
			ret = err
		}
	}()
	input := filepath.Join(output, "inputs.bin")
	f, err := os.Create(input)
	if err != nil {
		return err
	}
	writeErr := binary.Write(f, binary.LittleEndian, []uint32{0x43524631, uint32(len(fixtures))})
	for _, c := range fixtures {
		if writeErr != nil {
			break
		}
		writeErr = binary.Write(f, binary.LittleEndian, []uint32{uint32(c.Blocks), uint32(c.States), math.Float32bits(c.Blank)})
		if writeErr == nil {
			writeErr = binary.Write(f, binary.LittleEndian, c.Scores)
		}
	}
	closeErr := f.Close()
	if writeErr != nil {
		return writeErr
	}
	if closeErr != nil {
		return closeErr
	}
	m.Inputs[input], err = Hash(input)
	if err != nil {
		return err
	}
	m.Inputs[exe], err = Hash(exe)
	if err != nil {
		return err
	}
	if err := writeJSON(filepath.Join(output, "fixtures.json"), fixtures); err != nil {
		return err
	}
	command, err := execute(output, "native-run", exe, input, output)
	m.Commands = append(m.Commands, command)
	if err != nil {
		return err
	}
	m.Outputs = map[string]string{}
	for i := range fixtures {
		for _, stage := range []string{"scores", "forward", "backward", "posts"} {
			path := filepath.Join(output, fmt.Sprint(i), stage+".bin")
			m.Outputs[path], err = Hash(path)
			if err != nil {
				return err
			}
		}
	}
	m.Status = "produced"
	return nil
}
