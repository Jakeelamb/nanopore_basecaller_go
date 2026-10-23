// Command build produces the CPU tensor library and a complete local build receipt.
package main

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

const archivePin = "8680441dbe3b0990fa7a99a70319affe25bc51f0cd6e105c3e07d522d7355929"

type receipt struct {
	Command  []string `json:"command"`
	Started  string   `json:"started_utc"`
	Finished string   `json:"finished_utc"`
	Exit     int      `json:"exit_code"`
	Log      string   `json:"log"`
	LogSHA   string   `json:"log_sha256"`
}
type manifest struct {
	ABI            int               `json:"abi_version"`
	Qualification  string            `json:"qualification"`
	Torch          string            `json:"torch_root"`
	DoradoSource   string            `json:"dorado_source"`
	DoradoCommit   string            `json:"dorado_commit"`
	Archive        string            `json:"archive"`
	ArchiveSHA     string            `json:"archive_sha256"`
	ArchiveMatched map[string]string `json:"extracted_archive_member_sha256"`
	Inputs         map[string]string `json:"input_sha256"`
	Runtime        map[string]string `json:"runtime_sha256"`
	Output         string            `json:"output"`
	OutputSHA      string            `json:"output_sha256"`
	Commands       []receipt         `json:"commands"`
	Status         string            `json:"status"`
	Error          string            `json:"error,omitempty"`
}

func hash(path string) (string, error) {
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
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run() (ret error) {
	torch := flag.String("torch", "", "read-only extracted qualified libtorch root")
	archive := flag.String("archive", "", "qualified CUDA 13.0 ONT archive")
	out := flag.String("out", filepath.Join("artifacts", "tensor", time.Now().UTC().Format("20060102T150405.000000000Z")), "new build directory")
	cxx := flag.String("cxx", "g++", "C++ compiler")
	weight := flag.String("weight", "", "optional read-only real Dorado weight for independent float32 snapshot")
	dorado := flag.String("dorado-source", os.Getenv("DORADO_SOURCE"), "read-only pinned Dorado checkout for initialization and archive contracts")
	flag.Parse()
	if *torch == "" || *archive == "" || *dorado == "" {
		return fmt.Errorf("-torch, -archive, and -dorado-source are required; no downloads are performed")
	}
	for _, v := range []*string{torch, archive, out, dorado} {
		p, err := filepath.Abs(*v)
		if err != nil {
			return err
		}
		*v = p
	}
	if *weight != "" {
		var err error
		*weight, err = filepath.Abs(*weight)
		if err != nil {
			return err
		}
	}
	if _, err := os.Stat(*out); err == nil {
		return fmt.Errorf("output directory already exists; preserve attempts with a fresh -out")
	}
	if err := os.MkdirAll(*out, 0755); err != nil {
		return err
	}
	m := manifest{ABI: 3, Qualification: "CPU development backend using ONT LibTorch CUDA13 headers and CPU libraries; not official stock CUDA12.8 runtime parity", Torch: *torch, Archive: *archive, Inputs: map[string]string{}, Runtime: map[string]string{}, ArchiveMatched: map[string]string{}, Status: "failed"}
	defer func() {
		if ret != nil {
			m.Error = ret.Error()
		}
		data, err := json.MarshalIndent(m, "", "  ")
		if err == nil {
			err = os.WriteFile(filepath.Join(*out, "manifest.json"), append(data, '\n'), 0644)
		}
		if ret == nil && err != nil {
			ret = err
		}
	}()
	var err error
	m.DoradoSource = *dorado
	pinsRaw, err := os.ReadFile("verification/tensor/source-pins.json")
	if err != nil {
		return err
	}
	var sourcePins struct {
		Commit string            `json:"commit"`
		Files  map[string]string `json:"files"`
	}
	if err := json.Unmarshal(pinsRaw, &sourcePins); err != nil {
		return err
	}
	m.DoradoCommit = sourcePins.Commit
	for rel, want := range sourcePins.Files {
		path := filepath.Join(*dorado, rel)
		actual, err := hash(path)
		if err != nil {
			return err
		}
		if actual != want {
			return fmt.Errorf("Dorado source pin mismatch: %s", path)
		}
		m.Inputs[path] = actual
	}
	m.ArchiveSHA, err = hash(*archive)
	if err != nil {
		return err
	}
	if m.ArchiveSHA != archivePin {
		return fmt.Errorf("archive SHA256 mismatch: %s", m.ArchiveSHA)
	}
	inputs := []string{"verification/tensor/backend.cpp", "verification/tensor/fixtures.cpp", "verification/tensor/runtime.h", "verification/tensor/source-pins.json", "verification/tensor/build/main.go", "internal/tensor/abi.h", filepath.Join(*torch, "build-version"), filepath.Join(*torch, "include/ATen/ATen.h"), filepath.Join(*torch, "include/torch/csrc/api/include/torch/serialize.h"), filepath.Join(*torch, "lib/libtorch_cpu.so"), filepath.Join(*torch, "lib/libc10.so")}
	if *weight != "" {
		inputs = append(inputs, *weight)
	}
	for _, path := range inputs {
		h, err := hash(path)
		if err != nil {
			return err
		}
		m.Inputs[path] = h
	}
	execute := func(label string, args ...string) (string, error) {
		logPath := filepath.Join(*out, label+".log")
		r := receipt{Command: args, Started: time.Now().UTC().Format(time.RFC3339Nano), Log: logPath}
		command := exec.Command(args[0], args[1:]...)
		data, err := command.CombinedOutput()
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		r.Exit = -1
		if command.ProcessState != nil {
			r.Exit = command.ProcessState.ExitCode()
		}
		sum := sha256.Sum256(data)
		r.LogSHA = hex.EncodeToString(sum[:])
		if writeErr := os.WriteFile(logPath, data, 0644); writeErr != nil {
			return "", writeErr
		}
		m.Commands = append(m.Commands, r)
		if err != nil {
			return string(data), fmt.Errorf("%s failed (%v); see %s", label, err, logPath)
		}
		return string(data), nil
	}
	compiler, err := exec.LookPath(*cxx)
	if err != nil {
		return err
	}
	compiler, err = filepath.EvalSymlinks(compiler)
	if err != nil {
		return err
	}
	m.Inputs[compiler], err = hash(compiler)
	if err != nil {
		return err
	}
	if _, err = execute("compiler", compiler, "--version"); err != nil {
		return err
	}
	libdir := filepath.Join(*torch, "lib")
	common := []string{"-std=c++17", "-O1", "-g0", "-fPIC", "-pthread", "-D_GLIBCXX_USE_CXX11_ABI=1", "-Iinternal/tensor", "-I" + filepath.Join(*torch, "include"), "-I" + filepath.Join(*torch, "include/torch/csrc/api/include")}
	link := []string{"-L" + libdir, "-Wl,-rpath," + libdir, "-ltorch_cpu", "-lc10"}
	m.Output = filepath.Join(*out, "libnanopore_tensor.so")
	args := append([]string{compiler}, common...)
	args = append(args, "-MMD", "-MF", filepath.Join(*out, "backend.d"), "-shared", "verification/tensor/backend.cpp", "-o", m.Output)
	args = append(args, link...)
	if _, err = execute("compile-backend", args...); err != nil {
		return err
	}
	m.OutputSHA, err = hash(m.Output)
	if err != nil {
		return err
	}
	ldd, err := execute("runtime-linkage", "ldd", m.Output)
	if err != nil {
		return err
	}
	if strings.Contains(ldd, "not found") {
		return fmt.Errorf("runtime library missing; see runtime-linkage.log")
	}
	re := regexp.MustCompile(`(?:=>\s+|^\s*)(/[^\s]+)`)
	for _, match := range re.FindAllStringSubmatch(ldd, -1) {
		path := match[1]
		m.Runtime[path], err = hash(path)
		if err != nil {
			return err
		}
	}
	fixtures := filepath.Join(*out, "create-fixtures")
	args = append([]string{compiler}, common...)
	args = append(args, "-MMD", "-MF", filepath.Join(*out, "fixtures.d"), "verification/tensor/fixtures.cpp", "-o", fixtures)
	args = append(args, link...)
	if _, err = execute("compile-fixtures", args...); err != nil {
		return err
	}
	fixtureArgs := []string{fixtures, *out}
	if *weight != "" {
		fixtureArgs = append(fixtureArgs, *weight)
	}
	if _, err = execute("create-fixtures", fixtureArgs...); err != nil {
		return err
	}
	m.Inputs[fixtures], err = hash(fixtures)
	if err != nil {
		return err
	}
	for _, name := range []string{"float32.tensor", "int64.tensor", "multiple.tensor", "empty.tensor", "primitive-reference.json", "argsort-reference.json", "attention-reference.json"} {
		path := filepath.Join(*out, name)
		m.Inputs[path], err = hash(path)
		if err != nil {
			return err
		}
	}
	if *weight != "" {
		path := filepath.Join(*out, "real-weight.json")
		m.Inputs[path], err = hash(path)
		if err != nil {
			return err
		}
	}
	for _, name := range []string{"backend.d", "fixtures.d"} {
		data, err := os.ReadFile(filepath.Join(*out, name))
		if err != nil {
			return err
		}
		for _, path := range strings.Fields(strings.ReplaceAll(string(data), "\\\n", "")) {
			if !strings.HasPrefix(path, *torch+string(filepath.Separator)) {
				continue
			}
			if _, ok := m.Inputs[path]; ok {
				continue
			}
			m.Inputs[path], err = hash(path)
			if err != nil {
				return err
			}
		}
	}
	z, err := zip.OpenReader(*archive)
	if err != nil {
		return err
	}
	defer z.Close()
	members := map[string]*zip.File{}
	for _, f := range z.File {
		members[f.Name] = f
	}
	for path, want := range m.Inputs {
		if !strings.HasPrefix(path, *torch+string(filepath.Separator)) {
			continue
		}
		rel, err := filepath.Rel(*torch, path)
		if err != nil {
			return err
		}
		member := "libtorch/" + filepath.ToSlash(rel)
		f, ok := members[member]
		if !ok {
			return fmt.Errorf("distribution archive lacks %s", member)
		}
		r, err := f.Open()
		if err != nil {
			return err
		}
		h := sha256.New()
		_, copyErr := io.Copy(h, r)
		closeErr := r.Close()
		if copyErr != nil {
			return copyErr
		}
		if closeErr != nil {
			return closeErr
		}
		got := hex.EncodeToString(h.Sum(nil))
		if got != want {
			return fmt.Errorf("extracted distribution differs from pinned archive at %s", path)
		}
		m.ArchiveMatched[member] = got
	}
	for _, name := range []string{"libtorch_cpu.so", "libc10.so"} {
		expected := m.Inputs[filepath.Join(libdir, name)]
		found := false
		for path, got := range m.Runtime {
			if filepath.Base(path) == name {
				if got != expected {
					return fmt.Errorf("runtime %s differs from selected distribution", name)
				}
				found = true
			}
		}
		if !found {
			return fmt.Errorf("runtime linkage lacks %s", name)
		}
	}
	m.Status = "built"
	fmt.Println(m.Output)
	return nil
}
