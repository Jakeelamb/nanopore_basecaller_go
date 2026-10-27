// Command build produces the CPU tensor library and a complete local build receipt.
package main

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild/recipe"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/cudafixture"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const archivePin = "8680441dbe3b0990fa7a99a70319affe25bc51f0cd6e105c3e07d522d7355929"

type receipt = runner.CommandReceipt

type manifest struct {
	Head              string            `json:"head"`
	Started           string            `json:"started_utc"`
	Finished          string            `json:"finished_utc"`
	Environment       []string          `json:"environment"`
	ExtractedBlockSHA string            `json:"extracted_cuda_source_377_394_sha256,omitempty"`
	ToolkitVersion    int               `json:"cuda_toolkit_header_version,omitempty"`
	Capability        string            `json:"cuda_family"`
	ABI               int               `json:"abi_version"`
	Qualification     string            `json:"qualification"`
	Torch             string            `json:"torch_root"`
	DoradoSource      string            `json:"dorado_source"`
	DoradoCommit      string            `json:"dorado_commit"`
	Archive           string            `json:"archive"`
	ArchiveSHA        string            `json:"archive_sha256"`
	ArchiveMatched    map[string]string `json:"extracted_archive_member_sha256"`
	Inputs            map[string]string `json:"input_sha256"`
	Runtime           map[string]string `json:"runtime_sha256"`
	Output            string            `json:"output"`
	OutputSHA         string            `json:"output_sha256"`
	Commands          []receipt         `json:"commands"`
	Status            string            `json:"status"`
	Error             string            `json:"error,omitempty"`
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
		os.Exit(runner.ExitCode(err))
	}
}
func run() (ret error) {
	if err := runner.CheckRoot(); err != nil {
		return err
	}
	torch := flag.String("torch", "", "read-only extracted qualified libtorch root")
	archive := flag.String("archive", "", "qualified CUDA 13.0 ONT archive")
	out := flag.String("out", filepath.Join("artifacts", "tensor", time.Now().UTC().Format("20060102T150405.000000000Z")), "new build directory")
	cxx := flag.String("cxx", "g++", "C++ compiler")
	weight := flag.String("weight", "", "optional read-only real Dorado weight for independent float32 snapshot")
	family := flag.String("cuda-family", "none", "none or 13.0; build and link only, never capture")
	cudaRoot := flag.String("cuda-root", "/opt/cuda", "CUDA13 toolkit for compile and link only")
	koi := flag.String("koi", os.Getenv("LIBKOI_ROOT"), "read-only pinned libkoi prefix; required for a CUDA build because the HAC boundary calls host_convolution_f16")
	dorado := flag.String("dorado-source", os.Getenv("DORADO_SOURCE"), "read-only pinned Dorado checkout for initialization and archive contracts")
	planOnly := flag.Bool("plan-only", false, "print the declarative compiler recipe only; no filesystem outputs, compiler queries, or native execution")
	flag.Parse()
	if *family != "none" && *family != "13.0" {
		return fmt.Errorf("unsupported CUDA family %q", *family)
	}
	if *torch == "" || *archive == "" || *dorado == "" {
		return fmt.Errorf("-torch, -archive, and -dorado-source are required; no downloads are performed")
	}
	if *family == "13.0" && *koi == "" {
		return fmt.Errorf("-koi is required for a CUDA build: the HAC small-convolution boundary links the pinned libkoi")
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
	if *koi != "" {
		p, err := filepath.Abs(*koi)
		if err != nil {
			return err
		}
		*koi = p
	}
	compilerRecipe, err := recipe.New(recipe.Options{Compiler: *cxx, Torch: *torch, Archive: *archive, Dorado: *dorado, CUDA: *cudaRoot, Koi: *koi, Weight: *weight, Family: *family})
	if err != nil {
		return err
	}
	if *planOnly {
		return json.NewEncoder(os.Stdout).Encode(compilerRecipe)
	}
	attempt, err := runner.NewAttempt(*out)
	if err != nil {
		return err
	}
	*out = attempt
	m := manifest{ABI: 5, Qualification: "ABI5 candidate; compile/link evidence does not qualify CPU contracts, CUDA runtime, or decoder numerics", Torch: *torch, Archive: *archive, Inputs: map[string]string{}, Runtime: map[string]string{}, ArchiveMatched: map[string]string{}, Status: "failed", Capability: *family}
	defer func() {
		m.Finished = time.Now().UTC().Format(time.RFC3339Nano)
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
	m.Started = time.Now().UTC().Format(time.RFC3339Nano)
	for _, key := range []string{"CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "DORADO_MATMUL_FP16_MODE", "LD_LIBRARY_PATH"} {
		m.Environment = append(m.Environment, key+"="+os.Getenv(key))
	}
	if head, err := exec.Command("git", "rev-parse", "HEAD").Output(); err == nil {
		m.Head = strings.TrimSpace(string(head))
	} else {
		return err
	}
	for _, root := range []string{"internal/tensor", "verification/tensor"} {
		if err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			h, err := hash(path)
			if err == nil {
				m.Inputs[path] = h
			}
			return err
		}); err != nil {
			return err
		}
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		m.Status = "admission_blocked"
		return err
	}
	defer lock.Close()
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
	sourceHead, err := exec.Command("git", "-C", *dorado, "rev-parse", "HEAD").Output()
	if err != nil {
		return err
	}
	if strings.TrimSpace(string(sourceHead)) != sourcePins.Commit {
		return fmt.Errorf("Dorado commit pin mismatch")
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
	inputs := []string{"verification/tensor/backend.cpp", "verification/tensor/cuda_records.h", "verification/tensor/cuda_backend.h", "verification/tensor/cuda_storage.inc", "verification/tensor/hac_smallconv.inc", "verification/tensor/sup_primitive.inc", "verification/tensor/hac_lastconv.inc", "verification/tensor/hac_flstm.inc", "verification/tensor/hac_output.inc", "verification/tensor/hac_lstm_step.inc", "verification/tensor/decoder_guide.inc", "verification/tensor/decoder/guide_ptx.h", "verification/tensor/storage_byte_interval.h", "verification/tensor/drain_ownership.h", "verification/tensor/cuda_failure_test.cpp", "verification/tensor/cuda_oracle.cpp", "verification/tensor/fixtures.cpp", "verification/tensor/runtime.h", "verification/tensor/source-pins.json", "verification/tensor/build/main.go", "internal/tensor/abi.h", filepath.Join(*torch, "build-version"), filepath.Join(*torch, "include/ATen/ATen.h"), filepath.Join(*torch, "include/torch/csrc/api/include/torch/serialize.h"), filepath.Join(*torch, "lib/libtorch_cpu.so"), filepath.Join(*torch, "lib/libc10.so"), "verification/tensor/decoder_beam.inc", "verification/tensor/decoder/beam_ptx.h", "verification/tensor/decoder_posts.inc", "verification/tensor/decoder/forward_ptx.h", "verification/tensor/decoder/posterior_ptx.h", "verification/tensor/decoder/quality_ptx.h"}
	if *weight != "" {
		inputs = append(inputs, *weight)
	}
	if *family == "13.0" {
		// The HAC boundary calls libkoi's host_convolution_f16. Both the header
		// and the static archive are pinned inputs, so a capture can never run
		// against a different Koi build.
		inputs = append(inputs, filepath.Join(*koi, "include/koi.h"), filepath.Join(*koi, "lib/libkoi.a"))
	}
	for _, path := range inputs {
		h, err := hash(path)
		if err != nil {
			return err
		}
		m.Inputs[path] = h
	}
	execute := func(label string, args ...string) (string, error) {
		r, err := runner.Execute(*out, label, nil, args...)
		m.Commands = append(m.Commands, r)
		if err != nil {
			return "", err
		}
		raw, err := os.ReadFile(r.Stdout)
		if err != nil {
			return "", fmt.Errorf("retained command output could not be read: %w", err)
		}
		return string(raw), nil
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
	compilerRecipe.Options.Compiler = compiler
	failureProbe := filepath.Join(*out, "cuda-failure-test")
	failureArgs, err := compilerRecipe.Combined("failure-test", *out, "")
	if err != nil {
		return err
	}
	if _, err = execute("compile-failure-test", failureArgs...); err != nil {
		return err
	}
	m.Inputs[failureProbe], err = hash(failureProbe)
	if err != nil {
		return err
	}
	if _, err = execute("failure-test", failureProbe); err != nil {
		return err
	}
	m.Inputs[failureProbe], err = hash(failureProbe)
	if err != nil {
		return err
	}
	libdir := filepath.Join(*torch, "lib")
	if *family == "13.0" {
		header, err := os.ReadFile(filepath.Join(*cudaRoot, "include/cuda.h"))
		if err != nil {
			return err
		}
		version := regexp.MustCompile(`(?m)^#define CUDA_VERSION\s+(\d+)\s*$`).FindSubmatch(header)
		if len(version) != 2 {
			return fmt.Errorf("missing CUDA toolkit version")
		}
		m.ToolkitVersion, err = strconv.Atoi(string(version[1]))
		if err != nil {
			return err
		}
		if m.ToolkitVersion/1000 != 13 {
			return fmt.Errorf("CUDA toolkit must be major family 13; got %d", m.ToolkitVersion)
		}
	}
	m.Output = filepath.Join(*out, "libnanopore_tensor.so")
	args, err := compilerRecipe.Combined("backend", *out, filepath.Join(*out, "backend.d"))
	if err != nil {
		return err
	}
	if _, err = execute("compile-backend", args...); err != nil {
		return err
	}
	m.Inputs[filepath.Join(*out, "backend.d")], err = hash(filepath.Join(*out, "backend.d"))
	if err != nil {
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
		if strings.HasPrefix(path, *torch+string(filepath.Separator)) {
			m.Inputs[path] = m.Runtime[path]
		}
	}
	if *family == "13.0" {
		m.ExtractedBlockSHA, err = runner.ExtractCUDA(*dorado, *out)
		if err != nil {
			return err
		}
		m.Inputs[filepath.Join(*out, "cuda_source_block.h")], err = hash(filepath.Join(*out, "cuda_source_block.h"))
		if err != nil {
			return err
		}
		oracle := filepath.Join(*out, "cuda-source-oracle")
		oracleArgs, err := compilerRecipe.Combined("cuda-oracle", *out, filepath.Join(*out, "cuda-oracle.d"))
		if err != nil {
			return err
		}
		if _, err = execute("compile-cuda-oracle", oracleArgs...); err != nil {
			return err
		}
		m.Inputs[filepath.Join(*out, "cuda-oracle.d")], err = hash(filepath.Join(*out, "cuda-oracle.d"))
		if err != nil {
			return err
		}
		m.Inputs[oracle], err = hash(oracle)
		if err != nil {
			return err
		}
		if err = cudafixture.Generate(filepath.Join(*out, "cuda-inputs")); err != nil {
			return err
		}
		if err = filepath.WalkDir(filepath.Join(*out, "cuda-inputs"), func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			m.Inputs[path], err = hash(path)
			return err
		}); err != nil {
			return err
		}
	}
	fixtures := filepath.Join(*out, "create-fixtures")
	args, err = compilerRecipe.Combined("fixtures", *out, filepath.Join(*out, "fixtures.d"))
	if err != nil {
		return err
	}
	if _, err = execute("compile-fixtures", args...); err != nil {
		return err
	}
	m.Inputs[fixtures], err = hash(fixtures)
	if err != nil {
		return err
	}
	m.Inputs[filepath.Join(*out, "fixtures.d")], err = hash(filepath.Join(*out, "fixtures.d"))
	if err != nil {
		return err
	}
	fixtureArgs := []string{fixtures, *out}
	if *weight != "" {
		fixtureArgs = append(fixtureArgs, *weight)
	}
	depfiles := []string{"backend.d", "fixtures.d"}
	if *family == "13.0" {
		depfiles = append(depfiles, "cuda-oracle.d")
	}
	for _, name := range depfiles {
		data, err := os.ReadFile(filepath.Join(*out, name))
		if err != nil {
			return err
		}
		for _, path := range strings.Fields(strings.ReplaceAll(string(data), "\\\n", "")) {
			if !strings.HasPrefix(path, *torch+string(filepath.Separator)) && !strings.HasPrefix(path, *cudaRoot+string(filepath.Separator)) {
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
	for path, want := range m.Inputs {
		if !strings.HasPrefix(path, *torch+string(filepath.Separator)) {
			continue
		}
		matched, err := runner.ArchiveMember(z, *torch, path, want)
		if err != nil {
			return err
		}
		for member, hash := range matched {
			m.ArchiveMatched[member] = hash
		}
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
	m.Status = "built"
	fmt.Println(m.Output)
	return nil
}
