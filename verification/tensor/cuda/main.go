// Command cuda captures or replays source half-GEMM fixtures only after explicit runtime admission.
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/cudafixture"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type savedCase struct {
	Case   string `json:"case"`
	Prefix string `json:"source_prefix"`
}
type consumerEvidence struct {
	Head        string            `json:"head"`
	Sources     map[string]string `json:"source_sha256"`
	TestBinary  string            `json:"test_binary"`
	TestSHA     string            `json:"test_binary_sha256"`
	StableAfter bool              `json:"stable_after_execution"`
}
type evidence struct {
	ProducerIdentity  string                  `json:"producer_device_runtime_handle_identity"`
	DeviceIdentity    string                  `json:"device_bookend_identity"`
	Environment       map[string]string       `json:"effective_runtime_environment"`
	Ordinal           int                     `json:"cuda_ordinal"`
	Consumer          *consumerEvidence       `json:"consumer,omitempty"`
	Schema            string                  `json:"schema"`
	ABI               int                     `json:"abi_version"`
	Status            string                  `json:"status"`
	Error             string                  `json:"error,omitempty"`
	Started           string                  `json:"started_utc"`
	Finished          string                  `json:"finished_utc"`
	Inputs            map[string]string       `json:"input_sha256"`
	Runtime           map[string]string       `json:"runtime_sha256"`
	OutputSHA         string                  `json:"output_sha256"`
	Cases             []savedCase             `json:"cases"`
	Commands          []runner.CommandReceipt `json:"commands"`
	MutantDifferences int                     `json:"precision_mutant_distinguished_cases"`
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
	weight := flag.String("weight", "", "read-only real HAC weight for CPU contracts against the CUDA-enabled library")
	backend := flag.String("backend", "", "ABI4 CUDA13 library produced by compile-only builder")
	capture := flag.String("capture", "", "saved capture manifest to replay; omitted creates fresh source capture")
	out := flag.String("out", "", "new artifacts/tensor attempt")
	admitted := flag.Bool("runtime-admitted", false, "coordinator has lifted the operational CUDA hold for this invocation")
	ordinal := flag.Int("ordinal", 0, "visible CUDA ordinal")
	flag.Parse()
	if *backend == "" || *out == "" {
		return fmt.Errorf("-backend and -out required")
	}
	if *ordinal < 0 || *ordinal > 2147483647 {
		return fmt.Errorf("invalid ordinal")
	}
	absolute, err := filepath.Abs(*backend)
	if err != nil {
		return err
	}
	*backend = absolute
	dir, err := runner.NewAttempt(*out)
	if err != nil {
		return err
	}
	m := evidence{Schema: "nanopore-cuda-half-capture/v1", ABI: 4, Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Inputs: map[string]string{}, Runtime: map[string]string{}}
	defer func() {
		m.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if ret != nil {
			m.Error = ret.Error()
		}
		raw, err := json.MarshalIndent(m, "", "  ")
		if err == nil {
			err = os.WriteFile(filepath.Join(dir, "manifest.json"), append(raw, '\n'), 0644)
		}
		if ret == nil {
			ret = err
		}
	}()
	if !*admitted {
		m.Status = "runtime_held"
		return fmt.Errorf("CUDA runtime operational hold remains in force; no producer or consumer was launched")
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		m.Status = "admission_blocked"
		return err
	}
	defer lock.Close()
	buildPath := filepath.Join(filepath.Dir(*backend), "manifest.json")
	raw, err := os.ReadFile(buildPath)
	if err != nil {
		return err
	}
	var build struct {
		ABI       int               `json:"abi_version"`
		Family    string            `json:"cuda_family"`
		Status    string            `json:"status"`
		OutputSHA string            `json:"output_sha256"`
		Inputs    map[string]string `json:"input_sha256"`
		Runtime   map[string]string `json:"runtime_sha256"`
		Torch     string            `json:"torch_root"`
	}
	if err = json.Unmarshal(raw, &build); err != nil {
		return err
	}
	if build.ABI != 4 || build.Family != "13.0" || build.Status != "built" {
		return fmt.Errorf("expected successfully built ABI4 CUDA13 artifact")
	}
	pin := func(path, want string) error {
		got, err := runner.SHA(path)
		if err != nil {
			return err
		}
		if want != "" && got != want {
			return fmt.Errorf("input pin mismatch: %s", path)
		}
		m.Inputs[path] = got
		return nil
	}
	if err = pin(*backend, build.OutputSHA); err != nil {
		return err
	}
	m.OutputSHA = build.OutputSHA
	if err = pin(buildPath, ""); err != nil {
		return err
	}
	for path, want := range build.Runtime {
		if err = pin(path, want); err != nil {
			return err
		}
		m.Runtime[path] = want
	}
	oracle := filepath.Join(filepath.Dir(*backend), "cuda-source-oracle")
	if build.Inputs[oracle] == "" {
		return fmt.Errorf("missing pinned independent oracle")
	}
	if err = pin(oracle, build.Inputs[oracle]); err != nil {
		return err
	}
	execute := func(label string, env []string, args ...string) (runner.CommandReceipt, error) {
		r, err := runner.Execute(dir, label, env, args...)
		m.Commands = append(m.Commands, r)
		return r, err
	}
	m.Ordinal = *ordinal
	m.Environment = map[string]string{"DORADO_MATMUL_FP16_MODE": "CUBLAS", "LD_LIBRARY_PATH": "", "LD_PRELOAD": ""}
	for _, key := range []string{"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG"} {
		value, present := os.LookupEnv(key)
		m.Environment[key] = value
		m.Environment[key+".present"] = strconv.FormatBool(present)
	}

	baseEnv := []string{"LD_LIBRARY_PATH=", "LD_PRELOAD=", "DORADO_MATMUL_FP16_MODE=CUBLAS", "NANOPORE_TENSOR_OLD_BACKEND="}
	deviceIdentity := ""
	loaded := func(log string) error {
		b, err := os.ReadFile(log)
		if err != nil {
			return err
		}
		identity := ""
		for _, line := range strings.Split(string(b), "\n") {
			if strings.HasPrefix(line, "driver=") {
				identity = line
			}
		}
		runtimeVersion := 0
		for _, field := range strings.Fields(identity) {
			if value, ok := strings.CutPrefix(field, "runtime="); ok {
				runtimeVersion, _ = strconv.Atoi(value)
			}
		}
		if runtimeVersion/1000 != 13 || !strings.Contains(identity, "stream_matches=1 ") {
			return fmt.Errorf("source CUDA family or stream diagnostics mismatch")
		}
		if deviceIdentity != "" && identity != deviceIdentity {
			return fmt.Errorf("source device or handle diagnostics changed between batches")
		}
		deviceIdentity = identity
		m.ProducerIdentity = identity
		for _, line := range strings.Split(string(b), "\n") {
			f := strings.Fields(line)
			if len(f) < 6 || !strings.Contains(f[0], "-") {
				continue
			}
			path := strings.Join(f[5:], " ")
			name := filepath.Base(path)
			family := strings.HasPrefix(name, "libtorch") || strings.HasPrefix(name, "libc10") || strings.HasPrefix(name, "libcu") || strings.HasPrefix(name, "libnv")
			if !family {
				continue
			}
			if strings.Contains(path, "CUDA-12") || strings.Contains(path, "cuda-12") {
				return fmt.Errorf("foreign CUDA12 loaded: %s", path)
			}
			if strings.HasPrefix(name, "libtorch") || strings.HasPrefix(name, "libc10") {
				if !strings.HasPrefix(path, build.Torch+"/") {
					return fmt.Errorf("foreign Torch distribution: %s", path)
				}
			}
			if want := build.Runtime[path]; want != "" {
				if err = pin(path, want); err != nil {
					return err
				}
			}
			got, err := runner.SHA(path)
			if err != nil {
				return err
			}
			if old := m.Runtime[path]; old != "" && old != got {
				return fmt.Errorf("loaded library changed: %s", path)
			}
			m.Runtime[path] = got
		}
		return nil
	}
	preIdentity := ""
	bookend := func(label string) error {
		state, err := execute(label+"-gpu-state", baseEnv, "nvidia-smi", "--query-gpu=index,uuid,driver_version,memory.total,memory.used,temperature.gpu", "--format=csv,noheader,nounits")
		if err != nil {
			return err
		}
		apps, err := execute(label+"-compute-apps", baseEnv, "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits")
		if err != nil {
			return err
		}
		raw, err := os.ReadFile(apps.Stdout)
		if err != nil {
			return err
		}
		if strings.TrimSpace(string(raw)) != "" {
			return fmt.Errorf("competing GPU process observed under admission; see %s", apps.Stdout)
		}
		raw, err = os.ReadFile(state.Stdout)
		if err != nil {
			return err
		}
		var identity []string
		for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
			parts := strings.Split(line, ",")
			if len(parts) != 6 {
				return fmt.Errorf("unexpected nvidia-smi state format")
			}
			identity = append(identity, strings.Join(parts[:3], ","))
		}
		joined := strings.Join(identity, "\n")
		if preIdentity != "" && preIdentity != joined {
			return fmt.Errorf("GPU device mapping or driver changed during runtime attempt")
		}
		preIdentity = joined
		m.DeviceIdentity = joined
		return nil
	}
	if err = bookend("before"); err != nil {
		return err
	}
	defer func() {
		if err := bookend("after"); err != nil {
			m.Status = "failed"
			if ret == nil {
				ret = err
			}
		}
	}()
	if *capture == "" {
		if err = captureBatches(&m, dir, *backend, oracle, *ordinal, build.Inputs, baseEnv, pin, execute, loaded); err != nil {
			return err
		}
		m.Status = "source_capture_exact_repeat"
		if m.MutantDifferences == 0 {
			m.Status = "source_capture_exact_repeat_mutant_gap"
		}
	} else {
		raw, err := os.ReadFile(*capture)
		if err != nil {
			return err
		}
		var prior evidence
		if err = json.Unmarshal(raw, &prior); err != nil {
			return err
		}
		if err = matchCaptureIdentity(prior, m); err != nil {
			return err
		}
		m.ProducerIdentity = prior.ProducerIdentity
		if prior.Schema != m.Schema || len(prior.Cases) != 37 || !strings.HasPrefix(prior.Status, "source_capture_exact_repeat") {
			return fmt.Errorf("unqualified source capture")
		}
		if err = pin(*capture, ""); err != nil {
			return err
		}
		for path, want := range prior.Inputs {
			if err = pin(path, want); err != nil {
				return err
			}
		}
		for path, want := range prior.Runtime {
			if err = pin(path, want); err != nil {
				return err
			}
			m.Runtime[path] = want
		}
		m.Cases = prior.Cases
		m.MutantDifferences = prior.MutantDifferences
		runtimeManifest := filepath.Join(dir, "runtime-pins.json")
		b, _ := json.Marshal(m)
		if err = os.WriteFile(runtimeManifest, b, 0644); err != nil {
			return err
		}
		env := append(append([]string{}, baseEnv...), "CGO_ENABLED=1", "NANOPORE_TENSOR_BACKEND="+*backend, "NANOPORE_TENSOR_CUDA=1", "NANOPORE_TENSOR_CUDA_DIAGNOSTICS=1", "NANOPORE_TENSOR_CUDA_ORDINAL="+strconv.Itoa(*ordinal), "NANOPORE_TENSOR_RUNTIME_MANIFEST="+runtimeManifest)
		if *weight == "" {
			return fmt.Errorf("CUDA consumer acceptance requires -weight for CPU contracts")
		}
		weightPath, err := filepath.Abs(*weight)
		if err != nil {
			return err
		}
		if build.Inputs[weightPath] == "" {
			return fmt.Errorf("CPU control weight lacks build pin")
		}
		if err = pin(weightPath, build.Inputs[weightPath]); err != nil {
			return err
		}
		env = append(env, "NANOPORE_TENSOR_WEIGHT="+weightPath)
		consumer, err := currentConsumer(dir)
		if err != nil {
			return err
		}
		m.Consumer = &consumer
		compileEnv := append(append([]string{}, baseEnv...), "CGO_ENABLED=1", "NANOPORE_TENSOR_BACKEND=", "NANOPORE_TENSOR_CUDA=")
		if _, err = execute("compile-go-consumer", compileEnv, "go", "test", "-c", "-o", consumer.TestBinary, "./internal/tensor"); err != nil {
			return err
		}
		consumer.TestSHA, err = runner.SHA(consumer.TestBinary)
		if err != nil {
			return err
		}
		if err = consumerStable(consumer); err != nil {
			return err
		}
		cpuTests := []string{"TestNativeTensorOperations", "TestNativeOwnershipAndCopies", "TestNativeMalformedInputs", "TestNativeLoadWeight", "TestNativeConcurrentClose", "TestNativeMissingLibrary", "TestNativeLoaderABIRejections", "TestNativeLoadedRuntimePins", "TestNativeDecoderPrimitives", "TestNativeCloseFailureRetainsLibrary", "TestNativeAttentionPrimitives", "TestNativeAttentionRejections", "TestNativeABIBoundary"}
		cpu, err := execute("cpu-contracts-cuda-library", env, "go", "tool", "test2json", "-t", "-p", "github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor", consumer.TestBinary, "-test.v=test2json", "-test.count=1", "-test.run", "^("+strings.Join(cpuTests, "|")+")$")
		if err != nil {
			return err
		}
		if err = runner.RequireGoTestRoots(cpu.Stdout, "github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor", cpuTests); err != nil {
			return err
		}
		ownership, err := execute("ownership-controls", env, "go", "tool", "test2json", "-t", "-p", "github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor", consumer.TestBinary, "-test.v=test2json", "-test.count=1", "-test.run", "^TestNativeCUDA(AllHalfBits|OwnershipAndLayouts|ABIBoundary)$")
		if err != nil {
			return err
		}
		const nativePackage = "github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
		if err = matchConsumerModes(ownership.Stdout, m.ProducerIdentity, 1); err != nil {
			return err
		}
		if err = runner.RequireGoTests(ownership.Stdout, nativePackage, []string{"TestNativeCUDAAllHalfBits", "TestNativeCUDAOwnershipAndLayouts", "TestNativeCUDAABIBoundary"}); err != nil {
			return err
		}
		replayEnv := append(append([]string{}, env...), "NANOPORE_TENSOR_CUDA_REPLAY="+*capture)
		replay, err := execute("source-corpus-go", replayEnv, "go", "tool", "test2json", "-t", "-p", "github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor", consumer.TestBinary, "-test.v=test2json", "-test.count=1", "-test.run", "^TestNativeCUDASourceFixtures$")
		if err != nil {
			return err
		}
		required := []string{"TestNativeCUDASourceFixtures"}
		seen := map[string]bool{}
		for _, saved := range prior.Cases {
			name := "TestNativeCUDASourceFixtures/" + filepath.Base(filepath.Dir(saved.Case))
			if seen[name] {
				return fmt.Errorf("duplicate source replay case")
			}
			seen[name] = true
			required = append(required, name)
		}
		if err = matchConsumerModes(replay.Stdout, m.ProducerIdentity, 36); err != nil {
			return err
		}
		if err = runner.RequireGoTests(replay.Stdout, nativePackage, required); err != nil {
			return err
		}
		if err = consumerStable(consumer); err != nil {
			return err
		}
		m.Consumer.StableAfter = true
		m.Status = "source_differential_passed"
	}
	fmt.Println(filepath.Join(dir, "manifest.json"))
	return nil
}
func checkGuards(c cudafixture.Case, data map[string][]uint16, role, path string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	v := c.Views[role]
	before := data[v.Allocation]
	if len(raw) != len(before)*2 {
		return fmt.Errorf("wrong complete parent output size")
	}
	allowed := map[uint64]bool{}
	out := c.Views["c"]
	if c.Mode == "gemm" && out.Allocation == v.Allocation {
		for i := int64(0); i < out.Shape[0]; i++ {
			for j := int64(0); j < out.Shape[1]; j++ {
				allowed[out.Offset+uint64(i*out.Strides[0]+j)] = true
			}
		}
	}
	for i, b := range before {
		if !allowed[uint64(i)] && binary.LittleEndian.Uint16(raw[2*i:]) != b {
			return fmt.Errorf("%s parent guard %d changed", role, i)
		}
	}
	return nil
}

func captureBatches(m *evidence, dir, backend, oracle string, ordinal int, buildInputs map[string]string, baseEnv []string, pin func(string, string) error, execute func(string, []string, ...string) (runner.CommandReceipt, error), loaded func(string) error) error {
	inputDir := filepath.Join(filepath.Dir(backend), "cuda-inputs")
	entries, err := os.ReadDir(inputDir)
	if err != nil {
		return err
	}
	if len(entries) != 37 {
		return fmt.Errorf("bounded corpus requires 37 cases")
	}
	type prepared struct {
		path string
		c    cudafixture.Case
		data map[string][]uint16
	}
	cases := make([]prepared, 0, 37)
	for _, entry := range entries {
		path := filepath.Join(inputDir, entry.Name(), "case.json")
		c, data, err := cudafixture.Load(path)
		if err != nil {
			return err
		}
		if buildInputs[path] == "" {
			return fmt.Errorf("case metadata lacks immutable build pin: %s", path)
		}
		if err = pin(path, buildInputs[path]); err != nil {
			return err
		}
		for _, a := range c.Allocations {
			p := filepath.Join(filepath.Dir(path), a.File)
			if err = pin(p, a.SHA); err != nil {
				return err
			}
		}
		if c.Name != entry.Name() {
			return fmt.Errorf("case name differs from directory")
		}
		cases = append(cases, prepared{path, c, data})
	}
	writePlan := func(label string, mutant bool) (string, error) {
		path := filepath.Join(dir, label+".plan")
		var plan strings.Builder
		for _, saved := range cases {
			c := saved.c
			if mutant && !strings.HasPrefix(c.Name, "precision-") {
				continue
			}
			v := c.Views
			pathFor := func(role string) string {
				return filepath.Join(filepath.Dir(saved.path), c.Allocations[v[role].Allocation].File)
			}
			shared := "0"
			if v["a"].Allocation == v["c"].Allocation {
				shared = "1"
			}
			dims := []int64{1, 1, 1, 1, 1, 1, int64(v["a"].Offset), int64(v["b"].Offset), int64(v["c"].Offset)}
			if c.Mode == "gemm" {
				dims = []int64{v["a"].Shape[0], v["b"].Shape[1], v["a"].Shape[1], v["a"].Strides[0], v["b"].Strides[0], v["c"].Strides[0], int64(v["a"].Offset), int64(v["b"].Offset), int64(v["c"].Offset)}
			}
			args := []string{c.Mode, filepath.Join(dir, c.Name+"-"+label), pathFor("a"), pathFor("b"), pathFor("c"), shared}
			for _, n := range dims {
				args = append(args, strconv.FormatInt(n, 10))
			}
			for i, arg := range args {
				if strings.ContainsAny(arg, "\r\n\t") {
					return "", fmt.Errorf("control character in oracle batch argument")
				}
				if i > 0 {
					plan.WriteByte(' ')
				}
				plan.WriteString("\"" + strings.ReplaceAll(strings.ReplaceAll(arg, "\\", "\\\\"), "\"", "\\\"") + "\"")
			}
			plan.WriteByte('\n')
		}
		if err := os.WriteFile(path, []byte(plan.String()), 0644); err != nil {
			return "", err
		}
		if err := pin(path, ""); err != nil {
			return "", err
		}
		return path, nil
	}
	for _, label := range []string{"source-1", "source-2", "mutant-f32"} {
		plan, err := writePlan(label, label == "mutant-f32")
		if err != nil {
			return err
		}
		endpoint := "source-f16"
		if label == "mutant-f32" {
			endpoint = "mutant-f32"
		}
		r, err := execute(label, baseEnv, oracle, "--batch", plan, endpoint, strconv.Itoa(ordinal))
		if err != nil {
			return err
		}
		if err = loaded(r.Stdout); err != nil {
			return err
		}
	}
	for _, saved := range cases {
		c := saved.c
		first, second := filepath.Join(dir, c.Name+"-source-1"), filepath.Join(dir, c.Name+"-source-2")
		for _, role := range []string{"a", "b", "c"} {
			a, b := first+"-"+role+".u16le", second+"-"+role+".u16le"
			if err = pin(a, ""); err != nil {
				return err
			}
			if err = pin(b, m.Inputs[a]); err != nil {
				return fmt.Errorf("native repeat differs: %w", err)
			}
			if err = checkGuards(c, saved.data, role, a); err != nil {
				return err
			}
		}
		m.Cases = append(m.Cases, savedCase{saved.path, first})
		if strings.HasPrefix(c.Name, "precision-") {
			prefix := filepath.Join(dir, c.Name+"-mutant-f32")
			for _, role := range []string{"a", "b", "c"} {
				if err = pin(prefix+"-"+role+".u16le", ""); err != nil {
					return err
				}
			}
			if m.Inputs[prefix+"-c.u16le"] != m.Inputs[first+"-c.u16le"] {
				m.MutantDifferences++
			}
		}
	}
	return nil
}

func currentConsumer(dir string) (consumerEvidence, error) {
	c := consumerEvidence{Sources: map[string]string{}, TestBinary: filepath.Join(dir, "go-cuda-consumer.test")}
	raw, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return c, err
	}
	c.Head = strings.TrimSpace(string(raw))
	for _, root := range []string{"internal/tensor", "verification/tensor/cuda", "verification/tensor/cudafixture", "verification/tensor/runner"} {
		if err = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			c.Sources[path], err = runner.SHA(path)
			return err
		}); err != nil {
			return c, err
		}
	}
	c.Sources["go.mod"], err = runner.SHA("go.mod")
	return c, err
}
func consumerStable(c consumerEvidence) error {
	current, err := currentConsumer(filepath.Dir(c.TestBinary))
	if err != nil {
		return err
	}
	if current.Head != c.Head || len(current.Sources) != len(c.Sources) {
		return fmt.Errorf("consumer source identity changed during comparison")
	}
	for path, want := range c.Sources {
		if current.Sources[path] != want {
			return fmt.Errorf("consumer source changed during comparison: %s", path)
		}
	}
	if c.TestSHA != "" {
		got, err := runner.SHA(c.TestBinary)
		if err != nil || got != c.TestSHA {
			return fmt.Errorf("consumer executable changed during comparison")
		}
	}
	return nil
}

func matchConsumerModes(path, identity string, minimum int) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	expected := map[string]string{}
	for _, field := range strings.Fields(identity) {
		key, value, ok := strings.Cut(field, "=")
		if ok {
			expected[key] = value
		}
	}
	count := 0
	for {
		var event struct{ Output string }
		if err := dec.Decode(&event); err == io.EOF {
			break
		} else if err != nil {
			return err
		}
		for _, line := range strings.Split(event.Output, "\n") {
			if !strings.Contains(line, "nt_cuda_gemm_f16 ") {
				continue
			}
			actual := map[string]string{}
			for _, field := range strings.Fields(line) {
				key, value, ok := strings.Cut(field, "=")
				if ok {
					actual[key] = value
				}
			}
			for _, key := range []string{"driver", "runtime", "cublas", "ordinal", "uuid", "stream_matches", "pointer_mode", "math_mode", "atomics_mode"} {
				if expected[key] == "" || actual[key] != expected[key] {
					return fmt.Errorf("consumer cuBLAS %s differs from source", key)
				}
			}
			count++
		}
	}
	if count < minimum {
		return fmt.Errorf("consumer reported %d cuBLAS diagnostics; need at least %d", count, minimum)
	}
	return nil
}

func matchCaptureIdentity(prior, current evidence) error {
	if prior.ProducerIdentity == "" || prior.DeviceIdentity == "" || prior.DeviceIdentity != current.DeviceIdentity || prior.Ordinal != current.Ordinal || len(prior.Environment) != len(current.Environment) {
		return fmt.Errorf("saved source GPU, ordinal or environment identity differs")
	}
	for key, value := range prior.Environment {
		if current.Environment[key] != value {
			return fmt.Errorf("saved source runtime environment differs: %s", key)
		}
	}
	return nil
}
