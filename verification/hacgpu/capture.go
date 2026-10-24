package hacgpu

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

// ReceiptSchema identifies an admitted H01 capture attempt.
const ReceiptSchema = "nanopore-hacgpu-stage-capture-receipt/v1"

// CaptureOptions describes one admitted H01 capture.
type CaptureOptions struct {
	Build    string // build manifest written by BuildOracle
	Dorado   string // read-only pinned Dorado checkout
	Model    string // pinned HAC model directory
	Fixture  string // pinned POD5 fixture manifest
	Read     int    // read index inside the fixture
	Signal   string // raw int16 signal file
	Batch    int    // fixed batch size, a multiple of the kernel granularity
	Chunk    int    // fixed chunk length in samples
	Out      string // fresh artifacts directory
	Ordinal  int
	Admitted bool
}

// Receipt is the admitted capture record.
type Receipt struct {
	Schema        string                  `json:"schema"`
	Status        string                  `json:"status"`
	Error         string                  `json:"error,omitempty"`
	Started       string                  `json:"started_utc"`
	Finished      string                  `json:"finished_utc"`
	Case          string                  `json:"case"`
	BuildManifest string                  `json:"build_manifest"`
	Oracle        string                  `json:"oracle"`
	OracleSHA     string                  `json:"oracle_sha256"`
	Model         string                  `json:"model_dir"`
	Fixture       string                  `json:"fixture_manifest"`
	Signal        string                  `json:"signal"`
	SignalSHA     string                  `json:"signal_sha256"`
	Normalisation Normalisation           `json:"normalisation"`
	Input         string                  `json:"input_file"`
	InputSHA      string                  `json:"input_sha256"`
	InputSamples  int                     `json:"input_samples"`
	Batch         int                     `json:"batch"`
	Stride        int                     `json:"stride"`
	TOut          int                     `json:"t_out"`
	StackSHA      map[string]string       `json:"stack_sha256"`
	Ordinal       int                     `json:"ordinal"`
	Device        string                  `json:"device_identity"`
	Stages        string                  `json:"stages"`
	Reservation   int64                   `json:"reservation_bytes"`
	ScoresSHA     string                  `json:"scores_sha256"`
	ScoresShape   []int64                 `json:"scores_shape"`
	Capture       string                  `json:"capture_manifest"`
	Inputs        map[string]string       `json:"input_sha256_all"`
	Runtime       map[string]string       `json:"runtime_sha256"`
	Commands      []runner.CommandReceipt `json:"commands"`
	Environment   map[string]string       `json:"effective_runtime_environment"`
}

// Capture runs the admitted oracle once under the existing stock host lock and
// verifies every recorded byte before it reports success.
func RunCapture(o CaptureOptions) (r Receipt, err error) {
	r = Receipt{
		Schema:      ReceiptSchema,
		Status:      "capture_failed",
		Started:     time.Now().UTC().Format(time.RFC3339Nano),
		Ordinal:     o.Ordinal,
		Inputs:      map[string]string{},
		Runtime:     map[string]string{},
		Environment: map[string]string{},
	}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
		}
	}()
	if !o.Admitted {
		return r, fmt.Errorf("H01 capture requires explicit coordinator admission")
	}
	if o.Chunk <= 0 {
		return r, fmt.Errorf("H01 capture requires a fixed positive chunk")
	}
	if o.Batch <= 0 {
		return r, fmt.Errorf("H01 capture requires a fixed positive batch")
	}
	if o.Batch%64 != 0 {
		// `koi.h` documents `host_factorised_linear` as "N % 64 = 0"; a batch that
		// is not a multiple of 64 returns a non-success status and leaves the
		// output untouched, which is not a graph run.
		return r, fmt.Errorf("H01 batch %d is not a multiple of the 64-wide kernel granularity", o.Batch)
	}
	for _, p := range []*string{&o.Build, &o.Dorado, &o.Model, &o.Fixture, &o.Signal, &o.Out} {
		v, aerr := filepath.Abs(*p)
		if aerr != nil {
			return r, aerr
		}
		*p = v
	}
	if err = runner.CheckRoot(); err != nil {
		return r, err
	}

	// Every declared pin must still match before a single external command runs.
	build, err := loadVerifiedBuild(o.Build)
	if err != nil {
		return r, err
	}
	r.BuildManifest = o.Build
	r.Oracle = build.Output
	r.OracleSHA = build.OutputSHA
	if err = PinStageSources(o.Dorado, r.Inputs); err != nil {
		return r, err
	}
	for path, want := range build.Inputs {
		got, herr := fileSHA(path)
		if herr != nil {
			return r, herr
		}
		if got != want {
			return r, fmt.Errorf("declared build pin changed: %s", path)
		}
		r.Inputs[path] = got
	}

	// The model directory is pinned by its config plus every weight tensor.
	modelPins, err := pinDirectory(o.Model)
	if err != nil {
		return r, err
	}
	for k, v := range modelPins {
		r.Inputs[k] = v
	}
	r.Model = o.Model

	normalisation, err := fixtureNormalisation(o.Fixture, o.Read)
	if err != nil {
		return r, err
	}
	normalisation.Mean, normalisation.Stdev, err = standardisation(o.Model)
	if err != nil {
		return r, err
	}
	r.Normalisation = normalisation
	r.Fixture = o.Fixture
	r.Inputs[o.Fixture], err = fileSHA(o.Fixture)
	if err != nil {
		return r, err
	}

	// Serialise against every other native capture on this host.
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return r, err
	}
	defer lock.Close()

	dir, err := runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	r.Capture = dir

	baseEnv := []string{"LD_LIBRARY_PATH=", "LD_PRELOAD=", "DORADO_MATMUL_FP16_MODE=CUBLAS"}
	execute := func(label string, args ...string) (runner.CommandReceipt, error) {
		rec, rerr := runner.Execute(dir, label, baseEnv, args...)
		r.Commands = append(r.Commands, rec)
		return rec, rerr
	}
	for _, key := range []string{"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG"} {
		value, present := os.LookupEnv(key)
		r.Environment[key] = value
		r.Environment[key+".present"] = strconv.FormatBool(present)
	}
	r.Environment["DORADO_MATMUL_FP16_MODE"] = "CUBLAS"

	if _, err = execute("before-gpu-state", "nvidia-smi", "--query-gpu=index,uuid,driver_version,memory.total,memory.used,temperature.gpu", "--format=csv,noheader,nounits"); err != nil {
		return r, err
	}
	if _, err = execute("before-compute-apps", "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"); err != nil {
		return r, err
	}

	// Normalise the pinned signal with the pinned scaler arithmetic.
	r.Signal = o.Signal
	r.SignalSHA, err = fileSHA(o.Signal)
	if err != nil {
		return r, err
	}
	raw, err := os.ReadFile(o.Signal)
	if err != nil {
		return r, err
	}
	samples, err := DecodeInt16LE(raw)
	if err != nil {
		return r, err
	}
	windowLen := o.Batch*o.Chunk + normalisation.TrimStart
	if windowLen > len(samples) {
		return r, fmt.Errorf("batch %d x chunk %d plus trim %d exceeds %d samples",
			o.Batch, o.Chunk, normalisation.TrimStart, len(samples))
	}
	window := samples[:windowLen]
	bits, err := normalisation.Normalise(window)
	if err != nil {
		return r, err
	}
	if len(bits) != o.Batch*o.Chunk {
		return r, fmt.Errorf("normalised %d samples, want %d", len(bits), o.Batch*o.Chunk)
	}
	r.Input = filepath.Join(dir, "input.f16")
	if err = os.WriteFile(r.Input, EncodeHalfLE(bits), 0o644); err != nil {
		return r, err
	}
	if r.InputSHA, err = fileSHA(r.Input); err != nil {
		return r, err
	}
	r.InputSamples = len(bits)
	inputMeta, err := json.MarshalIndent(map[string]any{
		"schema":        "nanopore-hacgpu-normalised-input/v1",
		"signal":        r.Signal,
		"signal_sha256": r.SignalSHA,
		"normalisation": normalisation,
		"batch":         o.Batch,
		"chunk":         o.Chunk,
		"samples":       len(bits),
		"input_sha256":  r.InputSHA,
	}, "", "  ")
	if err != nil {
		return r, err
	}
	if err = os.WriteFile(filepath.Join(dir, "input.json"), append(inputMeta, '\n'), 0o644); err != nil {
		return r, err
	}

	captureDir := filepath.Join(dir, "stages")
	if err = os.Mkdir(captureDir, 0o755); err != nil {
		return r, err
	}
	run, err := execute("hacgpu-oracle", build.Output,
		"--model-dir", o.Model,
		"--input", r.Input,
		"--out", captureDir,
		"--case", "hac-v6-n"+strconv.Itoa(o.Batch)+"-chunk"+strconv.Itoa(o.Chunk),
		"--ordinal", strconv.Itoa(o.Ordinal),
		"--batch", strconv.Itoa(o.Batch),
		"--chunk", strconv.Itoa(o.Chunk))
	if err != nil {
		return r, err
	}
	_ = run

	capture, err := Import(captureDir)
	if err != nil {
		return r, err
	}
	if capture.Chunk != o.Chunk || capture.Features != 1 || capture.Batch != o.Batch {
		return r, fmt.Errorf("capture geometry %d/%d/%d differs from the declared case", capture.Batch, capture.Features, capture.Chunk)
	}
	// The factorised projection is tiled 32 rows of time at a time; a T_out that
	// is not a multiple of 32 does not complete (observed: T_out 48 never
	// returns). Production T_out values are 2048 and 1024, so the declared case
	// exercises the same tile path.
	if capture.TOut%32 != 0 {
		return r, fmt.Errorf("capture T_out %d is not a multiple of the 32-row projection tile", capture.TOut)
	}
	if capture.Stride <= 0 || capture.TOut != capture.Chunk/capture.Stride {
		return r, fmt.Errorf("capture stride %d and T_out %d are inconsistent with chunk %d", capture.Stride, capture.TOut, capture.Chunk)
	}
	// The native config loader and the Go config reader must agree on the
	// standardisation the input was prepared with, or the capture is not the
	// input the graph saw.
	if float64(capture.StdMean) != normalisation.Mean || float64(capture.StdStdev) != normalisation.Stdev {
		return r, fmt.Errorf("model standardisation %v/%v differs from the prepared %v/%v",
			capture.StdMean, capture.StdStdev, normalisation.Mean, normalisation.Stdev)
	}
	r.Stages = capture.StageList()
	r.Batch = capture.Batch
	r.Stride = capture.Stride
	r.TOut = capture.TOut
	r.StackSHA = capture.StackSHA
	r.Reservation = capture.Reservation
	r.ScoresSHA = capture.ScoresSHA
	r.ScoresShape = capture.ScoresShape

	// Pin every produced byte and the runtime libraries the oracle linked.
	for _, name := range []string{"capture.json"} {
		path := filepath.Join(captureDir, name)
		if r.Inputs[path], err = fileSHA(path); err != nil {
			return r, err
		}
	}
	for _, s := range capture.Stages {
		for _, name := range []string{s.DataFile, s.BackingFile} {
			path := filepath.Join(captureDir, name)
			if r.Inputs[path], err = fileSHA(path); err != nil {
				return r, err
			}
		}
	}
	// The producer reported the shared objects it actually mapped. Hash those
	// exact files: a link-line soname is not evidence that a given file ran.
	observed, err := observedRuntime(captureDir)
	if err != nil {
		return r, err
	}
	for _, path := range observed {
		got, rerr := fileSHA(path)
		if rerr != nil {
			return r, rerr
		}
		r.Runtime[path] = got
	}

	if _, err = execute("after-compute-apps", "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"); err != nil {
		return r, err
	}
	state, err := execute("after-gpu-state", "nvidia-smi", "--query-gpu=index,uuid,driver_version,memory.total,memory.used,temperature.gpu", "--format=csv,noheader,nounits")
	if err != nil {
		return r, err
	}
	body, err := os.ReadFile(state.Stdout)
	if err != nil {
		return r, err
	}
	r.Device = strings.TrimSpace(string(body))

	rawReceipt, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return r, err
	}
	r.Status = "stage_capture_passed"
	rawReceipt, err = json.MarshalIndent(r, "", "  ")
	if err != nil {
		return r, err
	}
	if err = os.WriteFile(filepath.Join(dir, "receipt.json"), append(rawReceipt, '\n'), 0o644); err != nil {
		return r, err
	}
	return r, nil
}

func loadVerifiedBuild(path string) (BuildManifest, error) {
	var b BuildManifest
	raw, err := os.ReadFile(path)
	if err != nil {
		return b, err
	}
	if err = json.Unmarshal(raw, &b); err != nil {
		return b, err
	}
	if b.Schema != "nanopore-hacgpu-oracle-build/v1" || b.Status != "built" || b.ABI != 4 {
		return b, fmt.Errorf("build manifest is not a qualified ABI4 oracle build")
	}
	got, err := fileSHA(b.Output)
	if err != nil {
		return b, err
	}
	if got != b.OutputSHA {
		return b, fmt.Errorf("oracle binary changed since the build manifest")
	}
	return b, nil
}

func pinDirectory(dir string) (map[string]string, error) {
	pins := map[string]string{}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	if len(entries) == 0 {
		return nil, fmt.Errorf("model directory %s is empty", dir)
	}
	for _, e := range entries {
		if e.IsDir() {
			return nil, fmt.Errorf("model directory has an unexpected subdirectory %s", e.Name())
		}
		path := filepath.Join(dir, e.Name())
		got, herr := fileSHA(path)
		if herr != nil {
			return nil, herr
		}
		pins[path] = got
	}
	if _, ok := pins[filepath.Join(dir, "config.toml")]; !ok {
		return nil, fmt.Errorf("model directory %s has no config.toml", dir)
	}
	return pins, nil
}

// fixtureNormalisation reads the pinned POD5 fixture manifest and derives the
// scaler parameters for one read. The HAC v6 config supplies mean and stdev.
func fixtureNormalisation(path string, read int) (Normalisation, error) {
	var n Normalisation
	raw, err := os.ReadFile(path)
	if err != nil {
		return n, err
	}
	// `verification/pod5/generate.py:22-23` stores calibration values as the
	// uint32 IEEE-754 bit pattern (`struct.unpack("<I", struct.pack("<f", v))`),
	// not as JSON numbers. Decoding them as floats would silently produce a
	// ~1e9 calibration scale, so the type here is explicitly the bit pattern.
	var m struct {
		Fixtures []struct {
			Reads []struct {
				FloatBits map[string]uint32 `json:"float_bits"`
			} `json:"reads"`
		} `json:"fixtures"`
	}
	if err = json.Unmarshal(raw, &m); err != nil {
		return n, err
	}
	total := 0
	var chosen map[string]uint32
	for _, f := range m.Fixtures {
		for _, r := range f.Reads {
			if total == read {
				chosen = r.FloatBits
			}
			total++
		}
	}
	if chosen == nil {
		return n, fmt.Errorf("fixture manifest has no read %d", read)
	}
	n.Scaling = float64(math.Float32frombits(chosen["CalibrationScale"]))
	n.Offset = float64(math.Float32frombits(chosen["CalibrationOffset"]))
	n.TrimStart = 10
	if n.Scaling == 0 {
		return n, fmt.Errorf("fixture read %d has no calibration scale", read)
	}
	if !finite(n.Scaling) || !finite(n.Offset) {
		return n, fmt.Errorf("fixture read %d has a non-finite calibration", read)
	}
	return n, nil
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// standardisation reads `mean` and `stdev` from the pinned model config's
// `[standardisation]` section. They are not hardcoded: a different pinned model
// must change the normalisation or fail, never silently reuse HAC's values.
func standardisation(modelDir string) (mean, stdev float64, err error) {
	raw, err := os.ReadFile(filepath.Join(modelDir, "config.toml"))
	if err != nil {
		return 0, 0, err
	}
	section := ""
	seenMean, seenStdev := false, false
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "[") {
			section = strings.Trim(line, "[]")
			continue
		}
		if section != "standardisation" {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		if key != "mean" && key != "stdev" {
			continue
		}
		parsed, perr := strconv.ParseFloat(strings.TrimSpace(value), 64)
		if perr != nil {
			return 0, 0, fmt.Errorf("model config %s is not a number: %w", key, perr)
		}
		if key == "mean" {
			mean, seenMean = parsed, true
		} else {
			stdev, seenStdev = parsed, true
		}
	}
	if !seenMean || !seenStdev {
		return 0, 0, fmt.Errorf("model config has no [standardisation] mean/stdev")
	}
	if stdev <= 0 {
		return 0, 0, fmt.Errorf("model config stdev %v is not positive", stdev)
	}
	return mean, stdev, nil
}

// observedRuntime reads the mapped-object list the oracle dumped and requires the
// families that prove the run was the pinned CUDA13 stack on a real driver.
func observedRuntime(captureDir string) ([]string, error) {
	raw, err := os.ReadFile(filepath.Join(captureDir, "runtime-paths.json"))
	if err != nil {
		return nil, err
	}
	var report struct {
		Schema string   `json:"schema"`
		Paths  []string `json:"paths"`
	}
	if err = json.Unmarshal(raw, &report); err != nil {
		return nil, err
	}
	if report.Schema != "nanopore-hacgpu-runtime-paths/v1" {
		return nil, fmt.Errorf("runtime path report schema %q is not the expected one", report.Schema)
	}
	if len(report.Paths) < 8 {
		return nil, fmt.Errorf("runtime path report lists only %d objects", len(report.Paths))
	}
	// Prefixes, because the mapped sonames carry versions (`libcublas.so.13`,
	// `libcuda.so.615.78.08`).
	require := map[string]bool{
		"libtorch_cpu.so": false, "libc10.so": false, "libcudart.so": false,
		"libcublas.so": false, "libcuda.so": false,
	}
	for _, path := range report.Paths {
		if strings.Contains(path, "CUDA-12") || strings.Contains(path, "cuda-12") {
			return nil, fmt.Errorf("foreign CUDA12 object mapped: %s", path)
		}
		name := filepath.Base(path)
		for want := range require {
			if strings.HasPrefix(name, want) {
				require[want] = true
			}
		}
	}
	for want, ok := range require {
		if !ok {
			return nil, fmt.Errorf("oracle did not map the required runtime %s", want)
		}
	}
	return report.Paths, nil
}
