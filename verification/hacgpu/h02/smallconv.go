// Package h02 is the H02 differential consumer: it replays the two HAC
// small convolutions through the Go caller and compares every written byte with
// the admitted H01 native stage capture.
//
// It is verification-only. It is not a producer, it is not part of the caller,
// and it never runs without explicit coordinator admission. The Go caller it
// exercises is internal/model/hac_gpu_smallconv.go; the only native work is the
// single host_convolution_f16 call behind the typed boundary.
package h02

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

// ReceiptSchema identifies one H02 differential attempt.
const ReceiptSchema = "nanopore-hacgpu-smallconv-receipt/v1"

// FillPattern is the byte every unwritten element of the declared backing holds
// before the first kernel runs. It is declared here, before execution, so an
// unwritten region can be proven untouched instead of being compared.
const FillPattern = 0x5a

// Options names the admitted capture and the ABI4 backend that will replay it.
type Options struct {
	Capture  string
	Model    string
	Backend  string
	Out      string
	Ordinal  int
	Admitted bool
}

// ConvSpec is one `[[encoder.sublayers]] type = "convolution"` block of the
// pinned model config, exactly as declared.
type ConvSpec struct {
	Index          int
	CIn, COut      int64
	WinLen, Stride int64
	Padding        int64
	Activation     string
}

// LayerPlan is the recorded plan of one convolution.
type LayerPlan struct {
	Index              int     `json:"index"`
	CIn                int64   `json:"c_in"`
	COut               int64   `json:"c_out"`
	WinLen             int64   `json:"win_len"`
	Stride             int64   `json:"stride"`
	Padding            int64   `json:"padding"`
	Activation         string  `json:"activation"`
	Layout             string  `json:"layout"`
	OutputTPadding     int64   `json:"output_t_padding"`
	TIn                int64   `json:"t_in"`
	TOut               int64   `json:"t_out"`
	OffsetBytes        int64   `json:"offset_bytes"`
	Bytes              int64   `json:"bytes"`
	CropOffsetElements uint64  `json:"crop_offset_elements"`
	CropShape          []int64 `json:"crop_shape"`
	CropStrides        []int64 `json:"crop_strides"`
	BatchStride        int64   `json:"out_n_stride"`
	Executes           bool    `json:"executes"`
	CaptureOffsetBytes int64   `json:"capture_offset_bytes"`
	CaptureShape       []int64 `json:"capture_shape"`
	CaptureLayout      string  `json:"capture_layout"`
}

// Packing records the weight-packing differential for one layer.
type Packing struct {
	Index            int     `json:"index"`
	WeightFileSHA    string  `json:"weight_file_sha256"`
	WeightShape      []int64 `json:"weight_shape"`
	BiasFileSHA      string  `json:"bias_file_sha256"`
	GoWeightSHA      string  `json:"go_packed_weight_sha256"`
	ReferenceSHA     string  `json:"reference_packed_weight_sha256"`
	GoBiasSHA        string  `json:"go_packed_bias_sha256"`
	ReferenceBiasSHA string  `json:"reference_packed_bias_sha256"`
	Equal            bool    `json:"go_equals_reference"`
}

// Stage records one stage comparison.
type Stage struct {
	Name          string   `json:"name"`
	CaptureSHA    string   `json:"capture_payload_sha256"`
	GoSHA         string   `json:"go_payload_sha256"`
	Shape         []int64  `json:"shape"`
	WrittenBytes  int64    `json:"compared_bytes"`
	GoWrittenSHA  string   `json:"go_written_sha256"`
	CapWrittenSHA string   `json:"capture_written_sha256"`
	UnwrittenRows [][2]int `json:"declared_unwritten_t"`
	Equal         bool     `json:"equal"`
}

// Padding records that the rows outside the crop still hold the declared fill.
type Padding struct {
	Index       int   `json:"index"`
	Rows        int64 `json:"rows"`
	Bytes       int64 `json:"bytes"`
	AllFill     bool  `json:"all_fill_pattern"`
	ExampleByte uint8 `json:"observed_byte"`
}

// Lifetime records the view-lifetime proof.
type Lifetime struct {
	ClosedViews      []string `json:"closed_views"`
	ParentStillEqual bool     `json:"parent_still_equal_after_view_close"`
	BackingStillFill bool     `json:"unrelated_backing_still_fill_pattern"`
}

// Receipt is the admitted H02 differential record.
type Receipt struct {
	Schema              string            `json:"schema"`
	Status              string            `json:"status"`
	Error               string            `json:"error,omitempty"`
	Started             string            `json:"started_utc"`
	Finished            string            `json:"finished_utc"`
	Attempt             string            `json:"attempt"`
	Case                string            `json:"case"`
	Capture             string            `json:"capture_dir"`
	CaptureJSON         string            `json:"capture_manifest_sha256"`
	CaptureStages       string            `json:"capture_stage_list"`
	Model               string            `json:"model_dir"`
	Backend             string            `json:"backend"`
	BackendSHA          string            `json:"backend_sha256"`
	BuildManifest       string            `json:"backend_build_manifest"`
	BuildSHA            string            `json:"backend_build_manifest_sha256"`
	ABI                 int               `json:"abi_version"`
	TorchRoot           string            `json:"torch_root"`
	Batch               int               `json:"batch"`
	Chunk               int               `json:"chunk"`
	Features            int64             `json:"features"`
	Reservation         int64             `json:"reservation_bytes"`
	FillPattern         uint8             `json:"fill_pattern"`
	DeclaredPreparation []string          `json:"declared_preparation"`
	Ordinal             int               `json:"cuda_ordinal"`
	Device              string            `json:"device_bookend_identity"`
	Environment         map[string]string `json:"effective_runtime_environment"`
	Plan                []LayerPlan       `json:"plan"`
	Packing             []Packing         `json:"packing"`
	Stages              []Stage           `json:"stages"`
	Padding             []Padding         `json:"padding"`
	Lifetime            Lifetime          `json:"lifetime"`
	Inputs              map[string]string `json:"input_sha256"`
	// BackendInputs and BackendInputsSHA record that every declared backend
	// build input still matched on disk at replay time. The map itself lives in
	// the pinned build manifest; only the count and a digest of the verified set
	// are repeated here.
	BackendInputs    int                     `json:"backend_build_inputs_verified"`
	BackendInputsSHA string                  `json:"backend_build_inputs_sha256"`
	Runtime          map[string]string       `json:"runtime_sha256"`
	Commands         []runner.CommandReceipt `json:"commands"`
}

// Run replays the declared case once and returns the verified receipt.
func Run(o Options) (r Receipt, err error) {
	r = Receipt{
		Schema:      ReceiptSchema,
		Status:      "failed",
		Started:     time.Now().UTC().Format(time.RFC3339Nano),
		Ordinal:     o.Ordinal,
		FillPattern: FillPattern,
		Inputs:      map[string]string{},
		Runtime:     map[string]string{},
		Environment: map[string]string{},
	}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
		}
		if r.Attempt == "" {
			return
		}
		raw, merr := json.MarshalIndent(r, "", "  ")
		if merr == nil {
			merr = os.WriteFile(filepath.Join(r.Attempt, "receipt.json"), append(raw, '\n'), 0o644)
		}
		if merr != nil && err == nil {
			err = merr
		}
	}()
	if !o.Admitted {
		return r, fmt.Errorf("H02 replay requires explicit coordinator admission")
	}
	if o.Ordinal < 0 || o.Ordinal > math.MaxInt32 {
		return r, fmt.Errorf("invalid CUDA ordinal %d", o.Ordinal)
	}
	for _, p := range []*string{&o.Capture, &o.Model, &o.Backend, &o.Out} {
		v, aerr := filepath.Abs(*p)
		if aerr != nil {
			return r, aerr
		}
		*p = v
	}
	if err = runner.CheckRoot(); err != nil {
		return r, err
	}

	// The backend must be the qualified ABI4 CUDA13 build, byte for byte.
	build, buildPath, err := verifyBackend(o.Backend)
	if err != nil {
		return r, err
	}
	r.Backend, r.BackendSHA = o.Backend, build.OutputSHA
	r.BuildManifest, r.BuildSHA = buildPath, build.ManifestSHA
	r.ABI, r.TorchRoot = build.ABI, build.Torch
	paths := make([]string, 0, len(build.Inputs))
	for path := range build.Inputs {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	var pins strings.Builder
	for _, path := range paths {
		got, herr := fileSHA(path)
		if herr != nil {
			return r, herr
		}
		if got != build.Inputs[path] {
			return r, fmt.Errorf("declared backend build pin changed: %s", path)
		}
		fmt.Fprintf(&pins, "%s %s\n", path, got)
	}
	r.BackendInputs = len(paths)
	r.BackendInputsSHA = hashString(pins.String())

	// The capture is the oracle. Import verifies every recorded byte.
	capture, err := hacgpu.Import(o.Capture)
	if err != nil {
		return r, fmt.Errorf("capture %s: %w", o.Capture, err)
	}
	r.Capture = o.Capture
	r.Case = capture.Case
	r.CaptureStages = capture.StageList()
	r.Batch, r.Chunk, r.Features = capture.Batch, capture.Chunk, int64(capture.Features)
	r.Reservation = capture.Reservation
	r.CaptureJSON, err = fileSHA(filepath.Join(o.Capture, "capture.json"))
	if err != nil {
		return r, err
	}
	r.Inputs[filepath.Join(o.Capture, "capture.json")] = r.CaptureJSON
	for _, s := range capture.Stages {
		for _, name := range []string{s.DataFile, s.BackingFile} {
			path := filepath.Join(o.Capture, name)
			if r.Inputs[path], err = fileSHA(path); err != nil {
				return r, err
			}
		}
	}

	// The conv stack is declared by the pinned config, never assumed.
	convs, err := ModelConvs(o.Model)
	if err != nil {
		return r, err
	}
	if len(convs) != capture.Convs {
		return r, fmt.Errorf("model declares %d convolutions but the capture has %d", len(convs), capture.Convs)
	}
	params, err := convParams(convs)
	if err != nil {
		return r, err
	}
	plan, err := model.PlanHACSmallConvs(int64(capture.Batch), int64(capture.Chunk), int64(capture.Features), capture.Reservation, params)
	if err != nil {
		return r, err
	}
	r.Plan, err = recordPlan(plan, capture, convs)
	if err != nil {
		return r, err
	}
	r.Model = o.Model

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
	r.Attempt = dir
	execute := func(label string, args ...string) (runner.CommandReceipt, error) {
		rec, rerr := runner.Execute(dir, label, []string{"LD_LIBRARY_PATH=", "LD_PRELOAD=", "DORADO_MATMUL_FP16_MODE=CUBLAS"}, args...)
		r.Commands = append(r.Commands, rec)
		return rec, rerr
	}
	for _, key := range []string{"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG"} {
		value, present := os.LookupEnv(key)
		r.Environment[key] = value
		r.Environment[key+".present"] = strconv.FormatBool(present)
	}
	r.Environment["DORADO_MATMUL_FP16_MODE"] = "CUBLAS"
	if _, err = execute("before-gpu-state", "nvidia-smi",
		"--query-gpu=index,uuid,driver_version,memory.total,memory.used,temperature.gpu",
		"--format=csv,noheader,nounits"); err != nil {
		return r, err
	}

	if err = replay(dir, &r, plan, convs, capture); err != nil {
		return r, err
	}

	r.Runtime, err = runtimePins()
	if err != nil {
		return r, err
	}

	state, err := execute("after-gpu-state", "nvidia-smi",
		"--query-gpu=index,uuid,driver_version,memory.total,memory.used,temperature.gpu",
		"--format=csv,noheader,nounits")
	if err != nil {
		return r, err
	}
	body, err := os.ReadFile(state.Stdout)
	if err != nil {
		return r, err
	}
	r.Device = strings.TrimSpace(string(body))

	r.Status = "smallconv_passed"
	return r, nil
}
