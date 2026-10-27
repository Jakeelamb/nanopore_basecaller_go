// Package h04 verifies FLSTM weight caches. Replay authenticates retained
// witnesses. VerifyConsumer loads pinned archives through the native CPU loader
// and runs the Go packer against retained native bytes. Run also invokes the
// native producer and requires separate coordinator admission.
package h04

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
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
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h02"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

// ReceiptSchema identifies one H04 differential attempt.
const ReceiptSchema = "nanopore-hacgpu-flstm-weights-receipt/v1"

// OracleSource is the translation unit this card's oracle must have been built
// from. The build manifest records the absolute path, so only the base name is
// checked here; the digest comes from the manifest.
const OracleSource = "flstm_cache_oracle.cpp"

// Options names the pinned oracle build, the ABI4 backend, the model, the
// declared case input and a fresh attempt directory.
type Options struct {
	Build    string
	Backend  string
	Model    string
	Input    string
	Out      string
	Ordinal  int
	Batch    int
	Chunk    int
	Admitted bool
}

// TensorCompare is one published tensor's differential.
type TensorCompare struct {
	Role          string  `json:"role"`
	DType         string  `json:"dtype"`
	Shape         []int64 `json:"shape"`
	Bytes         int64   `json:"bytes"`
	GoSHA256      string  `json:"go_sha256"`
	NativeSHA     string  `json:"native_sha256"`
	Equal         bool    `json:"equal"`
	GoPayload     string  `json:"go_payload,omitempty"`
	NativePayload string  `json:"native_payload,omitempty"`
}

// QuantDomain is the measured quantization domain of one layer's intermediate
// scales, recomputed in Go and compared with the capture's own measurement.
type QuantDomain struct {
	ScaleMin        float32 `json:"scale_min"`
	ScaleMax        float32 `json:"scale_max"`
	ZeroRows        int     `json:"zero_scale_rows"`
	NonfiniteRows   int     `json:"nonfinite_scale_rows"`
	CaptureScaleMin float32 `json:"capture_scale_min"`
	CaptureScaleMax float32 `json:"capture_scale_max"`
	Matches         bool    `json:"matches_capture"`
}

// LayerResult is one recurrent layer's differential.
type LayerResult struct {
	Index      int             `json:"index"`
	Reverse    bool            `json:"reverse"`
	WeightBase string          `json:"weight_prefix"`
	WeightPins []WeightPin     `json:"weight_pins"`
	Tensors    []TensorCompare `json:"tensors"`
	Domain     QuantDomain     `json:"quantization_domain"`
	Equal      bool            `json:"equal"`
}

// WeightPin is one loaded parameter file and the manifest pin it matched.
type WeightPin struct {
	Name   string  `json:"name"`
	SHA256 string  `json:"sha256"`
	Shape  []int64 `json:"shape"`
}

// Control is one deliberate deviation from the source's construction. The native
// cache is the discriminator: a control that reproduces it is a failed control.
type Control struct {
	Mutant          string          `json:"mutant"`
	Outcome         string          `json:"outcome"`
	RolesDiffering  []string        `json:"roles_differing,omitempty"`
	LayersDiffering int             `json:"layers_differing"`
	Error           string          `json:"error,omitempty"`
	LayersRejected  int             `json:"layers_rejected,omitempty"`
	Comparisons     []MutantCompare `json:"comparisons,omitempty"`
}

type MutantCompare struct {
	Layer  int           `json:"layer"`
	Tensor TensorCompare `json:"tensor"`
}

// Receipt is the admitted H04 differential record.
type Receipt struct {
	Schema        string                       `json:"schema"`
	Status        string                       `json:"status"`
	Error         string                       `json:"error,omitempty"`
	Started       string                       `json:"started_utc"`
	Finished      string                       `json:"finished_utc"`
	Attempt       string                       `json:"attempt"`
	Case          string                       `json:"case"`
	Oracle        string                       `json:"oracle"`
	OracleSHA     string                       `json:"oracle_sha256"`
	OracleSource  string                       `json:"oracle_source"`
	OracleSrcSHA  string                       `json:"oracle_source_sha256"`
	BuildManifest string                       `json:"oracle_build_manifest"`
	BuildSHA      string                       `json:"oracle_build_manifest_sha256"`
	ABI           int                          `json:"abi_version"`
	OracleInputs  int                          `json:"oracle_build_inputs_verified"`
	OracleInSHA   string                       `json:"oracle_build_inputs_sha256"`
	Backend       string                       `json:"backend"`
	BackendSHA    string                       `json:"backend_sha256"`
	BackendBuild  string                       `json:"backend_build_manifest"`
	BackendBSHA   string                       `json:"backend_build_manifest_sha256"`
	TorchRoot     string                       `json:"torch_root"`
	BackendInputs int                          `json:"backend_build_inputs_verified"`
	BackendInSHA  string                       `json:"backend_build_inputs_sha256"`
	Model         string                       `json:"model_dir"`
	ModelPackage  string                       `json:"model_package"`
	ModelPins     string                       `json:"model_weight_manifest_sha256"`
	Capture       string                       `json:"native_capture_dir"`
	CaptureJSON   string                       `json:"native_capture_manifest_sha256"`
	CaptureInput  string                       `json:"capture_input_sha256"`
	Input         string                       `json:"input"`
	InputSHA      string                       `json:"input_sha256"`
	Ordinal       int                          `json:"cuda_ordinal"`
	Batch         int                          `json:"batch"`
	Chunk         int                          `json:"chunk"`
	Features      int                          `json:"features"`
	LSTMLayers    int                          `json:"lstm_layers"`
	C             int64                        `json:"C"`
	K             int64                        `json:"K"`
	FirstReverse  bool                         `json:"first_reverse"`
	Reservation   int64                        `json:"reservation_bytes"`
	Quantize      hacgpu.FLSTMQuantizeContract `json:"quantize"`
	Declared      []string                     `json:"declared_rules"`
	Layers        []LayerResult                `json:"layers"`
	Controls      []Control                    `json:"controls"`
	Device        string                       `json:"device_bookend_identity"`
	Environment   map[string]string            `json:"effective_runtime_environment"`
	Inputs        map[string]string            `json:"input_files_sha256"`
	Runtime       map[string]string            `json:"runtime_sha256"`
	Commands      []runner.CommandReceipt      `json:"commands"`
	ObservedDir   string                       `json:"observed_go_payload_dir,omitempty"`
}

// declaredRules are the source facts the differential relies on, named before
// execution so a reader can check each one against the pinned revision.
var declaredRules = []string{
	"quantize_tensor(weight.to(f16), dim=1): half cast before the reduction (FLSTMStack.cpp:168-171, tensor_utils.cpp:291-297)",
	"down caches are scaled.t.view({-1,2,2,2,2,2,C}).permute({2,0,4,1,3,5,6}).contiguous().view({K,C}) (FLSTMStack.cpp:172-176)",
	"down scales are scaled.scale.view({-1,2,2,2,2,2}).permute({2,0,4,1,3,5}).contiguous().view({K}).to(f16) (FLSTMStack.cpp:177-180)",
	"up weights are quantize_tensor(cat({up_weight_ih, up_weight_hh}, 1), 1) then .view({4,-1,2,2,2,2,2K}).permute({1,5,0,2,3,4,6}).contiguous().view({4C,2K}) (FLSTMStack.cpp:202-209)",
	"up scale and bias use .view({4,-1,2,2,2,2}).permute({1,5,2,3,4,0}).contiguous() (FLSTMStack.cpp:210-217)",
	"up_bias_hh is never packed; the Koi path reads only up_bias_ih (FLSTMStack.cpp:212-216)",
	"first_reverse is the literal true at CRFModel.cpp:38 and each layer is first_reverse ^ (layer & 1) (FLSTMStack.cpp:163)",
}

// Run performs one admitted native capture and differential and returns the
// verified receipt.
func Run(o Options) (r Receipt, err error) {
	r = Receipt{
		Schema: ReceiptSchema, Status: "failed",
		Started: time.Now().UTC().Format(time.RFC3339Nano),
		Ordinal: o.Ordinal, Declared: declaredRules,
		Inputs: map[string]string{}, Runtime: map[string]string{},
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
		return r, fmt.Errorf("H04 differential requires explicit coordinator admission")
	}
	if o.Ordinal < 0 || o.Ordinal > math.MaxInt32 {
		return r, fmt.Errorf("invalid CUDA ordinal %d", o.Ordinal)
	}
	if o.Batch <= 0 || o.Chunk <= 0 {
		return r, fmt.Errorf("the declared case needs a positive batch and chunk, got %d/%d", o.Batch, o.Chunk)
	}
	for _, p := range []*string{&o.Build, &o.Backend, &o.Model, &o.Input, &o.Out} {
		v, aerr := filepath.Abs(*p)
		if aerr != nil {
			return r, aerr
		}
		*p = v
	}
	if err = runner.CheckRoot(); err != nil {
		return r, err
	}

	build, err := hacgpu.VerifyBuild(o.Build)
	if err != nil {
		return r, fmt.Errorf("oracle build %s: %w", o.Build, err)
	}
	source, sourceSHA := build.Source()
	if filepath.Base(source) != OracleSource {
		return r, fmt.Errorf("oracle build %s was not built from %s", o.Build, OracleSource)
	}
	r.Oracle, r.OracleSHA = build.Manifest.Output, build.Manifest.OutputSHA
	r.OracleSource, r.OracleSrcSHA = source, sourceSHA
	r.BuildManifest, r.BuildSHA = build.Path, build.ManifestSHA
	r.ABI, r.OracleInputs, r.OracleInSHA = build.Manifest.ABI, build.Inputs, build.InputsSHA

	backend, err := hacgpu.VerifyBackend(o.Backend)
	if err != nil {
		return r, fmt.Errorf("backend %s: %w", o.Backend, err)
	}
	r.Backend, r.BackendSHA = o.Backend, backend.OutputSHA
	r.BackendBuild, r.BackendBSHA = backend.Path, backend.ManifestSHA
	r.TorchRoot = backend.Torch
	r.BackendInputs, r.BackendInSHA = backend.InputCount, backend.InputsSHA

	geom, err := ModelFLSTMConfig(o.Model)
	if err != nil {
		return r, err
	}
	r.Model = o.Model
	r.Features, r.LSTMLayers = geom.Features, geom.Layers
	r.Batch, r.Chunk = o.Batch, o.Chunk
	r.C, r.K = int64(geom.Size), int64(geom.InnerDim)

	inputSHA, err := runner.SHA(o.Input)
	if err != nil {
		return r, err
	}
	r.Input, r.InputSHA = o.Input, inputSHA
	r.Inputs[o.Input] = inputSHA

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
	native := filepath.Join(dir, "native")
	if err = os.Mkdir(native, 0o755); err != nil {
		return r, err
	}
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

	r.Case = fmt.Sprintf("hac-v6-n%d-chunk%d", o.Batch, o.Chunk)
	if _, err = execute("flstm-cache-oracle", build.Manifest.Output,
		"--model-dir", o.Model,
		"--input", o.Input,
		"--out", native,
		"--case", r.Case,
		"--ordinal", strconv.Itoa(o.Ordinal),
		"--batch", strconv.Itoa(o.Batch),
		"--chunk", strconv.Itoa(o.Chunk),
	); err != nil {
		return r, fmt.Errorf("native cache oracle: %w", err)
	}

	captured, err := hacgpu.ImportFLSTMCaches(native)
	if err != nil {
		return r, fmt.Errorf("native cache capture: %w", err)
	}
	r.Capture, r.CaptureInput = native, captured.InputSHA256
	r.CaptureJSON, err = runner.SHA(filepath.Join(native, "caches.json"))
	if err != nil {
		return r, err
	}
	r.Inputs[filepath.Join(native, "caches.json")] = r.CaptureJSON
	r.ModelPackage, r.Reservation, r.FirstReverse = captured.ModelPackage, captured.Reservation, captured.FirstReverse
	r.Quantize = captured.Quantize
	if err = crossCheck(&r, captured, geom); err != nil {
		return r, err
	}

	if r.Layers, err = differential(&r, captured, o.Model); err != nil {
		return r, err
	}
	if r.Controls, err = controls(&r, captured); err != nil {
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

	r.Status = "flstm_weights_passed"
	return r, nil
}

// crossCheck requires the capture to describe the case and the model this attempt
// declared, before any comparison runs.
func crossCheck(r *Receipt, c hacgpu.FLSTMCaches, geom ModelFLSTM) error {
	if c.ModelDir != r.Model {
		return fmt.Errorf("capture model dir %q is not %q", c.ModelDir, r.Model)
	}
	if c.Case != r.Case {
		return fmt.Errorf("capture case %q is not %q", c.Case, r.Case)
	}
	if c.Ordinal != r.Ordinal || c.Batch != r.Batch || c.Chunk != r.Chunk || c.Features != r.Features {
		return fmt.Errorf("capture case geometry ordinal %d batch %d chunk %d features %d is not the declared %d/%d/%d/%d",
			c.Ordinal, c.Batch, c.Chunk, c.Features, r.Ordinal, r.Batch, r.Chunk, r.Features)
	}
	if c.InputSHA256 != r.InputSHA {
		return fmt.Errorf("capture input %s is not the declared input %s", c.InputSHA256, r.InputSHA)
	}
	if c.LSTMLayers != geom.Layers || c.C != int64(geom.Size) || c.K != int64(geom.InnerDim) {
		return fmt.Errorf("capture geometry %d layers C=%d K=%d is not the model's %d/%d/%d",
			c.LSTMLayers, c.C, c.K, geom.Layers, geom.Size, geom.InnerDim)
	}
	// `CRFModel.cpp:38` passes `first_reverse = true` as a literal. A capture
	// that recorded anything else is not this graph.
	if !c.FirstReverse {
		return fmt.Errorf("capture recorded first_reverse false, but CRFModel.cpp:38 passes true")
	}
	for i, layer := range c.Layers {
		if want := c.FirstReverse != (i%2 == 1); layer.Reverse != want {
			return fmt.Errorf("capture layer %d reverse %v is not first_reverse ^ (layer & 1) = %v",
				i, layer.Reverse, want)
		}
	}
	return nil
}

// differential loads archives through the native CPU tensor loader, then
// quantizes and packs in Go and compares all fourteen tensors with the capture.
func differential(r *Receipt, c hacgpu.FLSTMCaches, modelDir string) ([]LayerResult, error) {
	e, err := tensor.Open(r.Backend)
	if err != nil {
		return nil, err
	}
	defer e.Close()
	if r.ObservedDir != "" {
		if err := os.MkdirAll(r.ObservedDir, 0755); err != nil {
			return nil, err
		}
		if r.Runtime, err = consumerRuntimePins(r.Backend); err != nil {
			return nil, err
		}
	}

	manifestSHA, err := runner.SHA(h02.WeightManifestPath)
	if err != nil {
		return nil, err
	}
	r.ModelPins = manifestSHA
	r.Inputs[h02.WeightManifestPath] = manifestSHA
	manifest, err := readWeightManifest(h02.WeightManifestPath)
	if err != nil {
		return nil, err
	}
	prefixes, err := rnnPrefixes(manifest, c.LSTMLayers)
	if err != nil {
		return nil, err
	}

	params := model.FLSTMParams{C: c.C, K: c.K}
	sets := make([]model.FLSTMWeightSet, len(prefixes))
	out := make([]LayerResult, len(prefixes))
	for i, prefix := range prefixes {
		set, pins, err := loadLayerWeights(e, modelDir, manifest, prefix, params)
		if err != nil {
			return nil, fmt.Errorf("layer %d (%s): %w", i, prefix, err)
		}
		sets[i] = set
		out[i] = LayerResult{Index: i, Reverse: c.FirstReverse != (i%2 == 1), WeightBase: prefix, WeightPins: pins}
	}

	caches, err := model.PackFLSTMStack(params, sets)
	if err != nil {
		return nil, err
	}
	for i := range sets {
		q, err := model.QuantizeFLSTMLayer(params, sets[i])
		if err != nil {
			return nil, fmt.Errorf("layer %d: %w", i, err)
		}
		packed := map[string]struct {
			bytes []byte
			dtype string
			shape []int64
		}{
			"dn-ih-quantized":   {encodeI8(q.DownIH), "Char", []int64{params.K, params.C}},
			"dn-ih-quant-scale": {encodeF32(q.DownScaleIH), "Float", []int64{params.K}},
			"dn-hh-quantized":   {encodeI8(q.DownHH), "Char", []int64{params.K, params.C}},
			"dn-hh-quant-scale": {encodeF32(q.DownScaleHH), "Float", []int64{params.K}},
			"up-quantized":      {encodeI8(q.Up), "Char", []int64{params.UpRows(), params.UpCols()}},
			"up-quant-scale":    {encodeF32(q.UpScale), "Float", []int64{params.UpRows()}},
			"up-bias-f16":       {encodeF16(q.UpBias), "Half", []int64{params.UpRows()}},
			"dn-ih-cache":       {encodeI8(caches[i].DownIH), "Char", []int64{params.K, params.C}},
			"dn-ih-cache-scale": {encodeF16(caches[i].DownScaleIH), "Half", []int64{params.K}},
			"dn-hh-cache":       {encodeI8(caches[i].DownHH), "Char", []int64{params.K, params.C}},
			"dn-hh-cache-scale": {encodeF16(caches[i].DownScaleHH), "Half", []int64{params.K}},
			"up-cache":          {encodeI8(caches[i].Up), "Char", []int64{params.UpRows(), params.UpCols()}},
			"up-cache-scale":    {encodeF16(caches[i].UpScale), "Half", []int64{params.UpRows()}},
			"up-cache-bias":     {encodeF16(caches[i].UpBias), "Half", []int64{params.UpRows()}},
		}
		layer := &out[i]
		layer.Equal = true
		for _, role := range hacgpu.FLSTMCacheRoleNames() {
			want, ok := packed[role]
			if !ok {
				return nil, fmt.Errorf("layer %d has no Go bytes for role %q", i, role)
			}
			native, err := c.Layers[i].Bytes(filepath.Dir(filepath.Join(r.Capture, "caches.json")), role)
			if err != nil {
				return nil, err
			}
			rec := TensorCompare{
				Role: role, DType: want.dtype, Shape: want.shape, Bytes: int64(len(want.bytes)),
				GoSHA256: hashBytes(want.bytes), NativeSHA: hashBytes(native),
				Equal: bytes.Equal(want.bytes, native),
			}
			if r.ObservedDir != "" {
				rec.GoPayload = filepath.Join(r.ObservedDir, fmt.Sprintf("layer%d-%s.bin", i, role))
				rec.NativePayload = filepath.Join(r.Capture, c.Layers[i].Tensors[len(layer.Tensors)].File)
				if err := os.WriteFile(rec.GoPayload, want.bytes, 0644); err != nil {
					return out, err
				}
			}
			if !rec.Equal {
				layer.Equal = false
			}
			layer.Tensors = append(layer.Tensors, rec)
		}
		layer.Domain = domain(q, c.Layers[i])
		if !layer.Domain.Matches {
			layer.Equal = false
		}
		if !layer.Equal {
			return out, fmt.Errorf("layer %d (%s) does not reproduce the native cache", i, layer.WeightBase)
		}
	}
	return out, nil
}

// domain recomputes the measured quantization domain from the Go intermediates,
// in the oracle's order (down-IH, down-HH, up), and compares it with the
// capture's own measurement.
func domain(q model.FLSTMQuantized, captured hacgpu.FLSTMCacheLayer) QuantDomain {
	d := QuantDomain{CaptureScaleMin: captured.ScaleMin, CaptureScaleMax: captured.ScaleMax}
	seen := false
	for _, scale := range [][]float32{q.DownScaleIH, q.DownScaleHH, q.UpScale} {
		for _, v := range scale {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				d.NonfiniteRows++
				continue
			}
			if v == 0 {
				d.ZeroRows++
			}
			if !seen || v < d.ScaleMin {
				d.ScaleMin = v
			}
			if !seen || v > d.ScaleMax {
				d.ScaleMax = v
			}
			seen = true
		}
	}
	d.Matches = seen && d.ScaleMin == captured.ScaleMin && d.ScaleMax == captured.ScaleMax &&
		d.ZeroRows == captured.ZeroScaleRows && d.NonfiniteRows == captured.NonfiniteScaleRows
	return d
}

// controls re-packs every layer with each declared mutant and requires the native
// cache to discriminate it: either the mutant's bytes differ from the native
// bytes, or the source's own view cannot hold the mutant's tensor at all. A
// mutant that reproduces the native cache proves nothing and fails the attempt.
func controls(r *Receipt, c hacgpu.FLSTMCaches) ([]Control, error) {
	e, err := tensor.Open(r.Backend)
	if err != nil {
		return nil, err
	}
	defer e.Close()
	manifest, err := readWeightManifest(h02.WeightManifestPath)
	if err != nil {
		return nil, err
	}
	prefixes, err := rnnPrefixes(manifest, c.LSTMLayers)
	if err != nil {
		return nil, err
	}
	params := model.FLSTMParams{C: c.C, K: c.K}
	sets := make([]model.FLSTMWeightSet, len(prefixes))
	for i, prefix := range prefixes {
		set, _, err := loadLayerWeights(e, r.Model, manifest, prefix, params)
		if err != nil {
			return nil, err
		}
		sets[i] = set
	}
	dir := filepath.Dir(filepath.Join(r.Capture, "caches.json"))

	mutants := []model.FLSTMMutant{
		model.FLSTMMutantReductionAxis0,
		model.FLSTMMutantFloat32Quantize,
		model.FLSTMMutantEpsilon,
		model.FLSTMMutantClip128,
		model.FLSTMMutantTruncate,
		model.FLSTMMutantIdentityPermute,
		model.FLSTMMutantHiddenBias,
	}
	var out []Control
	for _, m := range mutants {
		rec := Control{Mutant: m.String()}
		roles := map[string]bool{}
		var firstErr error
		for i := range sets {
			got, err := model.PackFLSTMLayerMutant(params, sets[i], m)
			if err != nil {
				rec.LayersRejected++
				if firstErr == nil {
					firstErr = err
				}
				continue
			}
			packed := map[string][]byte{
				"dn-ih-cache":       encodeI8(got.DownIH),
				"dn-ih-cache-scale": encodeF16(got.DownScaleIH),
				"dn-hh-cache":       encodeI8(got.DownHH),
				"dn-hh-cache-scale": encodeF16(got.DownScaleHH),
				"up-cache":          encodeI8(got.Up),
				"up-cache-scale":    encodeF16(got.UpScale),
				"up-cache-bias":     encodeF16(got.UpBias),
			}
			differed := false
			for _, role := range hacgpu.FLSTMCacheRoleNames() {
				body, exists := packed[role]
				if !exists {
					continue
				}
				native, err := c.Layers[i].Bytes(dir, role)
				if err != nil {
					return nil, err
				}
				if !bytes.Equal(body, native) {
					roles[role] = true
					differed = true
				}
				if r.ObservedDir != "" {
					t, _ := c.Layers[i].Tensor(role)
					comparison := TensorCompare{Role: role, DType: t.DType, Shape: t.Shape, Bytes: int64(len(body)),
						GoSHA256: hashBytes(body), NativeSHA: hashBytes(native), Equal: bytes.Equal(body, native),
						GoPayload:     filepath.Join(r.ObservedDir, fmt.Sprintf("mutant-%s-layer%d-%s.bin", m, i, role)),
						NativePayload: filepath.Join(dir, t.File)}
					if err = os.WriteFile(comparison.GoPayload, body, 0644); err != nil {
						return nil, err
					}
					rec.Comparisons = append(rec.Comparisons, MutantCompare{Layer: i, Tensor: comparison})
				}
			}
			if differed {
				rec.LayersDiffering++
			}
		}
		if rec.LayersDiffering+rec.LayersRejected != len(sets) {
			return nil, fmt.Errorf("control %s does not discriminate every layer", m)
		}
		for role := range roles {
			rec.RolesDiffering = append(rec.RolesDiffering, role)
		}
		sort.Strings(rec.RolesDiffering)
		switch {
		case len(rec.RolesDiffering) > 0:
			rec.Outcome = "differs"
		case firstErr != nil:
			rec.Outcome = "rejected"
			rec.Error = firstErr.Error()
		default:
			return nil, fmt.Errorf("control %s reproduces the native cache, so the step it switches off is not load-bearing", m)
		}
		out = append(out, rec)
	}
	return out, nil
}

// weightManifestPin is one entry of the caller's own weight pin file.
type weightManifestPin struct {
	Name   string  `json:"name"`
	SHA256 string  `json:"sha256"`
	Shape  []int64 `json:"shape"`
}

func readWeightManifest(path string) (map[string]weightManifestPin, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var pins []weightManifestPin
	if err = json.Unmarshal(raw, &pins); err != nil {
		return nil, err
	}
	out := make(map[string]weightManifestPin, len(pins))
	for _, p := range pins {
		out[p.Name] = p
	}
	return out, nil
}

// rnnPrefixes derives the recurrent module prefixes from the pin file rather than
// assuming them: every `N.rnn.<param>.tensor` name is grouped by N, and the
// resulting indices must be exactly the declared layer count.
func rnnPrefixes(manifest map[string]weightManifestPin, layers int) ([]string, error) {
	seen := map[int]bool{}
	for name := range manifest {
		rest, ok := strings.CutSuffix(name, ".rnn.dn_weight_ih.tensor")
		if !ok {
			continue
		}
		n, err := strconv.Atoi(rest)
		if err != nil {
			continue
		}
		seen[n] = true
	}
	if len(seen) != layers {
		return nil, fmt.Errorf("weight manifest pins %d recurrent modules, the model declares %d layers", len(seen), layers)
	}
	indices := make([]int, 0, len(seen))
	for n := range seen {
		indices = append(indices, n)
	}
	sort.Ints(indices)
	out := make([]string, len(indices))
	for i, n := range indices {
		out[i] = fmt.Sprintf("%d.rnn.", n)
	}
	return out, nil
}

// loadLayerWeights loads the six pinned parameters of one layer through the CPU
// ATen engine and requires each file digest and shape to match the caller's pin
// file before the packer sees a byte.
func loadLayerWeights(e *tensor.Engine, modelDir string, manifest map[string]weightManifestPin, prefix string, p model.FLSTMParams) (model.FLSTMWeightSet, []WeightPin, error) {
	var set model.FLSTMWeightSet
	want := []struct {
		name  string
		shape []int64
		dst   *[]float32
	}{
		{"dn_weight_ih", []int64{p.K, p.C}, &set.DownIH},
		{"dn_weight_hh", []int64{p.K, p.C}, &set.DownHH},
		{"up_weight_ih", []int64{p.UpRows(), p.K}, &set.UpIH},
		{"up_weight_hh", []int64{p.UpRows(), p.K}, &set.UpHH},
		{"up_bias_ih", []int64{p.UpRows()}, &set.UpBiasIH},
		{"up_bias_hh", []int64{p.UpRows()}, &set.UpBiasHH},
	}
	pins := make([]WeightPin, 0, len(want))
	for _, w := range want {
		name := prefix + w.name + ".tensor"
		pin, ok := manifest[name]
		if !ok {
			return set, nil, fmt.Errorf("weight manifest has no %q", name)
		}
		path := filepath.Join(modelDir, name)
		got, err := runner.SHA(path)
		if err != nil {
			return set, nil, err
		}
		if got != pin.SHA256 {
			return set, nil, fmt.Errorf("weight %s digest %s does not match the pinned %s", name, got, pin.SHA256)
		}
		t, err := e.LoadWeight(path)
		if err != nil {
			return set, nil, err
		}
		shape, err := t.Shape()
		if err != nil {
			t.Close()
			return set, nil, err
		}
		if !sameShape(shape, w.shape) {
			t.Close()
			return set, nil, fmt.Errorf("weight %s shape %v is not the declared %v", name, shape, w.shape)
		}
		if !sameShape(shape, pin.Shape) {
			t.Close()
			return set, nil, fmt.Errorf("weight %s shape %v does not match the pinned %v", name, shape, pin.Shape)
		}
		data, err := t.Data()
		if err != nil {
			t.Close()
			return set, nil, err
		}
		if err = t.Close(); err != nil {
			return set, nil, err
		}
		if int64(len(data)) != shapeProduct(shape) {
			return set, nil, fmt.Errorf("weight %s holds %d elements, shape %v needs %d", name, len(data), shape, shapeProduct(shape))
		}
		*w.dst = data
		pins = append(pins, WeightPin{Name: name, SHA256: pin.SHA256, Shape: shape})
	}
	return set, pins, nil
}

func sameShape(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func shapeProduct(shape []int64) int64 {
	total := int64(1)
	for _, d := range shape {
		total *= d
	}
	return total
}

func encodeI8(v []int8) []byte {
	out := make([]byte, len(v))
	for i, x := range v {
		out[i] = byte(x)
	}
	return out
}

func encodeF32(v []float32) []byte {
	out := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(x))
	}
	return out
}

func encodeF16(v []uint16) []byte {
	out := make([]byte, len(v)*2)
	for i, x := range v {
		binary.LittleEndian.PutUint16(out[i*2:], x)
	}
	return out
}

func hashBytes(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// runtimePins records the exact shared objects this process mapped, so the
// receipt names the libraries that ran rather than the sonames the link line
// asked for.
func runtimePins() (map[string]string, error) {
	raw, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		return nil, err
	}
	prefixes := []string{"libtorch_cpu.so", "libc10.so", "libtorch_cuda.so", "libc10_cuda.so",
		"libcudart.so", "libcublas.so", "libcuda.so", "libkoi"}
	seen := map[string]bool{}
	out := map[string]string{}
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		path := fields[5]
		if !filepath.IsAbs(path) || seen[path] {
			continue
		}
		base := filepath.Base(path)
		matched := false
		for _, prefix := range prefixes {
			if strings.HasPrefix(base, prefix) {
				matched = true
				break
			}
		}
		if !matched {
			continue
		}
		seen[path] = true
		sum, herr := runner.SHA(path)
		if herr != nil {
			return nil, herr
		}
		out[path] = sum
	}
	if len(out) < 6 {
		names := make([]string, 0, len(out))
		for path := range out {
			names = append(names, path)
		}
		sort.Strings(names)
		return nil, fmt.Errorf("only %d pinned runtime libraries mapped: %v", len(out), names)
	}
	return out, nil
}
