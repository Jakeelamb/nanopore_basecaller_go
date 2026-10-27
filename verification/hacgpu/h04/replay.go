package h04

import (
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"slices"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

const ReplaySchema = "nanopore-hacgpu-flstm-retained-replay/v1"

type ReplayOptions struct {
	Receipt       string
	ReceiptSHA256 string
	Out           string
	SourceRoot    string
}

type ReplayControl struct {
	Name         string `json:"name"`
	ChangedFile  string `json:"changed_file"`
	BeforeSHA256 string `json:"before_sha256"`
	AfterSHA256  string `json:"after_sha256"`
	Rejection    string `json:"rejection"`
}

type RetainedPair struct {
	Layer          int           `json:"layer"`
	Tensor         TensorCompare `json:"historical_comparison"`
	ObservedSHA256 string        `json:"observed_native_sha256"`
}

type ReplayReceipt struct {
	Schema            string            `json:"schema"`
	Status            string            `json:"status"`
	Error             string            `json:"error,omitempty"`
	Started           string            `json:"started_utc"`
	Finished          string            `json:"finished_utc"`
	Attempt           string            `json:"attempt"`
	Command           []string          `json:"command"`
	Environment       map[string]string `json:"environment"`
	Consumer          string            `json:"consumer_executable"`
	ConsumerSHA256    string            `json:"consumer_executable_sha256"`
	BuildInfo         string            `json:"consumer_go_build_info"`
	GoVersion         string            `json:"go_version"`
	SourceRoot        string            `json:"source_root"`
	SourceSHA256      map[string]string `json:"consumer_source_and_test_sha256"`
	HistoricalReceipt string            `json:"historical_receipt"`
	HistoricalSHA256  string            `json:"historical_receipt_sha256"`
	InputsSHA256      map[string]string `json:"retained_inputs_sha256"`
	NativePayloads    int               `json:"native_payloads_authenticated"`
	FinalCaches       int               `json:"final_caches_authenticated"`
	Pairs             []RetainedPair    `json:"historical_pairs_authenticated"`
	Controls          []ReplayControl   `json:"changed_copy_controls"`
	Limits            []string          `json:"limits"`
}

// Replay authenticates retained native bytes against a separately pinned
// historical comparison record. It runs no model arithmetic or native program.
func Replay(o ReplayOptions) (r ReplayReceipt, err error) {
	r = ReplayReceipt{Schema: ReplaySchema, Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano),
		Command: append([]string(nil), os.Args...), GoVersion: runtime.Version(), HistoricalReceipt: o.Receipt,
		HistoricalSHA256: o.ReceiptSHA256, InputsSHA256: map[string]string{}, Environment: map[string]string{},
		Limits: []string{
			"Retained witness authentication only. No new quantizer, packer, mutant arithmetic, native producer, compiler, or GPU execution.",
			"Historical Go comparison source and executable were not pinned by the producer. Current consumer pins do not repair that gap.",
			"Historical oracle compilation and link closure is incomplete. Manifest authentication does not establish a complete input set.",
			"Historical child runtime maps and device bookend agreement were not authenticated by the producer receipt.",
			"Pinned HAC cache evidence only. No general half quantizer, H03, H05, full caller, or biological qualification.",
		}}
	r.Attempt, err = runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
		}
		raw, werr := json.MarshalIndent(r, "", "  ")
		if werr == nil {
			werr = os.WriteFile(filepath.Join(r.Attempt, "receipt.json"), append(raw, '\n'), 0644)
		}
		if err == nil {
			err = werr
		}
	}()
	if len(o.ReceiptSHA256) != 64 {
		return r, fmt.Errorf("an independently retained historical receipt SHA256 is required")
	}
	if o.SourceRoot == "" {
		o.SourceRoot, err = os.Getwd()
		if err != nil {
			return r, err
		}
	}
	r.SourceRoot, err = filepath.Abs(o.SourceRoot)
	if err != nil {
		return r, err
	}
	if r.SourceSHA256, err = replaySourcePins(r.SourceRoot); err != nil {
		return r, err
	}
	if r.Consumer, err = os.Executable(); err != nil {
		return r, err
	}
	if r.ConsumerSHA256, err = runner.SHA(r.Consumer); err != nil {
		return r, err
	}
	if info, ok := debug.ReadBuildInfo(); ok {
		r.BuildInfo = info.String()
	}
	for _, key := range []string{"GOMAXPROCS", "GOMEMLIMIT", "CGO_ENABLED", "GOPROXY", "GOSUMDB"} {
		r.Environment[key] = os.Getenv(key)
	}
	if err = replayPin(o.Receipt, o.ReceiptSHA256); err != nil {
		return r, err
	}
	raw, err := os.ReadFile(o.Receipt)
	if err != nil {
		return r, err
	}
	var old Receipt
	if err = json.Unmarshal(raw, &old); err != nil {
		return r, err
	}
	if old.Schema != ReceiptSchema || old.Status != "flstm_weights_passed" {
		return r, fmt.Errorf("historical receipt is not a successful H04 producer record")
	}
	pins := map[string]string{old.Capture + "/caches.json": old.CaptureJSON, old.Oracle: old.OracleSHA,
		old.OracleSource: old.OracleSrcSHA, old.BuildManifest: old.BuildSHA, old.Backend: old.BackendSHA,
		old.BackendBuild: old.BackendBSHA, old.Input: old.InputSHA,
		filepath.Join(r.SourceRoot, "internal/model/hac_weights.json"): old.ModelPins}
	for path, pin := range old.Inputs {
		pins[path] = pin
	}
	for _, layer := range old.Layers {
		for _, w := range layer.WeightPins {
			if w.Name == "" || filepath.Base(w.Name) != w.Name {
				return r, fmt.Errorf("unsafe weight name %q", w.Name)
			}
			pins[filepath.Join(old.Model, w.Name)] = w.SHA256
		}
	}
	for _, cmd := range old.Commands {
		if cmd.Exit != 0 {
			return r, fmt.Errorf("historical child command failed")
		}
		pins[cmd.Stdout], pins[cmd.Stderr] = cmd.StdoutSHA, cmd.StderrSHA
	}
	for path, pin := range pins {
		if err = replayPin(path, pin); err != nil {
			return r, err
		}
		r.InputsSHA256[path] = pin
	}
	capture, err := hacgpu.ImportFLSTMCaches(old.Capture)
	if err != nil {
		return r, err
	}
	if r.Pairs, err = retainedPairs(old, capture); err != nil {
		return r, err
	}
	r.NativePayloads = len(r.Pairs)
	for _, pair := range r.Pairs {
		if strings.Contains(pair.Tensor.Role, "-cache") {
			r.FinalCaches++
		}
	}
	if r.NativePayloads != 98 || r.FinalCaches != 49 {
		return r, fmt.Errorf("H04 retained case needs 98 payloads and 49 final caches, got %d and %d", r.NativePayloads, r.FinalCaches)
	}
	if r.Controls, err = replayControls(r.Attempt, old, capture, r.SourceRoot); err != nil {
		return r, err
	}
	after, err := replaySourcePins(r.SourceRoot)
	if err != nil {
		return r, err
	}
	if !maps.Equal(after, r.SourceSHA256) {
		return r, fmt.Errorf("consumer source set or bytes changed during replay")
	}
	if err = replayPin(o.Receipt, o.ReceiptSHA256); err != nil {
		return r, err
	}
	r.Status = "retained_witnesses_authenticated"
	return r, nil
}

func retainedPairs(old Receipt, c hacgpu.FLSTMCaches) ([]RetainedPair, error) {
	if old.C != 1024 || old.K != 128 || old.LSTMLayers != 7 || old.Batch != 64 || old.Chunk != 192 || !old.FirstReverse ||
		old.C != c.C || old.K != c.K || old.LSTMLayers != c.LSTMLayers || old.Batch != c.Batch || old.Chunk != c.Chunk ||
		old.Case != c.Case || old.Features != c.Features || old.FirstReverse != c.FirstReverse || old.Quantize != c.Quantize ||
		old.InputSHA != c.InputSHA256 || old.CaptureInput != c.InputSHA256 || old.ModelPackage != c.ModelPackage ||
		len(old.Layers) != len(c.Layers) {
		return nil, fmt.Errorf("historical receipt and retained capture disagree on the pinned HAC case")
	}
	var out []RetainedPair
	for i, layer := range c.Layers {
		prior := old.Layers[i]
		if prior.Index != i || !prior.Equal || prior.Reverse != layer.Reverse || !prior.Domain.Matches ||
			prior.Domain.ScaleMin != layer.ScaleMin || prior.Domain.ScaleMax != layer.ScaleMax ||
			prior.Domain.ZeroRows != layer.ZeroScaleRows || prior.Domain.NonfiniteRows != layer.NonfiniteScaleRows ||
			len(prior.Tensors) != len(layer.Tensors) {
			return nil, fmt.Errorf("historical layer %d is inconsistent", i)
		}
		for j, t := range layer.Tensors {
			pair := prior.Tensors[j]
			body, err := layer.Bytes(old.Capture, t.Role)
			if err != nil {
				return nil, err
			}
			observed := hashBytes(body)
			if pair.Role != t.Role || pair.DType != t.DType || !slices.Equal(pair.Shape, t.Shape) ||
				pair.Bytes != int64(len(body)) || !pair.Equal || pair.GoSHA256 != observed || pair.NativeSHA != observed {
				return nil, fmt.Errorf("historical layer %d role %q does not match the retained native bytes", i, t.Role)
			}
			out = append(out, RetainedPair{Layer: i, Tensor: pair, ObservedSHA256: observed})
		}
	}
	return out, nil
}

func replayPin(path, want string) error {
	if path == "" || len(want) != 64 {
		return fmt.Errorf("missing path or SHA256 for retained input %q", path)
	}
	got, err := runner.SHA(path)
	if err != nil {
		return err
	}
	if got != want {
		return fmt.Errorf("input %s SHA256 %s does not match %s", path, got, want)
	}
	return nil
}

func replaySourcePins(root string) (map[string]string, error) {
	out := map[string]string{}
	for _, dir := range []string{"internal/signal", "internal/tensor", "internal/model", "verification/tensor/runner",
		"verification/hacgpu", "verification/hacgpu/h02", "verification/hacgpu/h04", "verification/hacgpu/h04/cmd/replay"} {
		entries, err := os.ReadDir(filepath.Join(root, dir))
		if err != nil {
			return nil, err
		}
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			path := filepath.Join(dir, entry.Name())
			if !slices.Contains([]string{".go", ".c", ".h", ".cpp", ".inc"}, filepath.Ext(path)) {
				continue
			}
			pin, err := runner.SHA(filepath.Join(root, path))
			if err != nil {
				return nil, err
			}
			out[path] = pin
		}
	}
	for _, path := range []string{"go.mod", "internal/model/hac_weights.json", "internal/model/sup_weights.json"} {
		pin, err := runner.SHA(filepath.Join(root, path))
		if err != nil {
			return nil, err
		}
		out[path] = pin
	}
	return out, nil
}
