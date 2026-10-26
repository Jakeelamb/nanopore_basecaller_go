package h04

import (
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type ConsumerOptions struct {
	ReplayOptions
	Backend  string
	Admitted bool
}

type ConsumerReceipt struct {
	Schema           string            `json:"schema"`
	Status           string            `json:"status"`
	Error            string            `json:"error,omitempty"`
	Started          string            `json:"started_utc"`
	Finished         string            `json:"finished_utc"`
	Attempt          string            `json:"attempt"`
	HostLock         string            `json:"host_lock"`
	LockAcquired     string            `json:"lock_acquired_utc"`
	ReplayReceipt    string            `json:"retained_replay_receipt"`
	ReplaySHA256     string            `json:"retained_replay_receipt_sha256"`
	Consumer         string            `json:"consumer_executable"`
	ConsumerSHA256   string            `json:"consumer_executable_sha256"`
	SourceSHA256     map[string]string `json:"consumer_source_and_test_sha256"`
	Backend          string            `json:"cpu_loader_backend"`
	BackendSHA256    string            `json:"cpu_loader_backend_sha256"`
	BackendManifest  string            `json:"cpu_loader_build_manifest"`
	BackendBuildSHA  string            `json:"cpu_loader_build_manifest_sha256"`
	BackendInputs    int               `json:"cpu_loader_declared_inputs_verified"`
	BackendInputsSHA string            `json:"cpu_loader_declared_inputs_sha256"`
	RuntimeSHA256    map[string]string `json:"mapped_loader_runtime_sha256"`
	ModelPinsSHA256  string            `json:"weight_manifest_sha256"`
	NativePayloads   int               `json:"native_payloads_compared"`
	FinalCaches      int               `json:"final_caches_compared"`
	Layers           []LayerResult     `json:"fresh_go_comparisons"`
	Controls         []Control         `json:"fresh_go_mutants"`
	PayloadControl   ReplayControl     `json:"changed_observed_go_payload_control"`
	Limits           []string          `json:"limits"`
}

// VerifyConsumer executes only CPU archive deserialization and Go arithmetic.
// It does not create a CUDA device, run model forward, or invoke the oracle.
func VerifyConsumer(o ConsumerOptions) (r ConsumerReceipt, err error) {
	return verifyConsumer(o, runner.HostLock)
}

func verifyConsumer(o ConsumerOptions, lockPath string) (r ConsumerReceipt, err error) {
	if !o.Admitted {
		return r, fmt.Errorf("H04 CPU loader execution requires explicit coordinator admission")
	}
	lock, err := runner.Acquire(lockPath)
	if err != nil {
		return r, err
	}
	defer lock.Close()
	r = ConsumerReceipt{Schema: "nanopore-hacgpu-flstm-current-consumer/v1", Status: "failed",
		Started: time.Now().UTC().Format(time.RFC3339Nano), Backend: o.Backend, HostLock: lockPath,
		LockAcquired: time.Now().UTC().Format(time.RFC3339Nano),
		Limits: []string{
			"Fresh Go quantization and packing against the pinned retained HAC native capture. Native CPU archive deserialization only.",
			"No native oracle, model forward, GPU computation, H03, H05, end-to-end caller, or biological qualification.",
			"Historical Go producer provenance and complete oracle compilation, link, child runtime and device bookend provenance remain missing.",
			"General CUDA half quantizer equivalence is not established outside the pinned weights and tested host cases.",
		}}
	witness, err := Replay(o.ReplayOptions)
	if err != nil {
		return r, err
	}
	r.Attempt, r.Consumer, r.ConsumerSHA256 = witness.Attempt, witness.Consumer, witness.ConsumerSHA256
	r.SourceSHA256 = witness.SourceSHA256
	r.ReplayReceipt = filepath.Join(r.Attempt, "receipt.json")
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
		}
		raw, werr := json.MarshalIndent(r, "", "  ")
		if werr == nil {
			werr = os.WriteFile(filepath.Join(r.Attempt, "consumer-receipt.json"), append(raw, '\n'), 0644)
		}
		if err == nil {
			err = werr
		}
	}()
	if r.ReplaySHA256, err = runner.SHA(r.ReplayReceipt); err != nil {
		return r, err
	}
	backend, err := hacgpu.VerifyBackend(o.Backend)
	if err != nil {
		return r, err
	}
	r.BackendSHA256, r.BackendManifest, r.BackendBuildSHA = backend.OutputSHA, backend.Path, backend.ManifestSHA
	r.BackendInputs, r.BackendInputsSHA = backend.InputCount, backend.InputsSHA
	raw, err := os.ReadFile(o.Receipt)
	if err != nil {
		return r, err
	}
	var prior Receipt
	if err = json.Unmarshal(raw, &prior); err != nil {
		return r, err
	}
	if err = replayPin(o.Receipt, o.ReceiptSHA256); err != nil {
		return r, err
	}
	capture, err := hacgpu.ImportFLSTMCaches(prior.Capture)
	if err != nil {
		return r, err
	}
	current := Receipt{Backend: backend.Output, Model: prior.Model, Capture: prior.Capture,
		Inputs: map[string]string{}, ObservedDir: filepath.Join(r.Attempt, "observed-go")}
	r.Layers, err = differential(&current, capture, prior.Model)
	r.ModelPinsSHA256, r.RuntimeSHA256 = current.ModelPins, current.Runtime
	if err != nil {
		return r, err
	}
	for _, layer := range r.Layers {
		for _, t := range layer.Tensors {
			r.NativePayloads++
			if strings.Contains(t.Role, "-cache") {
				r.FinalCaches++
			}
		}
	}
	if r.NativePayloads != 98 || r.FinalCaches != 49 {
		return r, fmt.Errorf("current consumer did not compare all 98 payloads and 49 caches")
	}
	if r.Controls, err = controls(&current, capture); err != nil {
		return r, err
	}
	for _, layer := range r.Layers {
		for _, t := range layer.Tensors {
			if err = verifyObservedPair(t); err != nil {
				return r, err
			}
		}
	}
	for _, control := range r.Controls {
		for _, comparison := range control.Comparisons {
			if err = verifyObservedPair(comparison.Tensor); err != nil {
				return r, err
			}
		}
	}
	first := r.Layers[0].Tensors[0]
	body, err := os.ReadFile(first.GoPayload)
	if err != nil {
		return r, err
	}
	body[0] ^= 1
	changed := filepath.Join(r.Attempt, "controls", "observed-go-payload.changed")
	if err = os.WriteFile(changed, body, 0644); err != nil {
		return r, err
	}
	if r.PayloadControl, err = rejectedCopy("observed-go-payload", changed, first.GoSHA256, replayPin(changed, first.GoSHA256)); err != nil {
		return r, err
	}
	after, err := replaySourcePins(witness.SourceRoot)
	if err != nil {
		return r, err
	}
	if !maps.Equal(after, r.SourceSHA256) {
		return r, fmt.Errorf("consumer source changed during arithmetic verification")
	}
	for _, pin := range []struct{ path, sha string }{{backend.Output, backend.OutputSHA}, {backend.Path, backend.ManifestSHA},
		{o.Receipt, o.ReceiptSHA256}, {r.Consumer, r.ConsumerSHA256}} {
		if err = replayPin(pin.path, pin.sha); err != nil {
			return r, err
		}
	}
	r.Status = "current_go_cache_parity_passed"
	return r, nil
}

func verifyObservedPair(t TensorCompare) error {
	for _, pin := range []struct{ path, sha string }{{t.GoPayload, t.GoSHA256}, {t.NativePayload, t.NativeSHA}} {
		if err := replayPin(pin.path, pin.sha); err != nil {
			return err
		}
		info, err := os.Stat(pin.path)
		if err != nil {
			return err
		}
		if info.Size() != t.Bytes {
			return fmt.Errorf("observed payload %s has %d bytes, expected %d", pin.path, info.Size(), t.Bytes)
		}
	}
	if t.Equal != (t.GoSHA256 == t.NativeSHA) {
		return fmt.Errorf("observed payload equality disagrees with its digests")
	}
	return nil
}

func consumerRuntimePins(backend string) (map[string]string, error) {
	raw, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		return nil, err
	}
	backend, err = filepath.Abs(backend)
	if err != nil {
		return nil, err
	}
	pins := map[string]string{}
	var torchCPU, c10 bool
	for _, line := range strings.Split(string(raw), "\n") {
		parts := strings.Fields(line)
		if len(parts) < 6 || !filepath.IsAbs(parts[5]) {
			continue
		}
		path := parts[5]
		if !strings.Contains(filepath.Base(path), ".so") || pins[path] != "" {
			continue
		}
		if pins[path], err = runner.SHA(path); err != nil {
			return nil, err
		}
		torchCPU = torchCPU || filepath.Base(path) == "libtorch_cpu.so"
		c10 = c10 || filepath.Base(path) == "libc10.so"
	}
	if pins[backend] == "" || !torchCPU || !c10 {
		return nil, fmt.Errorf("CPU loader backend, libtorch_cpu and libc10 must be mapped during loading")
	}
	return pins, nil
}
