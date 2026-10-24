// Typed storage unit: the only admitted path that exercises the storage ABI on a
// real device. It runs the compiled Go consumer, requires the exact payload and
// rejection controls from the native receipt, and records that receipt in the
// attempt manifest.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

var storageChecks = []string{
	"f16-roundtrip", "i8-roundtrip", "f32-scale-roundtrip", "e4m3-roundtrip",
	"i32-roundtrip", "u8-roundtrip", "strided-view-guard", "strided-view-fill", "device-copy",
}

var storageRejections = []string{
	"u8-has-no-koi-type", "overlapping-copy", "dtype-mismatch-copy",
	"non-contiguous-upload", "non-contiguous-download", "scale-requires-i8",
	"scale-requires-f32", "native-scale-requires-i8", "native-scale-requires-f32",
	"scale-self", "scale-unknown-tag", "scale-owner-release",
	"fill-pattern-width", "native-copy-dtype-mismatch", "native-fill-pattern-width",
	"view-exceeds-parent", "view-negative-stride",
	"foreign-engine-handle", "unknown-dtype", "closed-download",
}

type storageCheck struct {
	Name           string `json:"name"`
	ExpectedSHA256 string `json:"expected_sha256"`
	ObservedSHA256 string `json:"observed_sha256"`
	Bytes          int    `json:"bytes"`
}
type storageRejection struct {
	Name  string `json:"name"`
	Error string `json:"error"`
}
type storageReceipt struct {
	Schema     string             `json:"schema"`
	Status     string             `json:"status"`
	Device     uint64             `json:"device"`
	Ordinal    int32              `json:"ordinal"`
	Runtime    map[string]string  `json:"runtime_sha256"`
	Checks     []storageCheck     `json:"checks"`
	Rejections []storageRejection `json:"rejections"`
}

// storageIdentity extracts the admitted device-open identity from the probe log
// and requires CUDA family 13 on the GPU the bookend observed.
func storageIdentity(log, bookend string) (string, error) {
	raw, err := os.ReadFile(log)
	if err != nil {
		return "", err
	}
	identity := ""
	for _, line := range strings.Split(string(raw), "\n") {
		var event struct{ Output string }
		if json.Unmarshal([]byte(line), &event) == nil {
			line = event.Output
		}
		if strings.Contains(line, "nt_cuda_device driver=") {
			identity = strings.TrimSpace(line)
		}
	}
	if identity == "" {
		return "", fmt.Errorf("probe log lacks a device identity line")
	}
	runtimeVersion := 0
	uuid := ""
	for _, field := range strings.Fields(identity) {
		if value, ok := strings.CutPrefix(field, "runtime="); ok {
			runtimeVersion, _ = strconv.Atoi(value)
		}
		if value, ok := strings.CutPrefix(field, "uuid="); ok {
			uuid = value
		}
	}
	if runtimeVersion/1000 != 13 || len(uuid) != 32 {
		return "", fmt.Errorf("probe device identity is not a CUDA13 device")
	}
	want := ""
	for _, field := range strings.Split(bookend, ",") {
		if value, ok := strings.CutPrefix(strings.TrimSpace(field), "GPU-"); ok {
			want = strings.ReplaceAll(value, "-", "")
		}
	}
	if want == "" || uuid != want {
		return "", fmt.Errorf("probe device UUID differs from the observed GPU")
	}
	return identity, nil
}

func captureStorageUnit(m *evidence, dir, backend, torchRoot string, ordinal int, baseEnv []string, pin func(string, string) error, execute func(string, []string, ...string) (runner.CommandReceipt, error)) error {
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
	receiptPath := filepath.Join(dir, "storage-capture.json")
	base := append(append([]string{}, baseEnv...), "CGO_ENABLED=1", "NANOPORE_TENSOR_BACKEND="+backend, "NANOPORE_TENSOR_CUDA=1", "NANOPORE_TENSOR_CUDA_ORDINAL="+fmt.Sprint(ordinal))
	probePath := filepath.Join(dir, "storage-runtime.json")
	probeEnv := append(append([]string{}, base...), "NANOPORE_TENSOR_CUDA_DIAGNOSTICS=1", "NANOPORE_TENSOR_STORAGE_PROBE="+probePath)
	const probeName = "TestNativeCUDAStorageProbe"
	probeRun, err := execute("storage-koi-runtime-probe", probeEnv, "go", "tool", "test2json", "-t", "-p", "github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor", consumer.TestBinary, "-test.v=test2json", "-test.count=1", "-test.run", "^"+probeName+"$")
	if err != nil {
		return err
	}
	identity, err := storageIdentity(probeRun.Stdout, m.DeviceIdentity)
	if err != nil {
		return err
	}
	m.ProducerIdentity = identity
	rawProbe, err := os.ReadFile(probePath)
	if err != nil {
		return err
	}
	var probe struct {
		Runtime map[string]string `json:"runtime_sha256"`
	}
	if err = json.Unmarshal(rawProbe, &probe); err != nil {
		return err
	}
	if len(probe.Runtime) == 0 {
		return fmt.Errorf("runtime probe recorded no CUDA libraries")
	}
	outputSHA, err := runner.SHA(backend)
	if err != nil {
		return err
	}
	manifest := map[string]any{"runtime_sha256": probe.Runtime, "output": backend, "output_sha256": outputSHA}
	rawManifest, err := json.Marshal(manifest)
	if err != nil {
		return err
	}
	runtimeManifest := filepath.Join(dir, "storage-runtime-pins.json")
	if err = os.WriteFile(runtimeManifest, rawManifest, 0o644); err != nil {
		return err
	}
	env := append(append([]string{}, base...), "NANOPORE_TENSOR_CUDA_DIAGNOSTICS=1", "NANOPORE_TENSOR_RUNTIME_MANIFEST="+runtimeManifest, "NANOPORE_TENSOR_STORAGE_CAPTURE="+receiptPath)
	const name = "TestNativeCUDAStorageDevice"
	run, err := execute("storage-koi-device", env, "go", "tool", "test2json", "-t", "-p", "github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor", consumer.TestBinary, "-test.v=test2json", "-test.count=1", "-test.run", "^"+name+"$")
	if err != nil {
		return err
	}
	if err = runner.RequireGoTestRoots(run.Stdout, "github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor", []string{name}); err != nil {
		return err
	}
	raw, err := os.ReadFile(receiptPath)
	if err != nil {
		return err
	}
	var receipt storageReceipt
	if err = json.Unmarshal(raw, &receipt); err != nil {
		return err
	}
	if receipt.Schema != "nanopore-cuda-storage-koi/v1" || receipt.Status != "passed" {
		return fmt.Errorf("unqualified storage receipt")
	}
	if receipt.Ordinal != int32(ordinal) || receipt.Device == 0 {
		return fmt.Errorf("storage receipt device identity differs from the admitted ordinal")
	}
	for path, want := range receipt.Runtime {
		got, err := runner.SHA(path)
		if err != nil {
			return err
		}
		if got != want {
			return fmt.Errorf("mapped runtime library changed: %s", path)
		}
		name := filepath.Base(path)
		if (strings.HasPrefix(name, "libtorch") || strings.HasPrefix(name, "libc10")) && !strings.HasPrefix(path, torchRoot+"/") {
			return fmt.Errorf("foreign Torch distribution mapped: %s", path)
		}
		m.Runtime[path] = got
		if err = pin(path, want); err != nil {
			return err
		}
	}
	checks := map[string]storageCheck{}
	for _, c := range receipt.Checks {
		if c.ExpectedSHA256 == "" || c.ExpectedSHA256 != c.ObservedSHA256 || c.Bytes <= 0 {
			return fmt.Errorf("storage check %q did not roundtrip exactly", c.Name)
		}
		checks[c.Name] = c
	}
	for _, want := range storageChecks {
		if _, ok := checks[want]; !ok {
			return fmt.Errorf("storage receipt lacks check %q", want)
		}
	}
	rejects := map[string]string{}
	for _, r := range receipt.Rejections {
		if r.Error == "" {
			return fmt.Errorf("storage rejection %q lacks a reason", r.Name)
		}
		rejects[r.Name] = r.Error
	}
	for _, want := range storageRejections {
		if _, ok := rejects[want]; !ok {
			return fmt.Errorf("storage receipt lacks rejection control %q", want)
		}
	}
	if err = pin(receiptPath, ""); err != nil {
		return err
	}
	m.Unit = "storage-koi"
	m.UnitReceipt = filepath.Base(receiptPath)
	m.UnitReceiptSHA, err = runner.SHA(receiptPath)
	if err != nil {
		return err
	}
	m.UnitChecks = len(checks)
	m.UnitRejections = len(rejects)
	if err = consumerStable(consumer); err != nil {
		return err
	}
	m.Consumer.StableAfter = true
	m.Status = "storage_koi_device_passed"
	return nil
}
