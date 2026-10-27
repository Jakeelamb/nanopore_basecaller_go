package h07

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type OracleCaptureOptions struct {
	Build, BuildSHA, Capture, Model, Out string
	Ordinal                              int
	Admitted                             bool
}
type OracleCaptureReceipt struct {
	Outputs                                           map[string]string
	Schema, Status, Error, Started, Finished, Attempt string
	Build, BuildSHA, Oracle, OracleSHA, Output        string
	BeforeDevice, AfterDevice                         string
	Inputs, Runtime                                   map[string]string
	Commands                                          []runner.CommandReceipt
	Caches                                            []CacheTensor
	Environment                                       map[string]RuntimeEnvValue
}
type RuntimeEnvValue struct {
	Present bool
	Value   string
}

func runtimeEnvironment(producer bool) map[string]RuntimeEnvValue {
	out := map[string]RuntimeEnvValue{}
	for _, name := range []string{"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG", "DORADO_MATMUL_FP16_MODE", "LD_LIBRARY_PATH", "LD_PRELOAD"} {
		v, present := os.LookupEnv(name)
		out[name] = RuntimeEnvValue{Present: present, Value: v}
	}
	if producer {
		for name, value := range map[string]string{"LD_LIBRARY_PATH": "", "LD_PRELOAD": "", "DORADO_MATMUL_FP16_MODE": "CUBLAS"} {
			out[name] = RuntimeEnvValue{Present: true, Value: value}
		}
	}
	return out
}

// VerifyRuntimeMaps accepts only shared objects and the executable frozen by
// the build. Unknown or deleted code objects invalidate the attempt.
func VerifyRuntimeMaps(body string, expected map[string]string) (map[string]string, error) {
	observed := map[string]string{}
	for _, line := range strings.Split(body, "\n") {
		parts := strings.Fields(line)
		if len(parts) < 6 {
			continue
		}
		path := strings.Join(parts[5:], " ")
		if !filepath.IsAbs(path) {
			continue
		}
		code := strings.Contains(filepath.Base(path), ".so") || strings.Contains(parts[1], "x")
		if !code {
			continue
		}
		if strings.HasSuffix(path, " (deleted)") {
			return nil, fmt.Errorf("deleted runtime code mapping: %s", path)
		}
		actual, err := filepath.EvalSymlinks(path)
		if err != nil {
			return nil, err
		}
		actual, err = filepath.Abs(actual)
		if err != nil {
			return nil, err
		}
		pin, ok := expected[actual]
		if !ok {
			return nil, fmt.Errorf("runtime code absent from frozen closure: %s", actual)
		}
		if err = requirePin(actual, pin); err != nil {
			return nil, err
		}
		observed[actual] = pin
	}
	if len(observed) < 5 {
		return nil, fmt.Errorf("runtime mapping observation is incomplete")
	}
	return observed, nil
}
func projectionArchivePins(modelDir string) (map[string]string, error) {
	raw, err := os.ReadFile("internal/model/hac_weights.json")
	if err != nil {
		return nil, err
	}
	var weights []struct {
		Name, SHA256 string
		Shape        []int64
	}
	if err = json.Unmarshal(raw, &weights); err != nil {
		return nil, err
	}
	out := map[string]string{}
	for _, want := range []struct {
		name  string
		shape []int64
	}{{"11.linear.weight.tensor", []int64{128, 1024}}, {"12.linear.weight.tensor", []int64{1024, 128}}} {
		found := 0
		for _, pin := range weights {
			if pin.Name == want.name {
				found++
				if !slices.Equal(pin.Shape, want.shape) {
					return nil, fmt.Errorf("projection archive semantic shape mismatch")
				}
				path := filepath.Join(modelDir, want.name)
				if _, err = readPinned(path, pin.SHA256); err != nil {
					return nil, err
				}
				out[path] = pin.SHA256
			}
		}
		if found != 1 {
			return nil, fmt.Errorf("projection archive role missing or duplicated")
		}
	}
	if err = pinFile(out, "internal/model/hac_weights.json"); err != nil {
		return nil, err
	}
	return out, nil
}
func CaptureOracle(o OracleCaptureOptions) (r OracleCaptureReceipt, err error) {
	if !o.Admitted {
		return r, fmt.Errorf("native projection cache producer requires separate runtime admission")
	}
	if len(o.BuildSHA) != 64 {
		return r, fmt.Errorf("independently pinned oracle build SHA256 required")
	}
	if o.Ordinal < 0 || int64(o.Ordinal) > 2147483647 {
		return r, fmt.Errorf("invalid ordinal")
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return r, err
	}
	defer lock.Close()
	r = OracleCaptureReceipt{Schema: "nanopore-hac-output-cache-capture/v1", Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Inputs: map[string]string{}, Runtime: map[string]string{}}
	r.Environment = runtimeEnvironment(true)
	r.Attempt, err = runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
		}
		raw, e := json.MarshalIndent(r, "", "  ")
		if e == nil {
			e = os.WriteFile(filepath.Join(r.Attempt, "capture.json"), append(raw, '\n'), 0644)
		}
		if err == nil {
			err = e
		}
	}()
	r.Build = o.Build
	r.BuildSHA = o.BuildSHA
	var build OracleBuildReceipt
	raw, err := readPinned(o.Build, o.BuildSHA)
	if err != nil {
		return r, err
	}
	if err = json.Unmarshal(raw, &build); err != nil {
		return r, err
	}
	if build.Schema != "nanopore-hac-output-oracle-build/v1" || build.Status != "closed_build_passed" || len(build.Compile) != 3 || len(build.Link) != 1 || len(build.Dependencies) != 3 {
		return r, fmt.Errorf("complete direct-source projection build proof required")
	}
	if err = checkPins(build.Inputs); err != nil {
		return r, err
	}
	if err = checkPins(build.Objects); err != nil {
		return r, err
	}
	if _, err = readPinned(build.Output, build.OutputSHA); err != nil {
		return r, err
	}
	r.Oracle = build.Output
	r.OracleSHA = build.OutputSHA
	for p, h := range build.Inputs {
		r.Inputs[p] = h
	}
	r.Inputs[o.Build] = r.BuildSHA
	r.Inputs[build.Output] = build.OutputSHA
	w, err := ReadWitness(o.Capture)
	if err != nil {
		return r, err
	}
	for p, h := range w.Pins {
		r.Inputs[p] = h
	}
	archivePins, err := projectionArchivePins(o.Model)
	if err != nil {
		return r, err
	}
	for p, h := range archivePins {
		r.Inputs[p] = h
	}
	smi, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return r, err
	}
	smi, err = filepath.Abs(smi)
	if err != nil {
		return r, err
	}
	if err = pinFile(r.Inputs, smi); err != nil {
		return r, err
	}
	for name := range r.Inputs {
		actual, e := filepath.Abs(name)
		if e != nil {
			return r, e
		}
		if actual != name {
			r.Inputs[actual] = r.Inputs[name]
			delete(r.Inputs, name)
		}
	}
	freeze, err := json.MarshalIndent(r.Inputs, "", "  ")
	if err != nil {
		return r, err
	}
	if err = os.WriteFile(filepath.Join(r.Attempt, "before-producer-inputs.json"), append(freeze, '\n'), 0644); err != nil {
		return r, err
	}
	execute := func(label string, args ...string) (runner.CommandReceipt, error) {
		c, e := runner.Execute(r.Attempt, label, []string{"LD_LIBRARY_PATH=", "LD_PRELOAD=", "DORADO_MATMUL_FP16_MODE=CUBLAS"}, args...)
		r.Commands = append(r.Commands, c)
		return c, e
	}
	identity := func(label string) (string, error) {
		c, e := execute(label, smi, "-i", strconv.Itoa(o.Ordinal), "--query-gpu=index,uuid,pci.bus_id,driver_version", "--format=csv,noheader,nounits")
		if e != nil {
			return "", e
		}
		b, e := os.ReadFile(c.Stdout)
		return strings.TrimSpace(string(b)), e
	}
	r.BeforeDevice, err = identity("gpu-before")
	if err != nil {
		return r, err
	}
	if r.BeforeDevice == "" || strings.Contains(r.BeforeDevice, "\n") {
		return r, fmt.Errorf("one explicit GPU identity required")
	}
	if _, err = execute("apps-before", smi, "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"); err != nil {
		return r, err
	}
	r.Output = filepath.Join(r.Attempt, "oracle-output")
	if err = checkPins(r.Inputs); err != nil {
		return r, err
	}
	args := []string{build.Output, "--down-weight", filepath.Join(o.Model, "11.linear.weight.tensor"), "--up-weight", filepath.Join(o.Model, "12.linear.weight.tensor"),
		"--input-parent", filepath.Join(o.Capture, w.Input.BackingFile), "--expected-parent", filepath.Join(o.Capture, w.Output.BackingFile), "--out", r.Output, "--ordinal", strconv.Itoa(o.Ordinal)}
	_, producerErr := execute("source-projection", args...)
	r.AfterDevice, err = identity("gpu-after")
	if err != nil {
		return r, errors.Join(producerErr, err)
	}
	if _, err = execute("apps-after", smi, "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"); err != nil {
		return r, errors.Join(producerErr, err)
	}
	if producerErr != nil {
		return r, producerErr
	}
	if r.BeforeDevice != r.AfterDevice {
		return r, fmt.Errorf("GPU UUID bookends differ")
	}
	selected, err := os.ReadFile(filepath.Join(r.Output, "selected-device.txt"))
	if err != nil {
		return r, err
	}
	identityFields := strings.Split(r.BeforeDevice, ",")
	if len(identityFields) != 4 {
		return r, fmt.Errorf("GPU bookend identity has wrong field count")
	}
	if !strings.Contains(string(selected), "ordinal="+strconv.Itoa(o.Ordinal)+"\n") || !strings.Contains(string(selected), "uuid="+strings.TrimSpace(identityFields[1])+"\n") {
		return r, fmt.Errorf("selected CUDA UUID differs from bookends")
	}
	if err = checkPins(r.Inputs); err != nil {
		return r, err
	}
	policy := map[string]string{}
	for p, h := range build.Inputs {
		policy[p] = h
	}
	policy[build.Output] = build.OutputSHA
	for _, name := range []string{"runtime-before.maps", "runtime-ready.maps", "runtime-after.maps"} {
		body, e := os.ReadFile(filepath.Join(r.Output, name))
		if e != nil {
			return r, e
		}
		objects, e := VerifyRuntimeMaps(string(body), policy)
		if e != nil {
			return r, e
		}
		for p, h := range objects {
			r.Runtime[p] = h
		}
	}
	candidate, err := os.ReadFile(filepath.Join(r.Output, "candidate-backing.bin"))
	if err != nil {
		return r, err
	}
	if !slices.Equal(candidate, w.OutputParent) {
		return r, fmt.Errorf("source projection parent differs from independent H01 witness")
	}
	r.Caches, err = ReadCacheDescriptors(r.Output)
	if err != nil {
		return r, err
	}
	r.Outputs = map[string]string{}
	for _, name := range []string{"caches.tsv", "candidate-backing.bin", "runtime-before.maps", "runtime-ready.maps", "runtime-after.maps", "selected-device.txt", "down.bin", "down-scale.bin", "up.bin"} {
		path := filepath.Join(r.Output, name)
		r.Outputs[path], err = runner.SHA(path)
		if err != nil {
			return r, err
		}
	}
	r.Status = "source_projection_caches_passed"
	return r, nil
}
