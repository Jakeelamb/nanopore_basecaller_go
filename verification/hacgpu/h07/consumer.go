package h07

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type ConsumerOptions struct {
	NativeReceipt, NativeSHA, Backend, Model, Capture, Out string
	ControllerBuild, ControllerBuildSHA                    string
	RuntimeManifest, RuntimeManifestSHA                    string
	Ordinal                                                int
	Admitted, Execute                                      bool
}
type OutputComparison struct {
	Role, GoFile, GoSHA, NativeFile, NativeSHA string
	Bytes                                      int
	NativeOffset                               int
	Equal                                      bool
}
type ConsumerReceipt struct {
	Commands                                            []runner.CommandReceipt
	BeforeDevice, AfterDevice                           string
	Environment                                         map[string]RuntimeEnvValue
	Schema, Status, Error, Started, Finished, Attempt   string
	Invocation                                          []string
	Executable, ExecutableSHA, NativeReceipt, NativeSHA string
	ControllerBuild, ControllerBuildSHA                 string
	RuntimeManifest, RuntimeManifestSHA                 string
	Backend                                             hacgpu.BackendBuild
	Inputs, Outputs, Runtime                            map[string]string
	Comparisons                                         []OutputComparison
	Controls                                            []ConsumerControl
	ProductionExecuted                                  bool
	Limits                                              []string
}

func readCaptureReceipt(path, pin string) (r OracleCaptureReceipt, b OracleBuildReceipt, err error) {
	raw, err := readPinned(path, pin)
	if err != nil {
		return r, b, err
	}
	if err = json.Unmarshal(raw, &r); err != nil {
		return r, b, err
	}
	if r.Schema != "nanopore-hac-output-cache-capture/v1" || r.Status != "source_projection_caches_passed" || r.BeforeDevice == "" || r.BeforeDevice != r.AfterDevice || len(r.Caches) != 3 || len(r.Outputs) != 9 || len(r.Commands) != 5 {
		return r, b, fmt.Errorf("incomplete native projection cache proof")
	}
	if err = checkPins(r.Inputs); err != nil {
		return r, b, err
	}
	if err = checkPins(r.Outputs); err != nil {
		return r, b, err
	}
	if err = checkPins(r.Runtime); err != nil {
		return r, b, err
	}
	raw, err = readPinned(r.Build, r.BuildSHA)
	if err != nil {
		return r, b, err
	}
	if err = json.Unmarshal(raw, &b); err != nil {
		return r, b, err
	}
	if b.Schema != "nanopore-hac-output-oracle-build/v1" || b.Status != "closed_build_passed" || b.OutputSHA != r.OracleSHA || b.Output != r.Oracle {
		return r, b, fmt.Errorf("native cache build identity mismatch")
	}
	if err = checkPins(b.Inputs); err != nil {
		return r, b, err
	}
	if err = checkPins(b.Objects); err != nil {
		return r, b, err
	}
	for _, c := range r.Commands {
		start, e := time.Parse(time.RFC3339Nano, c.Started)
		if e != nil {
			return r, b, e
		}
		finish, e := time.Parse(time.RFC3339Nano, c.Finished)
		if e != nil || finish.Before(start) || c.Exit != 0 || len(c.Args) == 0 {
			return r, b, fmt.Errorf("invalid native command receipt")
		}
		if err = requirePin(c.Stdout, c.StdoutSHA); err != nil {
			return r, b, err
		}
		if err = requirePin(c.Stderr, c.StderrSHA); err != nil {
			return r, b, err
		}
	}
	actual, err := ReadCacheDescriptors(r.Output)
	if err != nil {
		return r, b, err
	}
	for i, c := range actual {
		if c.Role != r.Caches[i].Role || c.SHA256 != r.Caches[i].SHA256 {
			return r, b, fmt.Errorf("native cache publication changed")
		}
	}
	return r, b, nil
}
func Consumer(ctx context.Context, o ConsumerOptions) (r ConsumerReceipt, err error) {
	if !o.Admitted {
		return r, fmt.Errorf("genuine native loader/projection execution requires separate admission")
	}
	if len(o.NativeSHA) != 64 {
		return r, fmt.Errorf("independently pinned native cache receipt SHA256 required")
	}
	if len(o.ControllerBuildSHA) != 64 {
		return r, fmt.Errorf("independently pinned consumer controller build SHA256 required")
	}
	if len(o.RuntimeManifestSHA) != 64 {
		return r, fmt.Errorf("independently pinned prospective consumer runtime SHA256 required")
	}
	if ctx == nil {
		return r, fmt.Errorf("nil context")
	}
	if err = ctx.Err(); err != nil {
		return r, err
	}
	if o.Ordinal < 0 || int64(o.Ordinal) > 2147483647 {
		return r, fmt.Errorf("invalid CUDA ordinal")
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return r, err
	}
	defer lock.Close()
	r = ConsumerReceipt{Schema: "nanopore-hac-output-consumer/v1", Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Invocation: slices.Clone(os.Args), NativeReceipt: o.NativeReceipt, NativeSHA: o.NativeSHA, Inputs: map[string]string{}, Outputs: map[string]string{}, Runtime: map[string]string{}, Limits: []string{"Fixed factorised HAC output only. VCS row3 and two-LinearCRF remain H07 obligations.", "No convolution, recurrence, full model forward, basecalling, biological or speed qualification."}}
	r.Attempt, err = runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
			r.Status = "failed"
		}
		raw, e := json.MarshalIndent(r, "", "  ")
		if e == nil {
			e = os.WriteFile(filepath.Join(r.Attempt, "consumer.json"), append(raw, '\n'), 0644)
		}
		if err == nil {
			err = e
		}
	}()
	r.Executable, err = os.Executable()
	if err != nil {
		return r, err
	}
	r.ExecutableSHA, err = runner.SHA(r.Executable)
	if err != nil {
		return r, err
	}
	controller, err := ReadControllerBuild(o.ControllerBuild, o.ControllerBuildSHA, r.Executable, r.ExecutableSHA)
	if err != nil {
		return r, err
	}
	if err = checkPins(controller.SourceSHA256); err != nil {
		return r, err
	}
	r.ControllerBuild, r.ControllerBuildSHA = o.ControllerBuild, o.ControllerBuildSHA
	r.Inputs[o.ControllerBuild] = o.ControllerBuildSHA
	maps.Copy(r.Inputs, controller.SourceSHA256)
	maps.Copy(r.Inputs, controller.SnapshotSHA256)
	maps.Copy(r.Inputs, controller.ToolSHA256)
	native, build, err := readCaptureReceipt(o.NativeReceipt, o.NativeSHA)
	if err != nil {
		return r, err
	}
	r.Inputs[o.NativeReceipt] = o.NativeSHA
	w, err := ReadWitness(o.Capture)
	if err != nil {
		return r, err
	}
	maps.Copy(r.Inputs, w.Pins)
	archives, err := projectionArchivePins(o.Model)
	if err != nil {
		return r, err
	}
	for path, pin := range archives {
		absolute, e := filepath.Abs(path)
		if e != nil {
			return r, e
		}
		if native.Inputs[absolute] != pin {
			return r, fmt.Errorf("native/Go archive or weight-manifest identity differs")
		}
		r.Inputs[absolute] = pin
	}
	r.Backend, err = hacgpu.VerifyBackend(o.Backend)
	if err != nil {
		return r, err
	}
	r.Inputs[r.Backend.Path] = r.Backend.ManifestSHA
	r.Inputs[r.Backend.Output] = r.Backend.OutputSHA
	runtimeRaw, err := readPinned(o.RuntimeManifest, o.RuntimeManifestSHA)
	if err != nil {
		return r, err
	}
	var policy map[string]string
	if err = json.Unmarshal(runtimeRaw, &policy); err != nil {
		return r, err
	}
	if len(policy) < len(build.Runtime) || policy[r.Backend.Output] != r.Backend.OutputSHA || policy[r.Executable] != r.ExecutableSHA {
		return r, fmt.Errorf("prospective runtime lacks backend, consumer executable, or source runtime closure")
	}
	for path, pin := range build.Runtime {
		if policy[path] != pin {
			return r, fmt.Errorf("prospective consumer runtime differs from source runtime: %s", path)
		}
	}
	if err = checkPins(policy); err != nil {
		return r, err
	}
	r.RuntimeManifest, r.RuntimeManifestSHA = o.RuntimeManifest, o.RuntimeManifestSHA
	r.Inputs[o.RuntimeManifest] = o.RuntimeManifestSHA
	maps.Copy(r.Inputs, policy)
	engine, err := tensor.Open(o.Backend)
	if err != nil {
		return r, err
	}
	defer func() { err = errors.Join(err, engine.Close()) }()
	weights, err := model.LoadHACOutputWeights(engine, o.Model)
	if err != nil {
		return r, err
	}
	cache, err := model.PackHACOutput(weights)
	if err != nil {
		return r, err
	}
	snapshot := cache.Snapshot()
	encodeHalf := func(v []uint16) []byte {
		b := make([]byte, len(v)*2)
		for i, x := range v {
			binary.LittleEndian.PutUint16(b[i*2:], x)
		}
		return b
	}
	down := make([]byte, len(snapshot.Down))
	for i, v := range snapshot.Down {
		down[i] = byte(v)
	}
	for i, body := range [][]byte{down, encodeHalf(snapshot.DownScale), encodeHalf(snapshot.Up)} {
		n := native.Caches[i]
		path := filepath.Join(r.Attempt, "go-cache-"+n.Role+".bin")
		if err = os.WriteFile(path, body, 0644); err != nil {
			return r, err
		}
		r.Outputs[path] = sum(body)
		expected, e := readPinned(filepath.Join(native.Output, n.File), n.SHA256)
		if e != nil {
			return r, e
		}
		comparison := OutputComparison{Role: "cache-" + n.Role, GoFile: path, GoSHA: sum(body), NativeFile: filepath.Join(native.Output, n.File), NativeSHA: n.SHA256, Bytes: len(body), Equal: bytes.Equal(body, expected)}
		r.Comparisons = append(r.Comparisons, comparison)
		if !comparison.Equal {
			return r, fmt.Errorf("Go projection cache differs: %s", n.Role)
		}
	}
	if err = cpuConsumerControls(o, w, native, &r); err != nil {
		return r, err
	}
	if o.Execute {
		r.Environment = runtimeEnvironment(false)
		if !maps.Equal(r.Environment, native.Environment) {
			return r, fmt.Errorf("projection environment differs from source cache capture")
		}
		identity := func(label string) (string, error) {
			c, e := runner.Execute(r.Attempt, label, nil, native.Commands[0].Args[0], "-i", strconv.Itoa(o.Ordinal), "--query-gpu=index,uuid,pci.bus_id,driver_version", "--format=csv,noheader,nounits")
			r.Commands = append(r.Commands, c)
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
		if r.BeforeDevice != native.BeforeDevice {
			return r, fmt.Errorf("projection device differs from source cache witness")
		}
		projectionErr := executeProjection(ctx, engine, o, cache, w, &r)
		r.AfterDevice, err = identity("gpu-after")
		if err != nil || projectionErr != nil {
			return r, errors.Join(projectionErr, err)
		}
		if r.BeforeDevice != r.AfterDevice {
			return r, fmt.Errorf("projection UUID bookends differ")
		}
		r.ProductionExecuted = true
	}
	runtimeBody, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		return r, err
	}
	runtimePath := filepath.Join(r.Attempt, "consumer-runtime.maps")
	if err = os.WriteFile(runtimePath, runtimeBody, 0644); err != nil {
		return r, err
	}
	r.Outputs[runtimePath] = sum(runtimeBody)
	r.Runtime, err = VerifyRuntimeMaps(string(runtimeBody), policy)
	if err != nil {
		return r, err
	}
	if err = checkPins(r.Inputs); err != nil {
		return r, err
	}
	if err = checkPins(r.Outputs); err != nil {
		return r, err
	}
	for _, comparison := range r.Comparisons {
		if err = verifyOutputComparison(comparison); err != nil {
			return r, err
		}
	}
	for _, control := range r.Controls {
		if err = verifyConsumerControl(control); err != nil {
			return r, err
		}
	}
	if _, err = ReadControllerBuild(o.ControllerBuild, o.ControllerBuildSHA, r.Executable, r.ExecutableSHA); err != nil {
		return r, err
	}
	if _, err = hacgpu.VerifyBackend(o.Backend); err != nil {
		return r, err
	}
	r.Status = "go_projection_caches_passed"
	if o.Execute {
		r.Status = "fixed_projection_passed"
	}
	return r, nil
}
func verifyOutputComparison(c OutputComparison) error {
	goBody, err := readPinned(c.GoFile, c.GoSHA)
	if err != nil {
		return err
	}
	nativeBody, err := os.ReadFile(c.NativeFile)
	if err != nil {
		return err
	}
	if c.Bytes != len(goBody) || c.NativeOffset < 0 || c.NativeOffset > len(nativeBody) || c.Bytes > len(nativeBody)-c.NativeOffset {
		return fmt.Errorf("retained projection comparison extent failed: %s", c.Role)
	}
	nativeBody = nativeBody[c.NativeOffset : c.NativeOffset+c.Bytes]
	if sum(nativeBody) != c.NativeSHA || !bytes.Equal(goBody, nativeBody) || !c.Equal {
		return fmt.Errorf("retained projection comparison failed: %s", c.Role)
	}
	return nil
}
func executeProjection(ctx context.Context, e *tensor.Engine, o ConsumerOptions, cache model.HACOutputCache, w Witness, r *ConsumerReceipt) (err error) {
	d, err := e.NewCUDADevice(o.Ordinal)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, d.Close()) }()
	s, err := d.NewStream()
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, s.Close()) }()
	backing, err := d.NewStorage(s, tensor.DTypeU8, []int64{w.Capture.Reservation}, []tensor.KoiTag{tensor.TagC}, w.InputParent)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, backing.Close()) }()
	full, err := backing.Reinterpret(tensor.DTypeI8, uint64(w.Input.OffsetBytes), []int64{37, 64, 1024}, []int64{65536, 1024, 1}, []tensor.KoiTag{tensor.TagT, tensor.TagN, tensor.TagC})
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, full.Close()) }()
	g, err := model.NewGPUStorageGeometry(model.GPUStorageCUTLASSI8, 64, 32, 1024)
	if err != nil {
		return err
	}
	elems, err := model.HACOutputTempElements(64, 32)
	if err != nil {
		return err
	}
	tempGeometry, err := model.NewGPUScratchGeometry(model.GPUDtypeHalf, elems)
	if err != nil {
		return err
	}
	plan, err := model.PlanHACOutput(model.GPUMemoryReservation{Bytes: w.Capture.Reservation}, model.GPUView{Geometry: g, Orientation: model.GPUOrientationForward, Offset: w.Input.OffsetBytes, End: w.Input.OffsetBytes + w.Input.Bytes}, model.GPUView{Scratch: true, Geometry: tempGeometry, End: tempGeometry.Bytes})
	if err != nil {
		return err
	}
	projection, err := model.NewHACOutput(d, s, cache)
	if projection != nil {
		defer func() { err = errors.Join(err, projection.Close()) }()
	}
	if err != nil {
		return err
	}
	if err = nativeScalarControls(d, s, backing, full, cache.Snapshot(), r); err != nil {
		return err
	}
	scores, err := projection.Run(ctx, s, backing, full, plan)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, scores.Close()) }()
	output, err := scores.Download(s)
	if err != nil {
		return err
	}
	parent, err := backing.Download(s)
	if err != nil {
		return err
	}
	if len(output) != ScoresBytes || len(parent) != int(w.Capture.Reservation) {
		return fmt.Errorf("projection output or parent extent differs")
	}
	for _, record := range []struct {
		role             string
		actual, expected []byte
	}{{"scores", output, w.Scores}, {"down", parent[:DownBytes], w.Down}, {"whole-parent", parent, w.OutputParent}} {
		path := filepath.Join(r.Attempt, "go-"+record.role+".bin")
		if err = os.WriteFile(path, record.actual, 0644); err != nil {
			return err
		}
		r.Outputs[path] = sum(record.actual)
		nativeFile := filepath.Join(o.Capture, w.Output.BackingFile)
		if record.role == "scores" {
			nativeFile = filepath.Join(o.Capture, w.Output.DataFile)
		}
		comparison := OutputComparison{Role: record.role, GoFile: path, GoSHA: sum(record.actual), NativeFile: nativeFile, NativeSHA: sum(record.expected), Bytes: len(record.actual), Equal: bytes.Equal(record.actual, record.expected)}
		r.Comparisons = append(r.Comparisons, comparison)
		if !comparison.Equal {
			return fmt.Errorf("Go projection differs: %s", record.role)
		}
	}
	return nil
}
