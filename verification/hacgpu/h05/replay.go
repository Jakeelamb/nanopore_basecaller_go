// Package h05 consumes retained native stages and H04 caches. It never runs a
// producer. Native execution requires a separate explicit admission.
package h05

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type Options struct {
	Capture, Caches, Backend, Out string
	Ordinal                       int
	Admitted                      bool
	Controls                      bool
}
type StageResult struct {
	Name            string   `json:"name"`
	Equal           bool     `json:"equal"`
	ComparedBytes   int      `json:"compared_bytes"`
	GotSHA          string   `json:"got_sha256"`
	NativeSHA       string   `json:"native_sha256"`
	ExcludedRows    [][2]int `json:"excluded_rows"`
	CandidateFile   string   `json:"candidate_inout_file,omitempty"`
	FullSHA         string   `json:"candidate_inout_sha256,omitempty"`
	NativeRejection string   `json:"native_rejection,omitempty"`
}
type Control struct {
	Name              string       `json:"name"`
	Separated         bool         `json:"separated"`
	Error             string       `json:"error,omitempty"`
	ObservedRejection string       `json:"observed_rejection,omitempty"`
	Stage             *StageResult `json:"stage,omitempty"`
}
type Receipt struct {
	Schema                 string                  `json:"schema"`
	Status                 string                  `json:"status"`
	Error                  string                  `json:"error,omitempty"`
	Started                string                  `json:"started_utc"`
	Finished               string                  `json:"finished_utc"`
	Capture                string                  `json:"capture"`
	Caches                 string                  `json:"caches"`
	Backend                string                  `json:"backend"`
	Inputs                 map[string]string       `json:"input_sha256"`
	BackendBuild           hacgpu.BackendBuild     `json:"backend_build"`
	Stages                 []StageResult           `json:"production_stages"`
	Stack                  StageResult             `json:"production_stack"`
	Controls               []Control               `json:"controls"`
	Gaps                   []string                `json:"gaps"`
	Environment            map[string]string       `json:"environment"`
	Runtime                map[string]string       `json:"runtime_sha256"`
	Commands               []runner.CommandReceipt `json:"commands"`
	BackendInputsRechecked bool                    `json:"backend_inputs_rechecked_at_finish"`
	Scratch                []ScratchResult         `json:"production_scratch"`
	NativeControlsExecuted bool                    `json:"native_controls_executed"`
	Executable             string                  `json:"retained_executable"`
	ExecutableSHA          string                  `json:"executable_sha256"`
	Invocation             []string                `json:"consumer_argv"`
}
type prepared struct {
	capture    hacgpu.Capture
	caches     hacgpu.FLSTMCaches
	host       []model.FLSTMCache
	conv       hacgpu.Stage
	initial    []byte
	plan       model.HACFusedPlan
	last       model.HACLastConvPlan
	scratch    model.GPUView
	arenaBytes int64
}

func sum(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }

func prepare(o Options) (p prepared, err error) {
	p.capture, err = hacgpu.Import(o.Capture)
	if err != nil {
		return p, err
	}
	p.caches, err = hacgpu.ImportFLSTMCaches(o.Caches)
	if err != nil {
		return p, err
	}
	c, w := p.capture, p.caches
	if c.LSTMLayers != 7 || c.LSTMSize != 1024 || c.LSTMInnerDim != 128 || c.Batch%64 != 0 || c.TOut <= 0 || c.Chunk <= 0 || !w.FirstReverse || w.C != 1024 || w.K != 128 || w.LSTMLayers != 7 || c.Case != w.Case || c.ModelPackage != w.ModelPackage || c.SourceCommit != w.SourceCommit || c.Batch != w.Batch || c.Chunk != w.Chunk || c.Features != w.Features {
		return p, fmt.Errorf("H05 requires matching fixed C1024 K128 seven-layer captures")
	}
	input, ok := c.Stage("input")
	if !ok || input.DataSHA != w.InputSHA256 {
		return p, fmt.Errorf("H01/H04 input pins differ")
	}
	p.conv, ok = c.Stage("conv2")
	if !ok || p.conv.Layout != "CUTLASS_TNC_I8" || p.conv.DType != "Char" || !p.conv.InputToRevLSTM {
		return p, fmt.Errorf("H05 needs reverse-input I8 conv2")
	}
	p.initial, err = stageBytes(c, o.Capture, "conv2")
	if err != nil {
		return p, err
	}
	n, t := int64(c.Batch), int64(c.TOut)
	geometry, err := model.NewGPUStorageGeometry(model.GPUStorageCUTLASSI8, n, t, 1024)
	if err != nil {
		return p, err
	}
	if !reflect.DeepEqual(p.conv.Shape, []int64{t + 5, n, 1024}) || p.conv.Bytes != geometry.Bytes {
		return p, fmt.Errorf("conv2 full-allocation geometry mismatch")
	}
	// This isolated recurrence consumer uses one arena with current at its front.
	// Whole-model alternating placement remains the fixed graph caller's owner.
	p.last = model.HACLastConvPlan{Output: geometry, Layer: model.HACSmallConvLayer{TOut: t}, InputToRevLSTM: true}
	size, err := model.HACFusedScratchBytes(n, t)
	if err != nil {
		return p, err
	}
	scratch, err := model.NewGPUScratchGeometry(model.GPUDtypeInt8, size)
	if err != nil {
		return p, err
	}
	p.scratch = model.GPUView{Scratch: true, Geometry: scratch, Offset: geometry.Bytes, End: geometry.Bytes + scratch.Bytes}
	p.plan, err = model.PlanHACFusedStack(&p.last, p.scratch)
	if err != nil {
		return p, err
	}
	p.arenaBytes = p.scratch.End
	for i := 0; i < 7; i++ {
		s, ok := c.Stage(fmt.Sprintf("flstm%d", i))
		if !ok {
			return p, fmt.Errorf("missing layer %d", i)
		}
		want := [][2]int{{int(t + 3), 2}}
		if i%2 == 0 {
			want = [][2]int{{1, 1}, {int(t + 3), 2}}
		}
		if s.DType != "Char" || s.Layout != "CUTLASS_TNC_I8" || s.N != int(n) || s.T != int(t) || s.C != 1024 || !reflect.DeepEqual(s.Shape, []int64{t + 5, n, 1024}) || s.InputToRevLSTM != (i%2 == 1) || !reflect.DeepEqual(s.UnwrittenT, want) {
			return p, fmt.Errorf("layer %d does not match prospective source row/orientation contract", i)
		}
	}
	p.host = make([]model.FLSTMCache, 7)
	for i, l := range w.Layers {
		signed := func(role string) ([]int8, error) {
			b, e := l.Bytes(o.Caches, role)
			if e != nil {
				return nil, e
			}
			v := make([]int8, len(b))
			for j, x := range b {
				v[j] = int8(x)
			}
			return v, nil
		}
		half := func(role string) ([]uint16, error) {
			b, e := l.Bytes(o.Caches, role)
			if e != nil {
				return nil, e
			}
			if len(b)%2 != 0 {
				return nil, fmt.Errorf("odd half payload")
			}
			v := make([]uint16, len(b)/2)
			for j := range v {
				v[j] = binary.LittleEndian.Uint16(b[2*j:])
			}
			return v, nil
		}
		v := &p.host[i]
		v.Params = model.FLSTMParams{C: 1024, K: 128}
		if v.DownIH, err = signed("dn-ih-cache"); err != nil {
			return p, err
		}
		if v.DownHH, err = signed("dn-hh-cache"); err != nil {
			return p, err
		}
		if v.DownScaleIH, err = half("dn-ih-cache-scale"); err != nil {
			return p, err
		}
		if v.DownScaleHH, err = half("dn-hh-cache-scale"); err != nil {
			return p, err
		}
		if v.Up, err = signed("up-cache"); err != nil {
			return p, err
		}
		if v.UpScale, err = half("up-cache-scale"); err != nil {
			return p, err
		}
		if v.UpBias, err = half("up-cache-bias"); err != nil {
			return p, err
		}
	}
	return p, nil
}

// Inspect checks genuine retained inputs without loading the native backend or
// acquiring the host lock. It is a consumer preflight, not numerical parity.
func Inspect(o Options) error { _, err := prepare(o); return err }

func stageBytes(c hacgpu.Capture, dir, name string) ([]byte, error) {
	stage, ok := c.Stage(name)
	if !ok {
		return nil, fmt.Errorf("missing stage %s", name)
	}
	b, err := c.RawPayload(dir, name)
	if err != nil {
		return nil, err
	}
	if sum(b) != stage.DataSHA {
		return nil, fmt.Errorf("stage %s payload changed after import", name)
	}
	return b, nil
}

func compare(p prepared, dir, name string, observed []byte) (StageResult, error) {
	stage, ok := p.capture.Stage(name)
	if !ok {
		return StageResult{}, fmt.Errorf("missing %s", name)
	}
	expected, err := stageBytes(p.capture, dir, name)
	if err != nil {
		return StageResult{}, err
	}
	if len(observed) != len(expected) {
		return StageResult{}, fmt.Errorf("%s byte count differs", name)
	}
	rowBytes := p.capture.Batch * 1024
	var got, want []byte
	for row := 0; row < p.capture.TOut+5; row++ {
		skip := false
		for _, span := range stage.UnwrittenT {
			if row >= span[0] && row < span[0]+span[1] {
				skip = true
			}
		}
		if !skip {
			got = append(got, observed[row*rowBytes:(row+1)*rowBytes]...)
			want = append(want, expected[row*rowBytes:(row+1)*rowBytes]...)
		}
	}
	return StageResult{Name: name, Equal: bytes.Equal(got, want), ComparedBytes: len(got), GotSHA: sum(got), NativeSHA: sum(want), ExcludedRows: stage.UnwrittenT}, nil
}

func Run(ctx context.Context, o Options) (r Receipt, err error) {
	r = Receipt{Schema: "nanopore-hacgpu-fixed-flstm-receipt/v1", Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Capture: o.Capture, Caches: o.Caches, Backend: o.Backend, Inputs: map[string]string{}, Gaps: []string{"This recurrence consumer does not establish whole-model, GPU performance, or biological acceptance."}}
	var persist func() error
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Status = "failed"
			r.Error = err.Error()
		}
		if persist != nil {
			err = errors.Join(err, persist())
		}
	}()
	if !o.Admitted {
		return r, fmt.Errorf("H05 native recurrence requires explicit coordinator admission")
	}
	p, err := prepare(o)
	if err != nil {
		return r, err
	}
	r.BackendBuild, err = hacgpu.VerifyBackend(o.Backend)
	if err != nil {
		return r, err
	}
	abs, err := filepath.Abs(o.Backend)
	if err != nil {
		return r, err
	}
	pin, err := runner.SHA(abs)
	if err != nil {
		return r, err
	}
	if pin != r.BackendBuild.OutputSHA {
		return r, fmt.Errorf("requested backend is not manifest output")
	}
	found := false
	for path := range r.BackendBuild.Inputs {
		if filepath.Base(path) == "hac_flstm.inc" {
			found = true
		}
	}
	if !found {
		return r, fmt.Errorf("backend manifest does not pin the FLSTM boundary")
	}
	for _, path := range []string{filepath.Join(o.Capture, "capture.json"), filepath.Join(o.Caches, "caches.json"), "internal/model/hac_gpu_flstm.go", "internal/model/hac_gpu_flstm_weights.go", "internal/tensor/cuda_flstm.go", "verification/hacgpu/h05/replay.go", "verification/hacgpu/h05/observe.go", "verification/tensor/hac_flstm.inc"} {
		r.Inputs[path], err = runner.SHA(path)
		if err != nil {
			return r, err
		}
	}
	attempt, err := runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	originalExe, exeErr := os.Executable()
	if exeErr != nil {
		return r, exeErr
	}
	executable, exeErr := os.ReadFile(originalExe)
	if exeErr != nil {
		return r, exeErr
	}
	r.Executable = filepath.Join(attempt, "consumer")
	r.ExecutableSHA = sum(executable)
	r.Invocation = append([]string(nil), os.Args...)
	if e := os.WriteFile(r.Executable, executable, 0700); e != nil {
		return r, e
	}
	r.Inputs[r.Executable] = r.ExecutableSHA
	persist = func() error {
		b, e := json.MarshalIndent(r, "", "  ")
		if e != nil {
			return e
		}
		path := filepath.Join(attempt, "receipt.json")
		tmp := path + ".tmp"
		if e := os.WriteFile(tmp, append(b, '\n'), 0600); e != nil {
			return e
		}
		return os.Rename(tmp, path)
	}
	r.Status = "prepared"
	if e := persist(); e != nil {
		return r, e
	}

	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return r, err
	}
	defer lock.Close()
	r.Environment = map[string]string{}
	for _, key := range []string{"CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG", "DORADO_MATMUL_FP16_MODE", "LD_LIBRARY_PATH", "LD_PRELOAD", "OMP_NUM_THREADS", "GOMAXPROCS", "GOMEMLIMIT"} {
		value, present := os.LookupEnv(key)
		r.Environment[key] = value
		r.Environment[key+".present"] = fmt.Sprint(present)
	}
	bookend := func(label string) error {
		rec, e := runner.Execute(attempt, label, []string{"LD_LIBRARY_PATH=", "LD_PRELOAD="}, "nvidia-smi", "--query-gpu=index,uuid,driver_version,memory.total,memory.used,temperature.gpu", "--format=csv,noheader,nounits")
		r.Commands = append(r.Commands, rec)
		return e
	}
	if e := bookend("before-gpu-state"); e != nil {
		return r, e
	}
	defer func() { err = errors.Join(err, bookend("after-gpu-state")) }()
	e, err := tensor.Open(o.Backend)
	if err != nil {
		return r, err
	}
	defer func() { err = errors.Join(err, e.Close()) }()
	d, err := e.NewCUDADevice(o.Ordinal)
	if err != nil {
		return r, err
	}
	defer func() { err = errors.Join(err, d.Close()) }()
	s, err := d.NewStream()
	if err != nil {
		return r, err
	}
	defer func() { err = errors.Join(err, s.Close()) }()
	r.Runtime, err = runtimePins(r.BackendBuild)
	if err != nil {
		return r, err
	}
	stack, err := model.NewHACFusedStack(d, s, p.host)
	if stack != nil {
		defer func() { err = errors.Join(err, stack.Close()) }()
	}
	if err != nil {
		return r, err
	}
	arena := bytes.Repeat([]byte{0x5a}, int(p.arenaBytes))
	copy(arena, p.initial)
	backing, err := d.NewStorage(s, tensor.DTypeU8, []int64{p.arenaBytes}, []tensor.KoiTag{tensor.TagC}, arena)
	if err != nil {
		return r, err
	}
	defer func() { err = errors.Join(err, backing.Close()) }()
	r.Status = "running"
	if e := persist(); e != nil {
		return r, e
	}
	if e := progress(attempt, "production_start", -1); e != nil {
		return r, e
	}
	result, err := stack.RunObserved(ctx, s, backing, p.plan, func(i int) error { return progress(attempt, "production_layer_start", i) }, func(i int, full *tensor.Storage) error {
		b, e := full.Download(s)
		if e != nil {
			return e
		}
		if e := os.WriteFile(filepath.Join(attempt, fmt.Sprintf("flstm%d-inout.bin", i)), b, 0600); e != nil {
			return e
		}
		whole, e := backing.Download(s)
		if e != nil {
			return e
		}
		scratch, e := observeScratch(p, o.Capture, attempt, i, whole)
		if e != nil {
			return e
		}
		r.Scratch = append(r.Scratch, scratch)
		rec, e := compare(p, o.Capture, fmt.Sprintf("flstm%d", i), b)
		if e != nil {
			return e
		}
		rec.CandidateFile = filepath.Join(attempt, fmt.Sprintf("flstm%d-inout.bin", i))
		rec.FullSHA = sum(b)
		r.Stages = append(r.Stages, rec)
		if e := persist(); e != nil {
			return e
		}
		if e := progress(attempt, "production_layer_observed", i); e != nil {
			return e
		}
		if !scratch.Equal {
			return fmt.Errorf("production layer %d scratch differs", i)
		}
		if !rec.Equal {
			return fmt.Errorf("production layer %d differs", i)
		}
		return nil
	})
	if err != nil {
		return r, err
	}
	observed, err := result.Download(s)
	closeErr := result.Close()
	if err != nil || closeErr != nil {
		return r, errors.Join(err, closeErr)
	}
	r.Stack, err = compare(p, o.Capture, "flstm6", observed)
	if err != nil {
		return r, err
	}
	if !r.Stack.Equal {
		return r, fmt.Errorf("production stack differs from retained flstm6")
	}
	if e := progress(attempt, "production_complete", 6); e != nil {
		return r, e
	}
	if o.Controls {
		r.NativeControlsExecuted = true
		for _, name := range []string{"wrong-parity-pair", "parity-out-of-range", "cache-order"} {
			if e := progress(attempt, "control_start:"+name, -1); e != nil {
				return r, e
			}
			stage, e := singleLayerControl(ctx, d, s, p, o, attempt, name)
			control := Control{Name: name}
			if e != nil {
				control.Error = e.Error()
				r.Controls = append(r.Controls, control)
				return r, fmt.Errorf("control %s execution failed: %w", name, e)
			}
			control.Separated = !stage.Equal
			control.ObservedRejection = stage.NativeRejection
			if name == "cache-order" {
				control.Stage = &stage
			}
			r.Controls = append(r.Controls, control)
			if !control.Separated {
				return r, fmt.Errorf("control %s did not separate", name)
			}
		}
	}
	r.BackendInputsRechecked = true
	if _, e := hacgpu.VerifyBackend(o.Backend); e != nil {
		r.BackendInputsRechecked = false
		return r, e
	}
	for path, pin := range r.Inputs {
		got, e := runner.SHA(path)
		if e != nil {
			return r, e
		}
		if got != pin {
			return r, fmt.Errorf("input changed during execution: %s", path)
		}
	}
	r.Status = "fixed_recurrence_outputs_equal"
	return r, nil
}

// singleLayerControl deliberately changes one first-layer argument. It never
// runs a second layer schedule. The positive path uses RunObserved exclusively.
func singleLayerControl(ctx context.Context, d *tensor.Device, s *tensor.Stream, p prepared, o Options, attempt, mutant string) (stage StageResult, err error) {
	var owned []*tensor.Storage
	defer func() {
		for i := len(owned) - 1; i >= 0; i-- {
			err = errors.Join(err, owned[i].Close())
		}
	}()
	newStorage := func(dt tensor.DType, shape []int64, data []byte) (*tensor.Storage, error) {
		tags := make([]tensor.KoiTag, len(shape))
		for i := range tags {
			tags[i] = tensor.TagC
		}
		v, e := d.NewStorage(s, dt, shape, tags, data)
		if e == nil {
			owned = append(owned, v)
		}
		return v, e
	}
	n, t := int64(p.capture.Batch), int64(p.capture.TOut)
	initial := append([]byte(nil), p.initial...)
	row := int(n) * 1024
	clear(initial[:row])
	clear(initial[(int(t)+1)*row : (int(t)+2)*row])
	copy(initial[(int(t)+2)*row:(int(t)+3)*row], bytes.Repeat([]byte{1}, row))
	inout, err := newStorage(tensor.DTypeI8, []int64{t + 5, n, 1024}, initial)
	if err != nil {
		return stage, err
	}
	sizes := []int64{2 * n, 512 * (t + 1) * n, 4 * (t + 1) * n}
	scratch := make([]*tensor.Storage, 3)
	for i, size := range sizes {
		scratch[i], err = newStorage(tensor.DTypeU8, []int64{size}, make([]byte, int(size)))
		if err != nil {
			return stage, err
		}
	}

	if e := ctx.Err(); e != nil {
		return stage, e
	}
	layer := p.caches.Layers[0]
	if mutant == "cache-order" {
		layer = p.caches.Layers[1]
	}
	op := tensor.FactorisedLSTMOperands{InOut: inout, Workspace: scratch[0], Down: scratch[1], DownScale: scratch[2]}
	slots := []**tensor.Storage{&op.DownIH, &op.DownHH, &op.DownScaleIH, &op.DownScaleHH, &op.Up, &op.UpScale, &op.UpBias}
	roles := []string{"dn-ih-cache", "dn-hh-cache", "dn-ih-cache-scale", "dn-hh-cache-scale", "up-cache", "up-cache-scale", "up-cache-bias"}
	for j, role := range roles {
		spec, ok := layer.Tensor(role)
		if !ok {
			return stage, fmt.Errorf("missing cache %s", role)
		}
		b, e := layer.Bytes(o.Caches, role)
		if e != nil {
			return stage, e
		}
		dtype := tensor.DTypeI8
		if spec.DType == "Half" {
			dtype = tensor.DTypeF16
		}
		*slots[j], err = newStorage(dtype, spec.Shape, b)
		if err != nil {
			return stage, err
		}
	}
	direction, parity := int32(-1), int32(1)
	if mutant == "wrong-parity-pair" {
		parity = 0
	}
	if mutant == "parity-out-of-range" {
		parity = 2
	}
	if mutant == "wrong-parity-pair" || mutant == "parity-out-of-range" {
		e := d.FactorisedLSTMUnchecked(s, op, tensor.FactorisedLSTMGeometry{N: int32(n), T: int32(t), Direction: direction, Parity: parity})
		if e == nil {
			return stage, fmt.Errorf("invalid parity accepted by native guard")
		}
		if mutant == "wrong-parity-pair" {
			var rejected *tensor.CUDAUnsupportedError
			if !errors.As(e, &rejected) || !strings.Contains(e.Error(), "source direction/parity pair") {
				return stage, fmt.Errorf("unexpected pair rejection: %w", e)
			}
		} else {
			if !isNativeScalarGuard(e) {
				return stage, fmt.Errorf("unexpected scalar rejection: %w", e)
			}
		}
		return StageResult{Name: mutant, Equal: false, NativeRejection: e.Error()}, nil
	}
	if e := d.FactorisedLSTM(s, op, tensor.FactorisedLSTMGeometry{N: int32(n), T: int32(t), Direction: direction, Parity: parity}); e != nil {
		return stage, e
	}
	b, e := inout.Download(s)
	if e != nil {
		return stage, e
	}
	return retainCacheOrderEvidence(p, o.Capture, attempt, b)
}

func runtimePins(build hacgpu.BackendBuild) (map[string]string, error) {
	raw, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		return nil, err
	}
	out := map[string]string{}
	koi, torchCUDA, cudart := false, false, false
	// Koi is statically linked by the maintained build. Its archive pin is
	// verified with every build input and the resulting backend digest.
	for path := range build.Inputs {
		if filepath.Base(path) == "libkoi.a" {
			koi = true
		}
	}
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		path := fields[5]
		if !filepath.IsAbs(path) {
			continue
		}
		base := filepath.Base(path)
		if !strings.HasPrefix(base, "libtorch") && !strings.HasPrefix(base, "libc10") && !strings.HasPrefix(base, "libcuda") && !strings.HasPrefix(base, "libcublas") && !strings.HasPrefix(base, "libkoi") {
			continue
		}
		if _, ok := out[path]; ok {
			continue
		}
		pin, e := runner.SHA(path)
		if e != nil {
			return nil, e
		}
		out[path] = pin
		if strings.HasPrefix(base, "libtorch") || strings.HasPrefix(base, "libc10") {
			if !strings.HasPrefix(path, build.Torch+string(filepath.Separator)) {
				return nil, fmt.Errorf("mapped torch library outside qualified root: %s", path)
			}
		}
		if strings.HasPrefix(base, "libtorch_cuda") {
			torchCUDA = true
		}
		if strings.HasPrefix(base, "libcudart.so.13") {
			cudart = true
		}
		if strings.HasPrefix(base, "libkoi") {
			for p, want := range build.Inputs {
				if strings.HasPrefix(filepath.Base(p), "libkoi") && want == pin {
					koi = true
				}
			}
		}
	}
	if !koi || !torchCUDA || !cudart {
		return nil, fmt.Errorf("runtime maps lack matching pinned Koi/qualified Torch CUDA/CUDA13 runtime: koi=%t torch=%t cudart13=%t", koi, torchCUDA, cudart)
	}
	return out, nil
}

const nativeScalarGuardReason = "fixed FLSTM requires N64, positive T, direction +/-1, parity 0/1"

// NT_CUDA_ARGUMENT is ABI status 1. cudaStatus maps it to a boundary CUDAError,
// with validation stage and no CUDA runtime code. The pinned source owns the
// exact rejection reason, so unrelated boundary errors cannot satisfy a probe.
func isNativeScalarGuard(err error) bool {
	var rejected *tensor.CUDAError
	return errors.As(err, &rejected) && rejected.Op == "factorised_lstm" && rejected.Subsystem == "boundary" && rejected.Detail.Stage == 0 && rejected.Detail.NativeCode == 0 && rejected.Reason == nativeScalarGuardReason
}
