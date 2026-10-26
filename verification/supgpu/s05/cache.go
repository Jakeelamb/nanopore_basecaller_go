// Package s05 compares Go FP8 parameter caches with retained native SUP tensors.
package s05

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/supgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type Anchor struct{ Path, SHA256 string }
type Options struct {
	Capture, Prepared, Build, Runtime Anchor
	Model, Out                        string
}
type Comparison struct {
	Layer                                                   int
	Phase, Role, GoFile, GoSHA256, NativeFile, NativeSHA256 string
	Bytes                                                   int
	Equal                                                   bool
}
type Control struct {
	Name     string
	Rejected bool
	Error    string
}
type Receipt struct {
	Schema, Status, Error, Started, Finished, Attempt string
	Options                                           Options
	Inputs, Outputs                                   map[string]string
	Comparisons                                       []Comparison
	Controls                                          []Control
	Limits                                            []string
}

func digest(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }

func captureDirectory(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	if filepath.Base(abs) != "capture.json" {
		return "", fmt.Errorf("capture anchor must name capture.json")
	}
	dir, err := filepath.EvalSymlinks(filepath.Dir(abs))
	if err != nil {
		return "", err
	}
	actual, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", err
	}
	if actual != filepath.Join(dir, "capture.json") {
		return "", fmt.Errorf("capture anchor does not identify the imported manifest")
	}
	return dir, nil
}

// The archive hash comes from model.WeightManifest, not from archive metadata.
// Only its pinned single-parameter FloatStorage profile is supported. Pickle
// bytes are authenticated and checked as data; no pickle program is executed.
func readHalfArchive(path string, p model.WeightReceipt) ([]uint16, error) {
	if filepath.Base(path) != p.Name || len(p.Shape) != 2 || len(p.SHA256) != 64 {
		return nil, fmt.Errorf("pinned matrix parameter required")
	}
	count := int64(1)
	for _, d := range p.Shape {
		if d < 1 || d > (1<<23)/count {
			return nil, fmt.Errorf("matrix extent outside profile")
		}
		count *= d
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() > 32<<20 {
		return nil, fmt.Errorf("parameter archive extent/type differs")
	}
	raw, err := io.ReadAll(io.LimitReader(f, 32<<20+1))
	if err != nil {
		return nil, err
	}
	if digest(raw) != p.SHA256 {
		return nil, fmt.Errorf("parameter archive digest differs: %s", p.Name)
	}
	z, err := zip.NewReader(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		return nil, err
	}
	root := strings.TrimSuffix(p.Name, ".tensor") + "/"
	want := map[string]bool{"data/0": true, "data.pkl": true, "code/__torch__/torch/nn/modules/module.py": true, "code/__torch__/torch/nn/modules/module.py.debug_pkl": true, "constants.pkl": true, "version": true, "byteorder": true, ".data/serialization_id": true}
	files := map[string]*zip.File{}
	for _, e := range z.File {
		name := strings.TrimPrefix(e.Name, root)
		if !strings.HasPrefix(e.Name, root) || !want[name] || files[name] != nil || e.Mode()&os.ModeSymlink != 0 {
			return nil, fmt.Errorf("parameter archive membership differs")
		}
		files[name] = e
	}
	if len(files) != len(want) {
		return nil, fmt.Errorf("parameter archive incomplete")
	}
	read := func(name string, max int64) ([]byte, error) {
		e := files[name]
		if e.UncompressedSize64 > uint64(max) {
			return nil, fmt.Errorf("archive member extent differs: %s", name)
		}
		r, e2 := e.Open()
		if e2 != nil {
			return nil, e2
		}
		defer r.Close()
		b, e2 := io.ReadAll(io.LimitReader(r, max+1))
		if e2 != nil {
			return nil, e2
		}
		if int64(len(b)) > max {
			return nil, fmt.Errorf("archive member too large")
		}
		return b, nil
	}
	order, err := read("byteorder", 16)
	if err != nil || string(order) != "little" {
		return nil, fmt.Errorf("parameter byte order differs: %v", err)
	}
	version, err := read("version", 16)
	if err != nil || string(version) != "3\n" {
		return nil, fmt.Errorf("parameter archive version differs: %v", err)
	}
	metadata, err := read("data.pkl", 4096)
	if err != nil {
		return nil, err
	}
	if !bytes.HasPrefix(metadata, []byte("\x80\x02c__torch__.torch.nn.modules.module\nModule\n")) || !bytes.Contains(metadata, []byte("ctorch\nFloatStorage\n")) {
		return nil, fmt.Errorf("parameter storage profile differs")
	}
	code, err := read("code/__torch__/torch/nn/modules/module.py", 4096)
	if err != nil {
		return nil, err
	}
	if !bytes.Contains(code, []byte("__parameters__ = [\"0\", ]")) || !bytes.Contains(code, []byte("__buffers__ = []")) {
		return nil, fmt.Errorf("single parameter zero profile required")
	}
	data, err := read("data/0", count*4)
	if err != nil {
		return nil, err
	}
	if int64(len(data)) != count*4 {
		return nil, fmt.Errorf("parameter data size differs")
	}
	out := make([]uint16, int(count))
	for i := range out {
		v := math.Float32frombits(binary.LittleEndian.Uint32(data[4*i:]))
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("nonfinite parameter")
		}
		out[i] = signal.Float16Bits(v)
		if out[i]&0x7c00 == 0x7c00 {
			return nil, fmt.Errorf("parameter overflows half")
		}
	}
	return out, nil
}

func cacheTensor(c supgpu.Capture, layer int, phase, role string) (supgpu.Tensor, error) {
	name := fmt.Sprintf("encoder.%02d.cache.%s_remove_bits", layer, phase)
	var found *supgpu.Tensor
	for _, e := range c.Events {
		if e.Name == name && e.Layer == layer {
			for _, t := range e.Tensors {
				if t.Name == role {
					v := t
					if found != nil {
						return supgpu.Tensor{}, fmt.Errorf("duplicate cache tensor")
					}
					found = &v
				}
			}
		}
	}
	if found == nil {
		return supgpu.Tensor{}, fmt.Errorf("missing %s/%s", name, role)
	}
	return *found, nil
}

func compareCache(data []byte, shape []int64, t supgpu.Tensor, native []byte) error {
	if t.DType != "Float8_e4m3fn" || !slices.Equal(shape, t.Shape) || len(data) != len(native) || int64(len(native)) != t.Data.Bytes || digest(native) != t.Data.SHA256 {
		return fmt.Errorf("FP8 cache contract or native bytes differ")
	}
	if !bytes.Equal(data, native) {
		return fmt.Errorf("Go/native FP8 cache mismatch")
	}
	return nil
}

func Run(ctx context.Context, o Options) (r Receipt, err error) {
	if ctx == nil {
		return r, fmt.Errorf("context required")
	}
	if err = ctx.Err(); err != nil {
		return r, err
	}
	captureDir, err := captureDirectory(o.Capture.Path)
	if err != nil {
		return r, err
	}
	if err = runner.CheckRoot(); err != nil {
		return r, err
	}
	buildInfo, ok := debug.ReadBuildInfo()
	pureGo := false
	if ok {
		for _, s := range buildInfo.Settings {
			if s.Key == "CGO_ENABLED" && s.Value == "0" {
				pureGo = true
			}
		}
	}
	if !pureGo {
		return r, fmt.Errorf("this retained-cache consumer requires a CGO_ENABLED=0 build")
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return r, err
	}
	defer lock.Close()
	attempt, err := runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	r = Receipt{Schema: "go-sup-fp8-cache-comparison/v1", Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Attempt: attempt, Options: o, Inputs: map[string]string{}, Outputs: map[string]string{}, Limits: []string{"FP8 parameter cache and half-cast differential only; no Go SUP forward, device load, biological or performance qualification.", "Retained source-capture002 has an explicitly documented per-role watchdog observation gap; no new native computation occurs here."}}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
		}
		b, e := json.MarshalIndent(r, "", "  ")
		if e == nil {
			e = os.WriteFile(filepath.Join(attempt, "receipt.json"), append(b, '\n'), 0600)
		}
		if err == nil {
			err = e
		}
	}()
	executable, e := os.Executable()
	if e != nil {
		return r, e
	}
	executableBytes, e := os.ReadFile(executable)
	if e != nil {
		return r, e
	}
	copyPath := filepath.Join(attempt, "consumer-executable")
	if err = os.WriteFile(copyPath, executableBytes, 0500); err != nil {
		return r, err
	}
	r.Outputs[copyPath] = digest(executableBytes)
	infoPath := filepath.Join(attempt, "go-build-info.txt")
	if err = os.WriteFile(infoPath, []byte(buildInfo.String()), 0600); err != nil {
		return r, err
	}
	r.Outputs[infoPath] = digest([]byte(buildInfo.String()))
	for _, path := range []string{"internal/model/sup_gpu_fp8.go", "internal/model/sup_weights.json", "internal/signal/half.go", "verification/supgpu/s05/cache.go", "verification/supgpu/s05/cmd/main.go"} {
		absolute, e := filepath.Abs(path)
		if e != nil {
			return r, e
		}
		sha, e := runner.SHA(absolute)
		if e != nil {
			return r, e
		}
		r.Inputs[absolute] = sha
	}
	for _, p := range []Anchor{o.Capture, o.Prepared, o.Build, o.Runtime} {
		s, e := runner.SHA(p.Path)
		if e != nil {
			return r, e
		}
		if s != p.SHA256 {
			return r, fmt.Errorf("external anchor changed: %s", p.Path)
		}
		r.Inputs[p.Path] = s
	}
	r.Inputs[filepath.Join(captureDir, "capture.json")] = o.Capture.SHA256
	capture, err := supgpu.Import(captureDir, o.Prepared.Path, o.Prepared.SHA256, o.Build.Path, o.Build.SHA256, o.Runtime.Path, o.Runtime.SHA256)
	if err != nil {
		return r, err
	}
	if !capture.UseF8 || !capture.UseI8 {
		return r, fmt.Errorf("fixed F8/I8 source capture required")
	}
	manifest, err := model.WeightManifest(model.SUP)
	if err != nil {
		return r, err
	}
	weights := map[string]model.WeightReceipt{}
	for _, p := range manifest {
		weights[p.Name] = p
	}
	for layer := 0; layer < 18; layer++ {
		if err = ctx.Err(); err != nil {
			return r, err
		}
		var input [2][]uint16
		for i, role := range []string{"fc1", "fc2"} {
			name := fmt.Sprintf("transformer_encoder.%d.ff.%s.weight.tensor", layer, role)
			p, ok := weights[name]
			if !ok {
				return r, fmt.Errorf("missing authoritative parameter")
			}
			path := filepath.Join(o.Model, name)
			input[i], err = readHalfArchive(path, p)
			if err != nil {
				return r, err
			}
			r.Inputs[path] = p.SHA256
		}
		cache, e := model.PackSUPFeedForwardFP8(input[0], input[1])
		if e != nil {
			return r, e
		}
		snapshot := cache.Snapshot()
		for i, role := range []string{"fc1_fp8", "fc2_fp8"} {
			data := [][]byte{snapshot.FC1, snapshot.FC2}[i]
			shape := [][]int64{snapshot.FC1Shape, snapshot.FC2Shape}[i]
			path := filepath.Join(attempt, fmt.Sprintf("layer-%02d-%s.bin", layer, role))
			if err = os.WriteFile(path, data, 0600); err != nil {
				return r, err
			}
			r.Outputs[path] = digest(data)
			for _, phase := range []string{"before", "after"} {
				t, e := cacheTensor(capture, layer, phase, role)
				if e != nil {
					return r, e
				}
				nativePath := filepath.Join(captureDir, t.Data.Path)
				native, e := os.ReadFile(nativePath)
				if e != nil {
					return r, e
				}
				if e = compareCache(data, shape, t, native); e != nil {
					return r, fmt.Errorf("layer%d %s %s: %w", layer, phase, role, e)
				}
				r.Inputs[nativePath] = t.Data.SHA256
				r.Comparisons = append(r.Comparisons, Comparison{layer, phase, role, path, digest(data), nativePath, t.Data.SHA256, len(data), true})
				if layer == 0 && i == 0 && phase == "before" {
					bad := slices.Clone(data)
					bad[0] ^= 1
					changed := compareCache(bad, shape, t, native)
					badShape := slices.Clone(shape)
					badShape[0], badShape[1] = badShape[1], badShape[0]
					wrongShape := compareCache(data, badShape, t, native)
					for _, v := range []struct {
						name string
						e    error
					}{{"changed-go-byte", changed}, {"same-size-wrong-shape", wrongShape}} {
						r.Controls = append(r.Controls, Control{v.name, v.e != nil, fmt.Sprint(v.e)})
						if v.e == nil {
							return r, fmt.Errorf("ineffective control %s", v.name)
						}
					}
				}
			}
		}
	}
	if len(r.Comparisons) != 72 || len(r.Controls) != 2 {
		return r, fmt.Errorf("comparison/control coverage incomplete")
	}
	for p, s := range r.Inputs {
		actual, e := runner.SHA(p)
		if e != nil {
			return r, e
		}
		if actual != s {
			return r, fmt.Errorf("input changed during comparison: %s", p)
		}
	}
	r.Status = "native_fp8_caches_equal"
	return r, nil
}
