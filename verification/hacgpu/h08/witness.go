// Package h08 authenticates retained source observations and verifies the same
// production HACGPU.Forward used by callers. It contains no source producer.
package h08

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h03"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h04"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h07"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type Pin = nativeproof.Pin
type Sources struct {
	Capture                             string
	Convolution, Recurrence, Projection Pin
}

func DefaultSources() Sources {
	const base = "/home/jake/Projects/nanopore_basecaller_go/artifacts/tensor/"
	return Sources{h07.DefaultCapture, Pin{Path: base + "h03-post-overlap-differential-20261024-001/receipt.json", SHA256: "26a08388decee22d25ea0a6d65910690b99997f4144e1d8ab3c14868aae2baf8"}, Pin{Path: base + "h04-current-consumer-20261025-004/consumer-receipt.json", SHA256: "d54c560c6bacd5f6408daac07b021d1a0f9da4d1a2f2c152854dca1a8ec8c400"}, Pin{Path: base + "h07-source-caches-002/capture.json", SHA256: "39dbcd8f0e86c81ea2c78869363169475649d1e00fb2a7087777b856fde022dd"}}
}
func digest(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func readPin(p Pin) ([]byte, error) {
	if len(p.SHA256) != 64 {
		return nil, fmt.Errorf("independent SHA256 required: %s", p.Path)
	}
	b, e := os.ReadFile(p.Path)
	if e != nil {
		return nil, e
	}
	if digest(b) != p.SHA256 {
		return nil, fmt.Errorf("retained SHA256 mismatch: %s", p.Path)
	}
	return b, nil
}
func readJSON(p Pin, v any) error {
	b, e := readPin(p)
	if e != nil {
		return e
	}
	return json.Unmarshal(b, v)
}
func checkPins(pins map[string]string) error {
	for _, path := range hacgpu.SortedKeys(pins) {
		if _, err := readPin(Pin{Path: path, SHA256: pins[path]}); err != nil {
			return err
		}
	}
	return nil
}

type ByteRange struct{ Offset, Bytes int64 }
type StageContract struct {
	Name, Layout, DType string
	Shape               []int64
	Offset, Bytes       int64
	Orientation         model.GPUOrientation
	NativeUnwritten     [][2]int
	Excluded            [][2]int
	Defined             []ByteRange
}

func stageContracts() ([]StageContract, error) {
	p, err := model.PlanHACGPU(model.FixedHACGPUProfile())
	if err != nil {
		return nil, err
	}
	views := p.Views()
	out := []StageContract{}
	for i, name := range []string{"input", "conv0", "conv1", "conv2"} {
		v := views[i]
		g := v.Geometry
		s := StageContract{Name: name, Layout: g.Layout.String(), DType: "Half", Shape: []int64{g.N, g.Rows, g.C}, Offset: v.Offset, Bytes: g.LiveBytes, Orientation: v.Orientation}
		if i == 2 {
			s.NativeUnwritten = [][2]int{{0, 9}, {201, 9}}
		}
		if i == 3 {
			s.Shape = []int64{37, 64, 1024}
			s.DType = "Char"
			s.NativeUnwritten = [][2]int{{0, 1}, {33, 4}}
		}
		s.Excluded = slices.Clone(s.NativeUnwritten)
		out = append(out, s)
	}
	for i := 0; i < 7; i++ {
		unwritten := [][2]int{{35, 2}}
		orientation := model.GPUOrientationReverse
		if i%2 == 0 {
			unwritten = [][2]int{{1, 1}, {35, 2}}
			orientation = model.GPUOrientationForward
		}
		// Conv2 writes row1, and the stack initializes rows0,33,34. Every row
		// through34 is therefore source-initialized, including preserved row1.
		out = append(out, StageContract{Name: fmt.Sprintf("flstm%d", i), Layout: "CUTLASS_TNC_I8", DType: "Char", Shape: []int64{37, 64, 1024}, Offset: views[3].Offset, Bytes: views[3].Geometry.LiveBytes, Orientation: orientation, NativeUnwritten: unwritten, Excluded: [][2]int{{35, 2}}})
	}
	for _, name := range []string{"linear1", "scores"} {
		out = append(out, StageContract{Name: name, Layout: "NTC", DType: "Half", Shape: []int64{64, 32, 1024}, Offset: 524288, Bytes: 4194304, Orientation: model.GPUOrientationForward})
	}
	for i := range out {
		out[i].Defined, err = definedRanges(out[i])
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}
func definedRanges(s StageContract) ([]ByteRange, error) {
	if len(s.Shape) != 3 {
		return nil, fmt.Errorf("stage shape is not rank3")
	}
	elem := int64(2)
	if s.DType == "Char" {
		elem = 1
	} else if s.DType != "Half" {
		return nil, fmt.Errorf("unknown stage dtype")
	}
	n, t, c := s.Shape[0], s.Shape[1], s.Shape[2]
	leading := s.Layout == "CUTLASS_TNC_I8"
	if leading {
		n, t = s.Shape[1], s.Shape[0]
	} else if s.Layout != "NTC" {
		return nil, fmt.Errorf("unknown stage layout")
	}
	if n <= 0 || t <= 0 || c <= 0 || n*t*c*elem != s.Bytes {
		return nil, fmt.Errorf("inconsistent stage extent")
	}
	excluded := make([]bool, t)
	for _, r := range s.Excluded {
		if r[0] < 0 || r[1] <= 0 || int64(r[0])+int64(r[1]) > t {
			return nil, fmt.Errorf("invalid prospective exclusion")
		}
		for j := r[0]; j < r[0]+r[1]; j++ {
			if excluded[j] {
				return nil, fmt.Errorf("overlapping prospective exclusion")
			}
			excluded[j] = true
		}
	}
	out := []ByteRange{}
	add := func(offset, size int64) {
		if len(out) > 0 && out[len(out)-1].Offset+out[len(out)-1].Bytes == offset {
			out[len(out)-1].Bytes += size
		} else {
			out = append(out, ByteRange{offset, size})
		}
	}
	if leading {
		for ti := int64(0); ti < t; ti++ {
			if !excluded[ti] {
				add(ti*n*c*elem, n*c*elem)
			}
		}
	} else {
		for ni := int64(0); ni < n; ni++ {
			for ti := int64(0); ti < t; ti++ {
				if !excluded[ti] {
					add((ni*t+ti)*c*elem, c*elem)
				}
			}
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("stage comparison has no defined bytes")
	}
	return out, nil
}
func selectedBytes(s StageContract, b []byte) ([]byte, error) {
	if int64(len(b)) != s.Bytes {
		return nil, fmt.Errorf("stage%s byte extent differs", s.Name)
	}
	out := []byte{}
	for _, r := range s.Defined {
		if r.Offset < 0 || r.Bytes <= 0 || r.Offset > int64(len(b)) || r.Bytes > int64(len(b))-r.Offset {
			return nil, fmt.Errorf("invalid declared stage range")
		}
		out = append(out, b[r.Offset:r.Offset+r.Bytes]...)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("empty stage comparison")
	}
	return out, nil
}

type CacheContract struct {
	Role               string
	DType              tensor.DType
	Shape              []int64
	Bytes              int
	SHA256, NativeFile string
	Receipt            Pin
}
type Witness struct {
	Sources      Sources
	Capture      hacgpu.Capture
	Stages       []StageContract
	Caches       []CacheContract
	Inputs       map[string]string
	Environment  map[string]h07.RuntimeEnvValue
	Device       string
	IdentityTool Pin
	Limits       []string
}

func ReadWitness(s Sources) (w Witness, err error) {
	w = Witness{Sources: s, Inputs: map[string]string{}, Limits: []string{"Retained H01/H03/H04/H07 witnesses are reused; this consumer does not repair historical producer compiler/runtime custody gaps.", "Full fallback/VCS/other profiles/biological accuracy/performance remain outside H08 fixed composition."}}
	h, err := h07.ReadWitness(s.Capture)
	if err != nil {
		return w, err
	}
	w.Capture = h.Capture
	for p, pin := range h.Pins {
		w.Inputs[p] = pin
	}
	w.Stages, err = stageContracts()
	if err != nil {
		return w, err
	}
	if len(w.Capture.Stages) != len(w.Stages) {
		return w, fmt.Errorf("stage count differs")
	}
	for i, contract := range w.Stages {
		n := w.Capture.Stages[i]
		if n.Name != contract.Name || n.Layout != contract.Layout || n.DType != contract.DType || n.OffsetBytes != contract.Offset || n.Bytes != contract.Bytes || !slices.Equal(n.Shape, contract.Shape) || !slices.Equal(n.UnwrittenT, contract.NativeUnwritten) || n.InputToRevLSTM != (contract.Orientation == model.GPUOrientationReverse) {
			return w, fmt.Errorf("prospective stage contract differs: %s", contract.Name)
		}
		for file, pin := range map[string]string{n.DataFile: n.DataSHA, n.BackingFile: n.BackingSHA} {
			w.Inputs[filepath.Join(s.Capture, file)] = pin
		}
	}
	manifest, err := model.WeightManifest(model.HAC)
	if err != nil {
		return w, err
	}
	weights := map[string]model.WeightReceipt{}
	for _, p := range manifest {
		weights[p.Name] = p
	}
	var conv struct {
		Status  string        `json:"status"`
		Packing []h03.Packing `json:"packing"`
	}
	if err = readJSON(s.Convolution, &conv); err != nil {
		return w, err
	}
	w.Inputs[s.Convolution.Path] = s.Convolution.SHA256
	if conv.Status != "lastconv_passed" || len(conv.Packing) != 3 {
		return w, fmt.Errorf("accepted convolution cache witness required")
	}
	for i, c := range conv.Packing {
		if c.Index != i || !c.Equal || c.Transposed != (i == 2) || !slices.Equal(c.WeightShape, weights[fmt.Sprintf("%d.conv.weight.tensor", i)].Shape) || c.WeightFileSHA != weights[fmt.Sprintf("%d.conv.weight.tensor", i)].SHA256 || c.BiasFileSHA != weights[fmt.Sprintf("%d.conv.bias.tensor", i)].SHA256 || c.GoWeightSHA != c.ReferenceSHA || c.GoBiasSHA != c.ReferenceBiasSHA {
			return w, fmt.Errorf("convolution cache identity differs")
		}
		p := weights[fmt.Sprintf("%d.conv.weight.tensor", i)]
		shape := []int64{p.Shape[1] * p.Shape[2], p.Shape[0]}
		if i == 2 {
			shape = []int64{p.Shape[0], p.Shape[1] * p.Shape[2]}
		}
		w.Caches = append(w.Caches, CacheContract{fmt.Sprintf("conv%d.weight", i), tensor.DTypeF16, shape, int(shape[0] * shape[1] * 2), c.ReferenceSHA, "", s.Convolution}, CacheContract{fmt.Sprintf("conv%d.bias", i), tensor.DTypeF16, []int64{p.Shape[0]}, int(p.Shape[0] * 2), c.ReferenceBiasSHA, "", s.Convolution})
	}
	var recurrent h04.ConsumerReceipt
	if err = readJSON(s.Recurrence, &recurrent); err != nil {
		return w, err
	}
	w.Inputs[s.Recurrence.Path] = s.Recurrence.SHA256
	if recurrent.Status != "current_go_cache_parity_passed" || recurrent.FinalCaches != 49 || len(recurrent.Layers) != 7 {
		return w, fmt.Errorf("accepted49-cache recurrent witness required")
	}
	roles := []string{"dn-ih-cache", "dn-hh-cache", "dn-ih-cache-scale", "dn-hh-cache-scale", "up-cache", "up-cache-scale", "up-cache-bias"}
	for i, l := range recurrent.Layers {
		if l.Index != i || !l.Equal || l.WeightBase != fmt.Sprintf("%d.rnn.", 4+i) || len(l.WeightPins) != 6 {
			return w, fmt.Errorf("recurrent layer binding differs")
		}
		for _, p := range l.WeightPins {
			if weights[p.Name].SHA256 != p.SHA256 || !slices.Equal(weights[p.Name].Shape, p.Shape) {
				return w, fmt.Errorf("foreign recurrent weight witness")
			}
		}
		for _, role := range roles {
			found := 0
			for _, t := range l.Tensors {
				if t.Role != role {
					continue
				}
				found++
				if !t.Equal || t.GoSHA256 != t.NativeSHA || t.NativePayload == "" {
					return w, fmt.Errorf("unverified recurrent cache")
				}
				if _, err = readPin(Pin{Path: t.NativePayload, SHA256: t.NativeSHA}); err != nil {
					return w, err
				}
				w.Inputs[t.NativePayload] = t.NativeSHA
				dtype := tensor.DTypeF16
				if t.DType == "Char" {
					dtype = tensor.DTypeI8
				} else if t.DType != "Half" {
					return w, fmt.Errorf("unknown cache dtype")
				}
				w.Caches = append(w.Caches, CacheContract{fmt.Sprintf("flstm%d.%s", i, role), dtype, slices.Clone(t.Shape), int(t.Bytes), t.NativeSHA, t.NativePayload, s.Recurrence})
			}
			if found != 1 {
				return w, fmt.Errorf("missing/repeated recurrent cache role")
			}
		}
	}
	var projection h07.OracleCaptureReceipt
	if err = readJSON(s.Projection, &projection); err != nil {
		return w, err
	}
	w.Inputs[s.Projection.Path] = s.Projection.SHA256
	if projection.Status != "source_projection_caches_passed" || len(projection.Caches) != 3 || len(projection.Commands) != 5 || projection.BeforeDevice == "" || projection.BeforeDevice != projection.AfterDevice {
		return w, fmt.Errorf("accepted projection cache witness required")
	}
	if err = checkPins(projection.Outputs); err != nil {
		return w, err
	}
	descriptors, e := h07.ReadCacheDescriptors(projection.Output)
	if e != nil {
		return w, e
	}
	for p, pin := range projection.Outputs {
		w.Inputs[p] = pin
	}
	for _, c := range projection.Commands {
		if c.Exit != 0 {
			return w, fmt.Errorf("projection producer command failed")
		}
	}
	for i, c := range projection.Caches {
		if c.Role != descriptors[i].Role || c.SHA256 != descriptors[i].SHA256 || !slices.Equal(c.Shape, descriptors[i].Shape) || c.Bytes != descriptors[i].Bytes || c.DType != descriptors[i].DType {
			return w, fmt.Errorf("projection cache descriptor differs")
		}
		path := filepath.Join(projection.Output, c.File)
		if _, err = readPin(Pin{Path: path, SHA256: c.SHA256}); err != nil {
			return w, err
		}
		dtype := tensor.DTypeF16
		if c.DType == "Char" {
			dtype = tensor.DTypeI8
		}
		w.Caches = append(w.Caches, CacheContract{"projection." + c.Role, dtype, slices.Clone(c.Shape), int(c.Bytes), c.SHA256, path, s.Projection})
	}
	w.Environment, w.Device = projection.Environment, projection.BeforeDevice
	if len(projection.Commands[0].Args) == 0 {
		return w, fmt.Errorf("identity tool absent")
	}
	tool := projection.Commands[0].Args[0]
	pin := projection.Inputs[tool]
	w.IdentityTool = Pin{Path: tool, SHA256: pin}
	if _, err = readPin(w.IdentityTool); err != nil {
		return w, err
	}
	w.Inputs[tool] = pin
	if len(w.Caches) != 58 {
		return w, fmt.Errorf("expected exactly58 production cache owners")
	}
	if err = checkPins(w.Inputs); err != nil {
		return w, err
	}
	return w, nil
}

type ReplayReceipt struct {
	Schema, Status, Error, Started, Finished string
	Sources                                  Sources
	Inputs                                   map[string]string
	Stages                                   []StageContract
	Caches                                   []CacheContract
	ControlFile, ControlSHA, ControlError    string
	Limits                                   []string
}

func Replay(s Sources, out string) (r ReplayReceipt, err error) {
	dir, err := runner.NewAttempt(out)
	if err != nil {
		return r, err
	}
	r = ReplayReceipt{Schema: "nanopore-hac-fixed-replay/v1", Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Sources: s}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
		}
		raw, e := json.MarshalIndent(r, "", "  ")
		if e == nil {
			e = os.WriteFile(filepath.Join(dir, "replay.json"), append(raw, '\n'), 0644)
		}
		if err == nil {
			err = e
		}
	}()
	w, err := ReadWitness(s)
	if err != nil {
		return r, err
	}
	r.Inputs, r.Stages, r.Caches, r.Limits = w.Inputs, w.Stages, w.Caches, w.Limits
	input, err := w.Capture.RawPayload(s.Capture, "input")
	if err != nil {
		return r, err
	}
	changed := bytes.Clone(input)
	changed[0] ^= 1
	r.ControlFile = filepath.Join(dir, "changed-input.bin")
	r.ControlSHA = digest(changed)
	if err = os.WriteFile(r.ControlFile, changed, 0644); err != nil {
		return r, err
	}
	stage, _ := w.Capture.Stage("input")
	_, controlErr := readPin(Pin{Path: r.ControlFile, SHA256: stage.DataSHA})
	if controlErr == nil {
		return r, fmt.Errorf("changed input witness accepted")
	}
	r.ControlError = controlErr.Error()
	r.Status = "retained_fixed_witnesses_authenticated_native_held"
	return r, nil
}
