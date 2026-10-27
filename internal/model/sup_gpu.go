package model

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"sync"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

type SUPGPUDescriptorSnapshot struct {
	Name           string
	DType          tensor.DType
	Shape, Strides []int64
	Tags           []tensor.KoiTag
	ScaleTag       tensor.KoiTag
	ScaleDType     tensor.DType
}
type SUPGPUObservationOrigin uint8

const (
	SUPGPUUploaded SUPGPUObservationOrigin = iota
	SUPGPUPreparedCache
)

type SUPGPUObservation struct {
	Name        string
	Layer       int
	Tensors     []SUPGPUBufferSnapshot
	Descriptors []SUPGPUDescriptorSnapshot
	Origin      SUPGPUObservationOrigin
}

// Observers receive owned bytes and metadata. They cannot mutate live device
// storage, a parameter cache, or a later observation.
type SUPGPUObserver func(SUPGPUObservation) error

type SUPGPU struct {
	mu             sync.Mutex
	device         *tensor.Device
	deviceID       uint64
	weights        SUPGPUWeights
	cacheEntered   [18]bool
	convs          [5][2]*tensor.Storage
	encoders       [18]map[string]*tensor.Storage
	upsample       [2]*tensor.Storage
	crf, sincos    *tensor.Storage
	owned, pending []*tensor.Storage
	closed         bool
}

func closeSUPStorages(handles []*tensor.Storage) (failed []*tensor.Storage, err error) {
	for _, h := range handles {
		if h != nil && h.DType() == tensor.DTypeI8 {
			meta, e := h.Meta()
			if errors.Is(e, tensor.ErrClosed) {
				e = h.Close()
			} else if e == nil && meta.ScalePresent {
				e = h.BindScale(nil, 0)
			}
			err = errors.Join(err, e)
		}
	}
	if err != nil {
		return slices.Clone(handles), err
	}
	return closeGPUStorageOwners(handles)
}

func NewSUPGPU(ctx context.Context, d *tensor.Device, s *tensor.Stream, w SUPGPUWeights) (m *SUPGPU, err error) {
	if ctx == nil {
		return nil, fmt.Errorf("nil SUP context")
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if d == nil || s == nil {
		return nil, tensor.ErrClosed
	}
	pin, _ := Pinned(SUP)
	if w.spec != pin {
		return nil, fmt.Errorf("%w: loaded SUP weights required", ErrGPUWeights)
	}
	if _, err = supGPURoles(w.receipts); err != nil {
		return nil, err
	}
	m = &SUPGPU{device: d, weights: w}
	defer func() {
		if err != nil {
			cleanup := m.Close()
			err = errors.Join(err, cleanup)
			if cleanup == nil {
				m = nil
			}
		}
	}()
	upload := func(b supGPUBuffer) (*tensor.Storage, error) {
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		if e := validateSUPBuffer(b); e != nil {
			return nil, e
		}
		h, e := d.NewStorage(s, b.dtype, b.shape, b.tags, b.data)
		if h != nil {
			m.owned = append(m.owned, h)
		}
		return h, e
	}
	for i := range m.convs {
		for j := range m.convs[i] {
			m.convs[i][j], err = upload(w.convs[i][j])
			if err != nil {
				return m, err
			}
		}
	}
	for layer := range m.encoders {
		m.encoders[layer] = map[string]*tensor.Storage{}
		for _, b := range w.encoders[layer].Snapshot(true) {
			h, e := upload(supGPUBuffer{b.DType, b.Shape, b.Tags, b.Data})
			if e != nil {
				return m, e
			}
			m.encoders[layer][b.Name] = h
		}
		if err = m.encoders[layer]["qkv_int8"].BindScale(m.encoders[layer]["qkv_scale"], tensor.TagC); err != nil {
			return m, err
		}
	}
	for j := range m.upsample {
		m.upsample[j], err = upload(w.upsample[j])
		if err != nil {
			return m, err
		}
	}
	m.crf, err = upload(w.crf)
	if err != nil {
		return m, err
	}
	meta, err := m.crf.Meta()
	if err != nil {
		return m, err
	}
	m.deviceID = meta.Device
	m.sincos, err = m.makeRoPE(ctx, s)
	if err != nil {
		return m, err
	}
	return m, ctx.Err()
}

func validateSUPBuffer(b supGPUBuffer) error {
	count := int64(1)
	if len(b.shape) == 0 || len(b.shape) != len(b.tags) || b.dtype.Size() == 0 {
		return fmt.Errorf("%w: invalid SUP cache metadata", ErrGPUWeights)
	}
	for _, n := range b.shape {
		if n < 1 || n > math.MaxInt64/count {
			return fmt.Errorf("%w: SUP cache extent overflow", ErrGPUWeights)
		}
		count *= n
	}
	if count > math.MaxInt64/int64(b.dtype.Size()) || int64(len(b.data)) != count*int64(b.dtype.Size()) {
		return fmt.Errorf("%w: SUP cache bytes differ", ErrGPUWeights)
	}
	return validateSUPValues(b.dtype, b.data, false)
}
func validateSUPValues(dtype tensor.DType, data []byte, positive bool) error {
	switch dtype {
	case tensor.DTypeF16:
		if len(data)%2 != 0 {
			return fmt.Errorf("SUP half bytes are truncated")
		}
		for i := 0; i < len(data); i += 2 {
			if binary.LittleEndian.Uint16(data[i:])&0x7c00 == 0x7c00 {
				return fmt.Errorf("SUP half element%d is nonfinite", i/2)
			}
		}
	case tensor.DTypeF32:
		if len(data)%4 != 0 {
			return fmt.Errorf("SUP float bytes are truncated")
		}
		for i := 0; i < len(data); i += 4 {
			v := math.Float32frombits(binary.LittleEndian.Uint32(data[i:]))
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || (positive && v <= 0) {
				return fmt.Errorf("SUP float element%d is nonfinite or nonpositive", i/4)
			}
		}
	case tensor.DTypeE4M3:
		for i, v := range data {
			if v&0x7f == 0x7f {
				return fmt.Errorf("SUP FP8 element%d is nonfinite", i)
			}
		}
	}
	return nil
}

func (m *SUPGPU) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	all := append(slices.Clone(m.owned), m.pending...)
	failed, err := closeSUPStorages(all)
	m.owned = failed
	m.pending = nil
	return err
}

type SUPGPUScoreMetadata struct {
	QScale, QBias, BlankScore, DecoderClampValue float32
	StateLen                                     int64
	MeanQStart                                   int
}
type SUPGPUResult struct {
	Scores   *tensor.Storage
	Metadata SUPGPUScoreMetadata
}

func (r *SUPGPUResult) Close() error {
	if r == nil || r.Scores == nil {
		return nil
	}
	return r.Scores.Close()
}

type supGPUInvocation struct {
	model                                         *SUPGPU
	plan                                          SUPGPUPlan
	sincos                                        *tensor.Storage
	ctx                                           context.Context
	stream                                        *tensor.Stream
	observe                                       SUPGPUObserver
	owned                                         []*tensor.Storage
	residual, int8, scale                         *tensor.Storage
	qkv, attn, projection, rms, fc1, fc2, counter *tensor.Storage
	ctr                                           [4]*tensor.Storage
}

func (v *supGPUInvocation) alloc(dtype tensor.DType, shape []int64, tags string) (*tensor.Storage, error) {
	if err := v.ctx.Err(); err != nil {
		return nil, err
	}
	count := int64(1)
	for _, n := range shape {
		if n < 1 || n > math.MaxInt64/count {
			return nil, fmt.Errorf("SUP allocation overflow")
		}
		count *= n
	}
	if count > math.MaxInt64/int64(dtype.Size()) || count*int64(dtype.Size()) > math.MaxInt {
		return nil, fmt.Errorf("SUP allocation byte overflow")
	}
	h, err := v.model.device.NewStorage(v.stream, dtype, shape, tagValues(tags), make([]byte, int(count)*dtype.Size()))
	if h != nil {
		v.owned = append(v.owned, h)
	}
	return h, err
}
func tagValues(s string) []tensor.KoiTag {
	out := make([]tensor.KoiTag, 0, len(s))
	for _, c := range s {
		out = append(out, tensor.KoiTag(c))
	}
	return out
}
func (v *supGPUInvocation) view(h *tensor.Storage, offset uint64, shape, strides []int64, tags string) (*tensor.Storage, error) {
	x, err := h.View(offset, shape, strides, tagValues(tags))
	if x != nil {
		v.owned = append(v.owned, x)
	}
	return x, err
}
func (v *supGPUInvocation) reshape(h *tensor.Storage, shape []int64, tags string) (*tensor.Storage, error) {
	return v.view(h, 0, shape, supContiguous(shape), tags)
}
func (v *supGPUInvocation) copy(src, dst *tensor.Storage) error {
	return v.math(tensor.SUPCopyStrided, 0, src, dst)
}
func (v *supGPUInvocation) math(op tensor.SUPPrimitive, selector int, handles ...*tensor.Storage) error {
	if err := v.ctx.Err(); err != nil {
		return err
	}
	return v.model.device.SUPMath(v.stream, v.plan.geometry, op, selector, handles...)
}

func (m *SUPGPU) makeRoPE(ctx context.Context, s *tensor.Stream) (out *tensor.Storage, err error) {
	p, err := PlanSUPGPU(FullFixedSUPGPUProfile())
	if err != nil {
		return nil, err
	}
	v := &supGPUInvocation{model: m, plan: p, ctx: ctx, stream: s}
	defer func() {
		var cleanup error
		failed, cleanup := closeSUPStorages(v.owned)
		m.pending = append(m.pending, failed...)
		err = errors.Join(err, cleanup)
	}()
	b := supHalfBuffer(SUPRoPEFrequencies(), []int64{4096, 32}, "TD")
	freq, err := m.device.NewStorage(s, b.dtype, b.shape, b.tags, b.data)
	if freq != nil {
		v.owned = append(v.owned, freq)
	}
	if err != nil {
		return nil, err
	}
	sin, err := v.alloc(tensor.DTypeF16, []int64{4096, 32}, "TD")
	if err != nil {
		return nil, err
	}
	cos, err := v.alloc(tensor.DTypeF16, []int64{4096, 32}, "TD")
	if err != nil {
		return nil, err
	}
	if err = v.math(tensor.SUPSinCos, 0, freq, sin, cos); err != nil {
		return nil, err
	}
	interleaved, err := v.alloc(tensor.DTypeF16, []int64{4096, 32, 2}, "TDd")
	if err != nil {
		return nil, err
	}
	for i, h := range []*tensor.Storage{sin, cos} {
		target, e := v.view(interleaved, uint64(i), []int64{4096, 32}, []int64{64, 2}, "TD")
		if e != nil {
			return nil, e
		}
		if e = v.copy(h, target); e != nil {
			return nil, e
		}
	}
	tiled, err := v.view(interleaved, 0, []int64{256, 8, 16, 8}, []int64{1024, 8, 64, 1}, "TDtd")
	if err != nil {
		return nil, err
	}
	out, err = v.alloc(tensor.DTypeF16, []int64{256, 8, 16, 8}, "TDtd")
	if err != nil {
		return nil, err
	}
	if err = v.copy(tiled, out); err != nil {
		return nil, err
	}
	// The model keeps the generated table. It never reads an oracle payload.
	v.owned = v.owned[:len(v.owned)-1]
	m.owned = append(m.owned, out)
	return out, nil
}

func (v *supGPUInvocation) snapshot(name string, h *tensor.Storage, tags []tensor.KoiTag, scaleTag tensor.KoiTag) (SUPGPUBufferSnapshot, error) {
	if err := v.ctx.Err(); err != nil {
		return SUPGPUBufferSnapshot{}, err
	}
	m, err := h.Meta()
	if err != nil {
		return SUPGPUBufferSnapshot{}, err
	}
	data, err := h.Download(v.stream)
	if err != nil {
		return SUPGPUBufferSnapshot{}, err
	}
	if err = validateSUPValues(m.DType, data, name == "scale" || name == "qkv_scale"); err != nil {
		return SUPGPUBufferSnapshot{}, err
	}
	return SUPGPUBufferSnapshot{Name: name, DType: m.DType, Shape: slices.Clone(m.Shape), Strides: slices.Clone(m.Strides), Tags: slices.Clone(tags), ScaleTag: scaleTag, Data: data}, nil
}
func (v *supGPUInvocation) emit(name string, layer int, tensors []SUPGPUBufferSnapshot, descriptors []SUPGPUDescriptorSnapshot) error {
	if err := v.ctx.Err(); err != nil {
		return err
	}
	if v.observe == nil {
		return nil
	}
	return v.observe(SUPGPUObservation{Name: name, Layer: layer, Tensors: tensors, Descriptors: descriptors, Origin: SUPGPUUploaded})
}
func (v *supGPUInvocation) emitOne(name, role string, h *tensor.Storage) error {
	if v.observe == nil {
		return v.ctx.Err()
	}
	snap, err := v.snapshot(role, h, nil, 0)
	if err != nil {
		return err
	}
	return v.emit(name, -1, []SUPGPUBufferSnapshot{snap}, nil)
}
func (v *supGPUInvocation) descriptors(op tensor.SUPPrimitive, handles []*tensor.Storage) ([]SUPGPUDescriptorSnapshot, error) {
	indices := []int{}
	names := []string{}
	switch op {
	case tensor.SUPQKVRotary:
		indices = []int{0, 2, 4, 5}
		names = []string{"input", "weights", "sincos", "output"}
	case tensor.SUPMaskedAttention:
		indices = []int{0, 1}
		names = []string{"input", "output"}
	case tensor.SUPProjection:
		indices = []int{0, 1, 2, 3}
		names = []string{"input", "weight", "bias", "output"}
	case tensor.SUPNorm1:
		indices = []int{0, 1, 2, 1, 3}
		names = []string{"input", "residual", "weight", "output", "fp8"}
	case tensor.SUPSwiGLU, tensor.SUPFC2:
		indices = []int{0, 1, 2}
		names = []string{"input", "weight", "output"}
	case tensor.SUPNorm2:
		indices = []int{0, 1, 2, 1, 3}
		names = []string{"input", "residual", "weight", "output", "int8"}
	}
	out := make([]SUPGPUDescriptorSnapshot, len(indices))
	for i, j := range indices {
		m, err := handles[j].Meta()
		if err != nil {
			return nil, err
		}
		out[i] = SUPGPUDescriptorSnapshot{names[i], m.DType, slices.Clone(m.Shape), slices.Clone(m.Strides), slices.Clone(m.Tags), m.ScaleTag, m.ScaleDType}
	}
	return out, nil
}

func (v *supGPUInvocation) emitLayer(stage string, layer int, handles []*tensor.Storage, op tensor.SUPPrimitive) error {
	if v.observe == nil {
		return v.ctx.Err()
	}
	type item struct {
		name  string
		h     *tensor.Storage
		tags  string
		scale tensor.KoiTag
	}
	half := item{"residual", v.residual, "NTCtc", 0}
	i8 := item{"int8", v.int8, "NTCtc", tensor.TagC}
	scale := item{"scale", v.scale, "", 0}
	counter := item{"counter", v.counter, "", 0}
	var items []item
	switch stage {
	case "enter":
		items = []item{half, i8, scale, counter}
	case "qkv":
		items = []item{{"qkv", v.qkv, "NT\x00HDtd", 0}, counter}
	case "attention":
		items = []item{{"attention", v.attn, "NTHDtd", 0}}
	case "projection":
		items = []item{{"projection", v.projection, "NTCtc", 0}, counter}
	case "norm1":
		items = []item{half, {"rms_fp8", v.rms, "NTCtc", 0}}
	case "swiglu":
		items = []item{{"swiglu", v.fc1, "MNmn", 0}, counter}
	case "fc2":
		items = []item{{"fc2", v.fc2, "NTCtc", 0}, counter}
	case "norm2":
		items = []item{half, i8, scale}
	}
	var snaps []SUPGPUBufferSnapshot
	for _, x := range items {
		snap, err := v.snapshot(x.name, x.h, tagValues(x.tags), x.scale)
		if err != nil {
			return err
		}
		snaps = append(snaps, snap)
	}
	desc, err := v.descriptors(op, handles)
	if err != nil {
		return err
	}
	return v.emit(fmt.Sprintf("encoder.%02d.%s", layer, stage), layer, snaps, desc)
}

func (v *supGPUInvocation) encoderInput(x *tensor.Storage) error {
	d := v.plan.dimensions
	shape, strides, _ := SUPNTCTileStrides(v.plan.geometry, 8)
	source, err := v.view(x, 0, shape, strides, "NTCtc")
	if err != nil {
		return err
	}
	v.residual, err = v.alloc(tensor.DTypeF16, shape, "NTCtc")
	if err != nil {
		return err
	}
	if err = v.copy(source, v.residual); err != nil {
		return err
	}
	q, err := v.alloc(tensor.DTypeI8, []int64{d.Batch, d.EncoderT, 512}, "NTC")
	if err != nil {
		return err
	}
	v.scale, err = v.alloc(tensor.DTypeF32, []int64{d.Batch, d.EncoderT}, "\x00\x00")
	if err != nil {
		return err
	}
	if err = v.math(tensor.SUPQuantizeRows, 0, x, q, v.scale); err != nil {
		return err
	}
	shape, strides, _ = SUPNTCTileStrides(v.plan.geometry, 16)
	source, err = v.view(q, 0, shape, strides, "NTCtc")
	if err != nil {
		return err
	}
	v.int8, err = v.alloc(tensor.DTypeI8, shape, "NTCtc")
	if err != nil {
		return err
	}
	if err = v.copy(source, v.int8); err != nil {
		return err
	}
	if err = v.int8.BindScale(v.scale, tensor.TagC); err != nil {
		return err
	}
	if v.observe != nil {
		snaps := []SUPGPUBufferSnapshot{}
		for _, x := range []struct {
			name string
			h    *tensor.Storage
			tags string
		}{{"half", v.residual, "NTCtc"}, {"int8", v.int8, "NTCtc"}, {"scale", v.scale, ""}} {
			snap, e := v.snapshot(x.name, x.h, tagValues(x.tags), 0)
			if e != nil {
				return e
			}
			snaps = append(snaps, snap)
		}
		return v.emit("encoder.input", -1, snaps, nil)
	}
	return nil
}

func (v *supGPUInvocation) allocateEncoder() error {
	d := v.plan.dimensions
	tiles, rows := d.EncoderT/16, d.Batch*d.EncoderT/16
	var err error
	for _, a := range []struct {
		dst   **tensor.Storage
		dtype tensor.DType
		shape []int64
		tags  string
	}{{&v.qkv, tensor.DTypeF16, []int64{d.Batch, tiles, 3, 8, 8, 16, 8}, "NT\x00HDtd"}, {&v.attn, tensor.DTypeF16, []int64{d.Batch, tiles, 8, 8, 16, 8}, "NTHDtd"}, {&v.projection, tensor.DTypeF16, []int64{d.Batch, tiles, 64, 16, 8}, "NTCtc"}, {&v.rms, tensor.DTypeE4M3, []int64{d.Batch, tiles, 32, 16, 16}, "NTCtc"}, {&v.fc1, tensor.DTypeE4M3, []int64{rows, 128, 16, 16}, "MNmn"}, {&v.fc2, tensor.DTypeF16, []int64{d.Batch, tiles, 64, 16, 8}, "NTCtc"}, {&v.counter, tensor.DTypeI32, []int64{4}, "C"}} {
		*a.dst, err = v.alloc(a.dtype, a.shape, a.tags)
		if err != nil {
			return err
		}
	}
	for i := range v.ctr {
		v.ctr[i], err = v.view(v.counter, uint64(i), []int64{1}, []int64{1}, "C")
		if err != nil {
			return err
		}
	}
	return nil
}

func (v *supGPUInvocation) encoderLayer(layer int) error {
	d := v.plan.dimensions
	rows := d.Batch * d.EncoderT / 16
	if err := v.ctx.Err(); err != nil {
		return err
	}
	w := v.model.encoders[layer]
	if err := v.observeEncoderCache(layer); err != nil {
		return err
	}
	if err := v.ctx.Err(); err != nil {
		return err
	}
	if err := v.counter.Fill(v.stream, []byte{0, 0, 0, 0}); err != nil {
		return err
	}
	if err := v.emitLayer("enter", layer, nil, 0); err != nil {
		return err
	}
	attn, err := v.reshape(v.attn, []int64{rows, 64, 16, 8}, "MKmk")
	if err != nil {
		return err
	}
	proj, err := v.reshape(v.projection, []int64{rows, 64, 16, 8}, "MNmn")
	if err != nil {
		return err
	}
	rms, err := v.reshape(v.rms, []int64{rows, 32, 16, 16}, "MKmk")
	if err != nil {
		return err
	}
	fc1, err := v.reshape(v.fc1, []int64{rows, 128, 16, 16}, "MKmk")
	if err != nil {
		return err
	}
	fc2, err := v.reshape(v.fc2, []int64{rows, 64, 16, 8}, "MNmn")
	if err != nil {
		return err
	}
	operands := map[tensor.SUPPrimitive][]*tensor.Storage{
		tensor.SUPQKVRotary:       {v.int8, v.scale, w["qkv_int8"], w["qkv_scale"], v.sincos, v.qkv, v.ctr[0]},
		tensor.SUPMaskedAttention: {v.qkv, v.attn},
		tensor.SUPProjection:      {attn, w["proj_weight"], w["proj_bias"], proj, v.ctr[1]},
		tensor.SUPNorm1:           {v.projection, v.residual, w["norm1_weight"], v.rms},
		tensor.SUPSwiGLU:          {rms, w["fc1_fp8"], v.fc1, v.ctr[2]},
		tensor.SUPFC2:             {fc1, w["fc2_fp8"], fc2, v.ctr[3]},
		tensor.SUPNorm2:           {v.fc2, v.residual, w["norm2_weight"], v.int8, v.scale},
	}
	return runSUPEncoderStages(v.ctx, layer, func(stage supEncoderStage) error {
		handles := operands[stage.op]
		if err := v.math(stage.op, 0, handles...); err != nil {
			return err
		}
		return v.emitLayer(stage.name, layer, handles, stage.op)
	})
}

// The before-removal event describes Go host preparation. After-removal
// observations read current uploaded owners so a wrong upload or mutation is
// visible independently of the later model outputs.
func supCacheObservations(cache SUPEncoderGPUCache, after bool, read func(string) (SUPGPUBufferSnapshot, error)) ([]SUPGPUBufferSnapshot, error) {
	prepared := cache.Snapshot(after)
	if !after {
		return prepared, nil
	}
	if read == nil {
		return nil, fmt.Errorf("uploaded SUP cache reader required")
	}
	for i, want := range prepared {
		actual, err := read(want.Name)
		if err != nil {
			return nil, err
		}
		if actual.Name != want.Name || actual.DType != want.DType || !slices.Equal(actual.Shape, want.Shape) || !slices.Equal(actual.Strides, want.Strides) || !slices.Equal(actual.Tags, want.Tags) || len(actual.Data) != len(want.Data) {
			return nil, fmt.Errorf("uploaded SUP cache role or metadata differs: %s", want.Name)
		}
		prepared[i] = actual
	}
	return prepared, nil
}

func (m *SUPGPU) Forward(ctx context.Context, s *tensor.Stream, input *tensor.Storage, p SUPGPUPlan) (*SUPGPUResult, error) {
	return m.ForwardObserved(ctx, s, input, p, nil)
}
func (m *SUPGPU) ForwardObserved(ctx context.Context, s *tensor.Stream, input *tensor.Storage, p SUPGPUPlan, observe SUPGPUObserver) (result *SUPGPUResult, err error) {
	if m == nil {
		return nil, ErrClosed
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, ErrClosed
	}
	if ctx == nil {
		return nil, fmt.Errorf("nil SUP context")
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if len(m.pending) > 0 {
		return nil, fmt.Errorf("SUP cleanup pending; Close must recover owners")
	}
	want, planErr := PlanSUPGPU(p.profile)
	if planErr != nil || !reflect.DeepEqual(p, want) {
		return nil, fmt.Errorf("%w: initialized SUP forward plan required", ErrGPUPlanState)
	}
	if s == nil || input == nil {
		return nil, tensor.ErrClosed
	}
	d := p.dimensions
	meta, err := input.Meta()
	if err != nil {
		return nil, err
	}
	if meta.Device != m.deviceID || meta.DType != tensor.DTypeF16 || meta.ScalePresent || !slices.Equal(meta.Shape, []int64{d.Batch, 1, d.Chunk}) || !slices.Equal(meta.Strides, []int64{d.Chunk, d.Chunk, 1}) || !slices.Equal(meta.Tags, tagValues("NCT")) {
		return nil, fmt.Errorf("%w: fixed owned half NCT input required", ErrGPUPlanLayout)
	}
	v := &supGPUInvocation{model: m, plan: p, ctx: ctx, stream: s, observe: observe}
	defer func() {
		failed, cleanup := closeSUPStorages(v.owned)
		m.pending = append(m.pending, failed...)
		err = errors.Join(err, cleanup)
		if err != nil && result != nil {
			e := result.Close()
			err = errors.Join(err, e)
			if e != nil {
				m.pending = append(m.pending, result.Scores)
			}
			result = nil
		}
	}()
	v.sincos, err = v.view(m.sincos, 0, []int64{d.EncoderT / 16, 8, 16, 8}, []int64{1024, 128, 8, 1}, "TDtd")
	if err != nil {
		return nil, err
	}
	inputBytes, err := input.Download(s)
	if err != nil {
		return nil, err
	}
	if err = validateSUPValues(meta.DType, inputBytes, false); err != nil {
		return nil, err
	}
	x, err := v.alloc(tensor.DTypeF16, meta.Shape, "NCT")
	if err != nil {
		return nil, err
	}
	if err = v.copy(input, x); err != nil {
		return nil, err
	}
	if err = v.emitOne("input", "input", x); err != nil {
		return nil, err
	}
	for i := 0; i < 5; i++ {
		specs, _ := tensor.SUPPrimitiveSpecs(p.geometry, tensor.SUPConvSiLU, i)
		out, e := v.alloc(tensor.DTypeF16, specs[3].Shape, "NCT")
		if e != nil {
			return nil, e
		}
		if e = v.math(tensor.SUPConvSiLU, i, x, m.convs[i][0], m.convs[i][1], out); e != nil {
			return nil, e
		}
		x = out
		if e = v.emitOne(fmt.Sprintf("conv%d", i), "output", x); e != nil {
			return nil, e
		}
	}
	transposed, err := v.view(x, 0, []int64{d.Batch, d.EncoderT, 512}, []int64{d.EncoderT * 512, 1, d.EncoderT}, "NTC")
	if err != nil {
		return nil, err
	}
	ntc, err := v.alloc(tensor.DTypeF16, []int64{d.Batch, d.EncoderT, 512}, "NTC")
	if err != nil {
		return nil, err
	}
	if err = v.copy(transposed, ntc); err != nil {
		return nil, err
	}
	if err = v.encoderInput(ntc); err != nil {
		return nil, err
	}
	if err = v.allocateEncoder(); err != nil {
		return nil, err
	}
	for i := 0; i < 18; i++ {
		if err = v.encoderLayer(i); err != nil {
			return nil, err
		}
	}
	// Undo the tiled C/t transpose before reshaping NTC.
	logical, err := v.view(v.residual, 0, []int64{d.Batch, d.EncoderT / 16, 16, 64, 8}, []int64{d.EncoderT * 512, 8192, 8, 128, 1}, "NTtCc")
	if err != nil {
		return nil, err
	}
	linear, err := v.alloc(tensor.DTypeF16, []int64{d.Batch, d.EncoderT / 16, 16, 64, 8}, "NTtCc")
	if err != nil {
		return nil, err
	}
	if err = v.copy(logical, linear); err != nil {
		return nil, err
	}
	untiled, err := v.reshape(linear, []int64{d.Batch, d.EncoderT, 512}, "NTC")
	if err != nil {
		return nil, err
	}
	if err = v.emitOne("encoder.output", "output", untiled); err != nil {
		return nil, err
	}
	up, err := v.alloc(tensor.DTypeF16, []int64{d.Batch, d.EncoderT, 1024}, "NTC")
	if err != nil {
		return nil, err
	}
	if err = v.math(tensor.SUPLinear, 0, untiled, m.upsample[0], m.upsample[1], up); err != nil {
		return nil, err
	}
	upsampled, err := v.reshape(up, []int64{d.Batch, 2 * d.EncoderT, 512}, "NTC")
	if err != nil {
		return nil, err
	}
	if err = v.emitOne("upsample", "output", upsampled); err != nil {
		return nil, err
	}
	if err = v.emitOne("crf.weight", "weight", m.crf); err != nil {
		return nil, err
	}
	scores, err := v.alloc(tensor.DTypeF16, []int64{d.Batch, 2 * d.EncoderT, 4096}, "NTC")
	if err != nil {
		return nil, err
	}
	if err = v.math(tensor.SUPLinear, 1, upsampled, m.crf, scores); err != nil {
		return nil, err
	}
	data, err := scores.Download(s)
	if err != nil {
		return nil, err
	}
	if err = validateSUPValues(tensor.DTypeF16, data, false); err != nil {
		return nil, err
	}
	if err = v.emitOne("scores", "output", scores); err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	v.owned = v.owned[:len(v.owned)-1]
	spec := m.weights.spec
	result = &SUPGPUResult{scores, SUPGPUScoreMetadata{spec.QScale, spec.QBias, spec.BlankScore, 0, spec.StateLen, spec.MeanQStart}}
	return result, nil
}

func (v *supGPUInvocation) observeEncoderCache(layer int) error {
	if err := v.ctx.Err(); err != nil {
		return err
	}
	cold := !v.model.cacheEntered[layer]
	v.model.cacheEntered[layer] = true
	if v.observe == nil {
		return nil
	}
	w := v.model.encoders[layer]
	sc, err := v.snapshot("sincos", v.model.sincos, nil, 0)
	if err != nil {
		return err
	}
	phases := []bool{true}
	if cold {
		phases = []bool{false, true}
	}
	for _, after := range phases {
		snaps, e := supCacheObservations(v.model.weights.encoders[layer], after, func(name string) (SUPGPUBufferSnapshot, error) {
			actual, e := v.snapshot(name, w[name], nil, 0)
			if e != nil {
				return actual, e
			}
			meta, e := w[name].Meta()
			if e != nil {
				return actual, e
			}
			if meta.ScalePresent != (name == "qkv_int8") || meta.ScalePresent && (meta.ScaleTag != tensor.TagC || meta.ScaleDType != tensor.DTypeF32) {
				return actual, fmt.Errorf("uploaded SUP cache scale binding differs: %s", name)
			}
			actual.Tags = slices.Clone(meta.Tags)
			return actual, nil
		})
		if e != nil {
			return e
		}
		for i := range snaps {
			snaps[i].Tags = nil
			snaps[i].Strides = supContiguous(snaps[i].Shape)
		}
		copySC := sc
		copySC.Data = slices.Clone(sc.Data)
		copySC.Shape = slices.Clone(sc.Shape)
		copySC.Strides = slices.Clone(sc.Strides)
		snaps = append(snaps[:3], append([]SUPGPUBufferSnapshot{copySC}, snaps[3:]...)...)
		name := fmt.Sprintf("encoder.%02d.cache.warm", layer)
		origin := SUPGPUUploaded
		if cold {
			phase := "after"
			if !after {
				phase = "before"
				origin = SUPGPUPreparedCache
			}
			name = fmt.Sprintf("encoder.%02d.cache.%s_remove_bits", layer, phase)
		}
		if err = v.ctx.Err(); err != nil {
			return err
		}
		if err = v.observe(SUPGPUObservation{Name: name, Layer: layer, Tensors: snaps, Origin: origin}); err != nil {
			return err
		}
	}
	return v.ctx.Err()
}
