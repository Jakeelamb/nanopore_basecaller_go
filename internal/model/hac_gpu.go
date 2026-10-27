package model

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"reflect"
	"slices"
	"sync"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

var ErrHACGPUInactiveSlotsUnqualified = errors.New("model: one-active-slot HAC GPU policy awaits native independence qualification")

type HACGPUProfile struct {
	Batch, Chunk int64
	Layout       GPUStorageLayout
}

// FixedHACGPUProfile remains the small verification profile.
func FixedHACGPUProfile() HACGPUProfile { return HACGPUProfile{64, 192, GPUStorageCUTLASSI8} }

// SourceDefaultHACGPUProfile is the pinned HAC fixed-chunk source default.
// Planning this geometry does not qualify its native execution.
func SourceDefaultHACGPUProfile() HACGPUProfile {
	return HACGPUProfile{64, 12288, GPUStorageCUTLASSI8}
}

// SourceShortHACGPUProfile is the default simplex half-size queue geometry.
// Its fixed graph plan does not qualify variable packing or native execution.
func SourceShortHACGPUProfile() HACGPUProfile {
	return HACGPUProfile{64, 6144, GPUStorageCUTLASSI8}
}

func (p HACGPUProfile) validate() error {
	if p != FixedHACGPUProfile() && p != SourceDefaultHACGPUProfile() && p != SourceShortHACGPUProfile() {
		return fmt.Errorf("%w: HAC GPU admits only N64/chunk192, chunk6144 or chunk12288 I8", ErrGPUPlanLayout)
	}
	return nil
}

type HACGPUPlan struct {
	profile     HACGPUProfile
	reservation GPUMemoryReservation
	input       GPUView
	convs       *HACSmallConvPlan
	last        *HACLastConvPlan
	fused       HACFusedPlan
	output      HACOutputPlan
	views       []GPUView
}

func (p HACGPUPlan) Profile() HACGPUProfile            { return p.profile }
func (p HACGPUPlan) Reservation() GPUMemoryReservation { return p.reservation }
func (p HACGPUPlan) Views() []GPUView                  { return slices.Clone(p.views) }

func hacGPUConvParams() []SmallConvParams {
	return []SmallConvParams{{1, 16, 5, 1, SmallConvSwish, GPUStorageNTC, 0}, {16, 16, 5, 1, SmallConvSwish, GPUStorageNTC, 9}, {16, 1024, 19, 6, SmallConvTanh, GPUStorageCUTLASSI8, 0}}
}

// PlanHACGPU ports the source's separate reserve and use traversals.
// Reservation propagates conv1 padding; actual use crops it before conv2.
func PlanHACGPU(profile HACGPUProfile) (out HACGPUPlan, err error) {
	if err = profile.validate(); err != nil {
		return out, err
	}
	p, err := NewGPUMemoryPlan(profile.Batch)
	if err != nil {
		return out, err
	}
	if err = p.NextTC(profile.Chunk, 1, GPUStorageNTC, GPUOrientationReverse); err != nil {
		return out, err
	}
	params := hacGPUConvParams()
	reserveT := profile.Chunk
	for _, layer := range params {
		padding, e := gpuMul(2, layer.OutputTPadding)
		if e != nil {
			return out, e
		}
		reserveT, err = gpuAdd(reserveT/layer.Stride, padding)
		if err != nil {
			return out, err
		}
		if err = p.NextTC(reserveT, layer.COut, layer.OutputLayout, GPUOrientationReverse); err != nil {
			return out, err
		}
	}
	fusedReserve, err := HACFusedScratchBytes(profile.Batch, reserveT)
	if err != nil {
		return out, err
	}
	if err = p.Temp([]int64{fusedReserve}, GPUDtypeInt8); err != nil {
		return out, err
	}
	outputReserve, err := HACOutputTempElements(profile.Batch, reserveT)
	if err != nil {
		return out, err
	}
	if err = p.Temp([]int64{outputReserve}, GPUDtypeHalf); err != nil {
		return out, err
	}
	reservation, err := p.Reserve()
	if err != nil {
		return out, err
	}
	u := reservation.Use()
	input, err := u.NextTC(profile.Chunk, 1, GPUStorageNTC, GPUOrientationReverse)
	if err != nil {
		return out, err
	}
	convs, err := PlanHACSmallConvs(profile.Batch, profile.Chunk, 1, reservation.Bytes, params)
	if err != nil {
		return out, err
	}
	for _, l := range convs.Layers {
		v, e := u.NextTC(l.Geometry.T, l.Params.COut, l.Params.OutputLayout, GPUOrientationReverse)
		if e != nil {
			return out, e
		}
		if v.Offset != l.OffsetBytes || v.Geometry != l.Geometry {
			return out, fmt.Errorf("%w: convolution placement", ErrGPUPlanMismatch)
		}
	}
	last, err := PlanHACLastConv(convs, true)
	if err != nil {
		return out, err
	}
	scratchSize, err := HACFusedScratchBytes(last.Output.N, last.Output.T)
	if err != nil {
		return out, err
	}
	scratch, err := u.Temp([]int64{scratchSize}, GPUDtypeInt8)
	if err != nil {
		return out, err
	}
	fused, err := PlanHACFusedStack(last, scratch)
	if err != nil {
		return out, err
	}
	if err = u.Alive(scratch); err != nil {
		return out, err
	}
	outputSize, err := HACOutputTempElements(last.Output.N, last.Output.T)
	if err != nil {
		return out, err
	}
	temp, err := u.Temp([]int64{outputSize}, GPUDtypeHalf)
	if err != nil {
		return out, err
	}
	output, err := PlanHACOutput(reservation, fused.OutputView(), temp)
	if err != nil {
		return out, err
	}
	if err = u.Finish(); err != nil {
		return out, err
	}
	return HACGPUPlan{profile, reservation, input, convs, last, fused, output, u.Views()}, nil
}

type hacGPUConvWeights struct{ weight, bias []float32 }
type HACGPUWeights struct {
	spec     Spec
	receipts []WeightReceipt
	convs    [3]hacGPUConvWeights
	fused    []FLSTMCache
	output   HACOutputCache
}

func (w HACGPUWeights) Receipts() []WeightReceipt {
	out := slices.Clone(w.receipts)
	for i := range out {
		out[i].Shape = slices.Clone(out[i].Shape)
	}
	return out
}

// hacGPURoles derives identity pins from the authoritative manifest; expected
// names and shapes encode graph roles, never a second archive hash list.
func hacGPURoles(manifest []WeightReceipt) (map[string]WeightReceipt, error) {
	want := map[string][]int64{}
	for i, p := range hacGPUConvParams() {
		want[fmt.Sprintf("%d.conv.weight.tensor", i)] = []int64{p.COut, p.CIn, p.WinLen}
		want[fmt.Sprintf("%d.conv.bias.tensor", i)] = []int64{p.COut}
	}
	for i := 0; i < 7; i++ {
		for j, name := range []string{"dn_weight_ih", "dn_weight_hh", "up_weight_ih", "up_weight_hh", "up_bias_ih", "up_bias_hh"} {
			want[fmt.Sprintf("%d.rnn.%s.tensor", 4+i, name)] = [][]int64{{128, 1024}, {128, 1024}, {4096, 128}, {4096, 128}, {4096}, {4096}}[j]
		}
	}
	want["11.linear.weight.tensor"], want["12.linear.weight.tensor"] = []int64{128, 1024}, []int64{1024, 128}
	if len(manifest) != len(want) {
		return nil, fmt.Errorf("%w: HAC GPU needs exactly50 named weights", ErrGPUWeights)
	}
	authoritative, err := WeightManifest(HAC)
	if err != nil {
		return nil, err
	}
	identity := map[string]string{}
	for _, p := range authoritative {
		identity[p.Name] = p.SHA256
	}
	out := map[string]WeightReceipt{}
	for _, p := range manifest {
		shape, ok := want[p.Name]
		if !ok || !slices.Equal(shape, p.Shape) || len(p.SHA256) != 64 || identity[p.Name] != p.SHA256 {
			return nil, fmt.Errorf("%w: named HAC role %s", ErrGPUWeights, p.Name)
		}
		if _, ok := out[p.Name]; ok {
			return nil, fmt.Errorf("%w: repeated HAC weight %s", ErrGPUWeights, p.Name)
		}
		out[p.Name] = p
	}
	return out, nil
}

type hacGPUWeightSource struct {
	spec     Spec
	receipts []WeightReceipt
	roles    map[string]WeightReceipt
}

func openHACGPUWeightSource(ctx context.Context, e *tensor.Engine, directory string) (out hacGPUWeightSource, err error) {
	if ctx == nil {
		return out, fmt.Errorf("nil HAC GPU context")
	}
	if err = ctx.Err(); err != nil {
		return out, err
	}
	if e == nil {
		return out, tensor.ErrClosed
	}
	out.spec, err = Inspect(directory)
	if err != nil {
		return out, err
	}
	if out.spec.Architecture != HAC {
		return out, fmt.Errorf("%w: fixed HAC config required", ErrGPUWeights)
	}
	out.receipts, err = WeightManifest(HAC)
	if err != nil {
		return out, err
	}
	out.roles, err = hacGPURoles(out.receipts)
	return out, err
}
func readHACGPUWeightValues(ctx context.Context, e *tensor.Engine, directory string, pins []WeightReceipt) (map[string][]float32, error) {
	values := map[string][]float32{}
	for _, pin := range pins {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		w, err := loadWeight(e, filepath.Join(directory, pin.Name), pin)
		if err != nil {
			return nil, err
		}
		data, readErr := w.Data()
		if err = errors.Join(readErr, w.Close()); err != nil {
			return nil, err
		}
		values[pin.Name] = data
	}
	return values, ctx.Err()
}
func hacGPURecurrentNames(layer int) []string {
	out := []string{}
	for _, name := range []string{"dn_weight_ih", "dn_weight_hh", "up_weight_ih", "up_weight_hh", "up_bias_ih", "up_bias_hh"} {
		out = append(out, fmt.Sprintf("%d.rnn.%s.tensor", 4+layer, name))
	}
	return out
}
func hacGPURecurrentWeights(values map[string][]float32) []FLSTMWeightSet {
	out := make([]FLSTMWeightSet, 7)
	for i := range out {
		names := hacGPURecurrentNames(i)
		out[i] = FLSTMWeightSet{values[names[0]], values[names[1]], values[names[2]], values[names[3]], values[names[4]], values[names[5]]}
	}
	return out
}

// LoadHACFLSTMWeights shares the graph's verified role binding and CPU archive
// loader while reading only the42 named recurrent archives. No model is run.
func LoadHACFLSTMWeights(ctx context.Context, e *tensor.Engine, directory string) ([]FLSTMWeightSet, []WeightReceipt, error) {
	source, err := openHACGPUWeightSource(ctx, e, directory)
	if err != nil {
		return nil, nil, err
	}
	pins := []WeightReceipt{}
	for i := 0; i < 7; i++ {
		for _, name := range hacGPURecurrentNames(i) {
			pins = append(pins, source.roles[name])
		}
	}
	values, err := readHACGPUWeightValues(ctx, e, directory, pins)
	if err != nil {
		return nil, nil, err
	}
	return hacGPURecurrentWeights(values), pins, nil
}

func LoadHACGPUWeights(ctx context.Context, e *tensor.Engine, directory string) (out HACGPUWeights, err error) {
	source, err := openHACGPUWeightSource(ctx, e, directory)
	if err != nil {
		return out, err
	}
	values, err := readHACGPUWeightValues(ctx, e, directory, source.receipts)
	if err != nil {
		return out, err
	}
	out.spec, out.receipts = source.spec, source.receipts
	for i := range out.convs {
		out.convs[i] = hacGPUConvWeights{values[fmt.Sprintf("%d.conv.weight.tensor", i)], values[fmt.Sprintf("%d.conv.bias.tensor", i)]}
	}
	for i, weights := range hacGPURecurrentWeights(values) {
		if err = ctx.Err(); err != nil {
			return HACGPUWeights{}, err
		}
		cache, e2 := PackFLSTMLayer(FLSTMParams{1024, 128}, weights)
		if e2 != nil {
			return HACGPUWeights{}, fmt.Errorf("HAC GPU recurrent cache%d: %w", i, e2)
		}
		out.fused = append(out.fused, cache)
	}
	dn, up := "11.linear.weight.tensor", "12.linear.weight.tensor"
	out.output, err = PackHACOutput(HACOutputWeights{outputWeight{source.roles[dn], values[dn]}, outputWeight{source.roles[up], values[up]}})
	if err != nil {
		return HACGPUWeights{}, err
	}
	return out, ctx.Err()
}

type HACGPU struct {
	mu       sync.Mutex
	device   *tensor.Device
	deviceID uint64
	spec     Spec
	pending  []*tensor.Storage
	receipts []WeightReceipt
	small    [2]*HACSmallConv
	last     *HACLastConv
	fused    *HACFusedStack
	output   *HACOutput
	closed   bool
}

func NewHACGPU(d *tensor.Device, s *tensor.Stream, w HACGPUWeights) (m *HACGPU, err error) {
	pinned, _ := Pinned(HAC)
	if w.spec != pinned || len(w.fused) != 7 {
		return nil, fmt.Errorf("%w: verified HAC GPU weights required", ErrGPUWeights)
	}
	if _, err = hacGPURoles(w.receipts); err != nil {
		return nil, err
	}
	if d == nil || s == nil {
		return nil, tensor.ErrClosed
	}
	m = &HACGPU{device: d, spec: w.spec, receipts: w.Receipts()}
	defer func() {
		if err != nil {
			cleanup := m.Close()
			err = errors.Join(err, cleanup)
			if cleanup == nil {
				m = nil
			}
		}
	}()
	params := hacGPUConvParams()
	for i := range m.small {
		m.small[i], err = NewHACSmallConv(d, s, params[i], w.convs[i].weight, w.convs[i].bias)
		if err != nil {
			return m, err
		}
	}
	m.last, err = NewHACLastConv(d, s, params[2], DefaultLastConvDevOptions(), w.convs[2].weight, w.convs[2].bias)
	if err != nil {
		return m, err
	}
	m.fused, err = NewHACFusedStack(d, s, w.fused)
	if err != nil {
		return m, err
	}
	m.output, err = NewHACOutput(d, s, w.output)
	if err != nil {
		return m, err
	}
	meta, err := m.small[0].Weights.Meta()
	if err != nil {
		return m, err
	}
	m.deviceID = meta.Device
	return m, nil
}

func (m *HACGPU) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	var err error
	m.pending, err = closeGPUStorageOwners(m.pending)
	if m.output != nil {
		e := m.output.Close()
		err = errors.Join(err, e)
		if e == nil {
			m.output = nil
		}
	}
	if m.fused != nil {
		e := m.fused.Close()
		err = errors.Join(err, e)
		if e == nil {
			m.fused = nil
		}
	}
	if m.last != nil {
		e := m.last.Close()
		err = errors.Join(err, e)
		if e == nil {
			m.last = nil
		}
	}
	for i, c := range m.small {
		if c != nil {
			e := c.Close()
			err = errors.Join(err, e)
			if e == nil {
				m.small[i] = nil
			}
		}
	}
	return err
}

type HACGPUBatch struct{ ActiveSlots int }
type HACGPUScoreMetadata struct {
	QScale, QBias, BlankScore, ScoreMin, ScoreMax float32
	DecoderClampValue                             float32
	StateLen                                      int64
	MeanQStart                                    int
}

func hacGPUScoreMetadata(s Spec) HACGPUScoreMetadata {
	return HACGPUScoreMetadata{s.QScale, s.QBias, s.BlankScore, -5, 5, 0, s.StateLen, s.MeanQStart}
}

type HACGPUResult struct {
	Scores                     *tensor.Storage
	ActiveSlots, PhysicalSlots int
	Metadata                   HACGPUScoreMetadata
	LastConv                   LastConvDispatch
}

func (r *HACGPUResult) Close() error {
	if r == nil || r.Scores == nil {
		return nil
	}
	return r.Scores.Close()
}

// Observations borrow live storage only until the callback returns. They must
// not mutate or close it. Both entry points execute the same production graph.
type HACGPUObservation struct {
	Name               string
	Storage, Workspace *tensor.Storage
	View               GPUView
}
type HACGPUObserver func(HACGPUObservation) error

func validateHACGPUInput(meta tensor.StorageMeta, b HACGPUBatch, deviceID uint64, probe bool, profile HACGPUProfile) error {
	if err := profile.validate(); err != nil {
		return err
	}
	if b.ActiveSlots != 1 && int64(b.ActiveSlots) != profile.Batch {
		return fmt.Errorf("%w: H08 supports one logical slot or the64-active qualification case", ErrGPUPlanLayout)
	}
	if b.ActiveSlots == 1 && !probe {
		return ErrHACGPUInactiveSlotsUnqualified
	}
	if meta.DType != tensor.DTypeF16 || meta.Device != deviceID || meta.ScalePresent || !slices.Equal(meta.Shape, []int64{int64(b.ActiveSlots), 1, profile.Chunk}) || !slices.Equal(meta.Strides, []int64{profile.Chunk, profile.Chunk, 1}) || !slices.Equal(meta.Tags, []tensor.KoiTag{tensor.TagN, tensor.TagC, tensor.TagT}) {
		return fmt.Errorf("%w: owned contiguous half NCT input on the model device required", ErrGPUPlanLayout)
	}
	return nil
}

type hacGPUOps interface {
	input() error
	conv(int) error
	recurrent(context.Context) error
	project(context.Context) error
}

func runHACGPU(ctx context.Context, o hacGPUOps) error {
	if ctx == nil {
		return fmt.Errorf("nil HAC GPU context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := o.input(); err != nil {
		return err
	}
	for i := 0; i < 3; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := o.conv(i); err != nil {
			return fmt.Errorf("conv%d: %w", i, err)
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := o.recurrent(ctx); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := o.project(ctx); err != nil {
		return err
	}
	return ctx.Err()
}

func (m *HACGPU) Forward(ctx context.Context, s *tensor.Stream, input *tensor.Storage, p HACGPUPlan, b HACGPUBatch) (*HACGPUResult, error) {
	return m.forward(ctx, s, input, p, b, 0, nil, false)
}
func (m *HACGPU) ForwardObserved(ctx context.Context, s *tensor.Stream, input *tensor.Storage, p HACGPUPlan, b HACGPUBatch, pattern byte, observe HACGPUObserver) (*HACGPUResult, error) {
	return m.forward(ctx, s, input, p, b, pattern, observe, false)
}

// ProbeInactiveSlots uses the same Forward implementation solely to qualify the
// zero-padded one-logical-slot policy. It does not promote production admission.
func (m *HACGPU) ProbeInactiveSlots(ctx context.Context, s *tensor.Stream, input *tensor.Storage, p HACGPUPlan, pattern byte, observe HACGPUObserver) (*HACGPUResult, error) {
	return m.forward(ctx, s, input, p, HACGPUBatch{1}, pattern, observe, true)
}

type hacGPUInvocation struct {
	m                            *HACGPU
	plan                         HACGPUPlan
	s                            *tensor.Stream
	ctx                          context.Context
	batch                        HACGPUBatch
	observe                      HACGPUObserver
	workspace, source, inputView *tensor.Storage
	alloc                        [3]*tensor.Storage
	crop                         [2]*tensor.Storage
	lastOut, full, scores        *tensor.Storage
	dispatch                     LastConvDispatch
}

func (o *hacGPUInvocation) emit(name string, v *tensor.Storage, view GPUView) error {
	if err := o.ctx.Err(); err != nil {
		return err
	}
	if o.observe != nil {
		if err := o.observe(HACGPUObservation{name, v, o.workspace, view}); err != nil {
			return err
		}
	}
	return o.ctx.Err()
}
func (o *hacGPUInvocation) input() error {
	if err := o.inputView.Fill(o.s, []byte{0, 0}); err != nil {
		return err
	}
	if err := o.ctx.Err(); err != nil {
		return err
	}
	g := o.plan.input.Geometry
	dst, err := o.inputView.View(0, []int64{int64(o.batch.ActiveSlots), g.T, g.C}, []int64{g.T * g.C, g.C, 1}, []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC})
	if err != nil {
		return err
	}
	copyErr := o.source.CopyTo(o.s, dst)
	cleanup := dst.Close()
	if cleanup != nil {
		o.m.pending = append(o.m.pending, dst)
	}
	err = errors.Join(copyErr, cleanup)
	if err != nil {
		return err
	}
	return o.emit("input", o.inputView, o.plan.input)
}
func (o *hacGPUInvocation) conv(i int) error {
	p := o.plan
	if i < 2 {
		in := o.inputView
		if i == 1 {
			in = o.alloc[0]
		}
		if err := o.m.small[i].Run(o.s, in, o.crop[i], p.convs.Layers[i]); err != nil {
			return err
		}
	} else {
		failed, err := ZeroInputPadding(o.s, o.alloc[1], p.last.Input.N, p.last.Input.Rows, p.last.Input.C, p.convs.Layers[1].Params.OutputTPadding)
		o.m.pending = append(o.m.pending, failed...)
		if err != nil {
			return err
		}
		if err := o.ctx.Err(); err != nil {
			return err
		}
		o.dispatch, err = o.m.last.Run(o.s, o.alloc[1], o.lastOut, p.last)
		if err != nil {
			return err
		}
	}
	l := p.convs.Layers[i]
	return o.emit(fmt.Sprintf("conv%d", i), o.alloc[i], GPUView{Geometry: l.Geometry, Orientation: GPUOrientationReverse, Offset: l.OffsetBytes, End: l.OffsetBytes + l.Geometry.Bytes})
}

type HACGPUCacheTensor struct {
	Role  string
	DType tensor.DType
	Shape []int64
	Data  []byte
}

// CacheTensors downloads the actual uploaded owners, without repacking a second
// copy of their arithmetic. Returned payloads are independent host copies.
func (m *HACGPU) CacheTensors(s *tensor.Stream) (out []HACGPUCacheTensor, err error) {
	if m == nil {
		return nil, ErrClosed
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, ErrClosed
	}
	add := func(role string, v *tensor.Storage) error {
		meta, e := v.Meta()
		if e != nil {
			return e
		}
		data, e := v.Download(s)
		if e != nil {
			return e
		}
		out = append(out, HACGPUCacheTensor{role, meta.DType, slices.Clone(meta.Shape), data})
		return nil
	}
	for i, c := range m.small {
		if err = add(fmt.Sprintf("conv%d.weight", i), c.Weights); err != nil {
			return nil, err
		}
		if err = add(fmt.Sprintf("conv%d.bias", i), c.Bias); err != nil {
			return nil, err
		}
	}
	if err = add("conv2.weight", m.last.Weights); err != nil {
		return nil, err
	}
	if err = add("conv2.bias", m.last.Bias); err != nil {
		return nil, err
	}
	for i, c := range m.fused.caches {
		for j, v := range []*tensor.Storage{c.DownIH, c.DownHH, c.DownScaleIH, c.DownScaleHH, c.Up, c.UpScale, c.UpBias} {
			if err = add(fmt.Sprintf("flstm%d.%s", i, []string{"dn-ih-cache", "dn-hh-cache", "dn-ih-cache-scale", "dn-hh-cache-scale", "up-cache", "up-cache-scale", "up-cache-bias"}[j]), v); err != nil {
				return nil, err
			}
		}
	}
	for i, v := range []*tensor.Storage{m.output.down, m.output.downScale, m.output.up} {
		if err = add([]string{"projection.down", "projection.down-scale", "projection.up"}[i], v); err != nil {
			return nil, err
		}
	}
	return out, nil
}
func (o *hacGPUInvocation) recurrent(ctx context.Context) error {
	var err error
	o.full, err = o.m.fused.RunObserved(ctx, o.s, o.workspace, o.plan.fused, func(int) error { return ctx.Err() }, func(i int, v *tensor.Storage) error {
		view := o.plan.fused.OutputView()
		view.Orientation = GPUOrientationReverse
		if i%2 == 0 {
			view.Orientation = GPUOrientationForward
		}
		return o.emit(fmt.Sprintf("flstm%d", i), v, view)
	})
	return err
}
func (o *hacGPUInvocation) project(ctx context.Context) error {
	var err error
	o.scores, err = o.m.output.Run(ctx, o.s, o.workspace, o.full, o.plan.output)
	if err != nil {
		return err
	}
	g, err := NewGPUStorageGeometry(GPUStorageNTC, o.plan.output.n, o.plan.output.t, o.plan.last.Output.C)
	if err != nil {
		return err
	}
	view := GPUView{Geometry: g, Offset: o.plan.output.out.offset, End: o.plan.output.out.offset + g.Bytes}
	if err = o.emit("linear1", o.scores, view); err != nil {
		return err
	}
	return o.emit("scores", o.scores, view)
}

func (m *HACGPU) forward(ctx context.Context, s *tensor.Stream, input *tensor.Storage, p HACGPUPlan, b HACGPUBatch, pattern byte, observe HACGPUObserver, probe bool) (result *HACGPUResult, err error) {
	if m == nil {
		return nil, ErrClosed
	}
	if ctx == nil {
		return nil, fmt.Errorf("nil HAC GPU context")
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, ErrClosed
	}
	if len(m.pending) != 0 || m.fused != nil && len(m.fused.pending) != 0 || m.output != nil && len(m.output.pending) != 0 {
		return nil, fmt.Errorf("HAC cleanup pending; Close must recover owners")
	}
	want, planErr := PlanHACGPU(p.profile)
	if planErr != nil || !reflect.DeepEqual(p, want) {
		return nil, fmt.Errorf("%w: initialized HAC forward plan required", ErrGPUPlanState)
	}
	if input == nil {
		return nil, tensor.ErrClosed
	}
	meta, err := input.Meta()
	if err != nil {
		return nil, err
	}
	if err = validateHACGPUInput(meta, b, m.deviceID, probe, p.profile); err != nil {
		return nil, err
	}
	o := &hacGPUInvocation{m: m, plan: p, s: s, ctx: ctx, batch: b, observe: observe}
	var owned []*tensor.Storage
	defer func() {
		failed, cleanup := closeGPUStorageOwners(append(owned, o.full))
		m.pending = append(m.pending, failed...)
		err = errors.Join(err, cleanup)
		if err != nil {
			if o.scores != nil {
				if e := o.scores.Close(); e != nil {
					err = errors.Join(err, e)
					m.pending = append(m.pending, o.scores)
				}
			}
			result = nil
		}
	}()
	keep := func(v *tensor.Storage, e error) (*tensor.Storage, error) {
		if v != nil {
			owned = append(owned, v)
		}
		return v, e
	}
	o.workspace, err = keep(m.device.NewStorage(s, tensor.DTypeU8, []int64{p.reservation.Bytes}, []tensor.KoiTag{tensor.TagC}, bytes.Repeat([]byte{pattern}, int(p.reservation.Bytes))))
	if err != nil {
		return nil, err
	}
	actual, err := o.workspace.Meta()
	if err != nil {
		return nil, err
	}
	if actual.DType != tensor.DTypeU8 || actual.OffsetElements != 0 || actual.Device != m.deviceID || actual.StorageElements < uint64(p.reservation.Bytes) || !slices.Equal(actual.Shape, []int64{p.reservation.Bytes}) || !slices.Equal(actual.Strides, []int64{1}) {
		return nil, fmt.Errorf("%w: actual owned workspace does not cover checked P03 reservation", ErrGPUPlanUndersized)
	}
	bind := func(dtype tensor.DType, offset int64, shape, strides []int64, tags []tensor.KoiTag) (*tensor.Storage, error) {
		return keep(o.workspace.Reinterpret(dtype, uint64(offset/int64(dtype.Size())), shape, strides, tags))
	}
	g := p.input.Geometry
	o.inputView, err = bind(tensor.DTypeF16, p.input.Offset, []int64{g.N, g.T, g.C}, []int64{g.T * g.C, g.C, 1}, []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC})
	if err != nil {
		return nil, err
	}
	o.source, err = keep(input.View(0, []int64{int64(b.ActiveSlots), g.T, g.C}, []int64{g.T * g.C, g.C, 1}, []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC}))
	if err != nil {
		return nil, err
	}
	for i, l := range p.convs.Layers {
		if i < 2 {
			o.alloc[i], err = bind(tensor.DTypeF16, l.OffsetBytes, []int64{l.Geometry.N, l.Geometry.Rows, l.Geometry.C}, []int64{l.Geometry.Rows * l.Params.COut, l.Params.COut, 1}, []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC})
			if err != nil {
				return nil, err
			}
			o.crop[i], err = keep(o.alloc[i].View(l.CropOffsetElements(), l.CropShape(), l.CropStrides(), []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC}))
			if err != nil {
				return nil, err
			}
		} else {
			o.alloc[i], err = bind(tensor.DTypeI8, l.OffsetBytes, []int64{l.Geometry.Rows, l.Geometry.N, l.Geometry.C}, []int64{l.Geometry.N * l.Geometry.C, l.Geometry.C, 1}, []tensor.KoiTag{tensor.TagT, tensor.TagN, tensor.TagC})
			if err != nil {
				return nil, err
			}
			o.lastOut, err = keep(o.alloc[i].View(p.last.OutputViewOffsetElements(), p.last.OutputViewShape(), p.last.OutputViewStrides(), []tensor.KoiTag{tensor.TagN, tensor.TagT, tensor.TagC}))
			if err != nil {
				return nil, err
			}
		}
	}
	if err = runHACGPU(ctx, o); err != nil {
		return nil, err
	}
	return &HACGPUResult{o.scores, b.ActiveSlots, int(p.profile.Batch), hacGPUScoreMetadata(m.spec), o.dispatch}, nil
}
