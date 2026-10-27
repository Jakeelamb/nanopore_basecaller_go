package model

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

// HACFusedPlan is the fixed N64, C1024, K128 recurrence in one planned arena.
// Its output keeps all T+5 rows. Projection chooses the forward row-2 view.
type HACFusedPlan struct {
	n, t                                   int64
	inout, scratch, down, scale, workspace fusedSpan
}
type fusedTarget uint8

const (
	fusedInOut fusedTarget = iota
	fusedScratch
)

type fusedSpan struct {
	target        fusedTarget
	offset, bytes int64
}
type fusedLayerCall struct {
	layer                   int
	n, t, direction, parity int32
}
type fusedOps interface {
	fill(fusedSpan, byte) error
	layer(fusedLayerCall) error
}

func fusedScratchSizes(n, t int64) (down, scale, work int64, err error) {
	if n <= 0 || n%64 != 0 || n > math.MaxInt32 || t <= 0 || t > math.MaxInt32 {
		err = fmt.Errorf("%w: fixed FLSTM requires N64 and positive native-int T", ErrGPUPlanLayout)
		return
	}
	rows, e := gpuAdd(t, 1)
	if e != nil {
		err = e
		return
	}
	nt, e := gpuMul(rows, n)
	if e != nil {
		err = e
		return
	}
	down, err = gpuMul(nt, 512)
	if err != nil {
		return
	}
	scale, err = gpuMul(nt, 4)
	if err != nil {
		return
	}
	work, err = gpuMul(n/2, 4)
	return
}

// HACFusedScratchBytes matches FLSTMStack.cpp's combined U8 temp extent.
func HACFusedScratchBytes(n, t int64) (int64, error) {
	d, s, w, e := fusedScratchSizes(n, t)
	if e != nil {
		return 0, e
	}
	v, e := gpuAdd(d, s)
	if e != nil {
		return 0, e
	}
	return gpuAdd(v, w)
}

// PlanHACFusedStack admits only the fixed graph. Variable chunk encoding has
// no operand in this contract and needs its own source-derived execution path.
func PlanHACFusedStack(last *HACLastConvPlan, scratch GPUView) (HACFusedPlan, error) {
	var p HACFusedPlan
	if last == nil || !last.InputToRevLSTM {
		return p, fmt.Errorf("%w: fixed FLSTM must follow reverse-input conv2", ErrGPUPlanMismatch)
	}
	g := last.Output
	if g.Layout != GPUStorageCUTLASSI8 || g.Dtype != GPUDtypeInt8 || g.C != 1024 {
		return p, fmt.Errorf("%w: fixed FLSTM requires CUTLASS I8 C1024", ErrGPUPlanLayout)
	}
	expected, e := NewGPUStorageGeometry(GPUStorageCUTLASSI8, g.N, g.T, 1024)
	if e != nil {
		return p, e
	}
	if g != expected || last.Layer.TOut != g.T {
		return p, fmt.Errorf("%w: inconsistent conv2 geometry", ErrGPUPlanLayout)
	}
	size, e := HACFusedScratchBytes(g.N, g.T)
	if e != nil {
		return p, e
	}
	if !scratch.Scratch || scratch.Geometry.Layout != GPUStorageRaw || scratch.Geometry.Dtype != GPUDtypeInt8 || scratch.Offset < 0 || scratch.Geometry.Bytes < size || scratch.End < scratch.Offset || scratch.End-scratch.Offset != scratch.Geometry.Bytes {
		return p, fmt.Errorf("%w: FLSTM scratch is not a sufficient U8 temp", ErrGPUPlanUndersized)
	}
	end, e := gpuAdd(last.OutputOffsetBytes, g.Bytes)
	if e != nil {
		return p, e
	}
	if last.OutputOffsetBytes < 0 || last.OutputOffsetBytes%GPUWorkingMemoryAlignment != 0 || scratch.Offset%GPUWorkingMemoryAlignment != 0 {
		return p, fmt.Errorf("%w: FLSTM arena offsets must be aligned", ErrGPUPlanLayout)
	}
	if last.OutputOffsetBytes < scratch.End && scratch.Offset < end {
		return p, fmt.Errorf("%w: FLSTM scratch overlaps inout", ErrGPUPlanOverlap)
	}
	d, s, w, e := fusedScratchSizes(g.N, g.T)
	if e != nil {
		return p, e
	}
	p = HACFusedPlan{n: g.N, t: g.T, inout: fusedSpan{fusedInOut, last.OutputOffsetBytes, g.Bytes}, scratch: fusedSpan{fusedScratch, scratch.Offset, size}, down: fusedSpan{fusedScratch, scratch.Offset, d}, scale: fusedSpan{fusedScratch, scratch.Offset + d, s}, workspace: fusedSpan{fusedScratch, scratch.Offset + d + s, w}}
	return p, nil
}

// OutputView describes the complete allocation with the final forward orientation.
func (p HACFusedPlan) OutputView() GPUView {
	g, _ := NewGPUStorageGeometry(GPUStorageCUTLASSI8, p.n, p.t, 1024)
	return GPUView{Geometry: g, Orientation: GPUOrientationForward, Offset: p.inout.offset, End: p.inout.offset + p.inout.bytes}
}

func runHACFused(ctx context.Context, p HACFusedPlan, ops fusedOps) error {
	if p.n <= 0 || p.t <= 0 {
		return fmt.Errorf("%w: uninitialized FLSTM plan", ErrGPUPlanState)
	}
	row := p.n * 1024
	for _, init := range []struct {
		row   int64
		value byte
	}{{0, 0}, {p.t + 1, 0}, {p.t + 2, 1}} {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := ops.fill(fusedSpan{fusedInOut, p.inout.offset + init.row*row, row}, init.value); err != nil {
			return fmt.Errorf("FLSTM sentinel row %d: %w", init.row, err)
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := ops.fill(p.scratch, 0); err != nil {
		return fmt.Errorf("FLSTM scratch: %w", err)
	}
	for i := 0; i < 7; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		direction := int32(-1)
		if i%2 != 0 {
			direction = 1
		}
		if err := ops.layer(fusedLayerCall{i, int32(p.n), int32(p.t), direction, int32(1 - i%2)}); err != nil {
			return fmt.Errorf("FLSTM layer %d: %w", i, err)
		}
	}
	return ctx.Err()
}

// HACFusedStack owns seven uploaded H04 caches. One worker exclusively owns
// Run and Close together with its stream and invocation arena.
type HACFusedStack struct {
	device  *tensor.Device
	caches  [7]tensor.FactorisedLSTMOperands
	pending []*tensor.Storage
	closed  bool
}

type fusedCacheSpec struct {
	dtype tensor.DType
	shape []int64
	data  []byte
}

func fusedCacheSpecs(caches []FLSTMCache) ([7][7]fusedCacheSpec, error) {
	var out [7][7]fusedCacheSpec
	if len(caches) != 7 {
		return out, fmt.Errorf("%w: FLSTM needs seven caches", ErrGPUWeights)
	}
	signed := func(v []int8) []byte {
		b := make([]byte, len(v))
		for i, x := range v {
			b[i] = byte(x)
		}
		return b
	}
	for i, c := range caches {
		if c.Params != (FLSTMParams{C: 1024, K: 128}) || len(c.DownIH) != 128*1024 || len(c.DownHH) != 128*1024 || len(c.DownScaleIH) != 128 || len(c.DownScaleHH) != 128 || len(c.Up) != 4096*256 || len(c.UpScale) != 4096 || len(c.UpBias) != 4096 {
			return out, fmt.Errorf("%w: FLSTM cache %d geometry", ErrGPUWeights, i)
		}
		out[i] = [7]fusedCacheSpec{{tensor.DTypeI8, []int64{128, 1024}, signed(c.DownIH)}, {tensor.DTypeI8, []int64{128, 1024}, signed(c.DownHH)}, {tensor.DTypeF16, []int64{128}, HalfBytes(c.DownScaleIH)}, {tensor.DTypeF16, []int64{128}, HalfBytes(c.DownScaleHH)}, {tensor.DTypeI8, []int64{4096, 256}, signed(c.Up)}, {tensor.DTypeF16, []int64{4096}, HalfBytes(c.UpScale)}, {tensor.DTypeF16, []int64{4096}, HalfBytes(c.UpBias)}}
	}
	return out, nil
}

func NewHACFusedStack(d *tensor.Device, s *tensor.Stream, caches []FLSTMCache) (*HACFusedStack, error) {
	specs, err := fusedCacheSpecs(caches)
	if err != nil {
		return nil, err
	}
	stack := &HACFusedStack{device: d}
	for i, layer := range specs {
		c := &stack.caches[i]
		slots := []**tensor.Storage{&c.DownIH, &c.DownHH, &c.DownScaleIH, &c.DownScaleHH, &c.Up, &c.UpScale, &c.UpBias}
		for j, v := range layer {
			tags := make([]tensor.KoiTag, len(v.shape))
			for k := range tags {
				tags[k] = tensor.TagC
			}
			*slots[j], err = d.NewStorage(s, v.dtype, v.shape, tags, v.data)
			if err != nil {
				return stack, errors.Join(err, stack.Close())
			}
		}
	}
	return stack, nil
}
func (s *HACFusedStack) Close() error {
	if s == nil {
		return nil
	}
	s.closed = true
	var errs []error
	var pendingErr error
	s.pending, pendingErr = closeGPUStorageOwners(s.pending)
	errs = append(errs, pendingErr)
	for i := range s.caches {
		c := &s.caches[i]
		for _, p := range []**tensor.Storage{&c.DownIH, &c.DownHH, &c.DownScaleIH, &c.DownScaleHH, &c.Up, &c.UpScale, &c.UpBias} {
			if *p != nil {
				if err := (*p).Close(); err != nil {
					errs = append(errs, err)
				} else {
					*p = nil
				}
			}
		}
	}
	return errors.Join(errs...)
}

type fusedCUDAOps struct {
	stack       *HACFusedStack
	stream      *tensor.Stream
	fills       map[fusedSpan]*tensor.Storage
	operands    [7]tensor.FactorisedLSTMOperands
	afterLayer  func(int, *tensor.Storage) error
	beforeLayer func(int) error
}

func (o *fusedCUDAOps) fill(span fusedSpan, v byte) error {
	return o.fills[span].Fill(o.stream, []byte{v})
}
func (o *fusedCUDAOps) layer(c fusedLayerCall) error {
	if o.beforeLayer != nil {
		if err := o.beforeLayer(c.layer); err != nil {
			return err
		}
	}
	if err := o.stack.device.FactorisedLSTM(o.stream, o.operands[c.layer], tensor.FactorisedLSTMGeometry{N: c.n, T: c.t, Direction: c.direction, Parity: c.parity}); err != nil {
		return err
	}
	if o.afterLayer != nil {
		return o.afterLayer(c.layer, o.operands[c.layer].InOut)
	}
	return nil
}

// Run binds every view before writing. Each operation drains synchronously at
// the native boundary. After failure the caller must regenerate conv2 input.
func (s *HACFusedStack) Run(ctx context.Context, stream *tensor.Stream, backing *tensor.Storage, p HACFusedPlan) (*tensor.Storage, error) {
	return s.run(ctx, stream, backing, p, nil, nil)
}

// RunObserved executes the same production loop and borrows its full inout to
// the observer after each completed layer. The observer must not retain or
// mutate the borrowed owner. An observer error aborts the remaining layers.
func (s *HACFusedStack) RunObserved(ctx context.Context, stream *tensor.Stream, backing *tensor.Storage, p HACFusedPlan, beforeLayer func(int) error, afterLayer func(int, *tensor.Storage) error) (*tensor.Storage, error) {
	return s.run(ctx, stream, backing, p, beforeLayer, afterLayer)
}
func (s *HACFusedStack) run(ctx context.Context, stream *tensor.Stream, backing *tensor.Storage, p HACFusedPlan, beforeLayer func(int) error, afterLayer func(int, *tensor.Storage) error) (out *tensor.Storage, err error) {
	if s == nil || s.closed || len(s.pending) != 0 {
		return nil, tensor.ErrClosed
	}
	if e := ctx.Err(); e != nil {
		return nil, e
	}
	if p.n <= 0 || p.t <= 0 {
		return nil, fmt.Errorf("%w: uninitialized FLSTM plan", ErrGPUPlanState)
	}
	meta, e := backing.Meta()
	if e != nil {
		return nil, e
	}
	contiguous := int64(1)
	for i := len(meta.Shape) - 1; i >= 0; i-- {
		if meta.Strides[i] != contiguous {
			return nil, fmt.Errorf("%w: FLSTM arena must be contiguous", ErrGPUPlanLayout)
		}
		contiguous *= meta.Shape[i]
	}
	if p.inout.offset+p.inout.bytes > contiguous*int64(meta.DType.Size()) || p.scratch.offset+p.scratch.bytes > contiguous*int64(meta.DType.Size()) {
		return nil, fmt.Errorf("%w: FLSTM actual backing", ErrGPUPlanUndersized)
	}
	var aliases []*tensor.Storage
	defer func() {
		failed, cleanup := closeGPUStorageOwners(aliases)
		s.pending = append(s.pending, failed...)
		err = errors.Join(err, cleanup)
		if err != nil && out != nil {
			if e := out.Close(); e != nil {
				err = errors.Join(err, e)
				s.pending = append(s.pending, out)
			}
			out = nil
		}
	}()
	bind := func(dtype tensor.DType, span fusedSpan, shape, strides []int64, tags []tensor.KoiTag) (*tensor.Storage, error) {
		v, e := backing.Reinterpret(dtype, uint64(span.offset/int64(dtype.Size())), shape, strides, tags)
		if v != nil {
			aliases = append(aliases, v)
		}
		return v, e
	}
	full, e := bind(tensor.DTypeI8, p.inout, []int64{p.t + 5, p.n, 1024}, []int64{p.n * 1024, 1024, 1}, []tensor.KoiTag{tensor.TagT, tensor.TagN, tensor.TagC})
	if e != nil {
		return nil, e
	}
	byteView := func(span fusedSpan) (*tensor.Storage, error) {
		return bind(tensor.DTypeU8, span, []int64{span.bytes}, []int64{1}, []tensor.KoiTag{tensor.TagC})
	}
	down, e := byteView(p.down)
	if e != nil {
		return nil, e
	}
	scale, e := byteView(p.scale)
	if e != nil {
		return nil, e
	}
	work, e := byteView(p.workspace)
	if e != nil {
		return nil, e
	}
	ops := &fusedCUDAOps{stack: s, stream: stream, fills: make(map[fusedSpan]*tensor.Storage), afterLayer: afterLayer, beforeLayer: beforeLayer}
	row := p.n * 1024
	for _, r := range []int64{0, p.t + 1, p.t + 2} {
		span := fusedSpan{fusedInOut, p.inout.offset + r*row, row}
		v, e := bind(tensor.DTypeI8, span, []int64{row}, []int64{1}, []tensor.KoiTag{tensor.TagC})
		if e != nil {
			return nil, e
		}
		ops.fills[span] = v
	}
	v, e := byteView(p.scratch)
	if e != nil {
		return nil, e
	}
	ops.fills[p.scratch] = v
	for i, c := range s.caches {
		c.InOut = full
		c.Down = down
		c.DownScale = scale
		c.Workspace = work
		ops.operands[i] = c
		dir := int32(-1)
		if i%2 != 0 {
			dir = 1
		}
		if e := s.device.ValidateFactorisedLSTM(stream, c, tensor.FactorisedLSTMGeometry{N: int32(p.n), T: int32(p.t), Direction: dir, Parity: int32(1 - i%2)}); e != nil {
			return nil, e
		}
	}
	if e := runHACFused(ctx, p, ops); e != nil {
		return nil, e
	}
	out, e = full.View(0, []int64{p.t + 5, p.n, 1024}, []int64{p.n * 1024, 1024, 1}, []tensor.KoiTag{tensor.TagT, tensor.TagN, tensor.TagC})
	return out, e
}
