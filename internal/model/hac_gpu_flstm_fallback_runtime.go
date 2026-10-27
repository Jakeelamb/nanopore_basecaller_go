package model

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

type halfLayerStorage struct{ downIH, downHH, up, bias *tensor.Storage }
type HACHalfStack struct {
	mu     sync.Mutex
	device *tensor.Device
	layers [7]halfLayerStorage
	closed bool
}

func NewHACHalfStack(d *tensor.Device, stream *tensor.Stream, caches []FLSTMHalfCache) (*HACHalfStack, error) {
	if d == nil || stream == nil {
		return nil, tensor.ErrClosed
	}
	if len(caches) != 7 {
		return nil, fmt.Errorf("%w: seven half FLSTM caches required", ErrGPUWeights)
	}
	for i, c := range caches {
		if c.params != (FLSTMParams{1024, 128}) || len(c.downIH) != 1024*128 || len(c.downHH) != 1024*128 || len(c.up) != 256*4096 || len(c.bias) != 4096 {
			return nil, fmt.Errorf("%w: half cache %d geometry", ErrGPUWeights, i)
		}
	}
	stack := &HACHalfStack{device: d}
	for i, c := range caches {
		layer := &stack.layers[i]
		for _, item := range []struct {
			slot   **tensor.Storage
			shape  []int64
			values []uint16
		}{
			{&layer.downIH, []int64{1024, 128}, c.downIH}, {&layer.downHH, []int64{1024, 128}, c.downHH},
			{&layer.up, []int64{256, 4096}, c.up}, {&layer.bias, []int64{4096}, c.bias},
		} {
			tags := make([]tensor.KoiTag, len(item.shape))
			for j := range tags {
				tags[j] = tensor.TagC
			}
			h, err := d.NewStorage(stream, tensor.DTypeF16, item.shape, tags, HalfBytes(item.values))
			if err != nil {
				return stack, errors.Join(err, stack.Close())
			}
			*item.slot = h
		}
	}
	return stack, nil
}

func (s *HACHalfStack) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	var failures []error
	for i := range s.layers {
		for _, slot := range []**tensor.Storage{&s.layers[i].downIH, &s.layers[i].downHH, &s.layers[i].up, &s.layers[i].bias} {
			if *slot != nil {
				if err := (*slot).Close(); err != nil {
					failures = append(failures, err)
				} else {
					*slot = nil
				}
			}
		}
	}
	return errors.Join(failures...)
}

type HACHalfObservation struct {
	Layer                          int
	InputRow, HiddenRow, OutputRow int64
	Down, Gates, Cell, Output      *tensor.Storage
}

type halfRuntimeOps struct {
	stack                             *HACHalfStack
	stream                            *tensor.Stream
	plan                              HACHalfPlan
	rows                              []*tensor.Storage
	downIH, downHH, down, gates, cell *tensor.Storage
	observe                           func(HACHalfObservation) error
}

func (o *halfRuntimeOps) zeroRow(r int64) error { return o.rows[r].Fill(o.stream, []byte{0, 0}) }
func (o *halfRuntimeOps) zeroCell(int) error    { return o.cell.Fill(o.stream, []byte{0, 0}) }
func (o *halfRuntimeOps) downInput(c halfFLSTMStep) error {
	return o.stack.device.StorageGEMMHalf(o.stream, o.rows[c.input], o.stack.layers[c.layer].downIH, o.downIH)
}
func (o *halfRuntimeOps) downHidden(c halfFLSTMStep) error {
	return o.stack.device.StorageGEMMHalf(o.stream, o.rows[c.hidden], o.stack.layers[c.layer].downHH, o.downHH)
}
func (o *halfRuntimeOps) up(c halfFLSTMStep) error {
	return o.stack.device.StorageGEMMHalf(o.stream, o.down, o.stack.layers[c.layer].up, o.gates)
}
func (o *halfRuntimeOps) step(c halfFLSTMStep) error {
	err := o.stack.device.HalfLSTMStep(o.stream, tensor.HalfLSTMStepOperands{Bias: o.stack.layers[c.layer].bias, Gates: o.gates, Cell: o.cell, Out: o.rows[c.output]}, int32(o.plan.n))
	if err == nil && o.observe != nil {
		err = o.observe(HACHalfObservation{c.layer, c.input, c.hidden, c.output, o.down, o.gates, o.cell, o.rows[c.output]})
	}
	return err
}

// Run returns an independently retained full T+3-row owner. The final reverse
// layer leaves forward-oriented output at rows [2,T+2).
func (s *HACHalfStack) Run(ctx context.Context, stream *tensor.Stream, inout, temp *tensor.Storage, p HACHalfPlan) (*tensor.Storage, error) {
	return s.RunObserved(ctx, stream, inout, temp, p, nil)
}

// RunObserved borrows the completed timestep's buffers to the observer. The
// observer must not mutate, close or retain those borrowed handles.
func (s *HACHalfStack) RunObserved(ctx context.Context, stream *tensor.Stream, inout, temp *tensor.Storage, p HACHalfPlan, observe func(HACHalfObservation) error) (out *tensor.Storage, err error) {
	if s == nil {
		return nil, tensor.ErrClosed
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, tensor.ErrClosed
	}
	if ctx == nil {
		return nil, fmt.Errorf("%w: nil half recurrence context", ErrGPUPlanState)
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	want, err := PlanHACHalfStack(p.n, p.t)
	if err != nil || want != p {
		return nil, errors.Join(err, ErrGPUPlanState)
	}
	metas := make([]tensor.StorageMeta, 2)
	for i, h := range []*tensor.Storage{inout, temp} {
		metas[i], err = h.Meta()
		if err != nil {
			return nil, err
		}
		if metas[i].DType != tensor.DTypeF16 {
			return nil, fmt.Errorf("%w: half recurrence storage dtype", ErrGPUPlanLayout)
		}
		shape := []int64{p.t + 3, p.n, 1024}
		if i == 1 {
			shape = []int64{p.n * (256 + 4096)}
		}
		if !slices.Equal(metas[i].Shape, shape) {
			return nil, fmt.Errorf("%w: half recurrence storage shape", ErrGPUPlanLayout)
		}
		stride := int64(1)
		for j := len(shape) - 1; j >= 0; j-- {
			if metas[i].Strides[j] != stride {
				return nil, fmt.Errorf("%w: half recurrence contiguous owners required", ErrGPUPlanLayout)
			}
			stride *= shape[j]
		}
	}
	if metas[0].Allocation == metas[1].Allocation {
		a0, b0 := metas[0].OffsetElements, metas[1].OffsetElements
		aEnd := a0 + uint64((p.t+3)*p.n*1024)
		bEnd := b0 + uint64(p.n*(256+4096))
		if a0 < bEnd && b0 < aEnd {
			return nil, fmt.Errorf("%w: half recurrence input/temp overlap", ErrGPUPlanOverlap)
		}
	}
	var owned []*tensor.Storage
	defer func() {
		for i := len(owned) - 1; i >= 0; i-- {
			err = errors.Join(err, owned[i].Close())
		}
		if err != nil && out != nil {
			err = errors.Join(err, out.Close())
			out = nil
		}
	}()
	bind := func(parent *tensor.Storage, offset uint64, shape, strides []int64) (*tensor.Storage, error) {
		tags := make([]tensor.KoiTag, len(shape))
		for i := range tags {
			tags[i] = tensor.TagC
		}
		h, e := parent.View(offset, shape, strides, tags)
		if e == nil {
			owned = append(owned, h)
		}
		return h, e
	}
	o := &halfRuntimeOps{stack: s, stream: stream, plan: p, observe: observe}
	for row := int64(0); row < p.t+3; row++ {
		h, e := bind(inout, uint64(row*p.n*1024), []int64{p.n, 1024}, []int64{1024, 1})
		if e != nil {
			return nil, e
		}
		o.rows = append(o.rows, h)
	}
	if o.downIH, err = bind(temp, 0, []int64{p.n, 128}, []int64{256, 1}); err != nil {
		return nil, err
	}
	if o.downHH, err = bind(temp, 128, []int64{p.n, 128}, []int64{256, 1}); err != nil {
		return nil, err
	}
	if o.down, err = bind(temp, 0, []int64{p.n, 256}, []int64{256, 1}); err != nil {
		return nil, err
	}
	if o.gates, err = bind(temp, uint64(p.n*256), []int64{p.n, 4096}, []int64{4096, 1}); err != nil {
		return nil, err
	}
	o.cell, err = s.device.NewStorage(stream, tensor.DTypeF16, []int64{p.n, 1024}, []tensor.KoiTag{tensor.TagN, tensor.TagC}, make([]byte, p.n*1024*2))
	if err != nil {
		return nil, err
	}
	owned = append(owned, o.cell)
	for _, layer := range s.layers {
		for _, call := range []struct{ a, b, out *tensor.Storage }{{o.rows[0], layer.downIH, o.downIH}, {o.rows[0], layer.downHH, o.downHH}, {o.down, layer.up, o.gates}} {
			if err = s.device.ValidateStorageGEMMHalf(stream, call.a, call.b, call.out); err != nil {
				return nil, err
			}
		}
		if err = s.device.ValidateHalfLSTMStep(stream, tensor.HalfLSTMStepOperands{Bias: layer.bias, Gates: o.gates, Cell: o.cell, Out: o.rows[0]}, int32(p.n)); err != nil {
			return nil, err
		}
	}
	if err = runHACHalfStack(ctx, p, o); err != nil {
		return nil, err
	}
	out, err = inout.View(0, []int64{p.t + 3, p.n, 1024}, []int64{p.n * 1024, 1024, 1}, []tensor.KoiTag{tensor.TagT, tensor.TagN, tensor.TagC})
	err = errors.Join(err, ctx.Err())
	return out, err
}
