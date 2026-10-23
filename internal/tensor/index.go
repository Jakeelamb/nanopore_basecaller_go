package tensor

import (
	"errors"
	"math"
)

// IndexTensor holds CPU int64 indices. Float arithmetic cannot accept this type.
type IndexTensor struct{ handle *Tensor }

func (e *Engine) NewInt64(shape, data []int64) (*IndexTensor, error) {
	if e == nil || e.engineState == nil {
		return nil, ErrClosed
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.valid(); err != nil {
		return nil, err
	}
	n, err := elementCount(shape)
	if err != nil {
		return nil, err
	}
	if n > math.MaxInt/8 {
		return nil, errors.New("index bytes overflow int")
	}
	if n != len(data) {
		return nil, errors.New("shape and index data length differ")
	}
	id, err := e.backend.newInt64(shape, data)
	if err != nil {
		return nil, err
	}
	return &IndexTensor{handle: &Tensor{engine: e, id: id}}, nil
}
func (x *IndexTensor) Close() error {
	if x == nil {
		return nil
	}
	return x.handle.Close()
}
func (x *IndexTensor) Shape() ([]int64, error) {
	if x == nil {
		return nil, ErrClosed
	}
	return x.handle.Shape()
}
func (x *IndexTensor) Data() ([]int64, error) {
	if x == nil || x.handle == nil || x.handle.engine == nil || x.handle.engine.engineState == nil {
		return nil, ErrClosed
	}
	e := x.handle.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.valid(); err != nil {
		return nil, err
	}
	if err := e.check(x.handle); err != nil {
		return nil, err
	}
	shape, err := e.backend.shape(x.handle.id)
	if err != nil {
		return nil, err
	}
	n, err := elementCount(shape)
	if err != nil {
		return nil, err
	}
	if n > math.MaxInt/8 {
		return nil, errors.New("index bytes overflow int")
	}
	return e.backend.dataInt64(x.handle.id, n)
}
func (e *Engine) IndexSelect(x *Tensor, dim int64, indices *IndexTensor) (*Tensor, error) {
	if indices == nil {
		return nil, errors.New("nil index tensor")
	}
	return e.run(21, []*Tensor{x, indices.handle}, []int64{dim}, nil)
}
func (e *Engine) LogSumExp(x *Tensor, dim int64, keepdim bool) (*Tensor, error) {
	keep := int64(0)
	if keepdim {
		keep = 1
	}
	return e.run(22, []*Tensor{x}, []int64{dim, keep}, nil)
}
func (e *Engine) Softmax(x *Tensor, dim int64) (*Tensor, error) {
	return e.run(23, []*Tensor{x}, []int64{dim}, nil)
}
func (e *Engine) Unsqueeze(x *Tensor, dim int64) (*Tensor, error) {
	return e.run(24, []*Tensor{x}, []int64{dim}, nil)
}

// ArgSort preserves ATen's ascending, unstable default and its ordering of ties.
func (e *Engine) ArgSort(indices *IndexTensor, dim int64) (*IndexTensor, error) {
	if indices == nil {
		return nil, errors.New("nil index tensor")
	}
	x, err := e.run(25, []*Tensor{indices.handle}, []int64{dim}, nil)
	if err != nil {
		return nil, err
	}
	return &IndexTensor{handle: x}, nil
}
