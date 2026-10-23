// Package tensor owns explicit CPU float32 tensor handles for a separately built ATen backend.
package tensor

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
)

var ErrUnsupported = errors.New("tensor backend requires Linux and CGO_ENABLED=1")
var ErrClosed = errors.New("tensor or engine is closed")

// Engine serializes native calls and owns all of its tensor handles. Close releases them all.
type Engine struct {
	*engineState
}

type engineState struct {
	mu      sync.Mutex
	backend *nativeBackend
	closed  bool
}

// Tensor is an engine-owned handle. Shape and Data return independent copies.
// A tensor remains valid after its input tensors close, including view operations.
type Tensor struct {
	engine *Engine
	id     uint64
	closed bool
}

func Open(libraryPath string) (*Engine, error) {
	if libraryPath == "" || strings.ContainsRune(libraryPath, 0) {
		return nil, errors.New("invalid tensor library path")
	}
	b, err := openNative(libraryPath)
	if err != nil {
		return nil, err
	}
	return &Engine{engineState: &engineState{backend: b}}, nil
}
func (e *Engine) valid() error {
	if e == nil || e.engineState == nil || e.closed || e.backend == nil {
		return ErrClosed
	}
	return nil
}
func (e *Engine) check(t *Tensor) error {
	if t == nil {
		return errors.New("nil tensor")
	}
	if t.engine == nil || t.engine.engineState != e.engineState {
		return errors.New("tensor belongs to a different engine")
	}
	if t.closed {
		return ErrClosed
	}
	return nil
}
func (e *Engine) Close() error {
	if e == nil {
		return nil
	}
	if e.engineState == nil {
		return ErrClosed
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	if e.backend == nil {
		return ErrClosed
	}
	if err := e.backend.close(); err != nil {
		return err
	}
	e.closed = true
	return nil
}
func (t *Tensor) Close() error {
	if t == nil {
		return nil
	}
	if t.engine == nil || t.engine.engineState == nil {
		return ErrClosed
	}
	e := t.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if t.closed || e.closed {
		t.closed = true
		return nil
	}
	if err := e.backend.release(t.id); err != nil {
		return err
	}
	t.closed = true
	return nil
}
func elementCount(shape []int64) (int, error) {
	if len(shape) > 32 {
		return 0, errors.New("tensor rank exceeds 32")
	}
	n := uint64(1)
	for _, d := range shape {
		if d < 0 {
			return 0, errors.New("negative tensor dimension")
		}
		if d != 0 && n > uint64(math.MaxInt)/uint64(d) {
			return 0, errors.New("tensor shape overflows int")
		}
		n *= uint64(d)
	}
	if n > uint64(math.MaxInt)/4 {
		return 0, errors.New("tensor bytes overflow int")
	}
	return int(n), nil
}
func (e *Engine) NewFloat32(shape []int64, data []float32) (*Tensor, error) {
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
	if n != len(data) {
		return nil, errors.New("shape and data length differ")
	}
	id, err := e.backend.newTensor(shape, data)
	if err != nil {
		return nil, err
	}
	return &Tensor{engine: e, id: id}, nil
}

// LoadWeight reads Dorado's Torch vector archive and requires exactly one CPU float32 tensor.
func (e *Engine) LoadWeight(path string) (*Tensor, error) {
	if e == nil || e.engineState == nil {
		return nil, ErrClosed
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.valid(); err != nil {
		return nil, err
	}
	if path == "" || strings.ContainsRune(path, 0) {
		return nil, errors.New("invalid weight path")
	}
	id, err := e.backend.load(path)
	if err != nil {
		return nil, err
	}
	return &Tensor{engine: e, id: id}, nil
}
func (t *Tensor) Shape() ([]int64, error) {
	if t == nil || t.engine == nil || t.engine.engineState == nil {
		return nil, ErrClosed
	}
	e := t.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.valid(); err != nil {
		return nil, err
	}
	if err := e.check(t); err != nil {
		return nil, err
	}
	return e.backend.shape(t.id)
}
func (t *Tensor) Data() ([]float32, error) {
	if t == nil || t.engine == nil || t.engine.engineState == nil {
		return nil, ErrClosed
	}
	e := t.engine
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.valid(); err != nil {
		return nil, err
	}
	if err := e.check(t); err != nil {
		return nil, err
	}
	shape, err := e.backend.shape(t.id)
	if err != nil {
		return nil, err
	}
	n, err := elementCount(shape)
	if err != nil {
		return nil, err
	}
	return e.backend.data(t.id, n)
}
func (e *Engine) run(op int, ts []*Tensor, p []int64, f []float64) (*Tensor, error) {
	if e == nil || e.engineState == nil {
		return nil, ErrClosed
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.valid(); err != nil {
		return nil, err
	}
	ids := make([]uint64, len(ts))
	for i, t := range ts {
		if err := e.check(t); err != nil {
			return nil, fmt.Errorf("input %d: %w", i, err)
		}
		ids[i] = t.id
	}
	id, err := e.backend.run(op, ids, p, f)
	if err != nil {
		return nil, err
	}
	return &Tensor{engine: e, id: id}, nil
}
func (e *Engine) Add(a, b *Tensor) (*Tensor, error)    { return e.run(1, []*Tensor{a, b}, nil, nil) }
func (e *Engine) Mul(a, b *Tensor) (*Tensor, error)    { return e.run(2, []*Tensor{a, b}, nil, nil) }
func (e *Engine) MatMul(a, b *Tensor) (*Tensor, error) { return e.run(3, []*Tensor{a, b}, nil, nil) }
func (e *Engine) Clamp(a *Tensor, lo, hi float64) (*Tensor, error) {
	return e.run(4, []*Tensor{a}, nil, []float64{lo, hi})
}
func (e *Engine) Tanh(a *Tensor) (*Tensor, error)    { return e.run(5, []*Tensor{a}, nil, nil) }
func (e *Engine) Sigmoid(a *Tensor) (*Tensor, error) { return e.run(6, []*Tensor{a}, nil, nil) }
func (e *Engine) SiLU(a *Tensor) (*Tensor, error)    { return e.run(7, []*Tensor{a}, nil, nil) }
func (e *Engine) Transpose(a *Tensor, d0, d1 int64) (*Tensor, error) {
	return e.run(8, []*Tensor{a}, []int64{d0, d1}, nil)
}
func (e *Engine) Flip(a *Tensor, dims []int64) (*Tensor, error) {
	return e.run(9, []*Tensor{a}, dims, nil)
}
func (e *Engine) Reshape(a *Tensor, shape []int64) (*Tensor, error) {
	return e.run(10, []*Tensor{a}, shape, nil)
}
func (e *Engine) Narrow(a *Tensor, dim, start, length int64) (*Tensor, error) {
	return e.run(11, []*Tensor{a}, []int64{dim, start, length}, nil)
}
func (e *Engine) Concat(ts []*Tensor, dim int64) (*Tensor, error) {
	return e.run(12, ts, []int64{dim}, nil)
}
func (e *Engine) Stack(ts []*Tensor, dim int64) (*Tensor, error) {
	return e.run(13, ts, []int64{dim}, nil)
}
func (e *Engine) Conv1d(x, w, b *Tensor, stride, padding, dilation, groups int64) (*Tensor, error) {
	ts := []*Tensor{x, w}
	if b != nil {
		ts = append(ts, b)
	}
	return e.run(14, ts, []int64{stride, padding, dilation, groups}, nil)
}
func (e *Engine) Linear(x, w, b *Tensor) (*Tensor, error) {
	ts := []*Tensor{x, w}
	if b != nil {
		ts = append(ts, b)
	}
	return e.run(15, ts, nil, nil)
}
func (e *Engine) AddScalar(a *Tensor, b float64) (*Tensor, error) {
	return e.run(16, []*Tensor{a}, nil, []float64{b})
}
func (e *Engine) MulScalar(a *Tensor, b float64) (*Tensor, error) {
	return e.run(17, []*Tensor{a}, nil, []float64{b})
}
func (e *Engine) Permute(a *Tensor, dims []int64) (*Tensor, error) {
	return e.run(18, []*Tensor{a}, dims, nil)
}
func (e *Engine) Contiguous(a *Tensor) (*Tensor, error) { return e.run(19, []*Tensor{a}, nil, nil) }
func (e *Engine) Zeros(shape []int64) (*Tensor, error) {
	if _, err := elementCount(shape); err != nil {
		return nil, err
	}
	return e.run(20, nil, shape, nil)
}
