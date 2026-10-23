package tensor

import (
	"errors"
	"fmt"
)

func (e *Engine) Mean(x *Tensor, dim int64, keepDim bool) (*Tensor, error) {
	keep := int64(0)
	if keepDim {
		keep = 1
	}
	return e.run(26, []*Tensor{x}, []int64{dim, keep}, nil)
}
func (e *Engine) Rsqrt(x *Tensor) (*Tensor, error) {
	return e.run(27, []*Tensor{x}, nil, nil)
}
func (e *Engine) Reciprocal(x *Tensor) (*Tensor, error) {
	return e.run(28, []*Tensor{x}, nil, nil)
}
func (e *Engine) Sin(x *Tensor) (*Tensor, error) {
	return e.run(29, []*Tensor{x}, nil, nil)
}
func (e *Engine) Cos(x *Tensor) (*Tensor, error) {
	return e.run(30, []*Tensor{x}, nil, nil)
}
func (e *Engine) Sub(a, b *Tensor) (*Tensor, error) {
	return e.run(31, []*Tensor{a, b}, nil, nil)
}
func (e *Engine) Square(x *Tensor) (*Tensor, error) {
	return e.run(32, []*Tensor{x}, nil, nil)
}

// ScaledDotProductAttention copies a bool mask whose 1 bytes allow attention.
// Nil maskShape and mask omit the mask. ATen validates broadcasting and tensor dimensions.
// Dropout is zero, causal is false, scale is unset, and grouped-query attention is false.
func (e *Engine) ScaledDotProductAttention(q, k, v *Tensor, maskShape []int64, mask []byte) (*Tensor, error) {
	if e == nil || e.engineState == nil {
		return nil, ErrClosed
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.valid(); err != nil {
		return nil, err
	}
	for i, t := range []*Tensor{q, k, v} {
		if err := e.check(t); err != nil {
			return nil, fmt.Errorf("input %d: %w", i, err)
		}
	}
	if maskShape != nil || mask != nil {
		n, err := elementCount(maskShape)
		if err != nil {
			return nil, err
		}
		if n != len(mask) {
			return nil, errors.New("mask shape and data length differ")
		}
		for _, b := range mask {
			if b > 1 {
				return nil, errors.New("mask bytes must be 0 or 1")
			}
		}
	}
	id, err := e.backend.sdpa(q.id, k.id, v.id, maskShape, mask)
	if err != nil {
		return nil, err
	}
	return &Tensor{engine: e, id: id}, nil
}
