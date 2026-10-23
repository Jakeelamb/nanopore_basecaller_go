package model

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

type modelGraph interface {
	forward(context.Context, *tensor.Engine, *tensor.Tensor, trace) (*tensor.Tensor, error)
}

type rotaryEmbedding struct {
	dim, maxSequence int64
	cos, sin         *tensor.Tensor
}

type txEncoderLayer struct {
	qkv, outWeight, outBias, fc1, fc2, norm1, norm2 *tensor.Tensor
	channels, heads, hidden                         int64
	upper, lower                                    int64
}

type supGraph struct {
	convs                             []convLayer
	encoders                          []txEncoderLayer
	rotary                            rotaryEmbedding
	alpha                             *tensor.Tensor
	upsampleWeight, upsampleBias, crf *tensor.Tensor
}

func bindSUP(weights []*tensor.Tensor, rotary rotaryEmbedding, alpha *tensor.Tensor) *supGraph {
	g := &supGraph{rotary: rotary, alpha: alpha}
	for i, p := range [][2]int64{{1, 2}, {1, 2}, {3, 4}, {2, 4}, {2, 2}} {
		g.convs = append(g.convs, convLayer{weights[2*i], weights[2*i+1], p[0], p[1], silu})
	}
	for i := 0; i < 18; i++ {
		w := weights[10+7*i : 17+7*i]
		g.encoders = append(g.encoders, txEncoderLayer{w[0], w[1], w[2], w[3], w[4], w[5], w[6], 512, 8, 2048, 127, 128})
	}
	g.upsampleWeight, g.upsampleBias, g.crf = weights[136], weights[137], weights[138]
	return g
}

func newRotary(e *tensor.Engine, dim, maxSequence int64, theta float32) (rotaryEmbedding, error) {
	if dim < 2 || dim%2 != 0 || maxSequence < 1 {
		return rotaryEmbedding{}, errors.New("invalid rotary dimensions")
	}
	s := newScope()
	powers := make([]float32, dim/2)
	for j := range powers {
		powers[j] = float32(math.Pow(float64(theta), float64(float32(2*j)/float32(dim))))
	}
	p := s.own(e.NewFloat32([]int64{dim / 2}, powers))
	inv := s.own(e.Reciprocal(p))
	positions := make([]float32, maxSequence)
	for i := range positions {
		positions[i] = float32(i)
	}
	t := s.own(e.NewFloat32([]int64{maxSequence, 1, 1, 1}, positions))
	freq := s.own(e.Mul(t, inv))
	cos := s.own(e.Cos(freq))
	sin := s.own(e.Sin(freq))
	if s.err == nil {
		s.detach(cos)
		s.detach(sin)
	}
	_, err := s.finish(nil)
	if err != nil {
		_ = cos.Close()
		_ = sin.Close()
		return rotaryEmbedding{}, err
	}
	return rotaryEmbedding{dim, maxSequence, cos, sin}, nil
}

func (r rotaryEmbedding) forward(e *tensor.Engine, qkv *tensor.Tensor) (*tensor.Tensor, error) {
	shape, err := qkv.Shape()
	if err != nil {
		return nil, err
	}
	if len(shape) != 5 || shape[1] < 1 || shape[1] > r.maxSequence || shape[2] != 3 || shape[4] != r.dim {
		return nil, fmt.Errorf("invalid rotary shape %v", shape)
	}
	s := newScope()
	cos := s.own(e.Narrow(r.cos, 0, 0, shape[1]))
	sin := s.own(e.Narrow(r.sin, 0, 0, shape[1]))
	qk := s.own(e.Narrow(qkv, 2, 0, 2))
	a := s.own(e.Narrow(qk, 4, 0, r.dim/2))
	b := s.own(e.Narrow(qk, 4, r.dim/2, r.dim/2))
	ca := s.own(e.Mul(cos, a))
	sb := s.own(e.Mul(sin, b))
	first := s.own(e.Sub(ca, sb))
	sa := s.own(e.Mul(sin, a))
	cb := s.own(e.Mul(cos, b))
	second := s.own(e.Add(sa, cb))
	rotated := s.own(e.Concat([]*tensor.Tensor{first, second}, 4))
	v := s.own(e.Narrow(qkv, 2, 2, 1))
	nt3hd := s.own(e.Concat([]*tensor.Tensor{rotated, v}, 2))
	layout := s.own(e.Permute(nt3hd, []int64{2, 0, 3, 1, 4}))
	out := s.own(e.Contiguous(layout))
	return s.finish(out)
}

type attentionSplit struct{ qb, qe, kvb, kve int64 }

func attentionSplits(time, upper, lower int64) []attentionSplit {
	elements := ((time+11)/12 + 3) / 4 * 4
	var splits []attentionSplit
	for i := int64(0); i < 12; i++ {
		qb := i * elements
		if qb >= time {
			break
		}
		qe := min(time, qb+elements)
		splits = append(splits, attentionSplit{qb, qe, max(int64(0), qb-lower), min(time, qe+upper)})
	}
	return splits
}

func splitMask(split attentionSplit, upper, lower int64) []byte {
	mask := make([]byte, (split.qe-split.qb)*(split.kve-split.kvb))
	for q := split.qb; q < split.qe; q++ {
		for k := split.kvb; k < split.kve; k++ {
			if k-q >= -upper && k-q <= lower {
				mask[(q-split.qb)*(split.kve-split.kvb)+k-split.kvb] = 1
			}
		}
	}
	return mask
}

func splitAttention(ctx context.Context, e *tensor.Engine, qkv *tensor.Tensor, upper, lower int64) (*tensor.Tensor, error) {
	shape, err := qkv.Shape()
	if err != nil {
		return nil, err
	}
	if len(shape) != 5 || shape[0] != 3 || shape[3] < 1 {
		return nil, fmt.Errorf("invalid attention shape %v", shape)
	}
	s := newScope()
	var components []*tensor.Tensor
	for _, split := range attentionSplits(shape[3], upper, lower) {
		if err := ctx.Err(); err != nil {
			s.err = err
			return s.finish(nil)
		}
		part := newScope()
		selectPart := func(index, start, end int64) *tensor.Tensor {
			one := part.own(e.Narrow(qkv, 0, index, 1))
			nt := part.own(e.Reshape(one, shape[1:]))
			return part.own(e.Narrow(nt, 2, start, end-start))
		}
		q := selectPart(0, split.qb, split.qe)
		k := selectPart(1, split.kvb, split.kve)
		v := selectPart(2, split.kvb, split.kve)
		out := part.own(e.ScaledDotProductAttention(q, k, v, []int64{split.qe - split.qb, split.kve - split.kvb}, splitMask(split, upper, lower)))
		out, err = part.finish(out)
		if err != nil {
			s.err = err
			return s.finish(nil)
		}
		components = append(components, s.own(out, nil))
	}
	all := s.own(e.Concat(components, 2))
	ntHD := s.own(e.Transpose(all, 1, 2))
	contiguous := s.own(e.Contiguous(ntHD))
	ntc := s.own(e.Reshape(contiguous, []int64{shape[1], shape[3], shape[2] * shape[4]}))
	return s.finish(ntc)
}

func rmsNorm(e *tensor.Engine, x, weight *tensor.Tensor) (*tensor.Tensor, error) {
	s := newScope()
	squared := s.own(e.Square(x))
	mean := s.own(e.Mean(squared, -1, true))
	eps := s.own(e.AddScalar(mean, float64(float32(1e-5))))
	rstd := s.own(e.Rsqrt(eps))
	normalized := s.own(e.Mul(x, rstd))
	out := s.own(e.Mul(normalized, weight))
	return s.finish(out)
}

func gatedMLP(e *tensor.Engine, x, fc1, fc2 *tensor.Tensor, hidden int64, observe trace) (*tensor.Tensor, error) {
	s := newScope()
	t := s.own(e.Linear(x, fc1, nil))
	if observe != nil && s.err == nil {
		s.err = observe("fc1", t)
	}
	y := s.own(e.Narrow(t, -1, 0, hidden))
	gate := s.own(e.Narrow(t, -1, hidden, hidden))
	activated := s.own(e.SiLU(gate))
	gated := s.own(e.Mul(activated, y))
	if observe != nil && s.err == nil {
		s.err = observe("gate", gated)
	}
	out := s.own(e.Linear(gated, fc2, nil))
	return s.finish(out)
}

func (l txEncoderLayer) forward(ctx context.Context, e *tensor.Engine, x *tensor.Tensor, rotary rotaryEmbedding, alpha *tensor.Tensor, observe trace) (*tensor.Tensor, error) {
	shape, err := x.Shape()
	if err != nil {
		return nil, err
	}
	if len(shape) != 3 || shape[2] != l.channels {
		return nil, fmt.Errorf("invalid transformer shape %v", shape)
	}
	s := newScope()
	emit := func(name string, t *tensor.Tensor) {
		if observe != nil && s.err == nil {
			s.err = observe(name, t)
		}
	}
	projected := s.own(e.Linear(x, l.qkv, nil))
	qkv := s.own(e.Reshape(projected, []int64{shape[0], shape[1], 3, l.heads, l.channels / l.heads}))
	emit("qkv", qkv)
	rotated := s.own(rotary.forward(e, qkv))
	emit("rope", rotated)
	s.release(projected, qkv)
	attn := s.own(splitAttention(ctx, e, rotated, l.upper, l.lower))
	emit("attention", attn)
	s.release(rotated)
	out := s.own(linearNTCBias(e, attn, l.outWeight, l.outBias))
	emit("projection", out)
	s.release(attn)
	scaled := s.own(e.Mul(x, alpha))
	residual := s.own(e.Add(out, scaled))
	norm1 := s.own(rmsNorm(e, residual, l.norm1))
	emit("norm1", norm1)
	s.release(out, scaled, residual)
	if s.err != nil {
		return s.finish(nil)
	}
	if err := ctx.Err(); err != nil {
		s.err = err
		return s.finish(nil)
	}
	ff := s.own(gatedMLP(e, norm1, l.fc1, l.fc2, l.hidden, observe))
	emit("ff", ff)
	scaled = s.own(e.Mul(norm1, alpha))
	residual = s.own(e.Add(ff, scaled))
	norm2 := s.own(rmsNorm(e, residual, l.norm2))
	emit("norm2", norm2)
	return s.finish(norm2)
}

// The shipped C++ nn::functional::linear uses matmul then bias for 3D input.
// ATen linear may fuse that addition and changes the F32 result.
func linearNTCBias(e *tensor.Engine, x, weight, bias *tensor.Tensor) (*tensor.Tensor, error) {
	s := newScope()
	transposed := s.own(e.Transpose(weight, 0, 1))
	product := s.own(e.MatMul(x, transposed))
	out := s.own(e.Add(product, bias))
	return s.finish(out)
}

func linearUpsample(e *tensor.Engine, x, weight, bias *tensor.Tensor, scale int64) (*tensor.Tensor, error) {
	shape, err := x.Shape()
	if err != nil {
		return nil, err
	}
	if len(shape) != 3 {
		return nil, errors.New("invalid upsample input")
	}
	s := newScope()
	linear := s.own(linearNTCBias(e, x, weight, bias))
	out := s.own(e.Reshape(linear, []int64{shape[0], scale * shape[1], shape[2]}))
	return s.finish(out)
}

func (g *supGraph) forward(ctx context.Context, e *tensor.Engine, x *tensor.Tensor, observe trace) (*tensor.Tensor, error) {
	s := newScope()
	emit := func(name string, t *tensor.Tensor) {
		if observe != nil && s.err == nil {
			s.err = observe(name, t)
		}
	}
	for i, c := range g.convs {
		if err := ctx.Err(); err != nil {
			s.err = err
			return s.finish(nil)
		}
		next := s.own(c.forward(e, x))
		s.release(x)
		x = next
		emit(fmt.Sprintf("conv%d", i), x)
		if s.err != nil {
			return s.finish(nil)
		}
	}
	next := s.own(e.Transpose(x, 1, 2))
	s.release(x)
	x = next
	emit("conv_ntc", x)
	for i, l := range g.encoders {
		if err := ctx.Err(); err != nil {
			s.err = err
			return s.finish(nil)
		}
		var observer trace
		if observe != nil {
			observer = func(name string, t *tensor.Tensor) error { return observe(fmt.Sprintf("encoder%d.%s", i, name), t) }
		}
		next = s.own(l.forward(ctx, e, x, g.rotary, g.alpha, observer))
		s.release(x)
		x = next
		if s.err != nil {
			return s.finish(nil)
		}
	}
	next = s.own(linearUpsample(e, x, g.upsampleWeight, g.upsampleBias, 2))
	s.release(x)
	x = next
	emit("upsample", x)
	next = s.own(e.Linear(x, g.crf, nil))
	s.release(x)
	x = next
	emit("scores", x)
	return s.finish(x)
}
