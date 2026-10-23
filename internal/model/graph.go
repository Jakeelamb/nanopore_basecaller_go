package model

import (
	"context"
	"errors"
	"fmt"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

type activation uint8

const (
	silu activation = iota
	tanh
)

type convLayer struct {
	weight, bias    *tensor.Tensor
	stride, padding int64
	activation      activation
}
type flstmLayer struct {
	dnIH, dnHH, upIH, upHH, biasIH, biasHH *tensor.Tensor
	channels                               int64
}
type hacGraph struct {
	convs           []convLayer
	rnns            []flstmLayer
	bottleneck, crf *tensor.Tensor
	firstReverse    bool
}

func bindHAC(weights []*tensor.Tensor) *hacGraph {
	g := &hacGraph{firstReverse: true}
	for i := 0; i < 3; i++ {
		s, p, a := int64(1), int64(2), silu
		if i == 2 {
			s, p, a = 6, 9, tanh
		}
		g.convs = append(g.convs, convLayer{weights[2*i], weights[2*i+1], s, p, a})
	}
	for i := 0; i < 7; i++ {
		w := weights[6+6*i : 12+6*i]
		g.rnns = append(g.rnns, flstmLayer{w[0], w[1], w[2], w[3], w[4], w[5], 1024})
	}
	g.bottleneck, g.crf = weights[48], weights[49]
	return g
}

// scope owns only temporary handles, never borrowed weights or caller input.
// Recurrent step scopes are closed per step, bounding live gate intermediates.
type scope struct {
	err   error
	owned map[*tensor.Tensor]struct{}
}

func newScope() *scope { return &scope{owned: make(map[*tensor.Tensor]struct{})} }
func (s *scope) own(t *tensor.Tensor, err error) *tensor.Tensor {
	if err != nil {
		s.err = errors.Join(s.err, err)
	}
	if t != nil {
		s.owned[t] = struct{}{}
	}
	return t
}
func (s *scope) release(ts ...*tensor.Tensor) {
	for _, t := range ts {
		if _, ok := s.owned[t]; ok {
			s.err = errors.Join(s.err, t.Close())
			delete(s.owned, t)
		}
	}
}
func (s *scope) detach(t *tensor.Tensor) *tensor.Tensor { delete(s.owned, t); return t }
func (s *scope) finish(out *tensor.Tensor) (*tensor.Tensor, error) {
	if s.err == nil {
		s.detach(out)
	}
	for t := range s.owned {
		s.release(t)
	}
	if s.err != nil {
		if out != nil {
			_ = out.Close()
		}
		return nil, s.err
	}
	return out, nil
}

func (c convLayer) forward(e *tensor.Engine, x *tensor.Tensor) (*tensor.Tensor, error) {
	s := newScope()
	z := s.own(e.Conv1d(x, c.weight, c.bias, c.stride, c.padding, 1, 1))
	var y *tensor.Tensor
	if c.activation == tanh {
		y = s.own(e.Tanh(z))
	} else {
		y = s.own(e.SiLU(z))
	}
	return s.finish(y)
}

// trace is private verification instrumentation. Production Forward passes nil.
type trace func(string, *tensor.Tensor) error

func (g *hacGraph) forward(ctx context.Context, e *tensor.Engine, x *tensor.Tensor, observe trace) (*tensor.Tensor, error) {
	s := newScope()
	emit := func(name string, x *tensor.Tensor) {
		if observe != nil && s.err == nil {
			s.err = observe(name, x)
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
	next = s.own(flstmStackForward(ctx, e, g.rnns, g.firstReverse, x, observe))
	s.release(x)
	x = next
	if s.err != nil {
		return s.finish(nil)
	}
	next = s.own(e.Linear(x, g.bottleneck, nil))
	s.release(x)
	x = next
	emit("linear1", x)
	next = s.own(e.Linear(x, g.crf, nil))
	s.release(x)
	x = next
	emit("linear2", x)
	next = s.own(e.Tanh(x))
	s.release(x)
	x = next
	next = s.own(e.MulScalar(x, 5))
	s.release(x)
	x = next
	emit("scores", x)
	return s.finish(x)
}

func flstmStackForward(ctx context.Context, e *tensor.Engine, layers []flstmLayer, firstReverse bool, x *tensor.Tensor, observe trace) (*tensor.Tensor, error) {
	s := newScope()
	var next *tensor.Tensor
	emit := func(name string, x *tensor.Tensor) {
		if observe != nil && s.err == nil {
			s.err = observe(name, x)
		}
	}
	reversed := !firstReverse
	for i, l := range layers {
		if err := ctx.Err(); err != nil {
			s.err = err
			return s.finish(nil)
		}
		if i > 0 || firstReverse {
			next = s.own(e.Flip(x, []int64{1}))
			s.release(x)
			x = next
			reversed = !reversed
		}
		next = s.own(l.forward(ctx, e, x))
		s.release(x)
		x = next
		emit(fmt.Sprintf("rnn%d", i), x)
		if s.err != nil {
			return s.finish(nil)
		}
	}
	if reversed {
		next = s.own(e.Flip(x, []int64{1}))
		s.release(x)
		x = next
	}
	emit("rnns", x)
	return s.finish(x)
}

func (l flstmLayer) forward(ctx context.Context, e *tensor.Engine, x *tensor.Tensor) (*tensor.Tensor, error) {
	s := newScope()
	shape, err := x.Shape()
	if err != nil {
		return nil, err
	}
	if len(shape) != 3 || shape[1] < 1 || shape[2] != l.channels {
		return nil, errors.New("invalid FLSTM shape")
	}
	T, N, C := shape[1], shape[0], l.channels
	tnc := s.own(e.Transpose(x, 0, 1))
	xc := s.own(e.Contiguous(tnc))
	s.release(tnc)
	dni := s.own(e.Transpose(l.dnIH, 0, 1))
	upi := s.own(e.Transpose(l.upIH, 0, 1))
	dnh := s.own(e.Transpose(l.dnHH, 0, 1))
	uph := s.own(e.Transpose(l.upHH, 0, 1))
	p := s.own(e.MatMul(xc, dni))
	q := s.own(e.MatMul(p, upi))
	ih := s.own(e.Add(q, l.biasIH))
	s.release(p, q, xc, dni, upi)
	h := s.own(e.Zeros([]int64{N, C}))
	c := s.own(e.Zeros([]int64{N, C}))
	hs := make([]*tensor.Tensor, 0, T)
	for t := int64(0); t < T; t++ {
		if err := ctx.Err(); err != nil {
			s.err = err
			return s.finish(nil)
		}
		step := newScope()
		p := step.own(e.MatMul(h, dnh))
		q := step.own(e.MatMul(p, uph))
		z := step.own(e.Add(q, l.biasHH))
		it := step.own(e.Narrow(ih, 0, t, 1))
		it2 := step.own(e.Reshape(it, []int64{N, 4 * C}))
		gates := step.own(e.Add(z, it2))
		hardSigmoid := func(offset int64) *tensor.Tensor {
			a := step.own(e.Narrow(gates, 1, offset, C))
			b := step.own(e.MulScalar(a, float64(float32(.2))))
			d := step.own(e.AddScalar(b, .5))
			return step.own(e.Clamp(d, 0, 1))
		}
		i, f := hardSigmoid(0), hardSigmoid(C)
		gc := step.own(e.Narrow(gates, 1, 2*C, C))
		gg := step.own(e.Clamp(gc, -1, 1))
		o := hardSigmoid(3 * C)
		fc := step.own(e.Mul(f, c))
		ig := step.own(e.Mul(i, gg))
		cn := step.own(e.Add(fc, ig))
		ct := step.own(e.Tanh(cn))
		hn := step.own(e.Mul(o, ct))
		if step.err != nil {
			_, er := step.finish(nil)
			s.err = er
			return s.finish(nil)
		}
		step.detach(cn)
		step.detach(hn)
		_, er := step.finish(nil)
		if er != nil {
			s.err = er
			_ = cn.Close()
			_ = hn.Close()
			return s.finish(nil)
		}
		s.release(c)
		if t == 0 {
			s.release(h)
		}
		h, c = s.own(hn, nil), s.own(cn, nil)
		hs = append(hs, h)
	}
	outTNC := s.own(e.Stack(hs, 0))
	outNTC := s.own(e.Transpose(outTNC, 0, 1))
	out := s.own(e.Contiguous(outNTC))
	return s.finish(out)
}
