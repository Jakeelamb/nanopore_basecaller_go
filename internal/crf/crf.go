// Package crf ports Dorado's CPU CRF forward and backward scan schedules to Go.
package crf

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/decode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

const maxElements = 1 << 27

// Result owns contiguous row-major arrays. Scores is [Blocks,4*States].
// Forward, Backward and Posts are [Blocks+1,States].
type Result struct {
	Blocks, States                   int
	Scores, Forward, Backward, Posts []float32
}

// DecodeInput borrows Result storage. The caller must preserve it during decoding.
func (r Result) DecodeInput() decode.Input {
	return decode.Input{Blocks: r.Blocks, States: r.States, Scores: r.Scores, BackGuides: r.Backward, Posts: r.Posts}
}

func dimensions(shape []int64) (blocks, states int, err error) {
	if len(shape) != 3 || shape[0] < 1 || shape[1] != 1 || shape[2]%4 != 0 {
		return 0, 0, errors.New("scores must have shape [T,1,4*States] with T positive")
	}
	s := shape[2] / 4
	if s < 4 || s > 16384 {
		return 0, 0, errors.New("states must be a power of four in [4,16384]")
	}
	for n := s; n > 1; n /= 4 {
		if n%4 != 0 {
			return 0, 0, errors.New("states must be a power of four")
		}
	}
	// Scores plus all three result arrays share one allocation bound.
	if shape[0] > (maxElements-3*s)/(7*s) {
		return 0, 0, fmt.Errorf("CRF result exceeds %d float32 elements", maxElements)
	}
	return int(shape[0]), int(s), nil
}
func finite(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }
func validateValues(scores []float32, blocks int, blank float32) error {
	// Keep accumulated guides within the CPU beam decoder's stricter input bound.
	bound := float32(math.MaxFloat32 / (64 * float64(blocks+1) * float64(blocks+1)))
	if !finite(blank) || math.Abs(float64(blank)) > float64(bound) {
		return errors.New("blank score is nonfinite or exceeds safe accumulation range")
	}
	for i, v := range scores {
		if !finite(v) || math.Abs(float64(v)) > float64(bound) {
			return fmt.Errorf("score %d is nonfinite or exceeds safe accumulation range", i)
		}
	}
	return nil
}

type closer interface{ Close() error }
type scope struct {
	owned []closer
	err   error
}

func (s *scope) own(x *tensor.Tensor, err error) *tensor.Tensor {
	if err != nil && s.err == nil {
		s.err = err
	}
	if x != nil {
		s.owned = append(s.owned, x)
	}
	return x
}
func (s *scope) index(x *tensor.IndexTensor, err error) *tensor.IndexTensor {
	if err != nil && s.err == nil {
		s.err = err
	}
	if x != nil {
		s.owned = append(s.owned, x)
	}
	return x
}
func (s *scope) close() error {
	err := s.err
	for i := len(s.owned) - 1; i >= 0; i-- {
		err = errors.Join(err, s.owned[i].Close())
	}
	s.owned = nil
	return err
}
func (s *scope) finish(out *tensor.Tensor) (*tensor.Tensor, error) {
	if s.err == nil && out != nil {
		for i, x := range s.owned {
			if x == out {
				s.owned = append(s.owned[:i], s.owned[i+1:]...)
				break
			}
		}
	}
	if err := s.close(); err != nil {
		if out != nil {
			err = errors.Join(err, out.Close())
		}
		return nil, err
	}
	return out, nil
}

// Compute borrows scoresTNC and supports exactly one batch. Every loop and index
// construction belongs to Go; native calls perform individual tensor operations.
func Compute(ctx context.Context, e *tensor.Engine, scoresTNC *tensor.Tensor, blank float32) (result Result, ret error) {
	if ctx == nil {
		return Result{}, errors.New("nil context")
	}
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	shape, err := scoresTNC.Shape()
	if err != nil {
		return Result{}, err
	}
	blocks, states, err := dimensions(shape)
	if err != nil {
		return Result{}, err
	}
	s := &scope{}
	defer func() {
		ret = errors.Join(ret, s.close())
		if ret != nil {
			result = Result{}
		}
	}()
	// Contiguous also checks that this engine owns the borrowed tensor.
	scores := s.own(e.Contiguous(scoresTNC))
	if s.err != nil {
		return Result{}, s.err
	}
	values, err := scores.Data()
	if err != nil {
		return Result{}, err
	}
	if err := validateValues(values, blocks, blank); err != nil {
		return Result{}, err
	}
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	result = Result{Blocks: blocks, States: states, Scores: values}
	idx := make([]int64, 4*states)
	for state := 0; state < states; state++ {
		for base := 0; base < 4; base++ {
			idx[4*state+base] = int64(state/4 + base*(states/4))
		}
	}
	forwardIdx := s.index(e.NewInt64([]int64{int64(len(idx))}, idx))
	if s.err != nil {
		return Result{}, s.err
	}
	result.Forward, err = scan(ctx, e, scores, forwardIdx, nil, blocks, states, blank, false)
	if err != nil {
		return Result{}, err
	}
	order := s.index(e.ArgSort(forwardIdx, 0))
	if s.err != nil {
		return Result{}, s.err
	}
	orderData, err := order.Data()
	if err != nil {
		return Result{}, err
	}
	predecessor := make([]int64, len(orderData))
	for i, value := range orderData {
		predecessor[i] = value >> 2
	}
	backwardIdx := s.index(e.NewInt64([]int64{int64(len(predecessor))}, predecessor))
	if s.err != nil {
		return Result{}, s.err
	}
	result.Backward, err = scan(ctx, e, scores, backwardIdx, order, blocks, states, blank, true)
	if err != nil {
		return Result{}, err
	}
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	fwd := s.own(e.NewFloat32([]int64{int64(blocks + 1), 1, int64(states)}, result.Forward))
	bwd := s.own(e.NewFloat32([]int64{int64(blocks + 1), 1, int64(states)}, result.Backward))
	if s.err != nil {
		return Result{}, s.err
	}
	sums := s.own(e.Add(fwd, bwd))
	if s.err != nil {
		return Result{}, s.err
	}
	posts := s.own(e.Softmax(sums, -1))
	if s.err != nil {
		return Result{}, s.err
	}
	result.Posts, err = posts.Data()
	if err != nil {
		return Result{}, err
	}
	for i, v := range result.Posts {
		if !finite(v) || v < 0 || v > 1 {
			return Result{}, fmt.Errorf("invalid posterior at %d", i)
		}
	}
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	return result, nil
}

func scan(ctx context.Context, e *tensor.Engine, scores *tensor.Tensor, idx, scoreOrder *tensor.IndexTensor, blocks, states int, blank float32, reverse bool) (out []float32, ret error) {
	alpha, err := e.Zeros([]int64{1, int64(states)})
	if err != nil {
		return nil, err
	}
	defer func() {
		ret = errors.Join(ret, alpha.Close())
		if ret != nil {
			out = nil
		}
	}()
	out = make([]float32, (blocks+1)*states)
	for step := 0; step < blocks; step++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		time := step
		if reverse {
			time = blocks - 1 - step
		}
		next, err := scanStep(e, scores, alpha, idx, scoreOrder, time, states, blank)
		if err != nil {
			return nil, err
		}
		if err := alpha.Close(); err != nil {
			return nil, errors.Join(err, next.Close())
		}
		alpha = next
		row, err := alpha.Data()
		if err != nil {
			return nil, err
		}
		position := step + 1
		if reverse {
			position = blocks - 1 - step
		}
		copy(out[position*states:(position+1)*states], row)
	}
	return out, nil
}
func scanStep(e *tensor.Engine, scores, alpha *tensor.Tensor, idx, scoreOrder *tensor.IndexTensor, time, states int, blank float32) (*tensor.Tensor, error) {
	s := &scope{}
	slice := s.own(e.Narrow(scores, 0, int64(time), 1))
	if s.err != nil {
		return s.finish(nil)
	}
	var stepScores *tensor.Tensor
	if scoreOrder != nil {
		stepScores = s.own(e.IndexSelect(slice, 2, scoreOrder))
	} else {
		stepScores = slice
	}
	stepScores = s.own(e.Reshape(stepScores, []int64{1, int64(states), 4}))
	if s.err != nil {
		return s.finish(nil)
	}
	gathered := s.own(e.IndexSelect(alpha, 1, idx))
	gathered = s.own(e.Reshape(gathered, []int64{1, int64(states), 4}))
	if s.err != nil {
		return s.finish(nil)
	}
	scoredSteps := s.own(e.Add(gathered, stepScores))
	stay := s.own(e.AddScalar(alpha, float64(blank)))
	stay = s.own(e.Unsqueeze(stay, -1))
	if s.err != nil {
		return s.finish(nil)
	}
	transitions := s.own(e.Concat([]*tensor.Tensor{stay, scoredSteps}, -1))
	if s.err != nil {
		return s.finish(nil)
	}
	next := s.own(e.LogSumExp(transitions, -1, false))
	return s.finish(next)
}
