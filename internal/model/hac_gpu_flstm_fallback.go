package model

import (
	"context"
	"fmt"
	"math"
	"slices"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
)

type FLSTMHalfCache struct {
	params                   FLSTMParams
	downIH, downHH, up, bias []uint16
}

type FLSTMHalfCacheSnapshot struct {
	Params                   FLSTMParams
	DownIH, DownHH, Up, Bias []uint16
}

func (c FLSTMHalfCache) Snapshot() FLSTMHalfCacheSnapshot {
	return FLSTMHalfCacheSnapshot{c.params, slices.Clone(c.downIH), slices.Clone(c.downHH), slices.Clone(c.up), slices.Clone(c.bias)}
}

// PackFLSTMHalfLayer implements the caches in FLSTMStack.cpp:264-274.
func PackFLSTMHalfLayer(p FLSTMParams, w FLSTMWeightSet) (FLSTMHalfCache, error) {
	var out FLSTMHalfCache
	if err := p.validate(); err != nil {
		return out, err
	}
	for _, item := range []struct {
		name   string
		values []float32
		count  int64
	}{
		{"down-ih", w.DownIH, p.K * p.C}, {"down-hh", w.DownHH, p.K * p.C},
		{"up-ih", w.UpIH, 4 * p.C * p.K}, {"up-hh", w.UpHH, 4 * p.C * p.K},
		{"bias-ih", w.UpBiasIH, 4 * p.C},
	} {
		if int64(len(item.values)) != item.count {
			return out, fmt.Errorf("%w: half %s extent", ErrGPUWeights, item.name)
		}
		for _, v := range item.values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || signal.Float16Bits(v)&0x7c00 == 0x7c00 {
				return out, fmt.Errorf("%w: half %s finite domain", ErrGPUWeights, item.name)
			}
		}
	}
	transpose := func(values []float32, rows, cols int64) []uint16 {
		result := make([]uint16, len(values))
		for r := int64(0); r < rows; r++ {
			for c := int64(0); c < cols; c++ {
				result[c*rows+r] = signal.Float16Bits(values[r*cols+c])
			}
		}
		return result
	}
	out.params = p
	out.downIH = transpose(w.DownIH, p.K, p.C)
	out.downHH = transpose(w.DownHH, p.K, p.C)
	out.up = transpose(concatRows(w.UpIH, w.UpHH, 4*p.C, p.K), 4*p.C, 2*p.K)
	out.bias = CastFLSTMHalf(w.UpBiasIH)
	return out, nil
}

type HACHalfPlan struct{ n, t int64 }

func PlanHACHalfStack(n, t int64) (HACHalfPlan, error) {
	if n <= 0 || t <= 0 || n > math.MaxInt32 || t > math.MaxInt32-3 {
		return HACHalfPlan{}, fmt.Errorf("%w: half FLSTM positive native dimensions", ErrGPUPlanLayout)
	}
	if _, err := flstmElementCount(t+3, n, 1024); err != nil {
		return HACHalfPlan{}, err
	}
	if n > math.MaxInt32/4096 {
		return HACHalfPlan{}, fmt.Errorf("%w: half FLSTM native matrix extent", ErrGPUPlanLayout)
	}
	return HACHalfPlan{n, t}, nil
}

type halfFLSTMStep struct {
	layer                 int
	input, hidden, output int64
}
type halfFLSTMOps interface {
	zeroRow(int64) error
	zeroCell(int) error
	downInput(halfFLSTMStep) error
	downHidden(halfFLSTMStep) error
	up(halfFLSTMStep) error
	step(halfFLSTMStep) error
}

// The Go loop owns all timestep and layer decisions; each operation is one
// matrix multiplication or one cell update, never an entire recurrent layer.
func runHACHalfStack(ctx context.Context, p HACHalfPlan, ops halfFLSTMOps) error {
	if ctx == nil || ops == nil || p.n <= 0 || p.t <= 0 {
		return fmt.Errorf("%w: uninitialized half recurrence", ErrGPUPlanLayout)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	for _, row := range []int64{0, p.t + 2} {
		if err := ops.zeroRow(row); err != nil {
			return err
		}
	}
	for layer := 0; layer < 7; layer++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := ops.zeroCell(layer); err != nil {
			return err
		}
		for tick := int64(0); tick < p.t; tick++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			reverse := layer%2 == 0
			ti := 2 + tick
			to, th := ti-1, ti-2
			if reverse {
				ti = p.t - tick
				to, th = ti+1, ti+2
			}
			call := halfFLSTMStep{layer, ti, th, to}
			for _, op := range []func(halfFLSTMStep) error{ops.downInput, ops.downHidden, ops.up, ops.step} {
				if err := ctx.Err(); err != nil {
					return err
				}
				if err := op(call); err != nil {
					return err
				}
			}
		}
	}
	return ctx.Err()
}
