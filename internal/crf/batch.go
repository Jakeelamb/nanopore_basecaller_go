package crf

import (
	"context"
	"errors"
	"fmt"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

func batchDimensions(shape []int64) (blocks, lanes, states int, err error) {
	if len(shape) != 3 || shape[1] < 1 || shape[1] > 4 {
		return 0, 0, 0, errors.New("scores must have shape [T,N,4*States] with N in [1,4]")
	}
	blocks, states, err = dimensions([]int64{shape[0], 1, shape[2]})
	if err != nil {
		return 0, 0, 0, err
	}
	lanes = int(shape[1])
	// Divide the budget before multiplication, including all retained lane arrays.
	if blocks > (maxElements/lanes-3*states)/(7*states) {
		return 0, 0, 0, fmt.Errorf("CRF batch result exceeds %d float32 elements", maxElements)
	}
	return blocks, lanes, states, nil
}

// ComputeBatch borrows TNC scores and returns N owned single-lane results in
// input order. N=1..4 matches CPUDecoder's one-lane-per-worker decomposition.
// Any error or cancellation returns nil, including after a completed lane.
func ComputeBatch(ctx context.Context, e *tensor.Engine, scoresTNC *tensor.Tensor, blank float32) ([]Result, error) {
	if ctx == nil {
		return nil, errors.New("nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	shape, err := scoresTNC.Shape()
	if err != nil {
		return nil, err
	}
	_, lanes, _, err := batchDimensions(shape)
	if err != nil {
		return nil, err
	}
	results := make([]Result, 0, lanes)
	for lane := 0; lane < lanes; lane++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		view, err := e.Narrow(scoresTNC, 1, int64(lane), 1)
		if err != nil {
			return nil, err
		}
		result, computeErr := Compute(ctx, e, view, blank)
		if err := errors.Join(computeErr, view.Close()); err != nil {
			return nil, err
		}
		results = append(results, result)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return results, nil
}
