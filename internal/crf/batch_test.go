package crf

import (
	"context"
	"errors"
	"fmt"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
	"math"
	"reflect"
	"testing"
)

func TestBatchDimensions(t *testing.T) {
	for _, shape := range [][]int64{nil, {1}, {1, 1}, {1, 1, 16, 1}, {0, 1, 16}, {-1, 1, 16}, {1, 0, 16}, {1, 5, 16}, {1, 1, 15}, {1, 1, 4}, {1, 1, 32}, {1, 1, 65537}, {math.MaxInt64, 4, 16}, {1, 1, math.MaxInt64 - 3}} {
		if _, _, _, err := batchDimensions(shape); err == nil {
			t.Fatalf("accepted %v", shape)
		}
	}
	for n := 1; n <= 4; n++ {
		for _, s := range []int{4, 16, 64, 256, 1024, 4096, 16384} {
			limit := (maxElements/n - 3*s) / (7 * s)
			for _, blocks := range []int{1, 9, limit} {
				b, lanes, states, err := batchDimensions([]int64{int64(blocks), int64(n), int64(4 * s)})
				if err != nil || b != blocks || lanes != n || states != s {
					t.Fatalf("shape %d,%d,%d: %d %d %d %v", blocks, n, s, b, lanes, states, err)
				}
			}
			if _, _, _, err := batchDimensions([]int64{int64(limit + 1), int64(n), int64(4 * s)}); err == nil {
				t.Fatalf("accepted aggregate overflow for N=%d S=%d", n, s)
			}
		}
	}
}

func TestBatchCanceledBeforeNative(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if got, err := ComputeBatch(ctx, nil, nil, 2); got != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-canceled: %v %v", got, err)
	}
	if got, err := ComputeBatch(nil, nil, nil, 2); got != nil || err == nil {
		t.Fatalf("nil context: %v %v", got, err)
	}
	if got, err := ComputeBatch(context.Background(), nil, nil, 2); got != nil || err == nil {
		t.Fatalf("nil tensor: %v %v", got, err)
	}
}

func assertResultBits(t *testing.T, got, want Result) {
	t.Helper()
	if got.Blocks != want.Blocks || got.States != want.States {
		t.Fatal("result dimensions changed")
	}
	for stage, pair := range [][2][]float32{{got.Scores, want.Scores}, {got.Forward, want.Forward}, {got.Backward, want.Backward}, {got.Posts, want.Posts}} {
		if len(pair[0]) != len(pair[1]) {
			t.Fatalf("stage %d length differs", stage)
		}
		for i, v := range pair[0] {
			if math.Float32bits(v) != math.Float32bits(pair[1][i]) {
				t.Fatalf("stage %d element %d differs", stage, i)
			}
		}
	}
}

func TestNativeBatchComposition(t *testing.T) {
	e := engine(t)
	const blocks, channels = 3, 16
	for n := 1; n <= 4; n++ {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			data := make([]float32, blocks*n*channels)
			want := make([]Result, n)
			for lane := 0; lane < n; lane++ {
				single := make([]float32, blocks*channels)
				for time := 0; time < blocks; time++ {
					for c := 0; c < channels; c++ {
						v := float32(100*lane+20*time+c) / 32
						data[(time*n+lane)*channels+c] = v
						single[time*channels+c] = v
					}
				}
				x := inputTensor(t, e, []int64{blocks, 1, channels}, single)
				var err error
				want[lane], err = Compute(context.Background(), e, x, .25)
				if err != nil {
					t.Fatal(err)
				}
			}
			x := inputTensor(t, e, []int64{blocks, int64(n), channels}, data)
			got, err := ComputeBatch(context.Background(), e, x, .25)
			if err != nil || len(got) != n {
				t.Fatalf("batch: %d %v", len(got), err)
			}
			for lane := range got {
				assertResultBits(t, got[lane], want[lane])
			}
			after, err := x.Data()
			if err != nil || !reflect.DeepEqual(after, data) {
				t.Fatalf("input changed: %v", err)
			}
			permuted := make([]float32, len(data))
			for time := 0; time < blocks; time++ {
				for lane := 0; lane < n; lane++ {
					copy(permuted[(time*n+lane)*channels:], data[(time*n+n-1-lane)*channels:(time*n+n-lane)*channels])
				}
			}
			p := inputTensor(t, e, []int64{blocks, int64(n), channels}, permuted)
			reversed, err := ComputeBatch(context.Background(), e, p, .25)
			if err != nil {
				t.Fatal(err)
			}
			for lane := range reversed {
				assertResultBits(t, reversed[lane], want[n-1-lane])
			}
			saved := got[0].Scores[0]
			got[0].Scores[0] = -12345
			afterMutation, err := x.Data()
			if err != nil || !reflect.DeepEqual(afterMutation, data) {
				t.Fatalf("result mutation changed input: %v", err)
			}
			got[0].Scores[0] = saved
			if err := x.Close(); err != nil {
				t.Fatal(err)
			}
			for lane := range got {
				assertResultBits(t, got[lane], want[lane])
				stages := [][]float32{got[lane].Scores, got[lane].Forward, got[lane].Backward, got[lane].Posts}
				expected := [][]float32{want[lane].Scores, want[lane].Forward, want[lane].Backward, want[lane].Posts}
				for index, stage := range stages {
					stage[0] = -12345
					for sibling := index + 1; sibling < len(stages); sibling++ {
						if !reflect.DeepEqual(stages[sibling], expected[sibling]) {
							t.Fatal("result stages alias")
						}
					}
					for other := lane + 1; other < n; other++ {
						assertResultBits(t, got[other], want[other])
					}
				}
			}
			again, err := ComputeBatch(context.Background(), e, p, .25)
			if err != nil {
				t.Fatal(err)
			}
			for lane := range again {
				assertResultBits(t, again[lane], want[n-1-lane])
				assertResultBits(t, reversed[lane], want[n-1-lane])
			}
		})
	}
}

type countedContext struct {
	context.Context
	checks int64
}

func (c *countedContext) Err() error { c.checks++; return c.Context.Err() }

func TestNativeBatchValidationAndCancellation(t *testing.T) {
	e := engine(t)
	valid := inputTensor(t, e, []int64{2, 4, 16}, make([]float32, 128))
	for lane := 0; lane < 4; lane++ {
		for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), math.MaxFloat32} {
			data := make([]float32, 128)
			data[(4+lane)*16+15] = bad
			x := inputTensor(t, e, []int64{2, 4, 16}, data)
			if got, err := ComputeBatch(context.Background(), e, x, 2); err == nil || got != nil {
				t.Fatalf("invalid lane %d: %v %v", lane, got, err)
			}
		}
	}
	for _, blank := range []float32{float32(math.NaN()), float32(math.Inf(1)), math.MaxFloat32} {
		if got, err := ComputeBatch(context.Background(), e, valid, blank); err == nil || got != nil {
			t.Fatalf("invalid blank: %v %v", got, err)
		}
	}
	for _, shape := range [][]int64{{1, 5, 16}, {1, 0, 16}, {0, 2, 16}, {1, 2, 32}, {2, 16}} {
		n := 1
		for _, d := range shape {
			n *= int(d)
		}
		x := inputTensor(t, e, shape, make([]float32, n))
		if got, err := ComputeBatch(context.Background(), e, x, 2); err == nil || got != nil {
			t.Fatalf("invalid shape %v: %v %v", shape, got, err)
		}
	}
	other := engine(t)
	foreign := inputTensor(t, other, []int64{1, 2, 16}, make([]float32, 32))
	if got, err := ComputeBatch(context.Background(), e, foreign, 2); err == nil || got != nil {
		t.Fatalf("foreign tensor: %v %v", got, err)
	}
	closed := inputTensor(t, e, []int64{1, 2, 16}, make([]float32, 32))
	if err := closed.Close(); err != nil {
		t.Fatal(err)
	}
	if got, err := ComputeBatch(context.Background(), e, closed, 2); !errors.Is(err, tensor.ErrClosed) || got != nil {
		t.Fatalf("closed tensor: %v %v", got, err)
	}
	if got, err := ComputeBatch(context.Background(), nil, valid, 2); !errors.Is(err, tensor.ErrClosed) || got != nil {
		t.Fatalf("nil engine: %v %v", got, err)
	}
	single := inputTensor(t, e, []int64{2, 1, 16}, make([]float32, 32))
	singleLaneChecks := &countedContext{Context: context.Background()}
	if got, err := ComputeBatch(singleLaneChecks, e, single, 2); err != nil || len(got) != 1 {
		t.Fatalf("single-lane batch: %v %v", got, err)
	}
	twoLanes := inputTensor(t, e, []int64{2, 2, 16}, make([]float32, 64))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	betweenLanes := &cancelAfter{Context: ctx, cancel: cancel}
	betweenLanes.remaining.Store(singleLaneChecks.checks)
	if got, err := ComputeBatch(betweenLanes, e, twoLanes, 2); !errors.Is(err, context.Canceled) || got != nil {
		t.Fatalf("between-lane cancellation: %v %v", got, err)
	}
	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	inside := &cancelAfter{Context: ctx2, cancel: cancel2}
	inside.remaining.Store(6)
	if got, err := ComputeBatch(inside, e, valid, 2); !errors.Is(err, context.Canceled) || got != nil {
		t.Fatalf("inside-lane cancellation: %v %v", got, err)
	}
	if _, err := ComputeBatch(context.Background(), e, valid, 2); err != nil {
		t.Fatal(err)
	}
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	if got, err := ComputeBatch(context.Background(), e, valid, 2); !errors.Is(err, tensor.ErrClosed) || got != nil {
		t.Fatalf("closed engine: %v %v", got, err)
	}
}

func TestNativeBatchStridedLayout(t *testing.T) {
	e := engine(t)
	for n := 1; n <= 4; n++ {
		data := make([]float32, 3*n*32)
		for time := 0; time < 3; time++ {
			for lane := 0; lane < n; lane++ {
				for channel := 0; channel < 32; channel++ {
					data[(time*n+lane)*32+channel] = float32(1000*time + 100*lane + channel)
				}
			}
		}
		x := inputTensor(t, e, []int64{3, int64(n), 32}, data)
		view, err := e.Narrow(x, 2, 8, 16)
		if err != nil {
			t.Fatal(err)
		}
		got, computeErr := ComputeBatch(context.Background(), e, view, 2)
		if err := errors.Join(computeErr, view.Close()); err != nil {
			t.Fatal(err)
		}
		for lane, result := range got {
			for time := 0; time < 3; time++ {
				for c := 0; c < 16; c++ {
					if result.Scores[time*16+c] != float32(1000*time+100*lane+c+8) {
						t.Fatalf("TNC identity lost: N=%d lane=%d time=%d channel=%d", n, lane, time, c)
					}
				}
			}
		}
	}
}
