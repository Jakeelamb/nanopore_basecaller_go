//go:build cgo && linux

package tensor

import (
	"fmt"
	"math"
	"runtime"
	"syscall"
	"testing"
)

func TestNativeForeignThreadInitialization(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	opener := syscall.Gettid()
	e := nativeEngine(t)
	values := make([]float32, 1024)
	for i := range values {
		values[i] = float32(i%17-8) / 4
	}
	x := newTensor(t, e, []int64{int64(len(values))}, values)
	const side = 256
	identity := make([]float32, side*side)
	for i := 0; i < side; i++ {
		identity[i*side+i] = 1
	}
	matrix := newTensor(t, e, []int64{side, side}, identity)
	type result struct {
		tid int
		err error
	}
	replies := make(chan result, 1)
	release := make(chan struct{})
	done := make(chan struct{}, 4)
	started := 0
	defer func() {
		close(release)
		for i := 0; i < started; i++ {
			<-done
		}
	}()
	seen := map[int]bool{opener: true}
	for worker := 0; worker < 4; worker++ {
		started++
		go func() {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()
			tid := syscall.Gettid()
			err := foreignThreadOperations(e, x, matrix, values, identity)
			replies <- result{tid, err}
			// Retain each OS thread so the next worker cannot reuse it.
			<-release
			done <- struct{}{}
		}()
		r := <-replies
		if seen[r.tid] {
			t.Fatalf("worker reused OS thread %d", r.tid)
		}
		seen[r.tid] = true
		if r.err != nil {
			t.Fatalf("worker %d tid=%d: %v", worker, r.tid, r.err)
		}
		t.Logf("worker=%d opener_tid=%d foreign_tid=%d tanh_elements=%d matmul=256x256 output=verified", worker, opener, r.tid, len(values))
	}
}

func foreignThreadOperations(e *Engine, x, matrix *Tensor, values, identity []float32) error {
	y, err := e.Tanh(x)
	if err != nil {
		return fmt.Errorf("tanh: %w", err)
	}
	data, readErr := y.Data()
	closeErr := y.Close()
	if readErr != nil {
		return readErr
	}
	if closeErr != nil {
		return closeErr
	}
	if len(data) != len(values) {
		return fmt.Errorf("tanh output length %d, want %d", len(data), len(values))
	}
	for i, v := range data {
		want := math.Tanh(float64(values[i]))
		if math.IsNaN(float64(v)) || math.Abs(float64(v)-want) > 2e-6 {
			return fmt.Errorf("tanh[%d]=%g, want %g", i, v, want)
		}
	}
	z, err := e.MatMul(matrix, matrix)
	if err != nil {
		return fmt.Errorf("MKL matmul: %w", err)
	}
	data, readErr = z.Data()
	closeErr = z.Close()
	if readErr != nil {
		return readErr
	}
	if closeErr != nil {
		return closeErr
	}
	if len(data) != len(identity) {
		return fmt.Errorf("matmul output length %d, want %d", len(data), len(identity))
	}
	for i, v := range data {
		if v != identity[i] {
			return fmt.Errorf("matmul[%d]=%g, want %g", i, v, identity[i])
		}
	}
	return nil
}
