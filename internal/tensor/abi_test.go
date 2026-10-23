//go:build cgo && linux

package tensor

import "testing"

func TestNativeABIBoundary(t *testing.T) {
	e := nativeEngine(t)
	other := nativeEngine(t)
	a := newTensor(t, e, []int64{2}, []float32{1, 2})
	indices := newIndices(t, e, []int64{2}, []int64{0, 1})
	if _, err := e.backend.data(indices.handle.id, 2); err == nil {
		t.Fatal("native ABI allowed int64 through float copy")
	}
	if _, err := e.backend.dataInt64(a.id, 2); err == nil {
		t.Fatal("native ABI allowed float32 through int64 copy")
	}
	if _, err := e.backend.run(16, []uint64{indices.handle.id}, nil, []float64{.5}); err == nil {
		t.Fatal("native ABI auto-cast int64 through floating arithmetic")
	}
	if _, err := e.backend.run(21, []uint64{a.id, a.id}, []int64{0}, nil); err == nil {
		t.Fatal("native ABI allowed float32 indices")
	}
	if _, err := other.backend.data(a.id, 2); err == nil {
		t.Fatal("native ABI accepted foreign handle")
	}
	if _, err := e.backend.data(^uint64(0), 2); err == nil {
		t.Fatal("native ABI accepted invalid ID")
	}
	if _, err := e.backend.data(a.id, 3); err == nil {
		t.Fatal("native ABI accepted wrong buffer count")
	}
	if _, err := e.backend.newTensor([]int64{-2}, nil); err == nil {
		t.Fatal("native ABI accepted negative dimension")
	}
	if _, err := e.backend.newTensor([]int64{3}, []float32{1}); err == nil {
		t.Fatal("native ABI accepted data length mismatch")
	}
	if _, err := e.backend.run(1, []uint64{a.id}, nil, nil); err == nil {
		t.Fatal("native ABI accepted wrong arity")
	}
	if _, err := e.backend.run(999, []uint64{a.id}, nil, nil); err == nil {
		t.Fatal("native ABI accepted unknown operation")
	}
	for _, op := range []int{26, 27, 28, 29, 30, 31, 32} {
		if _, err := e.backend.run(op, []uint64{indices.handle.id}, nil, nil); err == nil {
			t.Fatalf("native ABI allowed int64 through attention operation %d", op)
		}
	}
	if _, err := e.backend.run(26, []uint64{a.id}, []int64{0, 2}, nil); err == nil {
		t.Fatal("native mean accepted non-bool keepdim flag")
	}
	attention := newTensor(t, e, []int64{1, 1, 2, 2}, []float32{1, 2, 3, 4})
	for _, tt := range []struct {
		name    string
		q, k, v uint64
		shape   []int64
		mask    []byte
	}{
		{"mask byte", attention.id, attention.id, attention.id, []int64{2, 2}, []byte{1, 2, 0, 1}},
		{"mask count", attention.id, attention.id, attention.id, []int64{2, 2}, []byte{1}},
		{"mask shape", attention.id, attention.id, attention.id, []int64{-1}, nil},
		{"mask rank", attention.id, attention.id, attention.id, make([]int64, 33), nil},
		{"mask overflow", attention.id, attention.id, attention.id, []int64{1 << 62, 8}, nil},
		{"mask broadcast", attention.id, attention.id, attention.id, []int64{3}, []byte{1, 0, 1}},
		{"foreign tensor", attention.id, foreignID(t, other), attention.id, nil, nil},
		{"index tensor", attention.id, attention.id, indices.handle.id, nil, nil},
		{"invalid tensor", attention.id, attention.id, ^uint64(0), nil, nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := e.backend.sdpa(tt.q, tt.k, tt.v, tt.shape, tt.mask); err == nil {
				t.Fatal("native SDPA accepted invalid input")
			}
		})
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := e.backend.data(a.id, 2); err == nil {
		t.Fatal("native ABI accepted released ID")
	}
	if err := e.backend.release(a.id); err == nil {
		t.Fatal("native ABI accepted double release")
	}
	fresh := newTensor(t, e, []int64{2}, []float32{3, 4})
	if fresh.id == a.id {
		t.Fatal("native tensor ID was reused")
	}
	if _, err := e.backend.run(1, []uint64{a.id, fresh.id}, nil, nil); err == nil {
		t.Fatal("native ABI accepted stale operand")
	}
	staleEngine := e.backend.id
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	borrowed := *other.backend
	borrowed.id = staleEngine
	if _, err := borrowed.newTensor([]int64{1}, []float32{1}); err == nil {
		t.Fatal("native ABI accepted closed engine ID")
	}
}

func foreignID(t *testing.T, e *Engine) uint64 {
	t.Helper()
	return newTensor(t, e, []int64{1, 1, 2, 2}, []float32{1, 2, 3, 4}).id
}
