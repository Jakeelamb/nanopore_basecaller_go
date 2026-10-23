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
