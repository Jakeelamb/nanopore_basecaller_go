package gpubatch

import (
	"reflect"
	"testing"
)

func TestLSTMPackingOwnsInitializedSpans(t *testing.T) {
	cfg := hacConfig(true)
	cfg.ChunkSize = 12288
	a, err := NewAux(cfg, []int{768, 1536, 12288})
	if err != nil {
		t.Fatal(err)
	}
	before := a.SizesOut()
	first, err := a.PackLSTM()
	if err != nil {
		t.Fatal(err)
	}
	if len(first.InLayout) != 2432 || len(first.OutLayout) != 131200 || len(first.ForwardEncoding) != 131200 || len(first.BackwardEncoding) != 131264 || first.OutLayoutCapacity != 131264 {
		t.Fatalf("initialized lengths or destination capacity differ")
	}
	seen := make([]bool, len(first.InLayout))
	for _, value := range first.OutLayout {
		if value == -1 {
			continue
		}
		if value < 0 || int(value) >= len(seen) || seen[value] {
			t.Fatalf("invalid or duplicate compacted output coordinate %d", value)
		}
		seen[value] = true
	}
	for i, found := range seen {
		if !found {
			t.Fatalf("missing compacted output coordinate %d", i)
		}
	}
	if !reflect.DeepEqual(before, a.SizesOut()) {
		t.Fatal("packing changed input chunk sizes")
	}
	second, err := a.PackLSTM()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatal("same input changed packing words")
	}
	first.InLayout[0] ^= 1
	if !reflect.DeepEqual(first.OutLayout, second.OutLayout) || !reflect.DeepEqual(first.ForwardEncoding, second.ForwardEncoding) || !reflect.DeepEqual(first.BackwardEncoding, second.BackwardEncoding) {
		t.Fatal("InLayout aliases another upload span")
	}
	first.OutLayout[0] ^= 1
	if !reflect.DeepEqual(first.ForwardEncoding, second.ForwardEncoding) || !reflect.DeepEqual(first.BackwardEncoding, second.BackwardEncoding) {
		t.Fatal("OutLayout aliases a recurrence encoding")
	}
	first.ForwardEncoding[0] ^= 1
	if !reflect.DeepEqual(first.BackwardEncoding, second.BackwardEncoding) {
		t.Fatal("forward and backward recurrence encodings alias")
	}
	first.BackwardEncoding[0] ^= 1
	third, err := a.PackLSTM()
	if err != nil || !reflect.DeepEqual(second, third) {
		t.Fatal("packing results share mutable state", err)
	}
}

func TestLSTMPackingRejectsUnrepresentableBatches(t *testing.T) {
	t.Run("wrong lane", func(t *testing.T) {
		a, err := NewAux(supConfig(true), []int{768})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := a.PackLSTM(); err == nil {
			t.Fatal("SUP packing accepted")
		}
	})
	t.Run("nil", func(t *testing.T) {
		var a *Aux
		if _, err := a.PackLSTM(); err == nil {
			t.Fatal("nil packing accepted")
		}
	})
	t.Run("sentinel capacity", func(t *testing.T) {
		cfg := hacConfig(true)
		cfg.ChunkSize = 768
		sizes := make([]int, 8192)
		for i := range sizes {
			sizes[i] = 6
		}
		a, err := NewAux(cfg, sizes)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := a.PackLSTM(); err == nil {
			t.Fatal("sentinel workspace overflow accepted")
		}
	})
	t.Run("root encoding", func(t *testing.T) {
		cfg := hacConfig(true)
		cfg.BatchSize, cfg.ChunkSize = 65536, 6
		sizes := make([]int, cfg.BatchSize)
		for i := range sizes {
			sizes[i] = 6
		}
		a, err := NewAux(cfg, sizes)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := a.PackLSTM(); err == nil {
			t.Fatal("root encoding end above 65535 accepted")
		}
	})
	t.Run("coordinate overflow", func(t *testing.T) {
		cfg := hacConfig(true)
		cfg.BatchSize, cfg.ChunkSize = 192, 6*11184808
		a, err := NewAux(cfg, []int{6})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := a.PackLSTM(); err == nil {
			t.Fatal("source workspace coordinate overflow accepted")
		}
	})
}
