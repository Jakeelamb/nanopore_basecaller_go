package gpubatch

import (
	"reflect"
	"testing"
)

func hacConfig(variable bool) Config {
	return Config{Lane: HAC, ChunkSize: 768, BatchSize: 64, Stride: 6, InnerStride: 6, Granularity: 6, VariableChunks: variable}
}

func supConfig(variable bool) Config {
	return Config{Lane: SUP, ChunkSize: 768, BatchSize: 32, Stride: 6, InnerStride: 12, Granularity: 768,
		FirstConvPadding: 2, MaxConvPadding: 4, VariableChunks: variable}
}

func TestValidateRejectsUnsupportedLanesAndUnits(t *testing.T) {
	base := hacConfig(true)
	for name, mutate := range map[string]func(*Config){
		"lane":        func(c *Config) { c.Lane = 9 },
		"stride":      func(c *Config) { c.Stride = 0 },
		"chunk-unit":  func(c *Config) { c.ChunkSize = 767 },
		"batch-unit":  func(c *Config) { c.BatchSize = 63 },
		"inner":       func(c *Config) { c.InnerStride = 12 },
		"granularity": func(c *Config) { c.Granularity = 12 },
		"padding":     func(c *Config) { c.FirstConvPadding = 2 },
	} {
		c := base
		mutate(&c)
		if err := c.Validate(); err == nil {
			t.Fatalf("%s mutant accepted", name)
		}
	}
	sup := supConfig(true)
	for name, mutate := range map[string]func(*Config){
		"batch-unit": func(c *Config) { c.BatchSize = 31 },
		"inner":      func(c *Config) { c.InnerStride = 6 },
		"gran":       func(c *Config) { c.Granularity = 384 },
		"first-pad":  func(c *Config) { c.FirstConvPadding = 5 },
	} {
		c := sup
		mutate(&c)
		if err := c.Validate(); err == nil {
			t.Fatalf("SUP %s mutant accepted", name)
		}
	}
}

func TestGranularityTransitions(t *testing.T) {
	got, err := supConfig(true).Granularities([]int{1, 1, 3, 2, 2})
	want := []int{768, 768, 256, 128, 64}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, %v want %v", got, err, want)
	}
	if _, err := supConfig(true).Granularities([]int{1, 1, 5}); err == nil {
		t.Fatal("non-dividing stride accepted")
	}
	if _, err := supConfig(true).Granularities([]int{1, 0}); err == nil {
		t.Fatal("zero stride accepted")
	}
}

func TestFixedBatchUsesRealSlots(t *testing.T) {
	for _, cfg := range []Config{hacConfig(false), supConfig(false)} {
		b, err := NewBatch(cfg)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < cfg.BatchSize; i++ {
			flushed, err := b.Add(Chunk{ID: "c", Length: cfg.ChunkSize / 2})
			if err != nil || flushed {
				t.Fatalf("%v add %d flushed=%v err=%v", cfg.Lane, i, flushed, err)
			}
		}
		if !b.Full() || b.Real() != cfg.BatchSize {
			t.Fatalf("%v fixed batch not full", cfg.Lane)
		}
		ps := b.Placements()
		for i, p := range ps {
			if p.Slot != i || p.Offset != i*cfg.ChunkSize || p.Length != cfg.ChunkSize {
				t.Fatalf("%v placement %d = %+v", cfg.Lane, i, p)
			}
		}
		if _, err := b.Add(Chunk{Length: cfg.ChunkSize}); err == nil {
			t.Fatalf("%v accepted a chunk beyond fixed capacity", cfg.Lane)
		}
	}
}

func TestHACVariableMembershipAndPartCarry(t *testing.T) {
	b, err := NewBatch(hacConfig(true))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 64; i++ {
		flushed, err := b.Add(Chunk{ID: "c", Length: 768})
		if err != nil || flushed {
			t.Fatalf("chunk %d flushed=%v err=%v", i, flushed, err)
		}
	}
	if b.Full() {
		t.Fatal("64 chunks marked full before the part accumulator reaches capacity")
	}
	if b.Real() != 64 || b.PackedSamples() != 64*768 {
		t.Fatalf("real=%d packed=%d", b.Real(), b.PackedSamples())
	}
	flushed, err := b.Add(Chunk{ID: "next", Length: 768})
	if err != nil || !flushed {
		t.Fatalf("65th flushed=%v err=%v", flushed, err)
	}
	if b.Real() != 1 || b.PackedSamples() != 768 || b.Placements()[0].Offset != 0 {
		t.Fatalf("new batch = %+v packed=%d", b.Placements(), b.PackedSamples())
	}
}

func TestHACVariableRoundsToStride(t *testing.T) {
	b, err := NewBatch(hacConfig(true))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.Add(Chunk{Length: 767}); err != nil {
		t.Fatal(err)
	}
	if got := b.Placements()[0].Length; got != 768 {
		t.Fatalf("rounded length = %d", got)
	}
	if _, err := b.Add(Chunk{Length: 0}); err == nil {
		t.Fatal("zero chunk accepted")
	}
	if _, err := b.Add(Chunk{Length: 769}); err == nil {
		t.Fatal("oversized chunk accepted")
	}
}

func TestSUPVariableBoundaryPaddingAndFlush(t *testing.T) {
	b, err := NewBatch(supConfig(true))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 31; i++ {
		flushed, err := b.Add(Chunk{Length: 768})
		if err != nil || flushed {
			t.Fatalf("chunk %d flushed=%v err=%v", i, flushed, err)
		}
	}
	ps := b.Placements()
	if ps[0].Offset != 2 {
		t.Fatalf("first chunk offset = %d, want first-conv padding 2", ps[0].Offset)
	}
	if ps[1].Offset != 772 {
		t.Fatalf("second chunk offset = %d", ps[1].Offset)
	}
	flushed, err := b.Add(Chunk{Length: 768})
	if err != nil || !flushed {
		t.Fatalf("32nd flushed=%v err=%v", flushed, err)
	}
	if b.Real() != 1 || b.Placements()[0].Offset != 2 {
		t.Fatalf("new SUP batch = %+v", b.Placements())
	}
}

func TestSUPVariableRoundsToGranularity(t *testing.T) {
	b, err := NewBatch(supConfig(true))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.Add(Chunk{Length: 700}); err != nil {
		t.Fatal(err)
	}
	if got := b.Placements()[0].Length; got != 768 {
		t.Fatalf("rounded length = %d", got)
	}
}

func TestBatchCopiesPlacements(t *testing.T) {
	b, err := NewBatch(hacConfig(true))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.Add(Chunk{ID: "a", Length: 6}); err != nil {
		t.Fatal(err)
	}
	ps := b.Placements()
	ps[0].Chunk.ID = "mutated"
	if b.Placements()[0].Chunk.ID != "a" {
		t.Fatal("placements alias batch state")
	}
}
