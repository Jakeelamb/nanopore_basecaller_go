package caller

import (
	"reflect"
	"testing"
)

func twoChunkShapes() map[string]ReadShape {
	return map[string]ReadShape{
		"read-a": {RawSamples: 12, Stride: 6, Chunks: 2},
		"read-b": {RawSamples: 12, Stride: 6, Chunks: 2},
	}
}

func work(id string, index int, offset uint64, base byte) DecodedWork {
	return DecodedWork{ReadID: id, ChunkIndex: index, Offset: offset, RawLength: 6, Granularity: 6,
		Sequence: string(base), Qstring: "I", Moves: []uint8{1}}
}

func TestGPUAssemblyRestoresSourceOrder(t *testing.T) {
	c, err := NewCollector(twoChunkShapes())
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Add(work("read-a", 1, 6, 'C')); err != nil {
		t.Fatal(err)
	}
	if c.Ready("read-a") {
		t.Fatal("premature ready")
	}
	if _, err := c.Assemble("read-a"); err == nil {
		t.Fatal("assembled before every chunk arrived")
	}
	if err := c.Add(work("read-a", 0, 0, 'A')); err != nil {
		t.Fatal(err)
	}
	got, err := c.Assemble("read-a")
	if err != nil || got.Sequence != "AC" || got.Qstring != "II" || !reflect.DeepEqual(got.Moves, []uint8{1, 1}) {
		t.Fatalf("assembled %+v %v", got, err)
	}
}

func TestGPUAssemblyIndependentReads(t *testing.T) {
	c, err := NewCollector(twoChunkShapes())
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []DecodedWork{work("read-a", 0, 0, 'A'), work("read-b", 1, 6, 'G'), work("read-a", 1, 6, 'C'), work("read-b", 0, 0, 'T')} {
		if err := c.Add(w); err != nil {
			t.Fatal(err)
		}
	}
	for id, want := range map[string]string{"read-a": "AC", "read-b": "TG"} {
		got, err := c.Assemble(id)
		if err != nil || got.Sequence != want {
			t.Fatalf("%s = %+v %v want %s", id, got, err, want)
		}
	}
}

func TestGPUAssemblyRejectsBadWork(t *testing.T) {
	c, err := NewCollector(twoChunkShapes())
	if err != nil {
		t.Fatal(err)
	}
	bad := []DecodedWork{
		work("unknown", 0, 0, 'A'),
		work("read-a", -1, 0, 'A'),
		work("read-a", 2, 0, 'A'),
		{ReadID: "read-a", ChunkIndex: 0, Offset: 0, RawLength: 0, Granularity: 6, Sequence: "A", Qstring: "I", Moves: []uint8{1}},
		{ReadID: "read-a", ChunkIndex: 0, Offset: 8, RawLength: 6, Granularity: 6, Sequence: "A", Qstring: "I", Moves: []uint8{1}},
		{ReadID: "read-a", ChunkIndex: 0, Offset: 0, RawLength: 6, Granularity: 4, Sequence: "A", Qstring: "I", Moves: []uint8{1}},
	}
	for i, w := range bad {
		if err := c.Add(w); err == nil {
			t.Fatalf("bad work %d accepted", i)
		}
	}
	if err := c.Add(work("read-a", 0, 0, 'A')); err != nil {
		t.Fatal(err)
	}
	if err := c.Add(work("read-a", 0, 0, 'A')); err == nil {
		t.Fatal("duplicate chunk accepted")
	}
}

func TestGPUAssemblyDelegatesCoverage(t *testing.T) {
	c, err := NewCollector(map[string]ReadShape{"read-a": {RawSamples: 18, Stride: 6, Chunks: 2}})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Add(work("read-a", 0, 0, 'A')); err != nil {
		t.Fatal(err)
	}
	if err := c.Add(work("read-a", 1, 6, 'C')); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Assemble("read-a"); err == nil {
		t.Fatal("incomplete signal coverage accepted")
	}
}

func TestGPUAssemblyPreservesPaddingGranularity(t *testing.T) {
	c, err := NewCollector(map[string]ReadShape{"read-a": {RawSamples: 700, Stride: 6, Variable: true, Chunks: 1}})
	if err != nil {
		t.Fatal(err)
	}
	moves := make([]uint8, 128)
	moves[116] = 1
	w := DecodedWork{ReadID: "read-a", ChunkIndex: 0, Offset: 0, RawLength: 700, Granularity: 768,
		Sequence: "A", Qstring: "I", Moves: moves}
	if err := c.Add(w); err != nil {
		t.Fatal(err)
	}
	got, err := c.Assemble("read-a")
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Moves) != 116 || len(got.Sequence) != 0 {
		t.Fatalf("moves=%d sequence=%q", len(got.Moves), got.Sequence)
	}
}

func TestGPUAssemblyCollectorRejectsBadShapes(t *testing.T) {
	for _, shapes := range []map[string]ReadShape{
		nil,
		{"": {RawSamples: 1, Stride: 1, Chunks: 1}},
		{"r": {RawSamples: 0, Stride: 1, Chunks: 1}},
		{"r": {RawSamples: 1, Stride: 0, Chunks: 1}},
		{"r": {RawSamples: 1, Stride: 1, Chunks: 0}},
	} {
		if _, err := NewCollector(shapes); err == nil {
			t.Fatalf("shapes %v accepted", shapes)
		}
	}
}
