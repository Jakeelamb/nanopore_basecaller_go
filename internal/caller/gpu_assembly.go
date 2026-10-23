package caller

import (
	"fmt"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/stitch"
)

// DecodedWork is one completed GPU chunk. Completion order is not read order.
// RawLength excludes repeat padding; Granularity is the padding alignment the
// source used for this lane.
type DecodedWork struct {
	ReadID      string
	ChunkIndex  int
	Offset      uint64
	RawLength   uint64
	Granularity uint64
	Sequence    string
	Qstring     string
	Moves       []uint8
}

// ReadShape fixes the expected chunk count and stitch coordinates for one read.
type ReadShape struct {
	RawSamples uint64
	Stride     uint64
	Variable   bool
	Chunks     int
}

// Collector restores source chunk order before the accepted stitch unit.
type Collector struct {
	shapes map[string]ReadShape
	works  map[string]map[int]DecodedWork
}

func NewCollector(shapes map[string]ReadShape) (*Collector, error) {
	if len(shapes) == 0 {
		return nil, fmt.Errorf("empty read inventory")
	}
	c := &Collector{shapes: make(map[string]ReadShape, len(shapes)), works: make(map[string]map[int]DecodedWork, len(shapes))}
	for id, shape := range shapes {
		if id == "" || shape.RawSamples == 0 || shape.Stride == 0 || shape.Chunks <= 0 {
			return nil, fmt.Errorf("invalid read shape for %q", id)
		}
		c.shapes[id] = shape
		c.works[id] = make(map[int]DecodedWork, shape.Chunks)
	}
	return c, nil
}

func (c *Collector) Add(w DecodedWork) error {
	shape, ok := c.shapes[w.ReadID]
	if !ok {
		return fmt.Errorf("unknown read %q", w.ReadID)
	}
	if w.ChunkIndex < 0 || w.ChunkIndex >= shape.Chunks {
		return fmt.Errorf("read %q chunk index %d outside 0..%d", w.ReadID, w.ChunkIndex, shape.Chunks-1)
	}
	if _, exists := c.works[w.ReadID][w.ChunkIndex]; exists {
		return fmt.Errorf("duplicate read %q chunk %d", w.ReadID, w.ChunkIndex)
	}
	if w.RawLength == 0 || w.RawLength > shape.RawSamples || w.Offset > shape.RawSamples-w.RawLength {
		return fmt.Errorf("read %q chunk %d coordinate outside signal", w.ReadID, w.ChunkIndex)
	}
	if w.Granularity == 0 || w.Granularity%shape.Stride != 0 {
		return fmt.Errorf("read %q chunk %d granularity %d is not stride-aligned", w.ReadID, w.ChunkIndex, w.Granularity)
	}
	w.Moves = append([]uint8(nil), w.Moves...)
	c.works[w.ReadID][w.ChunkIndex] = w
	return nil
}

func (c *Collector) Ready(readID string) bool {
	shape, ok := c.shapes[readID]
	return ok && len(c.works[readID]) == shape.Chunks
}

// Assemble orders the completed chunks by source index and delegates coverage,
// overlap, move and quality validation to stitch.Stitch.
func (c *Collector) Assemble(readID string) (stitch.Result, error) {
	shape, ok := c.shapes[readID]
	if !ok {
		return stitch.Result{}, fmt.Errorf("unknown read %q", readID)
	}
	if len(c.works[readID]) != shape.Chunks {
		return stitch.Result{}, fmt.Errorf("read %q has %d of %d chunks", readID, len(c.works[readID]), shape.Chunks)
	}
	ordered := make([]DecodedWork, 0, shape.Chunks)
	for i := 0; i < shape.Chunks; i++ {
		w, ok := c.works[readID][i]
		if !ok {
			return stitch.Result{}, fmt.Errorf("read %q missing chunk %d", readID, i)
		}
		ordered = append(ordered, w)
	}
	chunks := make([]stitch.Chunk, len(ordered))
	for i, w := range ordered {
		chunks[i] = stitch.Chunk{Offset: w.Offset, RawSize: w.RawLength, Granularity: w.Granularity,
			Sequence: w.Sequence, Qstring: w.Qstring, Moves: w.Moves}
	}
	return stitch.Stitch(stitch.Config{RawSamples: shape.RawSamples, Stride: shape.Stride, Variable: shape.Variable}, chunks)
}
