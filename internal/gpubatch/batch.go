// Package gpubatch ports the pinned source batch geometry for the HAC and SUP
// GPU callers. It owns real chunk membership, packed offsets, padding slots and
// the granularity transitions consumed by the model lanes.
package gpubatch

import (
	"fmt"
	"math"
)

// Source fields are signed 32-bit. Coordinates and sizes are bounded before
// any allocation so a malformed request cannot wrap a native dimension.
const MaxCoordinate = math.MaxInt32

type Lane uint8

const (
	HAC Lane = iota + 1
	SUP
)

// Config fixes one supported model lane and its explicit batch. The caller
// derives these values from the pinned model spec; this package does not read
// model files.
type Config struct {
	Lane             Lane
	ChunkSize        int
	BatchSize        int
	Stride           int
	InnerStride      int
	Granularity      int
	FirstConvPadding int
	MaxConvPadding   int
	VariableChunks   bool
}

func (c Config) Validate() error {
	if c.Lane != HAC && c.Lane != SUP {
		return fmt.Errorf("unsupported model lane %d", c.Lane)
	}
	if c.Stride <= 0 || c.InnerStride <= 0 || c.Granularity <= 0 {
		return fmt.Errorf("invalid stride or granularity")
	}
	if c.ChunkSize <= 0 || c.ChunkSize > MaxCoordinate || c.ChunkSize%c.Granularity != 0 {
		return fmt.Errorf("invalid chunk size %d for granularity %d", c.ChunkSize, c.Granularity)
	}
	if c.BatchSize <= 0 || c.BatchSize > MaxCoordinate {
		return fmt.Errorf("invalid batch size %d", c.BatchSize)
	}
	if c.BatchSize%BatchGranularity(c.Lane) != 0 {
		return fmt.Errorf("batch size %d is not a multiple of %d", c.BatchSize, BatchGranularity(c.Lane))
	}
	switch c.Lane {
	case HAC:
		if c.InnerStride != c.Stride || c.Granularity != c.InnerStride {
			return fmt.Errorf("HAC requires granularity equal to stride")
		}
		if c.FirstConvPadding != 0 || c.MaxConvPadding != 0 {
			return fmt.Errorf("HAC does not use convolution padding")
		}
	case SUP:
		if c.InnerStride != 2*c.Stride || c.Granularity != 64*c.InnerStride {
			return fmt.Errorf("SUP requires doubled stride and 64 inner-stride granules")
		}
		if c.FirstConvPadding < 0 || c.MaxConvPadding < c.FirstConvPadding {
			return fmt.Errorf("invalid SUP convolution padding")
		}
		if c.MaxConvPadding > 0 && c.MaxConvPadding > c.ChunkSize {
			return fmt.Errorf("convolution padding exceeds chunk size")
		}
	}
	return nil
}

// BatchGranularity is the pinned source GPU batch multiple.
func BatchGranularity(l Lane) int {
	if l == SUP {
		return 32
	}
	return 64
}

// EncoderT is the transformer encoder granularity asserted by TxEncoderStack.
func (c Config) EncoderT() int { return c.Granularity / c.InnerStride }

// Granularities applies the five convolution strides in source order and returns
// the granularity after each layer. The final value is the transformer encoder
// input granularity.
func (c Config) Granularities(convStrides []int) ([]int, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	out := make([]int, 0, len(convStrides))
	g := c.Granularity
	for i, stride := range convStrides {
		if stride <= 0 || g%stride != 0 {
			return nil, fmt.Errorf("convolution %d stride %d does not divide granularity %d", i, stride, g)
		}
		g /= stride
		out = append(out, g)
	}
	return out, nil
}

// Chunk is one logical chunk with its raw valid sample length. Padding is
// derived, never supplied by the caller.
type Chunk struct {
	ID     string
	Length int
}

// Placement identifies where the source writes a chunk. Fixed batches use Slot;
// variable batches use the packed sample Offset.
type Placement struct {
	Chunk  Chunk
	Slot   int
	Offset int
	Length int
}

// Batch mirrors the source membership and flush rules. Reset preserves the HAC
// part accumulator because the pinned worker thread keeps it across flushes.
type Batch struct {
	cfg        Config
	placements []Placement
	offset     int
	chunksSize int
	part       int
}

func NewBatch(cfg Config) (*Batch, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	b := &Batch{cfg: cfg}
	b.Reset()
	return b, nil
}

// Reset starts an empty batch. HAC keeps the part accumulator across flushes.
func (b *Batch) Reset() {
	b.placements = nil
	b.chunksSize = 0
	if b.cfg.Lane == SUP {
		b.offset = b.cfg.FirstConvPadding
		b.chunksSize = b.cfg.MaxConvPadding
	} else {
		b.offset = 0
	}
}

func (b *Batch) Placements() []Placement { return append([]Placement(nil), b.placements...) }
func (b *Batch) Real() int               { return len(b.placements) }
func (b *Batch) PackedSamples() int      { return b.offset }
func (b *Batch) Empty() bool             { return len(b.placements) == 0 }

func (b *Batch) capacity() int {
	return b.cfg.BatchSize * b.cfg.ChunkSize
}

// Full reports whether the source worker must flush before accepting another
// chunk.
func (b *Batch) Full() bool {
	if !b.cfg.VariableChunks {
		return len(b.placements) == b.cfg.BatchSize
	}
	if b.cfg.Lane == SUP {
		return b.chunksSize > b.capacity()
	}
	return b.chunksSize == b.maxChunksSize()
}

func (c Config) Transformer() bool { return c.Lane == SUP }

func (b *Batch) maxChunksSize() int {
	return b.cfg.BatchSize * ((b.cfg.ChunkSize / b.cfg.Stride) + 2)
}

func (b *Batch) partSize() int {
	return 32 * ((b.cfg.ChunkSize / b.cfg.Stride) + 2)
}

func roundUp(n, unit int) int {
	if n <= 0 || unit <= 0 || n > MaxCoordinate-unit {
		return 0
	}
	return (n + unit - 1) / unit * unit
}

// Add places one chunk. flushed reports that the source would dispatch the
// previous batch first. The chunk then belongs to the new batch.
func (b *Batch) Add(c Chunk) (flushed bool, err error) {
	if err := b.cfg.Validate(); err != nil {
		return false, err
	}
	if c.Length <= 0 || c.Length > b.cfg.ChunkSize {
		return false, fmt.Errorf("chunk %q length %d outside 1..%d", c.ID, c.Length, b.cfg.ChunkSize)
	}
	if b.Full() {
		return false, fmt.Errorf("batch is full")
	}
	if !b.cfg.VariableChunks {
		offset := len(b.placements) * b.cfg.ChunkSize
		b.offset = offset + b.cfg.ChunkSize
		b.placements = append(b.placements, Placement{Chunk: c, Slot: len(b.placements), Offset: offset, Length: b.cfg.ChunkSize})
		return false, nil
	}
	switch b.cfg.Lane {
	case HAC:
		if b.cfg.Transformer() {
			return false, fmt.Errorf("HAC has no transformer lane")
		}
		padded := roundUp(c.Length, b.cfg.Stride)
		if padded == 0 {
			return false, fmt.Errorf("chunk %q length overflows padding", c.ID)
		}
		slice := padded/b.cfg.Stride + 2
		if b.part+slice > b.partSize() {
			b.chunksSize += b.partSize()
			b.part = 0
		}
		if b.chunksSize == b.maxChunksSize() {
			b.Reset()
			flushed = true
		}
		b.part += slice
		offset := b.offset
		b.offset += padded
		if b.offset > b.capacity() {
			return false, fmt.Errorf("packed chunks exceed batch capacity")
		}
		b.placements = append(b.placements, Placement{Chunk: c, Offset: offset, Length: padded})
	case SUP:
		padded := roundUp(c.Length, b.cfg.Granularity)
		if padded == 0 {
			return false, fmt.Errorf("chunk %q length overflows padding", c.ID)
		}
		if b.chunksSize+padded+b.cfg.MaxConvPadding > b.capacity() {
			b.Reset()
			flushed = true
		}
		offset := b.offset
		b.chunksSize += padded + b.cfg.MaxConvPadding
		b.offset += padded + b.cfg.FirstConvPadding
		limit := b.capacity() + (b.cfg.BatchSize+1)*b.cfg.MaxConvPadding
		if b.offset > limit {
			return false, fmt.Errorf("packed chunks exceed padded input capacity")
		}
		b.placements = append(b.placements, Placement{Chunk: c, Offset: offset, Length: padded})
	default:
		return false, fmt.Errorf("unsupported model lane %d", b.cfg.Lane)
	}
	return flushed, nil
}
