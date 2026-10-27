package gpubatch

import "fmt"

// Aux ports nn::AuxiliaryData for one packed batch. Tables, intervals and
// layout sizes are derived from the real chunk lengths in source units.
type Aux struct {
	cfg       Config
	sizes     []int
	sizesOut  []int
	table     [][2]int
	intervals [][2]int
	total     int
	max       int
	tLSTM     int
}

func NewAux(cfg Config, chunkSizes []int) (*Aux, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if !cfg.VariableChunks {
		return nil, fmt.Errorf("auxiliary data requires variable chunks")
	}
	if len(chunkSizes) == 0 {
		return nil, fmt.Errorf("empty chunk batch")
	}
	a := &Aux{
		cfg:   cfg,
		sizes: make([]int, len(chunkSizes)),
		table: make([][2]int, len(chunkSizes)),
	}
	a.max = cfg.BatchSize * cfg.ChunkSize / cfg.Granularity
	total := 0
	for i, size := range chunkSizes {
		if size <= 0 || size > cfg.ChunkSize || size%cfg.Granularity != 0 {
			return nil, fmt.Errorf("chunk %d size %d is not a positive multiple of %d within %d", i, size, cfg.Granularity, cfg.ChunkSize)
		}
		blocks := size / cfg.Granularity
		if total > MaxCoordinate-blocks {
			return nil, fmt.Errorf("chunk granularity total overflows")
		}
		a.table[i] = [2]int{total, blocks}
		a.sizes[i] = size
		total += blocks
	}
	if total > a.max {
		return nil, fmt.Errorf("total granularity %d exceeds maximum %d", total, a.max)
	}
	a.total = total
	a.intervals = make([][2]int, len(chunkSizes))
	for i, entry := range a.table {
		a.intervals[i] = [2]int{entry[0] * cfg.Granularity, (entry[0] + entry[1]) * cfg.Granularity}
	}
	if cfg.Lane == HAC {
		a.sizesOut = make([]int, len(chunkSizes))
		for i, size := range a.sizes {
			a.sizesOut[i] = size / cfg.Stride
		}
		tOut := cfg.ChunkSize / cfg.Stride
		a.tLSTM = 1 + tOut + 1
		if a.tLSTM%2 != 0 {
			a.tLSTM++
		}
	}
	return a, nil
}

func (a *Aux) ChunkTable() [][2]int { return append([][2]int(nil), a.table...) }
func (a *Aux) Intervals() [][2]int  { return append([][2]int(nil), a.intervals...) }
func (a *Aux) Total() int           { return a.total }
func (a *Aux) Max() int             { return a.max }
func (a *Aux) Sizes() []int         { return append([]int(nil), a.sizes...) }
func (a *Aux) SizesOut() []int      { return append([]int(nil), a.sizesOut...) }
func (a *Aux) TLSTM() int           { return a.tLSTM }

func (a *Aux) PackLSTM() (LSTMPacking, error) {
	if a == nil || a.cfg.Lane != HAC {
		return LSTMPacking{}, fmt.Errorf("LSTM packing is HAC-only")
	}
	workspace, err := a.Workspace()
	if err != nil {
		return LSTMPacking{}, err
	}
	return packLSTM(a.cfg.BatchSize, a.tLSTM, workspace, a.sizesOut)
}

// DeviceTable is the flattened int32 table the transformer lane uploads. The
// pinned source scales both columns by the chunk granularity.
func (a *Aux) DeviceTable() ([]int32, error) {
	if a.cfg.Lane != SUP {
		return nil, fmt.Errorf("device table is transformer-only")
	}
	out := make([]int32, 0, 2*len(a.table))
	for _, entry := range a.table {
		start, length := entry[0]*a.cfg.Granularity, entry[1]*a.cfg.Granularity
		if start > MaxCoordinate || length > MaxCoordinate {
			return nil, fmt.Errorf("scaled device table overflows int32")
		}
		out = append(out, int32(start), int32(length))
	}
	return out, nil
}

// RestoredIntervals is the HAC convolution postprocess table after the source
// multiplies every interval by stride_in.
func (a *Aux) RestoredIntervals() ([][2]int, error) {
	if a.cfg.Lane != HAC {
		return nil, fmt.Errorf("restored intervals are LSTM-only")
	}
	out := make([][2]int, len(a.intervals))
	for i, iv := range a.intervals {
		start, end := iv[0]*a.cfg.InnerStride, iv[1]*a.cfg.InnerStride
		if start > MaxCoordinate || end > MaxCoordinate {
			return nil, fmt.Errorf("restored interval overflows int32")
		}
		out[i] = [2]int{start, end}
	}
	return out, nil
}

// QKVRopeLUTSize is padded to the Koi MatMul multiple of 256 input values.
func (a *Aux) QKVRopeLUTSize() (int, error) {
	if a.cfg.Lane != SUP {
		return 0, fmt.Errorf("QKV RoPE LUT is transformer-only")
	}
	t := a.cfg.EncoderT()
	if t <= 0 || 256%t != 0 {
		return 0, fmt.Errorf("encoder granularity %d does not divide 256", t)
	}
	p := 256 / t
	if a.total > MaxCoordinate-p {
		return 0, fmt.Errorf("QKV RoPE LUT size overflows")
	}
	return (a.total + p - 1) / p * p, nil
}

// LSTMLayout returns the pinned auxiliary allocation sizes. All four are
// expressed in int32 elements.
func (a *Aux) LSTMLayout() (chunkSum, outLayout, forward, backward int, err error) {
	if a.cfg.Lane != HAC {
		return 0, 0, 0, 0, fmt.Errorf("LSTM layout is HAC-only")
	}
	for _, n := range a.sizesOut {
		if chunkSum > MaxCoordinate-n {
			return 0, 0, 0, 0, fmt.Errorf("LSTM chunk sum overflows")
		}
		chunkSum += n
	}
	n := a.cfg.BatchSize
	if n > MaxCoordinate/a.tLSTM {
		return 0, 0, 0, 0, fmt.Errorf("LSTM layout overflows")
	}
	outLayout = n * (a.tLSTM + 1)
	forward = n * a.tLSTM
	backward = outLayout
	return chunkSum, outLayout, forward, backward, nil
}

// Workspace is the pinned Koi LSTM workspace element count.
func (a *Aux) Workspace() (int, error) {
	if a.cfg.Lane != HAC {
		return 0, fmt.Errorf("LSTM workspace is HAC-only")
	}
	t := a.cfg.ChunkSize/a.cfg.Stride + 3
	if t > MaxCoordinate/6 || 6*t > MaxCoordinate/a.cfg.BatchSize {
		return 0, fmt.Errorf("LSTM workspace overflows")
	}
	return 6 * t * a.cfg.BatchSize, nil
}
