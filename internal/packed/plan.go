// Package packed describes source host geometry only. It cannot authorize a
// device operation, bind a numeric program, or establish native memory spans.
package packed

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpubatch"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/scheduler"
)

type Samples int64
type ParentOffset uint64
type Blocks int64
type RecurrentLane int32
type RecurrentTime int64
type EncoderStep int64
type ScoreRow int64
type GuideRow int64
type OutputOffset int64
type HistoryRow int64
type Elements int64
type LogicalID int32

var ErrPlan = errors.New("packed: initialized host plan required")

// Pins are supplied by the model owner, not discovered from a model catalog.
// They identify source/config data; they do not authenticate executable code.
type Pins struct{ Source, Model, Config string }
type Convolution struct{ In, Out, Window, Stride int }
type DecoderParameters struct {
	Clamp, Blank, BeamCut, QScale, QShift float32
	BeamWidth                             int
	MovePad                               bool
}

// DefinitionInput is the interpreted model boundary. The later model adapter
// supplies these values from its validated config, including convolution order.
type DefinitionInput struct {
	Config       gpubatch.Config
	Pins         Pins
	MaxChunk     Samples // basecaller maximum, independent of this full/short queue
	StateLen     int
	Convolutions []Convolution
	Decoder      DecoderParameters
}
type Definition struct{ data *DefinitionInput }

func NewDefinition(in DefinitionInput) (Definition, error) {
	c := in.Config
	for _, v := range []int{c.ChunkSize, c.BatchSize, c.Stride, c.InnerStride, c.Granularity, c.FirstConvPadding, c.MaxConvPadding} {
		if v < 0 || int64(v) > gpubatch.MaxCoordinate {
			return Definition{}, fmt.Errorf("packed: config outside source integer domain")
		}
	}
	if err := c.Validate(); err != nil {
		return Definition{}, err
	}
	if !c.VariableChunks || in.Pins.Source == "" || in.Pins.Model == "" || in.Pins.Config == "" {
		return Definition{}, fmt.Errorf("packed: variable config and source/model/config pins required")
	}
	if in.MaxChunk < Samples(c.ChunkSize) || in.MaxChunk > gpubatch.MaxCoordinate || in.MaxChunk%Samples(c.Granularity) != 0 {
		return Definition{}, fmt.Errorf("packed: invalid configured maximum chunk")
	}
	capacity := int64(c.BatchSize) * int64(c.ChunkSize)
	if capacity > gpubatch.MaxCoordinate || c.Stride > gpubatch.MaxCoordinate || c.InnerStride > gpubatch.MaxCoordinate || c.Granularity > gpubatch.MaxCoordinate {
		return Definition{}, fmt.Errorf("packed: source capacity overflows")
	}
	wantState, wantConv := 4, 3
	if c.Lane == gpubatch.SUP {
		wantState, wantConv = 5, 5
	}
	if in.StateLen != wantState || len(in.Convolutions) != wantConv {
		return Definition{}, fmt.Errorf("packed: cross-family state length or convolution count")
	}
	g, channels, maxPadding := c.Granularity, 1, 0
	for i, conv := range in.Convolutions {
		if conv.In != channels || conv.Out <= 0 || conv.Out > gpubatch.MaxCoordinate || conv.Window <= 0 || conv.Window > gpubatch.MaxCoordinate || conv.Window%2 != 1 || conv.Stride <= 0 || g%conv.Stride != 0 {
			return Definition{}, fmt.Errorf("packed: invalid convolution %d", i)
		}
		if c.Lane == gpubatch.SUP {
			// ConvStack.cpp evaluates winlen*C_in as int before the tensor
			// view. Bound that product and the complete int64 tensor count.
			weightInputs := int64(conv.Window) * int64(conv.In)
			if weightInputs > gpubatch.MaxCoordinate || weightInputs > math.MaxInt64/int64(conv.Out) {
				return Definition{}, fmt.Errorf("packed: VCS weight product overflows %d", i)
			}
			// The output view uses C_out/8; the first weight view uses
			// C_out/2. Later views tile both axes by 16, with final SMEM_N=128.
			if conv.Out%8 != 0 || (i == 0 && conv.Out%2 != 0) || (i > 0 && (weightInputs%16 != 0 || conv.Out%16 != 0)) || (i == 4 && conv.Out%128 != 0) {
				return Definition{}, fmt.Errorf("packed: unsupported VCS weight tiling %d", i)
			}
		}
		g /= conv.Stride
		channels = conv.Out
		maxPadding = max(maxPadding, conv.Window/2)
	}
	if c.Granularity/g != c.InnerStride || (c.Lane == gpubatch.SUP && (g != 64 || in.Convolutions[4].Stride <= 1)) {
		return Definition{}, fmt.Errorf("packed: convolution strides disagree with inner stride")
	}
	if c.Lane == gpubatch.SUP && (c.FirstConvPadding != in.Convolutions[0].Window/2 || c.MaxConvPadding != maxPadding) {
		return Definition{}, fmt.Errorf("packed: first/capacity gaps disagree with convolution padding")
	}
	p := in.Decoder
	for _, v := range []float32{p.Clamp, p.Blank, p.BeamCut, p.QScale, p.QShift} {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return Definition{}, fmt.Errorf("packed: nonfinite decoder scalar")
		}
	}
	if p.Clamp < 0 || p.BeamWidth != 32 || p.BeamCut != 100 {
		return Definition{}, fmt.Errorf("packed: nonnegative clamp, beam32 and cut100 required")
	}
	// Bound all products used by the existing int-based auxiliary authority.
	if c.Lane == gpubatch.HAC {
		l := int64(c.ChunkSize/c.Stride + 2)
		if l%2 != 0 {
			l++
		}
		if int64(c.BatchSize)*(l+5) > gpubatch.MaxCoordinate || 6*int64(c.BatchSize)*(int64(c.ChunkSize/c.Stride)+3) > gpubatch.MaxCoordinate {
			return Definition{}, fmt.Errorf("packed: recurrent source capacity overflows")
		}
		// CUDADecoder.cpp: Ts >= NT_out_max + 4*N, regardless of
		// membership. Reject impossible scratch before any packing arrays.
		states := int64(1) << (2 * in.StateLen)
		minPath := capacity/int64(c.Stride) + 4*int64(c.BatchSize)
		if minPath > gpubatch.MaxCoordinate/(4*states+4*int64(p.BeamWidth)) {
			return Definition{}, fmt.Errorf("packed: minimum decoder scratch capacity exceeded")
		}
	}

	in.Convolutions = slices.Clone(in.Convolutions)
	return Definition{&in}, nil
}

func (d Definition) Input() (DefinitionInput, error) {
	if d.data == nil {
		return DefinitionInput{}, ErrPlan
	}
	in := *d.data
	in.Convolutions = slices.Clone(in.Convolutions)
	return in, nil
}

// Member carries no samples or tensor offsets. Parent identity and offset are
// copied independently from scheduler placement and remain publication inputs.
type Member struct {
	ChunkID, ParentID string
	ParentOffset      ParentOffset
	Raw               Samples
}
type Segment struct {
	Logical                       LogicalID
	ScoreStart, Steps             ScoreRow
	GuideStart, Terminal          GuideRow
	HistoryStart                  HistoryRow
	HistoryRows                   HistoryRow
	OutputStart                   OutputOffset
	PosteriorPrefix, ScoreBacking Elements
}
type PlannedMember struct {
	Member
	Rounded, InputStart Samples
	Segment             Segment
}
type Extent struct{ Initialized, Capacity Elements }
type HACGeometry struct {
	Lanes                                                  RecurrentLane
	ForwardT, ReverseT, BackingT                           RecurrentTime
	ProjectionRow                                          RecurrentTime
	ZeroRows                                               [2]RecurrentTime
	InLayout, OutLayout, ForwardEncoding, BackwardEncoding []int32
	InExtent, OutExtent, ForwardExtent, BackwardExtent     Extent
	// Only pristine sample intervals are bound. Device revisions are unknown.
	SampleIntervals [][2]Samples
}
type SUPConvolution struct {
	Definition                    Convolution
	GranularityIn, GranularityOut Samples
	MInput, MOutput               int64
	Padding, NextPadding          int
	LUTEntries                    Blocks
	// MOutput is a source view size, not initialized or allocated capacity.
}
type SUPGeometry struct {
	Active, Padded, Maximum Blocks
	EncoderT, MaxPosition   EncoderStep
	ScoreRowsPerBlock       ScoreRow
	FirstGap, CapacityGap   Samples
	SampleTable             [][2]Samples
	DecoderTable            [][2]ScoreRow
	Convolutions            []SUPConvolution
	// Source allocation is zero-filled before native LUT fill; no fill is run.
	RopeZeroFillEntries Blocks
}
type graph interface{ packedGraph() }
type hacGraph struct{ HACGeometry }
type supGraph struct{ SUPGeometry }

func (hacGraph) packedGraph() {}
func (supGraph) packedGraph() {}

type InputGeometry struct {
	Raw, Rounded, Physical, Capacity Samples
	// Capacity charge uses MaxConvPadding; Physical advances FirstConvPadding.
	CapacityCharge Samples
}
type ActiveRow struct {
	Member        LogicalID
	Local, Source ScoreRow
	Guide         GuideRow
}

// FrameOffsets is CSR by ascending local time. Reverse guide traversal uses the
// same rows in reverse time; row order does not decide beam collision ordering.
type ActiveFrames struct {
	FrameOffsets []int
	Rows         []ActiveRow
}
type NumericRequirement struct {
	Stage                 string
	States, ScoreChannels int64
	MaxActive             int
	ScoreType, GuideType  string
}
type NativeReference struct {
	DescriptorRows, DescriptorInitializedRows   int64
	PathElements, AuxBytes, OutputPlaneCapacity int64
}
type DecoderGeometry struct {
	Parameters                                   DecoderParameters
	States, ScoreChannels                        int64
	ScoreRows, ScoreCapacity                     ScoreRow
	GuideRows                                    GuideRow
	OutputCapacity                               OutputOffset
	GuideElements, AlphaElements, HistoryRecords Elements
	Requirements                                 []NumericRequirement
	Native                                       NativeReference // allocation formulas only, never native read masks
}
type Plan struct{ data *planData }
type planData struct {
	definition Definition
	sequence   int
	members    []PlannedMember
	input      InputGeometry
	graph      graph
	decoder    DecoderGeometry
	frames     ActiveFrames
}

// PlanBatch validates the maintained scheduler handoff. It does not replay
// NewBatch/Add: only scheduler owns persistent HAC subbatch admission state.
func PlanBatch(d Definition, b scheduler.Batch, identities []Member) (Plan, error) {
	if d.data == nil {
		return Plan{}, ErrPlan
	}
	in := d.data
	c := in.Config
	capacity := c.BatchSize * c.ChunkSize
	if b.Sequence <= 0 || !b.Variable || b.Lane != c.Lane || b.PaddedSlots != c.BatchSize || b.PaddedSamples != capacity || b.Real <= 0 || b.Real != len(b.Placements) || b.Real != len(identities) || int64(b.Real) > gpubatch.MaxCoordinate {
		return Plan{}, fmt.Errorf("packed: incompatible scheduler handoff")
	}
	byID := make(map[string]Member, len(identities))
	for _, m := range identities {
		if m.ChunkID == "" || m.ParentID == "" || m.Raw <= 0 || m.Raw > Samples(c.ChunkSize) || uint64(m.ParentOffset) > math.MaxUint64-uint64(m.Raw) {
			return Plan{}, fmt.Errorf("packed: invalid parent/chunk identity")
		}
		if _, exists := byID[m.ChunkID]; exists {
			return Plan{}, fmt.Errorf("packed: duplicate identity %q", m.ChunkID)
		}
		byID[m.ChunkID] = m
	}
	p := &planData{definition: d, sequence: b.Sequence}
	sizes := make([]int, len(b.Placements))
	gap, chargeGap := int64(c.FirstConvPadding), int64(c.MaxConvPadding)
	offset, charge, raw, rounded := gap, chargeGap, int64(0), int64(0)
	states := int64(1) << (2 * in.StateLen)
	var score, guide, maxSteps int64
	for i, placement := range b.Placements {
		m, ok := byID[placement.Chunk.ID]
		if !ok || int64(placement.Chunk.Length) != int64(m.Raw) {
			return Plan{}, fmt.Errorf("packed: missing or repeated membership %q", placement.Chunk.ID)
		}
		delete(byID, placement.Chunk.ID)
		r := (int64(m.Raw) + int64(c.Granularity) - 1) / int64(c.Granularity) * int64(c.Granularity)
		if r > int64(c.ChunkSize) || placement.Slot != 0 || int64(placement.Length) != r || int64(placement.Offset) != offset {
			return Plan{}, fmt.Errorf("packed: source rounded placement mismatch for %q", m.ChunkID)
		}
		t := r / int64(c.Stride)
		if t < 1 || (t+1)*states > t*4*states {
			return Plan{}, fmt.Errorf("packed: unsafe posterior prefix")
		}
		s := Segment{Logical: LogicalID(i), ScoreStart: ScoreRow(score), Steps: ScoreRow(t), GuideStart: GuideRow(guide), Terminal: GuideRow(guide + t), HistoryStart: HistoryRow(guide), HistoryRows: HistoryRow(t + 1), OutputStart: OutputOffset(score), PosteriorPrefix: Elements((t + 1) * states), ScoreBacking: Elements(t * 4 * states)}
		p.members = append(p.members, PlannedMember{Member: m, Rounded: Samples(r), InputStart: Samples(offset), Segment: s})
		sizes[i] = int(r)
		offset += r + gap
		charge += r + chargeGap
		raw += int64(m.Raw)
		rounded += r
		score += t
		guide += t + 1
		if t > maxSteps {
			maxSteps = t
		}
		if rounded > int64(capacity) || offset > gpubatch.MaxCoordinate || charge > int64(capacity) || guide > gpubatch.MaxCoordinate {
			return Plan{}, fmt.Errorf("packed: input or decoder source capacity exceeded")
		}
	}
	if len(byID) != 0 {
		return Plan{}, fmt.Errorf("packed: unused identities")
	}
	a, err := gpubatch.NewAux(c, sizes)
	if err != nil {
		return Plan{}, err
	}
	// Ordinary CudaModelRunner owns [1,1,N*chunk], unlike the separately
	// padded create_empty_input benchmark helper. Capacity charge ensures gaps
	// fit this ordinary runner allocation; do not manufacture extra backing.
	inputCapacity := int64(capacity)
	if offset > inputCapacity {
		return Plan{}, fmt.Errorf("packed: physical input exceeds source allocation")
	}
	p.input = InputGeometry{Samples(raw), Samples(rounded), Samples(offset), Samples(inputCapacity), Samples(charge)}
	dec := DecoderGeometry{Parameters: in.Decoder, States: states, ScoreChannels: 4 * states, ScoreRows: ScoreRow(score), GuideRows: GuideRow(guide), OutputCapacity: OutputOffset(score), GuideElements: Elements(guide * states), AlphaElements: Elements(2 * int64(b.Real) * states), HistoryRecords: Elements(guide * int64(in.Decoder.BeamWidth))}
	if c.Lane == gpubatch.HAC {
		n := c.BatchSize
		dec.ScoreCapacity = ScoreRow(capacity / c.Stride)
		m := int64(capacity / c.Stride)
		ts := max(m+4*int64(n), score+int64(b.Real))
		dec.Native = NativeReference{max(int64(b.Real), 4*int64(n)), int64(b.Real), ts, ts * (4*states + 4*int64(in.Decoder.BeamWidth)), m}
	} else {
		padded, err := a.QKVRopeLUTSize()
		if err != nil {
			return Plan{}, err
		}
		if padded > a.Max() {
			return Plan{}, fmt.Errorf("packed: padded encoder blocks exceed source maximum")
		}
		s := SUPGeometry{Active: Blocks(a.Total()), Padded: Blocks(padded), Maximum: Blocks(a.Max()), EncoderT: EncoderStep(c.EncoderT()), MaxPosition: EncoderStep(in.MaxChunk / Samples(c.InnerStride)), ScoreRowsPerBlock: ScoreRow(c.Granularity / c.Stride), FirstGap: Samples(gap), CapacityGap: Samples(chargeGap), RopeZeroFillEntries: Blocks(padded)}
		table, err := a.DeviceTable()
		if err != nil {
			return Plan{}, err
		}
		for i := 0; i < len(table); i += 2 {
			s.SampleTable = append(s.SampleTable, [2]Samples{Samples(table[i]), Samples(table[i+1])})
			s.DecoderTable = append(s.DecoderTable, [2]ScoreRow{ScoreRow(int64(table[i]) / int64(c.Stride)), ScoreRow(int64(table[i+1]) / int64(c.Stride))})
		}
		// TxModel receives the full runner input; unlike HAC it does not narrow
		// to the populated prefix. Unpopulated input is not declared initialized.
		m, g := inputCapacity, int64(c.Granularity)
		for i, conv := range in.Convolutions {
			next := 0
			if i+1 < len(in.Convolutions) {
				next = in.Convolutions[i+1].Window / 2
			}
			out := m
			if conv.Stride > 1 {
				out = m/int64(conv.Stride) + int64(a.Max())*int64(next)
				if i == len(in.Convolutions)-1 {
					out = int64(padded) * int64(c.EncoderT())
				}
			}
			if out > gpubatch.MaxCoordinate || out*int64(conv.Out) > gpubatch.MaxCoordinate {
				return Plan{}, fmt.Errorf("packed: VCS convolution extent overflows")
			}
			s.Convolutions = append(s.Convolutions, SUPConvolution{conv, Samples(g), Samples(g / int64(conv.Stride)), m, out, conv.Window / 2, next, Blocks(a.Total())})
			m, g = out, g/int64(conv.Stride)
		}
		p.graph = supGraph{s}
		dec.ScoreCapacity = ScoreRow(int64(padded) * int64(s.ScoreRowsPerBlock))
		path := int64(a.Total()) * (int64(s.ScoreRowsPerBlock) + 1)
		dec.Native = NativeReference{int64(b.Real), int64(b.Real), path, path * (4*states + 4*int64(in.Decoder.BeamWidth)), int64(a.Total()) * int64(s.ScoreRowsPerBlock)}
	}
	if score > int64(dec.ScoreCapacity) || dec.Native.AuxBytes > gpubatch.MaxCoordinate {
		return Plan{}, fmt.Errorf("packed: source score/scratch capacity exceeded")
	}
	if c.Lane == gpubatch.HAC {
		// All membership-dependent scalar bounds precede the four-array
		// allocation. PackLSTM remains the sole packing/admission authority.
		pack, err := a.PackLSTM()
		if err != nil {
			return Plan{}, err
		}
		l, n := a.TLSTM(), c.BatchSize
		h := HACGeometry{Lanes: RecurrentLane(n), ForwardT: RecurrentTime(l), ReverseT: RecurrentTime(l + 1), BackingT: RecurrentTime(l + 5), ProjectionRow: 3, ZeroRows: [2]RecurrentTime{RecurrentTime(l + 2), RecurrentTime(l + 3)}, InLayout: pack.InLayout, OutLayout: pack.OutLayout, ForwardEncoding: pack.ForwardEncoding, BackwardEncoding: pack.BackwardEncoding, InExtent: Extent{Elements(len(pack.InLayout)), Elements(len(pack.InLayout))}, OutExtent: Extent{Elements(len(pack.OutLayout)), Elements(pack.OutLayoutCapacity)}, ForwardExtent: Extent{Elements(len(pack.ForwardEncoding)), Elements(len(pack.ForwardEncoding))}, BackwardExtent: Extent{Elements(len(pack.BackwardEncoding)), Elements(len(pack.BackwardEncoding))}}
		for _, iv := range a.Intervals() {
			h.SampleIntervals = append(h.SampleIntervals, [2]Samples{Samples(iv[0]), Samples(iv[1])})
		}
		p.graph = hacGraph{h}
	}
	for _, stage := range []string{"guides", "beam", "posteriors", "quality"} {
		dec.Requirements = append(dec.Requirements, NumericRequirement{stage, states, 4 * states, b.Real, "f16", "f32"})
	}
	p.decoder = dec
	p.frames.FrameOffsets = make([]int, int(maxSteps)+1)
	// Histogram segment ends, prefix frame counts, then fill in member order.
	// Work is O(logicalCount + maxSteps + sumSteps), including unequal lengths.
	ends := make([]int, int(maxSteps)+1)
	for _, member := range p.members {
		ends[member.Segment.Steps]++
	}
	active := len(p.members)
	for t := 0; t < int(maxSteps); t++ {
		active -= ends[t]
		p.frames.FrameOffsets[t+1] = p.frames.FrameOffsets[t] + active
	}
	p.frames.Rows = make([]ActiveRow, int(score))
	cursors := slices.Clone(p.frames.FrameOffsets[:int(maxSteps)])
	for _, member := range p.members {
		s := member.Segment
		for t := ScoreRow(0); t < s.Steps; t++ {
			p.frames.Rows[cursors[t]] = ActiveRow{s.Logical, t, s.ScoreStart + t, s.GuideStart + GuideRow(t)}
			cursors[t]++
		}
	}
	return Plan{p}, nil
}

func (p Plan) Validate() error {
	if p.data == nil {
		return ErrPlan
	}
	return nil
}
func (p Plan) Definition() (Definition, error) {
	if p.data == nil {
		return Definition{}, ErrPlan
	}
	return p.data.definition, nil
}
func (p Plan) Sequence() (int, error) {
	if p.data == nil {
		return 0, ErrPlan
	}
	return p.data.sequence, nil
}
func (p Plan) Members() ([]PlannedMember, error) {
	if p.data == nil {
		return nil, ErrPlan
	}
	return slices.Clone(p.data.members), nil
}
func (p Plan) Input() (InputGeometry, error) {
	if p.data == nil {
		return InputGeometry{}, ErrPlan
	}
	return p.data.input, nil
}
func (p Plan) Decoder() (DecoderGeometry, error) {
	if p.data == nil {
		return DecoderGeometry{}, ErrPlan
	}
	d := p.data.decoder
	d.Requirements = slices.Clone(d.Requirements)
	return d, nil
}
func (p Plan) Frames() (ActiveFrames, error) {
	if p.data == nil {
		return ActiveFrames{}, ErrPlan
	}
	return ActiveFrames{slices.Clone(p.data.frames.FrameOffsets), slices.Clone(p.data.frames.Rows)}, nil
}
func (p Plan) HAC() (HACGeometry, error) {
	if p.data == nil {
		return HACGeometry{}, ErrPlan
	}
	g, ok := p.data.graph.(hacGraph)
	if !ok {
		return HACGeometry{}, fmt.Errorf("packed: HAC geometry requested for SUP")
	}
	h := g.HACGeometry
	h.InLayout = slices.Clone(h.InLayout)
	h.OutLayout = slices.Clone(h.OutLayout)
	h.ForwardEncoding = slices.Clone(h.ForwardEncoding)
	h.BackwardEncoding = slices.Clone(h.BackwardEncoding)
	h.SampleIntervals = slices.Clone(h.SampleIntervals)
	return h, nil
}
func (p Plan) SUP() (SUPGeometry, error) {
	if p.data == nil {
		return SUPGeometry{}, ErrPlan
	}
	g, ok := p.data.graph.(supGraph)
	if !ok {
		return SUPGeometry{}, fmt.Errorf("packed: SUP geometry requested for HAC")
	}
	s := g.SUPGeometry
	s.SampleTable = slices.Clone(s.SampleTable)
	s.DecoderTable = slices.Clone(s.DecoderTable)
	s.Convolutions = slices.Clone(s.Convolutions)
	return s, nil
}

type SourceCoordinate struct {
	Block Blocks
	Local ScoreRow
}

// SourceOrder maps only logical rows, excluding SUP padded score capacity.
// HAC is compact [1,Q,C]; SUP is [paddedBlocks,rowsPerBlock,C].
func (p Plan) SourceOrder(row ScoreRow) (SourceCoordinate, error) {
	if p.data == nil {
		return SourceCoordinate{}, ErrPlan
	}
	if row < 0 || row >= p.data.decoder.ScoreRows {
		return SourceCoordinate{}, fmt.Errorf("packed: score row outside logical source")
	}
	if s, ok := p.data.graph.(supGraph); ok {
		return SourceCoordinate{Blocks(row / s.ScoreRowsPerBlock), row % s.ScoreRowsPerBlock}, nil
	}
	return SourceCoordinate{0, row}, nil
}

// UnboundContracts lists evidence obligations, not zero-valued executable
// spans. There is intentionally no NativeContract or program ID constructor.
func (p Plan) UnboundContracts() ([]string, error) {
	if p.data == nil {
		return nil, ErrPlan
	}
	return []string{"kernel read/write and initialization spans", "device interval revisions and replay restoration", "attention read windows", "decoder posterior native physical map and scratch", "beam collision ordering", "numeric program identities and segmented shape qualification", "allocation/stream/weight ownership and completion"}, nil
}
