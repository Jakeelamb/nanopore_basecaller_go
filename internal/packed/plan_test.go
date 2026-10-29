package packed_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/gpubatch"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/packed"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/scheduler"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacvcsaux"
)

func definitionInput(lane gpubatch.Lane) packed.DefinitionInput {
	in := packed.DefinitionInput{
		Config:   gpubatch.Config{Lane: lane, ChunkSize: 192, BatchSize: 64, Stride: 6, InnerStride: 6, Granularity: 6, VariableChunks: true},
		Pins:     packed.Pins{Source: "source-test", Model: "model-test", Config: "config-test"},
		MaxChunk: 12288, StateLen: 4,
		Convolutions: []packed.Convolution{{In: 1, Out: 16, Window: 5, Stride: 1}, {In: 16, Out: 16, Window: 5, Stride: 1}, {In: 16, Out: 1024, Window: 19, Stride: 6}},
		Decoder:      packed.DecoderParameters{Clamp: 5, Blank: 2, BeamCut: 100, BeamWidth: 32, QScale: 1.02, QShift: -.3, MovePad: true},
	}
	if lane == gpubatch.SUP {
		in.Config = gpubatch.Config{Lane: lane, ChunkSize: 1536, BatchSize: 32, Stride: 6, InnerStride: 12, Granularity: 768, FirstConvPadding: 2, MaxConvPadding: 4, VariableChunks: true}
		in.StateLen = 5
		in.Convolutions = []packed.Convolution{{In: 1, Out: 16, Window: 5, Stride: 1}, {In: 16, Out: 32, Window: 5, Stride: 1}, {In: 32, Out: 64, Window: 9, Stride: 3}, {In: 64, Out: 128, Window: 5, Stride: 2}, {In: 128, Out: 512, Window: 5, Stride: 2}}
	}
	return in
}

func mustDefinition(t *testing.T, in packed.DefinitionInput) packed.Definition {
	t.Helper()
	d, err := packed.NewDefinition(in)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// Use the maintained batch former; tests do not duplicate its admission policy.
func handoff(t *testing.T, cfg gpubatch.Config, lengths ...int) (scheduler.Batch, []packed.Member) {
	t.Helper()
	b, err := gpubatch.NewBatch(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var members []packed.Member
	for i, length := range lengths {
		id := fmt.Sprintf("chunk-%d", i)
		flushed, err := b.Add(gpubatch.Chunk{ID: id, Length: length})
		if err != nil || flushed {
			t.Fatalf("unexpected handoff flush/error: %v %v", flushed, err)
		}
		members = append(members, packed.Member{ChunkID: id, ParentID: "parent", ParentOffset: packed.ParentOffset(100000 + i*10000), Raw: packed.Samples(length)})
	}
	return scheduler.Batch{Sequence: 1, Lane: cfg.Lane, Variable: true, Real: b.Real(), PaddedSlots: cfg.BatchSize, PaddedSamples: cfg.BatchSize * cfg.ChunkSize, Placements: b.Placements()}, members
}

func TestHACHostArraysAndSegments(t *testing.T) {
	in := definitionInput(gpubatch.HAC)
	b, ids := handoff(t, in.Config, 1, 7, 192)
	d := mustDefinition(t, in)
	p, err := packed.PlanBatch(d, b, ids)
	if err != nil {
		t.Fatal(err)
	}
	members, err := p.Members()
	if err != nil {
		t.Fatal(err)
	}
	aux, err := gpubatch.NewAux(in.Config, []int{6, 12, 192})
	if err != nil {
		t.Fatal(err)
	}
	want, err := aux.PackLSTM()
	if err != nil {
		t.Fatal(err)
	}
	h, err := p.HAC()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(h.InLayout, want.InLayout) || !reflect.DeepEqual(h.OutLayout, want.OutLayout) || !reflect.DeepEqual(h.ForwardEncoding, want.ForwardEncoding) || !reflect.DeepEqual(h.BackwardEncoding, want.BackwardEncoding) {
		t.Fatal("four-array authority differs")
	}
	if h.OutExtent.Initialized != packed.Elements(64*34) || h.OutExtent.Capacity != packed.Elements(64*35) || len(h.OutLayout) != int(h.OutExtent.Initialized) {
		t.Fatal("OutLayout tail was manufactured")
	}
	if h.InExtent.Initialized != 35 || h.InExtent.Capacity != 35 || h.ForwardExtent != (packed.Extent{Initialized: 64 * 34, Capacity: 64 * 34}) || h.BackwardExtent != (packed.Extent{Initialized: 64 * 35, Capacity: 64 * 35}) {
		t.Fatal("independent upload extents")
	}
	if h.ForwardT != 34 || h.ReverseT != 35 || h.BackingT != 39 || h.ProjectionRow != 3 || h.ZeroRows != [2]packed.RecurrentTime{36, 37} {
		t.Fatal("source recurrent geometry")
	}
	g, err := p.Input()
	if err != nil {
		t.Fatal(err)
	}
	if g.Raw != 200 || g.Rounded != 210 || g.Physical != 210 || g.Capacity != 12288 {
		t.Fatalf("input %+v", g)
	}
	var score packed.ScoreRow
	var guide packed.GuideRow
	for i, m := range members {
		s := m.Segment
		if m.ParentOffset != ids[i].ParentOffset || m.ParentID != ids[i].ParentID || m.InputStart == packed.Samples(m.ParentOffset) {
			t.Fatal("parent coordinates conflated")
		}
		if s.ScoreStart != score || s.OutputStart != packed.OutputOffset(score) || s.GuideStart != guide || s.Terminal != guide+packed.GuideRow(s.Steps) || s.HistoryRows != packed.HistoryRow(s.Steps+1) || s.HistoryStart != packed.HistoryRow(guide) || s.PosteriorPrefix > s.ScoreBacking {
			t.Fatalf("segment %+v", s)
		}
		score += s.Steps
		guide += packed.GuideRow(s.Steps + 1)
	}
	dec, err := p.Decoder()
	if err != nil {
		t.Fatal(err)
	}
	if dec.Parameters != in.Decoder || dec.ScoreRows != score || dec.GuideRows != guide || dec.GuideElements != packed.Elements(guide)*256 || dec.AlphaElements != 2*3*256 || dec.HistoryRecords != packed.Elements(guide)*32 || len(dec.Requirements) != 4 || dec.OutputCapacity != packed.OutputOffset(score) {
		t.Fatal("decoder derivation")
	}
	if dec.Native.DescriptorRows != 256 || dec.Native.DescriptorInitializedRows != 3 || dec.Native.OutputPlaneCapacity != 2048 || dec.Native.PathElements != 2304 {
		t.Fatal("native source allocation reference")
	}
	assertFrames(t, p, members)
	if _, err := p.SUP(); err == nil {
		t.Fatal("cross-family getter")
	}
	if _, err := p.SourceOrder(score); err == nil {
		t.Fatal("logical score terminal exposed")
	}
	for row := packed.ScoreRow(0); row < score; row++ {
		c, err := p.SourceOrder(row)
		if err != nil || c.Block != 0 || c.Local != row {
			t.Fatal("HAC source order")
		}
	}
}

func assertFrames(t *testing.T, p packed.Plan, members []packed.PlannedMember) {
	t.Helper()
	f, err := p.Frames()
	if err != nil {
		t.Fatal(err)
	}
	if len(f.FrameOffsets) == 0 || f.FrameOffsets[0] != 0 {
		t.Fatal("invalid CSR start")
	}
	seen := map[[2]int64]bool{}
	for local := 0; local+1 < len(f.FrameOffsets); local++ {
		if f.FrameOffsets[local] > f.FrameOffsets[local+1] || f.FrameOffsets[local+1] > len(f.Rows) {
			t.Fatal("invalid CSR offsets")
		}
		rows := f.Rows[f.FrameOffsets[local]:f.FrameOffsets[local+1]]
		previous := packed.LogicalID(-1)
		for _, row := range rows {
			if row.Member <= previous || int(row.Member) >= len(members) {
				t.Fatal("CSR logical order")
			}
			previous = row.Member
			s := members[row.Member].Segment
			key := [2]int64{int64(row.Member), int64(local)}
			if seen[key] || row.Local != packed.ScoreRow(local) || row.Local >= s.Steps || row.Source != s.ScoreStart+row.Local || row.Guide != s.GuideStart+packed.GuideRow(row.Local) {
				t.Fatal("invalid active row")
			}
			seen[key] = true
		}
	}
	dec, err := p.Decoder()
	if err != nil || len(seen) != int(dec.ScoreRows) || f.FrameOffsets[len(f.FrameOffsets)-1] != len(f.Rows) {
		t.Fatal("active rows do not cover segments")
	}
}

func TestHACDefinitionRejectsScratchBeforePacking(t *testing.T) {
	// This checks definitions only: never hand the rejected N to a packer.
	// N8388608/L34 fits input, recurrent backing and workspace scalar bounds.
	const bytesPerPath = 4*256 + 4*32
	const pathPerLane = 192/6 + 4
	const largestBatch = (gpubatch.MaxCoordinate / bytesPerPath / pathPerLane / 64) * 64
	for _, n := range []int{64, largestBatch, largestBatch + 64, 8388608} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			in := definitionInput(gpubatch.HAC)
			in.Config.BatchSize = n
			if int64(n)*192 > gpubatch.MaxCoordinate || int64(n)*39 > gpubatch.MaxCoordinate || 6*int64(n)*35 > gpubatch.MaxCoordinate {
				t.Fatal("control must reach decoder scratch validation")
			}
			_, err := packed.NewDefinition(in)
			if n <= largestBatch {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), "minimum decoder scratch") {
				t.Fatalf("expected definition scratch rejection, got %v", err)
			}
		})
	}
}

func TestSUPDefinitionWeightTilings(t *testing.T) {
	for _, tc := range []struct {
		name      string
		outputs   [5]int
		wantError string
	}{
		{"minimum-later-tiles", [5]int{16, 16, 16, 16, 128}, ""},
		{"profile", [5]int{16, 32, 64, 128, 512}, ""},
		{"first-larger-tile", [5]int{32, 16, 16, 16, 256}, ""},
		{"later-output-24", [5]int{16, 24, 64, 128, 512}, "weight tiling 1"},
		{"later-input-product", [5]int{8, 32, 64, 128, 512}, "weight tiling 1"},
		{"first-output-view", [5]int{2, 32, 64, 128, 512}, "weight tiling 0"},
		{"first-weight-pair", [5]int{15, 32, 64, 128, 512}, "weight tiling 0"},
		{"final-tile", [5]int{16, 32, 64, 128, 144}, "weight tiling 4"},
		{"source-int-weight-product", [5]int{2147483632, 32, 64, 128, 512}, "weight product overflows 1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := definitionInput(gpubatch.SUP)
			for i, out := range tc.outputs {
				in.Convolutions[i].Out = out
				if i+1 < len(in.Convolutions) {
					in.Convolutions[i+1].In = out
				}
			}
			_, err := packed.NewDefinition(in)
			if tc.wantError == "" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.wantError) {
				t.Fatalf("expected %q, got %v", tc.wantError, err)
			}
		})
	}
}

func TestSUPGapsBlocksAndSourceOrder(t *testing.T) {
	in := definitionInput(gpubatch.SUP)
	for _, lengths := range [][]int{{1}, {1, 769}, {768, 768, 768, 768}, {1536, 1536, 1}} {
		b, ids := handoff(t, in.Config, lengths...)
		p, err := packed.PlanBatch(mustDefinition(t, in), b, ids)
		if err != nil {
			t.Fatal(err)
		}
		s, err := p.SUP()
		if err != nil {
			t.Fatal(err)
		}
		members, err := p.Members()
		if err != nil {
			t.Fatal(err)
		}
		var total packed.Samples
		var blocks packed.Blocks
		for i, m := range members {
			if m.InputStart != total+packed.Samples(i+1)*2 || s.SampleTable[i] != [2]packed.Samples{total, m.Rounded} || s.DecoderTable[i] != [2]packed.ScoreRow{packed.ScoreRow(total / 6), packed.ScoreRow(m.Rounded / 6)} {
				t.Fatal("gap/table units")
			}
			total += m.Rounded
			blocks += packed.Blocks(m.Rounded / 768)
		}
		g, err := p.Input()
		if err != nil {
			t.Fatal(err)
		}
		if g.Physical != total+packed.Samples(len(members)+1)*2 || g.CapacityCharge != total+packed.Samples(len(members)+1)*4 || g.Capacity != 32*1536 {
			t.Fatal("input gaps/capacity")
		}
		if s.Active != blocks || s.Padded != (blocks+3)/4*4 || s.RopeZeroFillEntries != s.Padded || s.EncoderT != 64 || s.MaxPosition != 1024 || s.ScoreRowsPerBlock != 128 {
			t.Fatal("encoder block geometry")
		}
		if s.Convolutions[0].MInput != int64(g.Capacity) || s.Convolutions[4].MOutput != int64(s.Padded)*64 {
			t.Fatal("source full-input M geometry")
		}
		for i, conv := range s.Convolutions {
			if conv.LUTEntries != blocks || conv.GranularityIn/packed.Samples(conv.Definition.Stride) != conv.GranularityOut {
				t.Fatal("conv units")
			}
			if i > 0 && conv.MInput != s.Convolutions[i-1].MOutput {
				t.Fatal("conv handoff")
			}
		}
		for row := packed.ScoreRow(0); row < packed.ScoreRow(total/6); row++ {
			coord, err := p.SourceOrder(row)
			if err != nil || coord.Block != packed.Blocks(row/128) || coord.Local != row%128 {
				t.Fatal("SUP source row map")
			}
		}
		if _, err := p.HAC(); err == nil {
			t.Fatal("cross-family getter")
		}
		assertFrames(t, p, members)
	}
}

func TestBoundaryRejections(t *testing.T) {
	for _, change := range []func(*packed.DefinitionInput){
		func(in *packed.DefinitionInput) { in.Config.VariableChunks = false },
		func(in *packed.DefinitionInput) { in.StateLen = 5 },
		func(in *packed.DefinitionInput) { in.Config.BatchSize = 2147483584 },
		func(in *packed.DefinitionInput) { in.Config.Granularity = 7 },
		func(in *packed.DefinitionInput) { in.Decoder.Clamp = float32(math.NaN()) },
		func(in *packed.DefinitionInput) { in.Decoder.BeamWidth = 1 },
		func(in *packed.DefinitionInput) { in.Decoder.BeamCut = 1 },
		func(in *packed.DefinitionInput) { in.Pins.Source = "" },
		func(in *packed.DefinitionInput) { in.MaxChunk = 1 },
		func(in *packed.DefinitionInput) { in.Convolutions[2].Stride = 5 },
	} {
		in := definitionInput(gpubatch.HAC)
		change(&in)
		if _, err := packed.NewDefinition(in); err == nil {
			t.Fatal("invalid definition accepted")
		}
	}
	in := definitionInput(gpubatch.HAC)
	d := mustDefinition(t, in)
	for _, change := range []func(*scheduler.Batch, []packed.Member){
		func(b *scheduler.Batch, _ []packed.Member) { b.Variable = false },
		func(b *scheduler.Batch, _ []packed.Member) { b.Lane = gpubatch.SUP },
		func(b *scheduler.Batch, _ []packed.Member) { b.Sequence = 0 },
		func(b *scheduler.Batch, _ []packed.Member) { b.Real++ },
		func(b *scheduler.Batch, _ []packed.Member) { b.PaddedSamples-- },
		func(b *scheduler.Batch, _ []packed.Member) { b.Placements[0].Offset++ },
		func(b *scheduler.Batch, _ []packed.Member) { b.Placements[0].Length++ },
		func(b *scheduler.Batch, _ []packed.Member) { b.Placements[0].Slot++ },
		func(b *scheduler.Batch, _ []packed.Member) { b.Placements[1] = b.Placements[0] },
		func(_ *scheduler.Batch, m []packed.Member) { m[1].ChunkID = m[0].ChunkID },
		func(_ *scheduler.Batch, m []packed.Member) { m[1].ChunkID = "missing" },
		func(_ *scheduler.Batch, m []packed.Member) { m[0].Raw = 0 },
		func(_ *scheduler.Batch, m []packed.Member) { m[0].ParentOffset = packed.ParentOffset(math.MaxUint64) },
	} {
		b, ids := handoff(t, in.Config, 1, 192)
		change(&b, ids)
		if _, err := packed.PlanBatch(d, b, ids); err == nil {
			t.Fatal("invalid handoff accepted")
		}
	}
	// Both placements independently valid but together exceed source capacity.
	in = definitionInput(gpubatch.SUP)
	b, ids := handoff(t, in.Config, 1536)
	b.Real = 32
	b.Placements = nil
	ids = nil
	for i := 0; i < 32; i++ {
		id := fmt.Sprintf("over-%d", i)
		b.Placements = append(b.Placements, gpubatch.Placement{Chunk: gpubatch.Chunk{ID: id, Length: 1536}, Offset: 2 + i*(1536+2), Length: 1536})
		ids = append(ids, packed.Member{ChunkID: id, ParentID: "p", Raw: 1536})
	}
	if _, err := packed.PlanBatch(mustDefinition(t, in), b, ids); err == nil {
		t.Fatal("SUP capacity gaps ignored")
	}
}

func TestZeroPlanAndDefensiveCopies(t *testing.T) {
	var zero packed.Plan
	checks := []error{zero.Validate()}
	_, e := zero.Members()
	checks = append(checks, e)
	_, e = zero.Decoder()
	checks = append(checks, e)
	_, e = zero.Frames()
	checks = append(checks, e)
	_, e = zero.HAC()
	checks = append(checks, e)
	_, e = zero.SUP()
	checks = append(checks, e)
	_, e = zero.Input()
	checks = append(checks, e)
	_, e = zero.Definition()
	checks = append(checks, e)
	_, e = zero.Sequence()
	checks = append(checks, e)
	_, e = zero.SourceOrder(0)
	checks = append(checks, e)
	_, e = zero.UnboundContracts()
	checks = append(checks, e)
	for _, err := range checks {
		if !errors.Is(err, packed.ErrPlan) {
			t.Fatal("zero plan getter accepted")
		}
	}
	for _, lane := range []gpubatch.Lane{gpubatch.HAC, gpubatch.SUP} {
		in := definitionInput(lane)
		d := mustDefinition(t, in)
		b, ids := handoff(t, in.Config, 1)
		p, err := packed.PlanBatch(d, b, ids)
		if err != nil {
			t.Fatal(err)
		}
		original, _ := p.Members()
		in.Convolutions[0].Window = 99
		copyIn, _ := d.Input()
		copyIn.Convolutions[0].Window = 101
		ids[0].ParentID = "mutated"
		b.Placements[0].Length = 0
		members, _ := p.Members()
		members[0].ParentID = "changed"
		frames, _ := p.Frames()
		frames.Rows[0].Local = 99
		frames.FrameOffsets[0] = 99
		dec, _ := p.Decoder()
		dec.Requirements[0].Stage = "changed"
		if lane == gpubatch.HAC {
			h, _ := p.HAC()
			h.InLayout[0] = -999
			h.OutLayout[0] = -999
			h.ForwardEncoding[0] = -999
			h.BackwardEncoding[0] = -999
			h.SampleIntervals[0][0] = -999
			h2, _ := p.HAC()
			if h2.InLayout[0] == -999 || h2.OutLayout[0] == -999 || h2.ForwardEncoding[0] == -999 || h2.BackwardEncoding[0] == -999 || h2.SampleIntervals[0][0] == -999 {
				t.Fatal("HAC alias")
			}
		} else {
			s, _ := p.SUP()
			s.SampleTable[0][0] = 99
			s.DecoderTable[0][0] = 99
			s.Convolutions[0].MInput = 99
			s2, _ := p.SUP()
			if s2.SampleTable[0][0] != 0 || s2.DecoderTable[0][0] != 0 || s2.Convolutions[0].MInput == 99 {
				t.Fatal("SUP alias")
			}
		}
		got, _ := p.Members()
		f2, _ := p.Frames()
		d2, _ := p.Decoder()
		in2, _ := d.Input()
		if !reflect.DeepEqual(got, original) || f2.Rows[0].Local != 0 || f2.FrameOffsets[0] != 0 || d2.Requirements[0].Stage != "guides" || in2.Convolutions[0].Window != 5 {
			t.Fatal("mutable plan alias")
		}
	}
}

type planningInvocation struct {
	t       *testing.T
	d       packed.Definition
	batches []scheduler.Batch
	plans   []packed.Plan
	logical int
}

func (i *planningInvocation) Call(_ context.Context, b scheduler.Batch) error {
	var ids []packed.Member
	for _, p := range b.Placements {
		ids = append(ids, packed.Member{ChunkID: p.Chunk.ID, ParentID: "parent", Raw: packed.Samples(p.Chunk.Length)})
	}
	p, err := packed.PlanBatch(i.d, b, ids)
	if err != nil {
		return err
	}
	members, err := p.Members()
	if err != nil {
		return err
	}
	assertFrames(i.t, p, members)
	i.batches = append(i.batches, b)
	i.plans = append(i.plans, p)
	i.logical += len(members)
	return nil
}
func (*planningInvocation) Drain(context.Context) error { return nil }

func TestMaintainedSchedulerHighlyUnequalCSR(t *testing.T) {
	in := definitionInput(gpubatch.HAC)
	in.Config.ChunkSize = 12288
	inv := &planningInvocation{t: t, d: mustDefinition(t, in)}
	s, err := scheduler.New(scheduler.Config{Geometry: in.Config, MaxPending: 1025, Selection: scheduler.SelectionExplicit}, inv)
	if err != nil {
		t.Fatal(err)
	}
	chunks := []gpubatch.Chunk{{ID: "long", Length: 12288}}
	for j := 0; j < 1024; j++ {
		chunks = append(chunks, gpubatch.Chunk{ID: fmt.Sprintf("tiny-%d", j), Length: 1})
	}
	ctx := context.Background()
	if err := s.Submit(ctx, chunks...); err != nil {
		t.Fatal(err)
	}
	if err := s.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	if err := s.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	if len(inv.plans) != 1 || inv.logical != 1025 {
		t.Fatalf("unequal handoff plans=%d logical=%d", len(inv.plans), inv.logical)
	}
	p := inv.plans[0]
	members, err := p.Members()
	if err != nil {
		t.Fatal(err)
	}
	f, err := p.Frames()
	if err != nil {
		t.Fatal(err)
	}
	if len(f.FrameOffsets) != 2049 || len(f.Rows) != 3072 || f.FrameOffsets[1] != 1025 || f.FrameOffsets[2048] != 3072 {
		t.Fatal("unequal CSR shape")
	}
	for j, member := range members {
		wantSteps := packed.ScoreRow(1)
		if j == 0 {
			wantSteps = 2048
		}
		segment := member.Segment
		if segment.Logical != packed.LogicalID(j) || segment.Steps != wantSteps || segment.Terminal != segment.GuideStart+packed.GuideRow(wantSteps) || segment.HistoryRows != packed.HistoryRow(wantSteps+1) {
			t.Fatal("unequal segment terminal")
		}
	}
	for local := 1; local < 2048; local++ {
		rows := f.Rows[f.FrameOffsets[local]:f.FrameOffsets[local+1]]
		if len(rows) != 1 || rows[0].Member != 0 || rows[0].Local != packed.ScoreRow(local) {
			t.Fatal("finished tiny members retained")
		}
	}
}

func TestMaintainedSchedulerConsecutiveHACParts(t *testing.T) {
	in := definitionInput(gpubatch.HAC)
	inv := &planningInvocation{t: t, d: mustDefinition(t, in)}
	s, err := scheduler.New(scheduler.Config{Geometry: in.Config, MaxPending: 1024, Selection: scheduler.SelectionExplicit}, inv)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	// Two flushes share the same scheduler and its persistent HAC part state.
	for round := 0; round < 2; round++ {
		var chunks []gpubatch.Chunk
		for j := 0; j < 70; j++ {
			chunks = append(chunks, gpubatch.Chunk{ID: fmt.Sprintf("%d-%d", round, j), Length: 192})
		}
		if err := s.Submit(ctx, chunks...); err != nil {
			t.Fatal(err)
		}
		if err := s.Flush(ctx); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	if inv.logical != 140 || len(inv.batches) != 4 {
		t.Fatalf("handoff logical=%d batches=%d", inv.logical, len(inv.batches))
	}
	// A fresh former on the second Flush would emit 64,6 instead of 58,12.
	for j, want := range []int{64, 6, 58, 12} {
		if inv.batches[j].Real != want {
			t.Fatalf("persistent part handoff %d: got %d want %d", j, inv.batches[j].Real, want)
		}
	}
	for j, b := range inv.batches {
		if b.Sequence != j+1 {
			t.Fatal("dispatch sequence")
		}
	}
	// Tiny lengths permit logical membership above physical lanes without masks.
	b, ids := handoff(t, in.Config, makeTinyLengths(100)...)
	p, err := packed.PlanBatch(inv.d, b, ids)
	if err != nil {
		t.Fatal(err)
	}
	members, _ := p.Members()
	if len(members) != 100 {
		t.Fatal("logical membership clipped to physical N")
	}
}

func makeTinyLengths(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

func TestTinyAndFullShortMaximumProfiles(t *testing.T) {
	for _, size := range []int{6, 192, 6144, 12288} {
		in := definitionInput(gpubatch.HAC)
		in.Config.ChunkSize = size
		b, ids := handoff(t, in.Config, 1, size)
		// Identity order is not source score order; scheduler placements are.
		ids[0], ids[1] = ids[1], ids[0]
		p, err := packed.PlanBatch(mustDefinition(t, in), b, ids)
		if err != nil {
			t.Fatal(err)
		}
		members, err := p.Members()
		if err != nil {
			t.Fatal(err)
		}
		if members[0].ChunkID != b.Placements[0].Chunk.ID || members[1].Rounded != packed.Samples(size) || members[1].Segment.Steps != packed.ScoreRow(size/6) {
			t.Fatal("source order/maximum")
		}
		assertFrames(t, p, members)
	}
	for _, size := range []int{768, 6144, 12288} {
		in := definitionInput(gpubatch.SUP)
		in.Config.ChunkSize = size
		b, ids := handoff(t, in.Config, 1, size)
		p, err := packed.PlanBatch(mustDefinition(t, in), b, ids)
		if err != nil {
			t.Fatal(err)
		}
		s, err := p.SUP()
		if err != nil || s.MaxPosition != 1024 {
			t.Fatal("maximum position must not shrink with queue chunk")
		}
	}
}

// This opt-in adapter consumes retained CPU evidence only. CompareNative owns
// its authenticated manifest, original path bindings and corruption controls.
// The second comparison proves that Plan preserves that same first fixture's
// four-array authority; it adds no native producer or GPU qualification.
func TestRetainedCPUAuxOptIn(t *testing.T) {
	path := os.Getenv("PACKED_HAC_AUX_EVIDENCE")
	sha := os.Getenv("PACKED_HAC_AUX_EVIDENCE_SHA256")
	if path == "" && sha == "" {
		t.Skip("set both retained CPU evidence path and independent SHA256")
	}
	if path == "" || sha == "" {
		t.Fatal("both retained evidence selectors required")
	}
	// Maintained CompareNative intentionally resolves producer source copies
	// relative to the checkout root. Restore cwd automatically after this test.
	t.Chdir("../..")
	comparison, err := hacvcsaux.CompareNative(path, sha)
	if err != nil {
		t.Fatal(err)
	}
	if comparison.Cases < 1 || comparison.OutputWordRejection == "" {
		t.Fatal("genuine first fixture and word-corruption rejection required")
	}
	if _, err := hacvcsaux.CompareNative(path, "invalid-independent-pin"); err == nil {
		t.Fatal("wrong independent pin accepted")
	}
	matrix, err := hacvcsaux.ReadMatrix()
	if err != nil {
		t.Fatal(err)
	}
	fixture := matrix.Cases[0] // included by both maintained first/all selections
	aux, err := fixture.Aux(matrix.InputStride)
	if err != nil {
		t.Fatal(err)
	}
	in := definitionInput(gpubatch.HAC)
	in.Config.ChunkSize = matrix.InputStride * fixture.FullChunkSizeOut
	in.Config.BatchSize = fixture.BatchSize
	in.MaxChunk = packed.Samples(in.Config.ChunkSize)
	in.Pins = packed.Pins{Source: "8b8fc5d36a9c0baab262a743cb175b5878e38ca6", Model: "dna_r10.4.1_e8.2_400bps_hac@v6.0.0", Config: "e550ccbedb68ed74d6ee88e2f510adf33c52a1c868bd5e3a2bfbb7dfe6154c3f"}
	b, ids := handoff(t, in.Config, aux.Sizes()...)
	p, err := packed.PlanBatch(mustDefinition(t, in), b, ids)
	if err != nil {
		t.Fatal(err)
	}
	h, err := p.HAC()
	if err != nil {
		t.Fatal(err)
	}
	want, err := aux.PackLSTM()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(h.InLayout, want.InLayout) || !reflect.DeepEqual(h.OutLayout, want.OutLayout) || !reflect.DeepEqual(h.ForwardEncoding, want.ForwardEncoding) || !reflect.DeepEqual(h.BackwardEncoding, want.BackwardEncoding) || int(h.OutExtent.Capacity) != want.OutLayoutCapacity {
		t.Fatal("plan differs from authenticated CPU auxiliary authority")
	}
}
