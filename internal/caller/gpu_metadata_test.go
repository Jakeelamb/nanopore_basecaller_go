//go:build cgo

package caller

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/postcall"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/splitter"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/stitch"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/subread"
)

func gpuFixture(sequence string) GPURead {
	r := testRead()
	moves := make([]uint8, len(sequence)+2)
	for i := 2; i < len(moves); i++ {
		moves[i] = 1
	}
	r.Signal = make([]int16, 6*len(moves)+3)
	r.NumSamples = uint64(len(r.Signal))
	r.StartSample = 5001
	r.AcquisitionStartTimeMS = 1000
	r.ProtocolStartTimeMS = 900
	r.RunID = "run"
	r.FlowCellID = "flow"
	r.Channel = 42
	return GPURead{
		Read:       r,
		Normalized: signal.Normalized{Samples: make([]uint16, len(r.Signal)), Trimmed: 10},
		Stride:     6,
		Assembled:  stitch.Result{Sequence: sequence, Qstring: strings.Repeat("I", len(sequence)), Moves: moves},
	}
}

func TestGPUStageReusesCoreMetadataAndMux(t *testing.T) {
	g := gpuFixture(strings.Repeat("A", 100))
	stage, err := gpuStage(testSpec(), g)
	if err != nil {
		t.Fatal(err)
	}
	if stage.CalledRead.ID != g.Read.ID.String() || stage.CalledRead.NumSamples != g.Read.NumSamples || stage.CalledRead.Trimmed != 10 {
		t.Fatalf("identity or sample accounting differs: %+v", stage.CalledRead)
	}
	if stage.CalledRead.Sequence != g.Assembled.Sequence || stage.CalledRead.Qstring != g.Assembled.Qstring || len(stage.CalledRead.Moves) != len(g.Assembled.Moves) {
		t.Fatalf("assembled result not preserved: %+v", stage.CalledRead)
	}
	m := stage.CalledRead.FASTQ
	if m == nil || m.MeanQStart != testSpec().MeanQStart || m.ModelName != testSpec().Name || m.Channel != 42 || m.StartTimeMS != 2000 || m.ProtocolStartTimeMS != 900 || m.RunID != "run" || m.FlowCellID != "flow" {
		t.Fatalf("FASTQ metadata differs: %+v", m)
	}
	if stage.CalledRead.TailTrimmedSamples != 0 {
		t.Fatalf("non-mux read trimmed %d samples", stage.CalledRead.TailTrimmedSamples)
	}
}

func TestGPUStageAppliesMuxTrim(t *testing.T) {
	g := gpuFixture(strings.Repeat("A", 100))
	g.Read.EndReason = 1
	g.Read.EndReasonName = "mux_change"
	g.Assembled.Qstring = strings.Repeat("I", 70) + strings.Repeat("!", 30)
	stage, err := gpuStage(testSpec(), g)
	if err != nil {
		t.Fatal(err)
	}
	if len(stage.CalledRead.Sequence) != 69 || len(stage.CalledRead.Moves) != 71 || stage.CalledRead.TailTrimmedSamples != 189 {
		t.Fatalf("mux trim differs: len=%d tail=%d", len(stage.CalledRead.Sequence), stage.CalledRead.TailTrimmedSamples)
	}
}

func TestGPUProcessFourPolicies(t *testing.T) {
	spec := testSpec()
	for _, trim := range []bool{false, true} {
		for _, split := range []bool{false, true} {
			policy := postcall.DefaultPolicy()
			policy.Trim, policy.Split = trim, split
			g := gpuFixture(strings.Repeat("A", 100))
			original := append([]uint8(nil), g.Assembled.Moves...)
			out, err := ProcessGPURead(spec, g, policy)
			if err != nil {
				t.Fatalf("trim=%v split=%v: %v", trim, split, err)
			}
			if len(out.Postcall.Children) != 1 || !out.Postcall.Children[0].Filter.Keep || len(out.Reads) != 1 {
				t.Fatalf("trim=%v split=%v: children=%d", trim, split, len(out.Postcall.Children))
			}
			if out.Reads[0].Sequence != g.Assembled.Sequence {
				t.Fatalf("trim=%v split=%v: sequence changed", trim, split)
			}
			want := TrimFlags(0)
			if trim {
				want = TrimAdapter | TrimPrimer
			}
			if out.Reads[0].FASTQ.TrimFlags != want {
				t.Fatalf("trim=%v split=%v: flags %v want %v", trim, split, out.Reads[0].FASTQ.TrimFlags, want)
			}
			if string(original) != string(g.Assembled.Moves) {
				t.Fatal("GPU postcall mutated assembled moves")
			}
		}
	}
}

func TestGPUProcessPositiveSplitOrdering(t *testing.T) {
	g := gpuFixture(strings.Repeat("C", 1100))
	g.Normalized.Samples = make([]uint16, 6615)
	g.Read.Signal = make([]int16, 6615)
	g.Read.NumSamples = 6615
	seq := []byte(g.Assembled.Sequence)
	copy(seq[865:], splitter.InternalAdapter)
	g.Assembled.Sequence = string(seq)
	q := []byte(g.Assembled.Qstring)
	copy(q[865:870], "+++++")
	g.Assembled.Qstring = string(q)
	g.Normalized.Samples[(865+2)*6] = 0x4200
	policy := postcall.DefaultPolicy()
	policy.Trim = false
	out, err := ProcessGPURead(testSpec(), g, policy)
	if err != nil {
		t.Fatal(err)
	}
	if out.Postcall.SplitDecision == nil || len(out.Postcall.Children) != 2 {
		t.Fatalf("split decision: %+v", out.Postcall.SplitDecision)
	}
	a, b := out.Postcall.Children[0].Read, out.Postcall.Children[1].Read
	if a.ID != subread.DeriveUUID(g.Read.ID.String(), 0) || b.ID != subread.DeriveUUID(g.Read.ID.String(), 1) || a.NextID != b.ID || b.PrevID != a.ID {
		t.Fatal("child identity or order differs")
	}
	if a.Sequence != g.Assembled.Sequence[:865] || b.Sequence != g.Assembled.Sequence[866:] {
		t.Fatal("child sequence boundaries differ")
	}
	if a.MeanQStart != 0 || b.MeanQStart != 0 || a.EndReason != "unknown" || b.EndReason != g.Read.EndReasonName {
		t.Fatalf("child metadata reset differs: %+v %+v", a, b)
	}
	if a.StartTimeMS <= g.Read.StartSample*1000/5000 || b.StartTimeMS <= a.StartTimeMS {
		t.Fatalf("child clock not advanced: root=%d a=%d b=%d", g.Read.StartSample*1000/5000, a.StartTimeMS, b.StartTimeMS)
	}
}

func TestGPUStageRejectsBadReads(t *testing.T) {
	spec := testSpec()
	bad := map[string]func(*GPURead){
		"stride-model":   func(g *GPURead) { g.Stride = 12 },
		"stride-zero":    func(g *GPURead) { g.Stride = 0 },
		"no-samples":     func(g *GPURead) { g.Read.NumSamples = 0 },
		"no-signal":      func(g *GPURead) { g.Normalized.Samples = nil },
		"quality-length": func(g *GPURead) { g.Assembled.Qstring = "I" },
		"move-length":    func(g *GPURead) { g.Assembled.Moves = g.Assembled.Moves[:1] },
		"rate":           func(g *GPURead) { g.Read.SampleRate = 4000 },
		"kit":            func(g *GPURead) { g.Read.SequencingKit = "SQK-NOPE" },
		"flowcell":       func(g *GPURead) { g.Read.FlowCellProductCode = "FLO-UNKNOWN" },
		"clock":          func(g *GPURead) { g.Read.AcquisitionStartTimeMS = -1 },
	}
	for name, mutate := range bad {
		g := gpuFixture(strings.Repeat("A", 100))
		mutate(&g)
		if _, err := gpuStage(spec, g); err == nil {
			t.Fatalf("%s accepted", name)
		}
	}
}

func TestProcessGPUReadRejectsPolicyBeforeWork(t *testing.T) {
	g := gpuFixture(strings.Repeat("A", 100))
	policy := postcall.DefaultPolicy()
	policy.Version = "future"
	if _, err := ProcessGPURead(testSpec(), g, policy); err == nil {
		t.Fatal("unknown policy accepted")
	}
	policy = postcall.DefaultPolicy()
	policy.MinQ = 7
	if _, err := ProcessGPURead(testSpec(), g, policy); err == nil {
		t.Fatal("unsupported minimum quality accepted")
	}
}

func TestProcessGPUClosedCaller(t *testing.T) {
	var c Caller
	if _, err := c.ProcessGPU(context.Background(), gpuFixture("A"), postcall.DefaultPolicy()); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.ProcessGPU(ctx, gpuFixture("A"), postcall.DefaultPolicy()); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
