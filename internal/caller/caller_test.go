package caller

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/model"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/pod5"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/tensor"
)

func testSpec() model.Spec {
	return model.Spec{Name: "dna_r10.4.1_e8.2_400bps_hac@v6.0.0", ConfigSHA256: model.HACConfigSHA256, Architecture: model.HAC, Features: 1, SampleRate: 5000, Stride: 6, InnerStride: 6, Granularity: 6, ChunkSize: 12288, Overlap: 600, StateLen: 4, OutputChannels: 1024, Mean: 94, Stdev: 24, QScale: 1.02, QBias: -.3, BlankScore: 2, MeanQStart: 60}
}
func testRead() pod5.Read {
	var id pod5.ID
	id[0] = 1
	return pod5.Read{ID: id, Signal: make([]int16, 22), NumSamples: 22, CalibrationScale: .187, CalibrationOffset: -270, OpenPoreLevel: float32(math.NaN()), RunInfo: pod5.RunInfo{SampleRate: 5000, FlowCellProductCode: "FLO-PRO114M", SequencingKit: "sqk-lsk114"}}
}
func TestConfiguration(t *testing.T) {
	s := testSpec()
	cc, opt, err := configuration(s, Config{})
	if err != nil || cc.ChunkSize != 12288 || cc.Overlap != 600 || opt.BeamWidth != 32 || opt.BlankScore != 2 {
		t.Fatalf("defaults: %+v %+v %v", cc, opt, err)
	}
	cc, _, err = configuration(s, Config{ChunkSize: 12, Overlap: 0})
	if err != nil || cc.Overlap != 0 {
		t.Fatalf("explicit zero overlap: %+v %v", cc, err)
	}
	for _, cfg := range []Config{{Overlap: 6}, {ChunkSize: 7}, {ChunkSize: 12, Overlap: 12}, {ChunkSize: 12, Overlap: 1}, {ChunkSize: 1 << 27}} {
		if _, _, err := configuration(s, cfg); err == nil {
			t.Fatalf("accepted config %+v", cfg)
		}
	}
	s.Architecture = model.SUP
	if _, _, err := configuration(s, Config{}); !errors.Is(err, ErrUnsupported) {
		t.Fatal(err)
	}
}
func TestChemistryAndAdjustment(t *testing.T) {
	r := testRead()
	if err := validateRead(r); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"sample", "flow", "kit", "length"} {
		bad := r
		switch field {
		case "sample":
			bad.SampleRate = 4000
		case "flow":
			bad.FlowCellProductCode = "FLO-MIN004RA"
		case "kit":
			bad.SequencingKit = "SQK-RNA004"
		case "length":
			bad.NumSamples++
		}
		if err := validateRead(bad); err == nil {
			t.Fatalf("accepted bad %s", field)
		}
	}
	r.FlowCellProductCode = "FLO-PRO114HD"
	r.SequencingKit = "SQK-RAD114"
	if err := validateRead(r); !errors.Is(err, ErrUnsupported) {
		t.Fatal("HD chemistry admitted unsupported kit")
	}
	for _, tc := range []struct {
		flow     string
		expected float32
	}{{"flo-pro114m", 199.21}, {"FLO-PRO114HD", 199.21}, {"FLO-MIN114HD", 197.61}, {"FLO-FLG114", 200}} {
		level := float32(205)
		scale := float32(.187)
		want := float32(float32(level-tc.expected) / scale)
		if got := adjustment(tc.flow, level, scale); math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("pore correction %s got %v want %v", tc.flow, got, want)
		}
	}
	if adjustment("UNKNOWN", 205, .187) != 0 || adjustment("FLO-PRO114M", float32(math.NaN()), .187) != 0 {
		t.Fatal("missing expected pore/NaN must use zero adjustment")
	}
}
func TestClosedCancellationAndRejectionBeforeInference(t *testing.T) {
	var c Caller
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Call(context.Background(), testRead()); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	active := &Caller{engine: &tensor.Engine{}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := active.Call(ctx, testRead()); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	bad := testRead()
	bad.SampleRate = 4000
	if _, err := active.Call(context.Background(), bad); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("unsupported chemistry reached missing inference model: %v", err)
	}
	if _, err := Open(ctx, Config{}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
