//go:build cgo

package postcall

import (
	"reflect"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/ends"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/splitter"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/subread"
)

func plain(sequence string) subread.Read {
	moves := make([]uint8, len(sequence)+2)
	for i := 2; i < len(moves); i++ {
		moves[i] = 1
	}
	return subread.Read{ID: "bb76c31a-36f0-48a0-abd2-107a10e1ade0", Sequence: sequence, Qstring: strings.Repeat("I", len(sequence)), Moves: moves, Signal: make([]uint16, len(moves)*6+3), Stride: 6, SampleRate: 5000, StartSample: 160723601, AcquisitionStartMS: 1718800934753, StartTimeMS: 32144720, StartTime: "2024-06-19T13:38:00.000000+00:00", LeadingTrim: 10, NumSamples: uint64(len(moves)*6 + 3), NumEvents: 71, PreTrimSeqLen: uint64(len(sequence)), MeanQStart: 60, ReadNumber: 19, EndReason: "signal_positive", Inherited: subread.Inherited{SequencingKit: "sqk-lsk114", RunID: "run", ModelName: "model", FlowcellID: "flow", Channel: 42}}
}

func TestFourPoliciesNoMatchAndIndependentFlags(t *testing.T) {
	for _, trim := range []bool{false, true} {
		for _, split := range []bool{false, true} {
			p := DefaultPolicy()
			p.Trim, p.Split = trim, split
			in := plain(strings.Repeat("A", 100))
			original := append([]uint8(nil), in.Moves...)
			got, err := Process(in, p)
			if err != nil {
				t.Fatal(err)
			}
			if len(got.Children) != 1 || !got.Children[0].Filter.Keep || !reflect.DeepEqual(got.Children[0].Read, in) {
				t.Fatalf("trim=%v split=%v: %+v", trim, split, got)
			}
			if (got.SplitDecision != nil) != split || (got.Children[0].Detection != nil) != trim {
				t.Fatal("flags changed another stage")
			}
			if !reflect.DeepEqual(original, in.Moves) {
				t.Fatal("input mutated")
			}
		}
	}
}

func TestSynchronizedEndTrimClassification(t *testing.T) {
	sequence := "TTTCTGTTGGTGCTGATATTGCTGGG" + strings.Repeat("C", 200) + "AAAGAAGATAGAGCGAGGCAAGT"
	in := plain(sequence)
	p := DefaultPolicy()
	p.Split = false
	got, err := Process(in, p)
	if err != nil {
		t.Fatal(err)
	}
	c := got.Children[0]
	if c.Detection.PrimerClassification.Orientation != ends.Forward || c.Detection.PrimerClassification.Name != "cDNA_FWD" {
		t.Fatalf("classification: %+v", c.Detection)
	}
	if c.Read.Sequence != strings.Repeat("C", 200) || c.Read.Qstring != strings.Repeat("I", 200) || len(c.Read.Moves) != 200 || len(c.Read.Signal) != 1200 || c.Read.LeadingTrim != 10+uint64(2+len("TTTCTGTTGGTGCTGATATTGCTGGG"))*6 {
		t.Fatalf("unsynchronized trim: %+v", c.Read)
	}
	if c.Read.StartSample != in.StartSample || c.Read.StartTime != in.StartTime || c.Read.MeanQStart != 60 {
		t.Fatal("root trim changed clock or mean-quality offset")
	}
	if !c.Filter.Keep {
		t.Fatal("trimmed read filtered")
	}
}

func TestPositiveSplitChildOrderingAndMetadata(t *testing.T) {
	in := plain(strings.Repeat("C", 1100))
	seq := []byte(in.Sequence)
	copy(seq[865:], splitter.InternalAdapter)
	in.Sequence = string(seq)
	q := []byte(in.Qstring)
	copy(q[865:870], "+++++")
	in.Qstring = string(q)
	in.Signal[(865+2)*6] = 0x4200
	p := DefaultPolicy()
	p.Trim = false
	got, err := Process(in, p)
	if err != nil {
		t.Fatal(err)
	}
	if got.SplitDecision == nil || !reflect.DeepEqual(got.SplitDecision.Spacers, []splitter.Interval{{Start: 865, End: 866}}) || len(got.Children) != 2 {
		t.Fatalf("split decision: %+v", got.SplitDecision)
	}
	a, b := got.Children[0].Read, got.Children[1].Read
	if a.Sequence != in.Sequence[:865] || b.Sequence != in.Sequence[866:] || a.ID != subread.DeriveUUID(in.ID, 0) || b.ID != subread.DeriveUUID(in.ID, 1) || a.NextID != b.ID || b.PrevID != a.ID {
		t.Fatal("child identity/order differs")
	}
	if a.StartSample != in.StartSample+in.LeadingTrim+12 || a.StartTimeMS != 32144725 || a.MeanQStart != 0 || b.MeanQStart != 0 || a.NumEvents != 0 || a.ReadNumber != -1 || a.EndReason != "unknown" || b.EndReason != in.EndReason || a.LeadingTrim != 0 {
		t.Fatalf("child metadata differs: %+v %+v", a, b)
	}
	if a.Inherited != in.Inherited || b.Inherited != in.Inherited || len(b.Signal)%6 != 3 {
		t.Fatal("inheritance or partial tail differs")
	}
	if !got.Children[0].Filter.Keep || !got.Children[1].Filter.Keep {
		t.Fatal("children filtered")
	}
}

func TestSingleChildAndFilteredChild(t *testing.T) {
	in := plain(strings.Repeat("C", 200))
	in.Moves = make([]uint8, 1100)
	for i := 900; i < 1100; i++ {
		in.Moves[i] = 1
	}
	in.Signal = make([]uint16, 6603)
	in.Signal[5400] = 0x4200
	in.Sequence = splitter.InternalAdapter + in.Sequence[len(splitter.InternalAdapter):]
	in.Qstring = "+++++" + in.Qstring[5:]
	p := DefaultPolicy()
	p.Trim = false
	got, err := Process(in, p)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Children) != 1 || got.Children[0].Read.ParentID != in.ID || got.Children[0].Read.ID != subread.DeriveUUID(in.ID, 0) || len(got.Children[0].Read.Sequence) != 199 {
		t.Fatalf("one-child endpoint: %+v", got)
	}
	short := plain("ACGT")
	got, err = Process(short, p)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Children) != 1 || got.Children[0].Filter.Keep {
		t.Fatal("short child should produce no writer output")
	}
	empty := plain("")
	got, err = Process(empty, p)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Children) != 1 || got.Children[0].Filter.Keep || got.SplitDecision != nil {
		t.Fatal("empty bypass/filter differs")
	}
}

func TestAdmissionAndBadCoreShapes(t *testing.T) {
	for _, field := range []string{"kit", "version", "minq", "ceil", "stride", "quality", "moves"} {
		in := plain(strings.Repeat("A", 100))
		p := DefaultPolicy()
		switch field {
		case "kit":
			in.Inherited.SequencingKit = "SQK-RAD114"
		case "version":
			p.Version = "future"
		case "minq":
			p.MinQ = 7
		case "ceil":
			in.Moves = append(in.Moves, 0)
		case "stride":
			in.Stride = 0
		case "quality":
			in.Qstring = in.Qstring[:99]
		case "moves":
			in.Moves[0] = 2
		}
		if _, err := Process(in, p); err == nil {
			t.Fatalf("accepted %s", field)
		}
	}
	in := plain("ACGTA")
	in.Inherited.SequencingKit = "SQK-RAD114"
	p := DefaultPolicy()
	p.Trim, p.Split = false, false
	if _, err := Process(in, p); err != nil {
		t.Fatal("disabled processing narrowed core kit contract", err)
	}
}
