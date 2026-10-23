package ends

import (
	"math"
	"strings"
	"sync"
	"testing"
)

func TestDetectDNA(t *testing.T) {
	body := strings.Repeat("G", 240)
	for _, c := range []struct {
		name, sequence string
		orientation    Orientation
		start, end     int
	}{
		{"no hit", body, Unknown, 0, len(body)},
		{"adapters", adapterFront + body + adapterRear, Unknown, len(adapterFront), len(adapterFront) + len(body)},
		{"forward", primerFront + body + reverseComplement(primerRear), Forward, len(primerFront), len(primerFront) + len(body)},
		{"reverse", primerRear + body + reverseComplement(primerFront), Reverse, len(primerRear), len(primerRear) + len(body)},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := Detect(c.sequence, "sqk-lsk114", DefaultOptions())
			if err != nil {
				t.Fatal(err)
			}
			if got.Retained != (BaseInterval{c.start, c.end}) || got.PrimerClassification.Orientation != c.orientation {
				t.Fatalf("unexpected result %+v", got)
			}
		})
	}
}

func TestAdmissionAndFlags(t *testing.T) {
	seq := adapterFront + strings.Repeat("G", 240) + adapterRear
	for _, kit := range []string{"", "ANY"} {
		got, err := Detect(seq, kit, DefaultOptions())
		if err != nil || got.Admission != KitMissing || got.Retained != (BaseInterval{0, len(seq)}) || got.Adapters.Raw.Front.Name != Unclassified {
			t.Fatalf("missing kit %+v %v", got, err)
		}
	}
	for _, kit := range []string{"SQK-PCS114", "SQK-LSK114-260", "ALL", "garbage", " SQK-LSK114"} {
		if _, err := Detect(seq, kit, DefaultOptions()); err == nil {
			t.Errorf("unsupported kit %q accepted", kit)
		}
	}
	for _, seq := range []string{"a", "U", "AC-GT", "A\x00C", "AC🙂"} {
		if _, err := Detect(seq, SupportedKit, DefaultOptions()); err == nil {
			t.Errorf("malformed DNA accepted %q", seq)
		}
	}
	got, err := Detect(seq, SupportedKit, Options{})
	if err != nil || got.Retained != (BaseInterval{0, len(seq)}) || got.PrimerClassification.Name != Unclassified {
		t.Fatalf("disabled flags %+v %v", got, err)
	}
}

func TestIntervalAcceptance(t *testing.T) {
	base := Pair{End{float32(.8), "front", 1, 5}, End{float32(.8), "rear", 10, 15}}
	pair, interval := trimInterval(base, 20)
	if pair != base || interval != (BaseInterval{6, 10}) {
		t.Fatalf("equality threshold rejected %+v %+v", pair, interval)
	}
	base.Front.Score = math.Float32frombits(math.Float32bits(.8) - 1)
	pair, interval = trimInterval(base, 20)
	if pair.Front.Name != Unclassified || interval != (BaseInterval{0, 10}) {
		t.Fatalf("below threshold accepted %+v %+v", pair, interval)
	}
	base.Front.Score = .8
	base.Front.End = 10
	pair, interval = trimInterval(base, 20)
	if pair.Front.Name != Unclassified || pair.Rear.Name != Unclassified || interval != (BaseInterval{0, 20}) {
		t.Fatalf("inverted interval not reset %+v %+v", pair, interval)
	}
}

func TestDisjointIntervalsLeaveReadUntrimmed(t *testing.T) {
	sequence := strings.Repeat("G", 10) + reverseComplement(primerRear) + strings.Repeat("G", 5) + adapterFront + strings.Repeat("G", 40)
	got, err := Detect(sequence, SupportedKit, DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	if !got.Conflict || got.Retained != (BaseInterval{0, len(sequence)}) || got.Adapters.Retained.Start <= got.Primers.Retained.End {
		t.Fatalf("disjoint accepted intervals were not preserved: %+v", got)
	}
}

func TestPrimerClassification(t *testing.T) {
	for _, c := range []struct {
		front, rear, name string
		orientation       Orientation
	}{
		{"cDNA_FWD_FRONT", "cDNA_FWD_REAR", "cDNA_FWD", Forward},
		{"cDNA_REV_FRONT", Unclassified, "cDNA_REV", Reverse},
		{Unclassified, "cDNA_FWD_REAR", "cDNA_FWD", Forward},
		{"cDNA_FWD_FRONT", "cDNA_REV_REAR", Unclassified, Unknown},
		{Unclassified, Unclassified, Unclassified, Unknown},
	} {
		got := classify(Pair{Front: End{Name: c.front}, Rear: End{Name: c.rear}})
		if got != (Classification{c.name, c.orientation}) {
			t.Fatalf("classification %+v", got)
		}
	}
}

func TestConcurrentDetection(t *testing.T) {
	seq := adapterFront + primerFront + strings.Repeat("G", 240) + reverseComplement(primerRear) + adapterRear
	want, err := Detect(seq, SupportedKit, DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 24; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := Detect(seq, SupportedKit, DefaultOptions())
			if err != nil || got != want {
				t.Errorf("concurrent result changed: %+v %v", got, err)
			}
		}()
	}
	wg.Wait()
}
