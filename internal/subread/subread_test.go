package subread_test

import (
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/subread"
)

func read() subread.Read {
	return subread.Read{ID: "parent", Sequence: "ACGT", Qstring: "IIII", Moves: []uint8{1, 1, 1, 1}, Signal: []uint16{0, 1, 2, 3, 4, 5, 6, 7}, Stride: 2, SampleRate: 2000, StartSample: 1, LeadingTrim: 2, NumSamples: 10, ReadNumber: 12, NumEvents: 14, PreTrimSeqLen: 15, MeanQStart: 16, SplitCount: 1, SplitPoint: math.MaxUint32, PrevID: "before", NextID: "after", EndReason: "mux_change", IsMuxChange: true, Inherited: subread.Inherited{RunID: "run", ModelName: "model", ClientToken: 7}}
}

func TestConcurrentCalls(t *testing.T) {
	in := read()
	want, err := subread.Materialize(in, []subread.Interval{{Start: 1, End: 2}})
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for range 12 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := subread.Materialize(in, []subread.Interval{{Start: 1, End: 2}})
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Errorf("concurrent materialization differs: %v", err)
			}
		}()
	}
	wg.Wait()
}
func TestMaterializedMetadata(t *testing.T) {
	in := read()
	out, err := subread.Materialize(in, []subread.Interval{{Start: 1, End: 2}})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("children=%d", len(out))
	}
	first, last := out[0], out[1]
	if first.Sequence != "A" || last.Sequence != "GT" || !reflect.DeepEqual(first.Signal, []uint16{0, 1}) || !reflect.DeepEqual(last.Moves, []uint8{1, 1}) {
		t.Fatal("child slicing differs")
	}
	if first.ID != subread.DeriveUUID("parent", 0) || last.ID != subread.DeriveUUID("parent", 1) || first.ParentID != "parent" || last.ParentID != "parent" {
		t.Fatal("child identity differs")
	}
	if first.PrevID != "before" || first.NextID != last.ID || last.PrevID != first.ID || last.NextID != "after" {
		t.Fatal("neighbor chain differs")
	}
	if first.StartSample != 3 || first.EndSample != 5 || first.StartTimeMS != 2 || first.StartTime != "1970-01-01T00:00:00.002000+00:00" || first.SplitPoint != 1 {
		t.Fatalf("clock or sample metadata differs: %+v", first)
	}
	if first.EndReason != "unknown" || first.IsMuxChange || last.EndReason != in.EndReason || !last.IsMuxChange {
		t.Fatal("last end reason differs")
	}
	for _, child := range out {
		if child.ReadNumber != -1 || child.NumEvents != 0 || child.LeadingTrim != 0 || child.MeanQStart != 0 || child.PreTrimSeqLen != uint64(len(child.Sequence)) || child.SplitCount != 2 || child.Inherited != in.Inherited {
			t.Fatal("child metadata reset or inheritance differs")
		}
	}
	first.Moves[0] = 0
	if in.Moves[0] != 1 {
		t.Fatal("child move storage aliases parent")
	}
	if &last.Signal[0] != &in.Signal[4] {
		t.Fatal("child signal is not the borrowed native-style view")
	}
}
func TestIdentityAndExistingParent(t *testing.T) {
	in := read()
	out, err := subread.Materialize(in, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(out, []subread.Read{in}) {
		t.Fatal("root identity changed")
	}
	in.ParentID = "root"
	out, err = subread.Materialize(in, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := in
	want.ID = subread.DeriveUUID("root", 0)
	want.SubreadID = 0
	want.SplitCount = 1
	if !reflect.DeepEqual(out, []subread.Read{want}) {
		t.Fatal("existing parent finalization differs")
	}
	in.Moves = nil
	out, err = subread.Materialize(in, []subread.Interval{{Start: 1, End: 2}})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(out, []subread.Read{in}) {
		t.Fatal("empty moves bypass changed")
	}
}
func TestRejectedBoundaries(t *testing.T) {
	for name, change := range map[string]func(*subread.Read){
		"missing-ID": func(r *subread.Read) { r.ID = "" }, "quality-length": func(r *subread.Read) { r.Qstring = "I" }, "RNA-base": func(r *subread.Read) { r.Sequence = "ACGU" }, "quality": func(r *subread.Read) { r.Qstring = " III" },
		"nonbinary-move": func(r *subread.Read) { r.Moves[0] = 2 }, "move-count": func(r *subread.Read) { r.Moves[0] = 0 }, "signal-count": func(r *subread.Read) { r.Signal = r.Signal[:5] }, "zero-stride": func(r *subread.Read) { r.Stride = 0 }, "zero-rate": func(r *subread.Read) { r.SampleRate = 0 }, "read-number": func(r *subread.Read) { r.ReadNumber = math.MaxInt32 + 1 },
		"sample-overflow": func(r *subread.Read) { r.StartSample = math.MaxUint64 }, "clock-overflow": func(r *subread.Read) { r.AcquisitionStartMS = math.MaxUint64 },
	} {
		t.Run(name, func(t *testing.T) {
			in := read()
			change(&in)
			if _, err := subread.Materialize(in, nil); err == nil {
				t.Fatal("malformed input accepted")
			}
		})
	}
	for _, spacers := range [][]subread.Interval{{{Start: 2, End: 1}}, {{Start: 0, End: 5}}, {{Start: 2, End: 3}, {Start: 1, End: 2}}, {{Start: 1, End: 3}, {Start: 2, End: 4}}} {
		if _, err := subread.Materialize(read(), spacers); err == nil {
			t.Fatal("malformed spacers accepted")
		}
	}
}

func TestNativeClockDomain(t *testing.T) {
	const lastMS = uint64(9223372036854)
	for _, tc := range []struct {
		name                string
		acquisition, sample uint64
		leading, rate       uint64
		spacer              subread.Interval
		wantRelative        uint64
		wantTime            string
		reject              bool
	}{
		{name: "current-era", acquisition: 1700000000123, rate: 2000, wantTime: "2023-11-14T22:13:20.123000+00:00"},
		{name: "last-acquisition", acquisition: lastMS, rate: 2000, wantTime: "2262-04-11T23:47:16.854000+00:00"},
		{name: "first-acquisition-rejected", acquisition: lastMS + 1, rate: 2000, reject: true},
		{name: "last-summed-rounded", acquisition: lastMS - 1, sample: 1, rate: 2000, wantRelative: 1, wantTime: "2262-04-11T23:47:16.854000+00:00"},
		{name: "first-summed-rounded-rejected", acquisition: lastMS, sample: 1, rate: 2000, reject: true},
		{name: "last-trim-and-spacer", acquisition: lastMS - 3, sample: 1, leading: 2, rate: 2000, spacer: subread.Interval{Start: 0, End: 1}, wantRelative: 3, wantTime: "2262-04-11T23:47:16.854000+00:00"},
		{name: "first-trim-and-spacer-rejected", acquisition: lastMS - 2, sample: 1, leading: 2, rate: 2000, spacer: subread.Interval{Start: 0, End: 1}, reject: true},
		{name: "later-child-rejected-no-partial-output", acquisition: lastMS, rate: 2000, spacer: subread.Interval{Start: 1, End: 1}, reject: true},
		{name: "last-relative", sample: lastMS, rate: 1000, wantRelative: lastMS, wantTime: "2262-04-11T23:47:16.854000+00:00"},
		{name: "first-relative-rejected", sample: lastMS + 1, rate: 1000, reject: true},
		{name: "year-9999-rejected", acquisition: 253402300799000, rate: 1000, reject: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := read()
			in.AcquisitionStartMS, in.StartSample, in.LeadingTrim, in.SampleRate = tc.acquisition, tc.sample, tc.leading, tc.rate
			out, err := subread.Materialize(in, []subread.Interval{tc.spacer})
			if tc.reject {
				if err == nil || out != nil {
					t.Fatalf("unrepresentable native timestamp accepted: %+v, %v", out, err)
				}
				return
			}
			if err != nil || len(out) != 1 {
				t.Fatalf("representable timestamp rejected: %+v, %v", out, err)
			}
			if out[0].StartTimeMS != tc.wantRelative || out[0].StartTime != tc.wantTime {
				t.Fatalf("clock differs: relative=%d absolute=%s", out[0].StartTimeMS, out[0].StartTime)
			}
		})
	}
}

func TestUnconvertedClockBypass(t *testing.T) {
	for _, emptyMoves := range []bool{false, true} {
		in := read()
		in.AcquisitionStartMS = 9223372036854
		in.StartSample = 9223372036855
		in.StartTimeMS, in.StartTime = math.MaxUint64, "original-clock"
		var spacers []subread.Interval
		if emptyMoves {
			in.Moves = nil
			spacers = []subread.Interval{{Start: 1, End: 2}}
		}
		out, err := subread.Materialize(in, spacers)
		if err != nil || !reflect.DeepEqual(out, []subread.Read{in}) {
			t.Fatalf("unconverted clock fields changed: %+v, %v", out, err)
		}
		in.AcquisitionStartMS++
		if out, err := subread.Materialize(in, spacers); err == nil || out != nil {
			t.Fatalf("out-of-domain acquisition accepted on bypass: %+v, %v", out, err)
		}
	}
}
