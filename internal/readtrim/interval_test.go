package readtrim_test

import (
	"errors"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/decode"
	"github.com/Jakeelamb/nanopore_basecaller_go/internal/readtrim"
)

func intervalFixture() readtrim.MuxRead {
	return readtrim.MuxRead{Sequence: "ACNGT", Qstring: "!5I+?", Moves: []uint8{0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0}, Signal: make([]uint16, 41), Stride: 3, IsMuxChange: true, NumMinknowEvents: 91, PreTrimSequenceLength: 71}
}
func TestIntervalRanges(t *testing.T) {
	for _, c := range []struct {
		start, end, ms, me, ss, se int
		lead                       uint64
		trim                       bool
	}{
		{0, 5, 0, 14, 0, 41, 10, false},
		{0, 1, 2, 4, 6, 12, 16, true},
		{1, 4, 4, 11, 12, 33, 22, true},
		{4, 5, 11, 14, 33, 41, 43, true},
		{2, 2, 0, 0, 0, 0, 10, true},
	} {
		r := intervalFixture()
		for i := range r.Signal {
			r.Signal[i] = uint16(i)
		}
		before := append([]uint16(nil), r.Signal...)
		got, err := readtrim.ApplyInterval(r, readtrim.Interval{Start: c.start, End: c.end}, 10)
		if err != nil {
			t.Fatal(err)
		}
		if got.MoveStart != c.ms || got.MoveEnd != c.me || got.SignalStart != c.ss || got.SignalEnd != c.se || got.LeadingTrimSamples != c.lead || got.Trimmed != c.trim {
			t.Fatalf("range %+v got %+v", c, got)
		}
		if got.Read.Sequence != r.Sequence[c.start:c.end] || got.Read.Qstring != r.Qstring[c.start:c.end] || !reflect.DeepEqual(got.Read.Moves, r.Moves[c.ms:c.me]) || !reflect.DeepEqual(got.Read.Signal, r.Signal[c.ss:c.se]) {
			t.Fatal("synchronized slice mismatch")
		}
		if got.Read.PreTrimSequenceLength != 71 || got.Read.NumMinknowEvents != 91 || !got.Read.IsMuxChange || got.Read.Stride != 3 || got.NumSamples != c.lead+uint64(c.se-c.ss) {
			t.Fatal("metadata changed")
		}
		if !reflect.DeepEqual(before, r.Signal) {
			t.Fatal("input modified")
		}
		if len(got.Read.Signal) > 0 && &got.Read.Signal[0] != &r.Signal[c.ss] {
			t.Fatal("signal copied")
		}
		if len(got.Read.Moves) > 0 && &got.Read.Moves[0] != &r.Moves[c.ms] {
			t.Fatal("moves copied")
		}
		if c.trim && (cap(got.Read.Signal) != len(got.Read.Signal) || cap(got.Read.Moves) != len(got.Read.Moves)) {
			t.Fatal("removed capacity exposed")
		}
	}
	got, err := readtrim.ApplyInterval(readtrim.MuxRead{Stride: 1}, readtrim.Interval{}, 0)
	if err != nil || got.Trimmed {
		t.Fatalf("empty identity %v %+v", err, got)
	}
	one := readtrim.MuxRead{Sequence: "N", Qstring: "I", Moves: []uint8{0, 1, 0}, Signal: []uint16{0, 1, 2}, Stride: 1}
	if _, err := readtrim.ApplyInterval(one, readtrim.Interval{End: 1}, 0); err != nil {
		t.Fatal(err)
	}
}
func TestIntervalRejections(t *testing.T) {
	for _, interval := range []readtrim.Interval{{Start: -1, End: 3}, {Start: 3, End: 2}, {Start: 0, End: 6}, {Start: 5, End: 5}} {
		if _, err := readtrim.ApplyInterval(intervalFixture(), interval, 0); err == nil {
			t.Fatalf("accepted %+v", interval)
		}
	}
	for _, change := range []func(*readtrim.MuxRead){
		func(r *readtrim.MuxRead) { r.Qstring = "!" }, func(r *readtrim.MuxRead) { r.Sequence = "ACXGT" },
		func(r *readtrim.MuxRead) { r.Moves[0] = 2 }, func(r *readtrim.MuxRead) { r.Moves[2] = 0 }, func(r *readtrim.MuxRead) { r.Stride = 0 },
		func(r *readtrim.MuxRead) { r.Signal = r.Signal[:4] }, func(r *readtrim.MuxRead) { r.Signal[0] = 0x7c00 }, func(r *readtrim.MuxRead) { r.Qstring = "\x7fIIII" },
	} {
		r := intervalFixture()
		change(&r)
		if _, err := readtrim.ApplyInterval(r, readtrim.Interval{End: 5}, 0); err == nil {
			t.Fatal("malformed read accepted")
		}
	}
	for _, c := range []struct {
		i    readtrim.Interval
		lead uint64
	}{{readtrim.Interval{Start: 1, End: 4}, math.MaxUint64 - 10}, {readtrim.Interval{End: 5}, math.MaxUint64 - 40}} {
		if _, err := readtrim.ApplyInterval(intervalFixture(), c.i, c.lead); err == nil {
			t.Fatal("overflow accepted")
		}
	}
}
func TestIntervalConcurrentProperties(t *testing.T) {
	r := intervalFixture()
	var wg sync.WaitGroup
	for k := 0; k < 8; k++ {
		wg.Add(1)
		go func(k int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(k)))
			for n := 0; n < 100; n++ {
				a := rng.Intn(5)
				b := a + rng.Intn(6-a)
				got, err := readtrim.ApplyInterval(r, readtrim.Interval{Start: a, End: b}, 10)
				if err != nil {
					t.Error(err)
					return
				}
				sum := 0
				for _, m := range got.Read.Moves {
					sum += int(m)
				}
				if sum != b-a || len(got.Read.Sequence) != b-a || got.NumSamples != got.LeadingTrimSamples+uint64(len(got.Read.Signal)) {
					t.Error("interval invariant failed")
					return
				}
			}
		}(k)
	}
	wg.Wait()
}
func TestFilterBoundaries(t *testing.T) {
	if _, err := readtrim.Filter("AAAAA", "IIIII", 0, 0); errors.Is(err, decode.ErrCGORequired) {
		t.Skip("native float libm bridge disabled")
	}
	for _, c := range []struct {
		n, start  int
		threshold float32
		reason    readtrim.FilterReason
	}{
		{0, 0, 0, readtrim.FilterShort}, {4, 0, 0, readtrim.FilterShort}, {5, 0, 0, readtrim.FilterPass},
		{5, 0, 39, readtrim.FilterPass}, {5, 0, 40, readtrim.FilterPass}, {5, 0, 41, readtrim.FilterLowQuality},
		{4, 0, 41, readtrim.FilterShort | readtrim.FilterLowQuality}, {5, 60, -1, readtrim.FilterPass},
	} {
		got, err := readtrim.Filter(strings.Repeat("N", c.n), strings.Repeat("I", c.n), c.start, c.threshold)
		if err != nil {
			t.Fatal(err)
		}
		if got.Reason != c.reason || got.Keep != (c.reason == readtrim.FilterPass) {
			t.Fatalf("%+v got %+v", c, got)
		}
	}
	for _, n := range []int{59, 60, 61, 121} {
		q := strings.Repeat("!", min(n, 60)) + strings.Repeat("I", max(0, n-60))
		selected := q
		if n > 60 {
			selected = q[60:]
		}
		want, err := decode.MeanQScore(selected)
		if err != nil {
			t.Fatal(err)
		}
		got, err := readtrim.Filter(strings.Repeat("A", n), q, 60, 0)
		if err != nil || math.Float32bits(got.MeanQ) != math.Float32bits(want) {
			t.Fatalf("offset n=%d got %+v err %v", n, got, err)
		}
	}
}
func TestFilterRejects(t *testing.T) {
	for _, c := range []struct {
		seq, q    string
		start     int
		threshold float32
	}{{"A", "", 0, 0}, {"N", "I", -1, 0}, {"X", "I", 0, 0}, {"A", "\x20", 0, 0}, {"A", "\x7f", 0, 0}, {"A", "I", 0, float32(math.NaN())}, {"A", "I", 0, float32(math.Inf(1))}} {
		if _, err := readtrim.Filter(c.seq, c.q, c.start, c.threshold); err == nil {
			t.Fatalf("accepted %+v", c)
		}
	}
	if _, err := decode.MeanQScore("I"); errors.Is(err, decode.ErrCGORequired) {
		if _, err := readtrim.Filter("AAAAA", "IIIII", 0, 0); !errors.Is(err, decode.ErrCGORequired) {
			t.Fatal("missing no-cgo error")
		}
	}
}
