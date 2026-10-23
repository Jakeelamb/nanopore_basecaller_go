package readtrim_test

import (
	"math"
	"math/rand"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/readtrim"
)

func fixture(q string, stride uint32, zeros, partial int) readtrim.MuxRead {
	moves := make([]uint8, zeros)
	for range q {
		moves = append(moves, 1)
		for j := 0; j < zeros; j++ {
			moves = append(moves, 0)
		}
	}
	samples := len(moves)*int(stride) + partial
	signal := make([]uint16, samples)
	for i := range signal {
		signal[i] = uint16(i % 0x7c00)
		if i%2 == 1 {
			signal[i] |= 0x8000
		}
	}
	return readtrim.MuxRead{Sequence: strings.Repeat("A", len(q)), Qstring: q, Moves: moves, Signal: signal, Stride: stride, IsMuxChange: true, NumMinknowEvents: math.MaxUint64, PreTrimSequenceLength: uint64(len(q) + 21)}
}
func tail(n, low int) string { return strings.Repeat("I", n-low) + strings.Repeat("!", low) }

func TestMuxBehavior(t *testing.T) {
	for _, tc := range []struct {
		name, q string
		wantEnd int
	}{
		{"short99", tail(99, 20), 99},
		{"length100", tail(100, 20), 79},
		{"no-low-tail", strings.Repeat("I", 100), 100},
		{"all-low-excessive", strings.Repeat("!", 100), 100},
		{"candidate-trim4", tail(100, 3), 100},
		{"candidate-trim5", tail(100, 4), 100},
		{"candidate-trim6", tail(100, 5), 94},
		{"excessive-below30", tail(100, 70), 100},
		{"excessive-at30", tail(100, 69), 30},
		{"excessive-above30", tail(100, 68), 31},
		{"low-threshold7", strings.Repeat("I", 90) + strings.Repeat("(", 10), 89},
		{"mid-threshold8", strings.Repeat("I", 90) + strings.Repeat(")", 10), 100},
		{"mid-threshold12", strings.Repeat("I", 90) + strings.Repeat("-", 10), 100},
		{"high-threshold13", strings.Repeat("I", 90) + strings.Repeat(".", 10), 100},
		{"tie-earlier-minimum", strings.Repeat("I", 90) + "!" + " )"[1:] + strings.Repeat("!", 8), 89},
		{"mid12-tie-earlier-minimum", strings.Repeat("I", 90) + "!-!!!!!!!!", 89},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := fixture(tc.q, 6, 2, 4)
			beforeMoves, beforeSignal := append([]uint8(nil), in.Moves...), append([]uint16(nil), in.Signal...)
			got, err := readtrim.MuxChangeTrim(in)
			if err != nil {
				t.Fatal(err)
			}
			if got.SequenceEnd != tc.wantEnd || got.Trimmed != (tc.wantEnd < len(tc.q)) {
				t.Fatalf("got end=%d trimmed=%v want end=%d", got.SequenceEnd, got.Trimmed, tc.wantEnd)
			}
			if !reflect.DeepEqual(beforeMoves, in.Moves) || !reflect.DeepEqual(beforeSignal, in.Signal) {
				t.Fatal("input storage changed")
			}
			if got.Read.Sequence != in.Sequence[:tc.wantEnd] || got.Read.Qstring != in.Qstring[:tc.wantEnd] {
				t.Fatal("sequence/quality prefix mismatch")
			}
			if got.Trimmed {
				wantMove := 2 + 3*tc.wantEnd
				if got.MoveEnd != wantMove || got.SignalEnd != wantMove*6 || got.NumSamples != uint64(got.SignalEnd) {
					t.Fatalf("bad trim ends %+v", got)
				}
				if got.Read.PreTrimSequenceLength != uint64(tc.wantEnd) || got.Read.NumMinknowEvents != 0 {
					t.Fatal("trim metadata not updated")
				}
				if cap(got.Read.Moves) != len(got.Read.Moves) || cap(got.Read.Signal) != len(got.Read.Signal) {
					t.Fatal("trimmed result exposes removed suffix capacity")
				}
			} else if !reflect.DeepEqual(got.Read, in) {
				t.Fatal("no-op changed read")
			}
			if len(got.Read.Moves) > 0 && &got.Read.Moves[0] != &in.Moves[0] {
				t.Fatal("moves were copied")
			}
			if len(got.Read.Signal) > 0 && &got.Read.Signal[0] != &in.Signal[0] {
				t.Fatal("signal was copied")
			}
		})
	}
}

func TestNonMuxIdentityAndEmpty(t *testing.T) {
	for _, in := range []readtrim.MuxRead{fixture(tail(100, 20), 6, 1, -1), {Stride: 6}} {
		in.IsMuxChange = false
		got, err := readtrim.MuxChangeTrim(in)
		if err != nil {
			t.Fatal(err)
		}
		if got.Trimmed || !reflect.DeepEqual(got.Read, in) || got.NumSamples != uint64(len(in.Signal)) {
			t.Fatal("non-mux changed")
		}
	}
}

func TestRejectMalformedEvenNonMux(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*readtrim.MuxRead)
	}{
		{"quality-length", func(r *readtrim.MuxRead) { r.Qstring = r.Qstring[:99] }},
		{"base", func(r *readtrim.MuxRead) { r.Sequence = "N" + r.Sequence[1:] }},
		{"quality-low", func(r *readtrim.MuxRead) { r.Qstring = "\x20" + r.Qstring[1:] }},
		{"quality-high", func(r *readtrim.MuxRead) { r.Qstring = "\x7f" + r.Qstring[1:] }},
		{"stride-zero", func(r *readtrim.MuxRead) { r.Stride = 0 }},
		{"stride-overflow", func(r *readtrim.MuxRead) { r.Stride = math.MaxUint32 }},
		{"nonbinary-move", func(r *readtrim.MuxRead) { r.Moves[0] = 2 }},
		{"move-count", func(r *readtrim.MuxRead) { r.Moves[0] = 0 }},
		{"signal-length", func(r *readtrim.MuxRead) { r.Signal = r.Signal[:50] }},
		{"nan", func(r *readtrim.MuxRead) { r.Signal[0] = 0x7e00 }},
		{"infinity", func(r *readtrim.MuxRead) { r.Signal[0] = 0xfc00 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := fixture(tail(100, 20), 6, 0, 0)
			in.IsMuxChange = false
			tc.mutate(&in)
			if _, err := readtrim.MuxChangeTrim(in); err == nil {
				t.Fatal("malformed input accepted")
			}
		})
	}
}

func floatBoundary(n int) readtrim.MuxRead {
	return fixture(strings.Repeat("I", 1048579)+strings.Repeat("!", n-1048579), 1, 0, 0)
}
func TestBinary32ExcessiveBoundary(t *testing.T) {
	for _, tc := range []struct {
		n       int
		trimmed bool
	}{{3495262, true}, {3495263, false}, {3495264, false}} {
		in := floatBoundary(tc.n)
		got, err := readtrim.MuxChangeTrim(in)
		if err != nil {
			t.Fatal(err)
		}
		if got.Trimmed != tc.trimmed {
			t.Fatalf("n=%d trimmed=%v want=%v", tc.n, got.Trimmed, tc.trimmed)
		}
		if got.Trimmed && got.SequenceEnd != 1048578 {
			t.Fatal("binary32 boundary trim end")
		}
	}
}

func syntheticCases() []readtrim.MuxRead {
	var cases []readtrim.MuxRead
	for _, n := range []int{0, 1, 4, 5, 99, 100, 101, 200, 333} {
		for _, low := range []int{0, 1, 3, 4, 5, 6, n / 2, n * 7 / 10, n} {
			if low > n {
				continue
			}
			for _, partial := range []int{-1, 0, 1, 5} {
				if n == 0 && partial < 0 {
					continue
				}
				cases = append(cases, fixture(tail(n, low), 6, 0, partial))
			}
		}
	}
	for _, q := range []string{
		strings.Repeat("I", 90) + "!)!!!!!!!!",
		strings.Repeat("I", 90) + "!-!!!!!!!!",
		strings.Repeat("I", 90) + "!.!!!!!!!!",
		strings.Repeat("!", 40) + strings.Repeat(")", 30) + strings.Repeat("!", 30),
		strings.Repeat("!", 49) + "." + strings.Repeat("!", 50),
	} {
		for zeros := 0; zeros < 4; zeros++ {
			cases = append(cases, fixture(q, 6, zeros, 4))
		}
	}
	random := rand.New(rand.NewSource(0x8b8fc5d))
	for i := 0; i < 600; i++ {
		n := 100 + random.Intn(500)
		q := []byte(tail(n, random.Intn(n+1)))
		for j := 0; j < random.Intn(n); j++ {
			q[random.Intn(n)] = []byte{'!', '(', ')', '-', '.', 'S', '~'}[random.Intn(7)]
		}
		r := fixture(string(q), uint32(1+random.Intn(10)), random.Intn(4), 0)
		r.Signal = r.Signal[:len(r.Signal)-random.Intn(int(r.Stride))]
		r.IsMuxChange = i%13 != 0
		cases = append(cases, r)
	}
	return cases
}

func TestDeterministicCasesAndConcurrentBorrow(t *testing.T) {
	for i, in := range syntheticCases() {
		got, err := readtrim.MuxChangeTrim(in)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		again, err := readtrim.MuxChangeTrim(in)
		if err != nil || !reflect.DeepEqual(got, again) {
			t.Fatal("nondeterministic transform")
		}
		if got.SequenceEnd != len(got.Read.Sequence) || got.MoveEnd != len(got.Read.Moves) || got.SignalEnd != len(got.Read.Signal) {
			t.Fatal("ends disagree with read")
		}
		bases := 0
		for _, m := range got.Read.Moves {
			bases += int(m)
		}
		if bases != len(got.Read.Sequence) {
			t.Fatal("output move base count mismatch")
		}
	}
	in := fixture(tail(100, 20), 6, 2, 1)
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 20 {
				if _, err := readtrim.MuxChangeTrim(in); err != nil {
					t.Error(err)
				}
			}
		}()
	}
	wg.Wait()
}
