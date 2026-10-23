package stitch

import (
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/chunk"
)

type testCase struct {
	name   string
	config Config
	chunks []Chunk
	want   Result
}

func upstreamCases() []testCase {
	sequences := []string{"ACGT", "ACGT", "ACGT", "ACGT", "ACGT", "ACGT", "ACGT"}
	moves := [][]uint8{
		{1, 0, 0, 1, 0, 0, 1, 0, 1, 0}, {1, 0, 0, 1, 0, 0, 0, 1, 0, 1},
		{1, 0, 0, 1, 0, 1, 1, 0, 0, 0}, {1, 0, 0, 1, 0, 0, 1, 0, 1, 0},
		{0, 1, 0, 1, 0, 0, 1, 0, 1, 0}, {1, 0, 0, 0, 0, 0, 1, 0, 1, 1},
		{1, 0, 0, 1, 0, 0, 1, 0, 1, 0},
	}
	offsets := []uint64{0, 7, 14, 21, 28, 35, 40}
	chunks := make([]Chunk, len(offsets))
	for i := range chunks {
		chunks[i] = Chunk{offsets[i], 10, 10, sequences[i], "!&.-", moves[i]}
	}
	return []testCase{
		{"upstream-odd-overlap", Config{50, 1, false}, chunks, Result{
			"ACGTCGCGTCGTCGTCCGT", "!&.-&.&.-&.-&.-&&.-",
			[]uint8{1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0},
		}},
		{"upstream-partial-stride-base", Config{7, 2, false}, []Chunk{{0, 10, 10, "ACGT", "abcd", []uint8{1, 1, 0, 1, 1}}}, Result{"AC", "ab", []uint8{1, 1, 0}}},
		{"upstream-partial-stride-blank", Config{7, 2, false}, []Chunk{{0, 10, 10, "ACG", "abc", []uint8{1, 1, 0, 0, 1}}}, Result{"AC", "ab", []uint8{1, 1, 0}}},
		{"upstream-variable-padded-tail", Config{10, 2, true}, []Chunk{
			{0, 8, 4, "AAAA", "!!!!", []uint8{1, 1, 1, 1}},
			{6, 4, 8, "WXYZ", "wxyz", []uint8{1, 1, 1, 1}},
		}, Result{"AAAAX", "!!!!x", []uint8{1, 1, 1, 1, 1}}},
		{"empty", Config{}, nil, Result{}},
		{"shorter-than-stride", Config{1, 2, false}, []Chunk{{0, 2, 2, "A", "!", []uint8{1}}}, Result{"", "", []uint8{}}},
	}
}

func TestStitchUpstream(t *testing.T) {
	for _, entry := range upstreamCases() {
		t.Run(entry.name, func(t *testing.T) {
			got, err := Stitch(entry.config, entry.chunks)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, entry.want) {
				t.Fatalf("got %#v want %#v", got, entry.want)
			}
		})
	}
}

// These fixtures exercise actual planner intervals, including balanced and Tx
// variable chunks. Their decoder outputs are deterministic synthetic fixtures.
func randomizedCases(t testing.TB) []testCase {
	t.Helper()
	rng := rand.New(rand.NewSource(836352))
	cases := make([]testCase, 0, 720)
	for mode := 0; mode < 3; mode++ {
		for i := 0; i < 240; i++ {
			stride := uint64(1 + rng.Intn(12))
			granularity := stride * uint64(1+rng.Intn(5))
			maxSize := granularity * uint64(5+rng.Intn(16))
			overlap := stride * uint64(1+rng.Intn(4))
			n := uint64(1 + rng.Intn(int(maxSize*5)))
			plan := chunk.Config{ChunkSize: maxSize, Stride: stride, Overlap: overlap, Granularity: granularity}
			var intervals []chunk.Interval
			var err error
			switch mode {
			case 0:
				var offsets []uint64
				offsets, err = chunk.Fixed(n, plan)
				for _, offset := range offsets {
					intervals = append(intervals, chunk.Interval{Start: offset, End: offset + maxSize})
				}
				granularity = maxSize
			case 1:
				intervals, err = chunk.Variable(n, plan)
				granularity = stride
			case 2:
				intervals, err = chunk.Transformer(n, plan)
			}
			if err != nil {
				t.Fatal(err)
			}
			entry := testCase{config: Config{n, stride, mode != 0}}
			for _, interval := range intervals {
				size := interval.End - interval.Start
				padded := ((size + granularity - 1) / granularity) * granularity
				moves := make([]uint8, padded/stride)
				var sequence, quality []byte
				for j := range moves {
					moves[j] = uint8(rng.Intn(2))
					if moves[j] == 1 {
						sequence = append(sequence, "ACGT"[rng.Intn(4)])
						quality = append(quality, byte(33+rng.Intn(70)))
					}
				}
				entry.chunks = append(entry.chunks, Chunk{interval.Start, size, granularity, string(sequence), string(quality), moves})
			}
			cases = append(cases, entry)
		}
	}
	return cases
}

func TestStitchCoverageAndOwnership(t *testing.T) {
	for i, entry := range randomizedCases(t) {
		before := make([]Chunk, len(entry.chunks))
		copy(before, entry.chunks)
		for j := range before {
			before[j].Moves = append([]uint8(nil), before[j].Moves...)
		}
		got, err := Stitch(entry.config, entry.chunks)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		if len(got.Moves) != int(entry.config.RawSamples/entry.config.Stride) {
			t.Fatalf("case %d: lost signal coverage", i)
		}
		bases := 0
		for _, move := range got.Moves {
			bases += int(move)
		}
		if bases != len(got.Sequence) || bases != len(got.Qstring) {
			t.Fatalf("case %d: inconsistent result", i)
		}
		if len(got.Moves) > 0 {
			got.Moves[0] ^= 1
		}
		if !reflect.DeepEqual(entry.chunks, before) {
			t.Fatalf("case %d: mutated or retained input backing data", i)
		}
	}
}

func TestStitchRejectsMalformed(t *testing.T) {
	one := []Chunk{{0, 4, 4, "AC", "!!", []uint8{1, 0, 1, 0}}}
	entries := []struct {
		name   string
		config Config
		chunks []Chunk
	}{
		{"empty-signal", Config{0, 1, false}, one},
		{"zero-stride", Config{4, 0, false}, one},
		{"signal-overflow", Config{math.MaxUint64, 1, false}, one},
		{"stride-overflow", Config{4, math.MaxUint64, false}, one},
		{"coordinate-overflow", Config{4, 1, false}, []Chunk{{math.MaxUint64, 4, 4, "AC", "!!", []uint8{1, 0, 1, 0}}}},
		{"raw-size-overflow", Config{4, 1, false}, []Chunk{{0, math.MaxUint64, 4, "AC", "!!", []uint8{1, 0, 1, 0}}}},
		{"padded-size-overflow", Config{4, 1, false}, []Chunk{{0, maxNativeInt, 2, "", "", nil}}},
		{"zero-granularity", Config{4, 1, false}, []Chunk{{0, 4, 0, "AC", "!!", []uint8{1, 0, 1, 0}}}},
		{"unaligned-granularity", Config{4, 2, false}, []Chunk{{0, 4, 3, "AC", "!!", []uint8{1, 1}}}},
		{"quality-length", Config{4, 1, false}, []Chunk{{0, 4, 4, "AC", "!", []uint8{1, 0, 1, 0}}}},
		{"move-length", Config{4, 1, false}, []Chunk{{0, 4, 4, "AC", "!!", []uint8{1, 0, 1}}}},
		{"move-sum", Config{4, 1, false}, []Chunk{{0, 4, 4, "A", "!", []uint8{1, 0, 1, 0}}}},
		{"non-binary", Config{4, 1, false}, []Chunk{{0, 4, 4, "AC", "!!", []uint8{2, 0, 0, 0}}}},
		{"not-origin", Config{4, 1, false}, []Chunk{{1, 4, 4, "AC", "!!", []uint8{1, 0, 1, 0}}}},
		{"missing-tail", Config{5, 1, false}, one},
		{"variable-beyond-signal", Config{3, 1, true}, one},
		{"unordered", Config{4, 1, false}, append(append([]Chunk(nil), one...), one[0])},
		{"gap", Config{9, 1, false}, []Chunk{one[0], {5, 4, 4, "AC", "!!", []uint8{1, 0, 1, 0}}}},
		{"unaligned-overlap", Config{7, 2, false}, []Chunk{{0, 4, 4, "A", "!", []uint8{1, 0}}, {3, 4, 4, "C", "!", []uint8{0, 1}}}},
	}
	for _, entry := range entries {
		t.Run(entry.name, func(t *testing.T) {
			if result, err := Stitch(entry.config, entry.chunks); err == nil {
				t.Fatalf("accepted malformed input: %#v", result)
			}
		})
	}
}

var benchmarkResult Result

func BenchmarkStitch(b *testing.B) {
	entry := randomizedCases(b)[239]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := Stitch(entry.config, entry.chunks)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkResult = result
	}
}
