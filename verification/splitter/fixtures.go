package splitteroracle

import (
	"math/rand"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/splitter"
)

type Case struct {
	Name  string
	Input splitter.Input
}

func fixture(samples int, stride uint64) splitter.Input {
	moves := make([]uint8, uint64(samples)/stride)
	for i := range moves {
		moves[i] = 1
	}
	return splitter.Input{Signal: make([]uint16, samples), Moves: moves, Sequence: strings.Repeat("C", len(moves)), Qstring: strings.Repeat("*", len(moves)), Stride: stride}
}
func spike(in *splitter.Input, start, end int, bits uint16) {
	for i := start; i < end; i++ {
		in.Signal[i] = bits
	}
}
func adapter(in *splitter.Input, start int, sequence string) {
	in.Sequence = in.Sequence[:start] + sequence + in.Sequence[start+len(sequence):]
}
func changed(sequence string, positions ...int) string {
	b := []byte(sequence)
	for _, i := range positions {
		if b[i] == 'A' {
			b[i] = 'G'
		} else {
			b[i] = 'A'
		}
	}
	return string(b)
}

// Cases separates native positive controls from genuine retained negative reads.
func Cases() []Case {
	var cases []Case
	add := func(name string, f func(*splitter.Input)) {
		in := fixture(18000, 6)
		f(&in)
		cases = append(cases, Case{name, in})
	}
	positive := func(in *splitter.Input) { spike(in, 6000, 6004, 0x4200); adapter(in, 1000, splitter.InternalAdapter) }
	add("no-pore", func(*splitter.Input) {})
	add("adapter-without-pore", func(in *splitter.Input) { adapter(in, 1000, splitter.InternalAdapter) })
	add("pore-without-adapter", func(in *splitter.Input) { spike(in, 6000, 6004, 0x4200) })
	add("positive-internal-adapter", positive)
	add("threshold-below", func(in *splitter.Input) { positive(in); spike(in, 6000, 6004, 0x4199) })
	add("threshold-equal", func(in *splitter.Input) { positive(in); spike(in, 6000, 6004, 0x419a) })
	add("threshold-above", func(in *splitter.Input) { positive(in); spike(in, 6000, 6004, 0x419b) })
	add("contiguous-equal-peak-later", func(in *splitter.Input) { spike(in, 6000, 6013, 0x4200); adapter(in, 1002, splitter.InternalAdapter) })
	add("lower-peak-preserves-argmax", func(in *splitter.Input) { positive(in); in.Signal[6000] = 0x4400 })
	add("merged-equal-peak-later", func(in *splitter.Input) {
		positive(in)
		spike(in, 6504, 6505, 0x4200)
		adapter(in, 1084, splitter.InternalAdapter)
	})
	add("merge-gap-500", func(in *splitter.Input) {
		positive(in)
		spike(in, 6504, 6505, 0x4400)
		adapter(in, 1084, splitter.InternalAdapter)
	})
	add("merge-gap-501", func(in *splitter.Input) {
		positive(in)
		spike(in, 6505, 6506, 0x4400)
		adapter(in, 1084, splitter.InternalAdapter)
	})
	add("mean-exactly-10", func(in *splitter.Input) { positive(in); in.Qstring = in.Qstring[:1000] + "+++++" + in.Qstring[1005:] })
	add("mean-above-10", func(in *splitter.Input) { positive(in); in.Qstring = in.Qstring[:1000] + ",++++" + in.Qstring[1005:] })
	add("mean-fractional-below-10", func(in *splitter.Input) { positive(in); in.Qstring = in.Qstring[:1000] + "*++++" + in.Qstring[1005:] })
	add("clipped-quality-window", func(in *splitter.Input) { spike(in, 17994, 17995, 0x4200) })
	add("exclusive-end-outside-moves", func(in *splitter.Input) { spike(in, 17999, 18000, 0x4200) })
	add("uncalled-prefix", func(in *splitter.Input) {
		positive(in)
		for i := 0; i < 1200; i++ {
			in.Moves[i] = 0
		}
		in.Sequence = in.Sequence[:1800]
		in.Qstring = in.Qstring[:1800]
	})
	add("four-adapter-edits", func(in *splitter.Input) {
		positive(in)
		adapter(in, 1000, changed(splitter.InternalAdapter, 2, 6, 10, 14))
	})
	add("five-adapter-edits", func(in *splitter.Input) {
		positive(in)
		adapter(in, 1000, changed(splitter.InternalAdapter, 2, 6, 10, 14, 18))
	})
	add("N-is-exact", func(in *splitter.Input) {
		positive(in)
		q := []byte(splitter.InternalAdapter)
		for _, i := range []int{2, 6, 10, 14, 18} {
			q[i] = 'N'
		}
		adapter(in, 1000, string(q))
	})
	add("adapter-search-right-four-deletions", func(in *splitter.Input) { spike(in, 6000, 6004, 0x4200); adapter(in, 1031, splitter.InternalAdapter) })
	add("adapter-search-right-five-deletions", func(in *splitter.Input) { spike(in, 6000, 6004, 0x4200); adapter(in, 1032, splitter.InternalAdapter) })
	add("adapter-first-native-tie", func(in *splitter.Input) {
		spike(in, 6000, 6004, 0x4200)
		adapter(in, 1000, splitter.InternalAdapter+"CCC"+splitter.InternalAdapter)
	})
	for _, offset := range []int{4999, 5000} {
		in := fixture(18000, 6)
		spike(&in, offset, offset+1, 0x4200)
		cases = append(cases, Case{map[int]string{4999: "prefix-4999", 5000: "prefix-5000"}[offset], in})
	}
	for _, width := range []int{499, 500} {
		in := fixture(8000, 1)
		spike(&in, 5500, 5500+width, 0x4200)
		name := "width-500-passes"
		if width == 500 {
			name = "width-501-rejected"
		}
		cases = append(cases, Case{name, in})
	}
	for _, highFirst := range []bool{false, true} {
		in := fixture(13000, 1)
		for i := 0; i < 12; i++ {
			at := 5100 + i*600
			spike(&in, at, at+1, 0x4200)
			adapter(&in, at, splitter.InternalAdapter)
		}
		name := "top-ten-equal-peaks-later-pairs"
		if highFirst {
			in.Signal[5100] = 0x4800
			name = "top-ten-higher-early-peak"
		}
		cases = append(cases, Case{name, in})
	}
	for _, n := range []int{9, 10, 11} {
		in := fixture(13000, 1)
		for i := 0; i < n; i++ {
			at := 5100 + i*600
			spike(&in, at, at+1, 0x4200)
			adapter(&in, at, splitter.InternalAdapter)
		}
		name := map[int]string{9: "top-count-nine", 10: "top-count-ten", 11: "top-count-eleven"}[n]
		cases = append(cases, Case{name, in})
	}
	for _, ceil := range []bool{false, true} {
		in := fixture(18001, 6)
		positive(&in)
		name := "partial-stride-floor-moves"
		if ceil {
			in.Moves = append(in.Moves, 1)
			in.Sequence += "C"
			in.Qstring += "*"
			name = "partial-stride-ceil-moves"
		}
		cases = append(cases, Case{name, in})
		in.Signal = append([]uint16(nil), in.Signal...)
		for i := range in.Signal {
			in.Signal[i] = 0
		}
		in.Signal[18000] = 0x4200
		tailName := "partial-tail-floor-rejected"
		if ceil {
			tailName = "partial-tail-ceil-admitted"
		}
		cases = append(cases, Case{tailName, in})
	}
	for _, samples := range []int{1, 4999, 5000} {
		in := fixture(samples, 1)
		spike(&in, 0, samples, 0x4200)
		cases = append(cases, Case{map[int]string{1: "one-sample", 4999: "below-prefix-size", 5000: "at-prefix-size"}[samples], in})
	}
	zero := fixture(6000, 1)
	for i := range zero.Moves {
		zero.Moves[i] = 0
	}
	zero.Sequence = ""
	zero.Qstring = ""
	spike(&zero, 5100, 5101, 0x4200)
	cases = append(cases, Case{"no-called-bases", zero})
	rng := rand.New(rand.NewSource(20261023))
	for i := 0; i < 32; i++ {
		in := fixture(18000+int(rng.Int31n(6)), 6)
		for j := range in.Signal {
			if rng.Intn(300) == 0 {
				in.Signal[j] = uint16(0x4198 + rng.Intn(16))
			} else {
				in.Signal[j] = uint16(rng.Intn(0x419a))
				if rng.Intn(2) == 0 {
					in.Signal[j] |= 0x8000
				}
			}
		}
		for j := range in.Moves {
			if rng.Intn(3) == 0 {
				in.Moves[j] = 0
			}
		}
		bases := 0
		for _, m := range in.Moves {
			bases += int(m)
		}
		in.Sequence = strings.Repeat("C", bases)
		in.Qstring = strings.Repeat("*", bases)
		cases = append(cases, Case{"seeded-sparse", in})
	}
	return cases
}
