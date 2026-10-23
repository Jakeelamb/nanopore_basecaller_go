//go:build cgo

package decode

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

type fixture struct {
	name string
	in   Input
	opt  Options
}

func constantFixture(name string, blocks, states, width int, score, blank float32) fixture {
	in := Input{Blocks: blocks, States: states, Scores: make([]float32, blocks*4*states), BackGuides: make([]float32, (blocks+1)*states), Posts: make([]float32, (blocks+1)*states)}
	for i := range in.Scores {
		in.Scores[i] = score
	}
	for i := range in.Posts {
		in.Posts[i] = 1 / float32(states)
	}
	return fixture{name, in, Options{width, 100, blank, 0, 1}}
}

func differentialFixtures() []fixture {
	cases := []fixture{
		constantFixture("one-event", 1, 4, 1, 0, 0),
		constantFixture("all-ties", 12, 4, 20, 0, 0),
		constantFixture("stay-preferred", 20, 16, 32, -4, 2),
		constantFixture("step-preferred", 20, 64, 80, 2, -4),
		constantFixture("cut-at-one", 8, 16, 13, 0, 0),
		constantFixture("no-cut", 12, 16, 13, 0, 0),
		constantFixture("beam-256", 15, 64, 256, 0, 0),
		constantFixture("initial-guide-ties", 8, 64, 17, -1, 0),
		constantFixture("sparse-post", 10, 64, 31, 0, 0),
		constantFixture("zero-post", 4, 16, 9, 0, 0),
		constantFixture("state-limit", 2, 16384, 2, -1, 0),
	}
	cases[4].opt.BeamCut = 1
	cases[5].opt.BeamCut = 0
	for i := range cases[7].in.BackGuides {
		cases[7].in.BackGuides[i] = float32((i % 7) - 3)
	}
	for i := range cases[8].in.Posts {
		cases[8].in.Posts[i] = 0
	}
	for row := 0; row <= cases[8].in.Blocks; row++ {
		cases[8].in.Posts[row*64+row%64] = 1
	}
	for i := range cases[9].in.Posts {
		cases[9].in.Posts[i] = 0
	}
	random := rand.New(rand.NewSource(8262026))
	for _, states := range []int{4, 16, 64, 256, 1024} {
		for _, width := range []int{1, 2, 7, 16, 31, 80, 256} {
			for variant := 0; variant < 4; variant++ {
				blocks := 1 + random.Intn(48)
				c := constantFixture("random", blocks, states, width, 0, 0)
				c.opt.BlankScore = float32(random.Float64()*6 - 3)
				c.opt.BeamCut = []float32{0, 1, 2, 100}[variant]
				c.opt.QShift = float32(random.Float64()*5 - 2)
				c.opt.QScale = float32(random.Float64()*1.5 + 0.25)
				for i := range c.in.Scores {
					c.in.Scores[i] = float32(random.NormFloat64() * 2.5)
				}
				for i := range c.in.BackGuides {
					c.in.BackGuides[i] = float32(random.NormFloat64() * 4)
				}
				for row := 0; row <= blocks; row++ {
					sum := float32(0)
					for j := 0; j < states; j++ {
						p := float32(random.ExpFloat64())
						c.in.Posts[row*states+j] = p
						sum += p
					}
					for j := 0; j < states; j++ {
						c.in.Posts[row*states+j] /= sum
					}
				}
				if variant == 3 {
					for i := range c.in.Scores {
						c.in.Scores[i] = float32(int(c.in.Scores[i]))
					}
					for i := range c.in.BackGuides {
						c.in.BackGuides[i] = float32(int(c.in.BackGuides[i]))
					}
				}
				cases = append(cases, c)
			}
		}
	}
	return cases
}

func TestDecodeSimple(t *testing.T) {
	c := constantFixture("stay", 4, 4, 4, -10, 2)
	got, err := Decode(c.in, c.opt)
	if err != nil {
		t.Fatal(err)
	}
	if got.Sequence != "A" || got.Qstring != "S" || !reflect.DeepEqual(got.Moves, []uint8{1, 0, 0, 0}) {
		t.Fatalf("unexpected result %#v", got)
	}
	if got.FinalScore != 8 {
		t.Fatalf("score %v want 8", got.FinalScore)
	}
}

func TestDecodeInvariantsAndOwnership(t *testing.T) {
	for i, c := range differentialFixtures() {
		inCopy := Input{c.in.Blocks, c.in.States, append([]float32(nil), c.in.Scores...), append([]float32(nil), c.in.BackGuides...), append([]float32(nil), c.in.Posts...)}
		result, err := Decode(c.in, c.opt)
		if err != nil {
			t.Fatalf("case %d %s: %v", i, c.name, err)
		}
		if !reflect.DeepEqual(c.in, inCopy) {
			t.Fatalf("case %d mutated input", i)
		}
		sum := 0
		for _, m := range result.Moves {
			if m > 1 {
				t.Fatal("nonbinary move")
			}
			sum += int(m)
		}
		if sum != len(result.Sequence) || sum != len(result.Qstring) || len(result.Moves) != c.in.Blocks || result.Moves[0] != 1 {
			t.Fatalf("case %d lengths or first move invalid", i)
		}
		if !finite(result.FinalScore) {
			t.Fatalf("case %d nonfinite score", i)
		}
		for _, q := range []byte(result.Qstring) {
			if q < 34 || q > 83 {
				t.Fatalf("case %d quality byte %d", i, q)
			}
		}
		again, err := Decode(c.in, c.opt)
		if err != nil || !reflect.DeepEqual(result, again) {
			t.Fatalf("case %d not deterministic: %v", i, err)
		}
	}
}

func TestRejectMalformed(t *testing.T) {
	mutations := map[string]func(*fixture){
		"empty":           func(c *fixture) { c.in.Blocks = 0 },
		"negative-blocks": func(c *fixture) { c.in.Blocks = -1 },
		"overflow-blocks": func(c *fixture) { c.in.Blocks = int(^uint(0) >> 1) },
		"non-dna-states":  func(c *fixture) { c.in.States = 8 },
		"small-states":    func(c *fixture) { c.in.States = 1 },
		"large-states":    func(c *fixture) { c.in.States = 65536 },
		"missing-scores":  func(c *fixture) { c.in.Scores = c.in.Scores[:3] },
		"missing-guides":  func(c *fixture) { c.in.BackGuides = nil },
		"missing-posts":   func(c *fixture) { c.in.Posts = nil },
		"zero-beam":       func(c *fixture) { c.opt.BeamWidth = 0 },
		"large-beam":      func(c *fixture) { c.opt.BeamWidth = 257 },
		"negative-cut":    func(c *fixture) { c.opt.BeamCut = -1 },
		"fractional-cut":  func(c *fixture) { c.opt.BeamCut = 0.5 },
		"nan-score":       func(c *fixture) { c.in.Scores[0] = float32(math.NaN()) },
		"infinite-guide":  func(c *fixture) { c.in.BackGuides[0] = float32(math.Inf(1)) },
		"nan-post":        func(c *fixture) { c.in.Posts[0] = float32(math.NaN()) },
		"negative-post":   func(c *fixture) { c.in.Posts[0] = -0.1 },
		"large-post":      func(c *fixture) { c.in.Posts[0] = 1.1 },
		"infinite-option": func(c *fixture) { c.opt.BlankScore = float32(math.Inf(-1)) },
		"unsafe-score":    func(c *fixture) { c.in.Scores[0] = math.MaxFloat32 },
		"unsafe-guide":    func(c *fixture) { c.in.BackGuides[0] = -math.MaxFloat32 },
		"unsafe-blank":    func(c *fixture) { c.opt.BlankScore = math.MaxFloat32 },
		"zero-qscale":     func(c *fixture) { c.opt.QScale = 0 },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			c := constantFixture(name, 3, 16, 4, 0, 0)
			mutate(&c)
			if _, err := Decode(c.in, c.opt); err == nil {
				t.Fatal("malformed input accepted")
			}
		})
	}
}

func TestCRC32CBitOrder(t *testing.T) {
	if got := crc32c(0x12345678, 0, 32); got != 0xfa745634 {
		t.Fatalf("CRC32C 0x%x", got)
	}
	for state := uint32(0); state < 256; state++ {
		seed := crc32c(0x12345678, state, 32)
		for base := uint32(0); base < 4; base++ {
			if crc32c(seed, base, 2) != crc32c(crc32c(seed, base&1, 1), base>>1, 1) {
				t.Fatal("bit order")
			}
		}
	}
}

func TestQualityMidpointRegression(t *testing.T) {
	base := differentialFixtures()[130]
	base.opt.QShift = 0.7775435
	base.opt.QScale = 1
	got, err := Decode(base.in, base.opt)
	if err != nil {
		t.Fatal(err)
	}
	const nativeSequence = "ATGAATTTCGGCA"
	const nativeQuality = "\x23\x22\x22\x22\x22\x22\x22\x22\x23\x22\x22\x23\x22"
	if got.Sequence != nativeSequence || got.Qstring != nativeQuality {
		t.Fatalf("native midpoint regression sequence=%q want %q quality=%x want %x", got.Sequence, nativeSequence, got.Qstring, nativeQuality)
	}
	base.opt.QShift = 0.27545434
	base.opt.QScale = 10
	got, err = Decode(base.in, base.opt)
	if err != nil {
		t.Fatal(err)
	}
	const nativeScaledQuality = "\x29\x27\x28\x27\x28\x26\x28\x28\x29\x28\x27\x29\x26"
	if got.Sequence != nativeSequence || got.Qstring != nativeScaledQuality {
		t.Fatalf("native scaled midpoint regression sequence=%q want %q quality=%x want %x", got.Sequence, nativeSequence, got.Qstring, nativeScaledQuality)
	}
}

func qualityThresholdFixtures(base fixture) []fixture {
	var cases []fixture
	adjacent := func(value float32, steps int) float32 {
		direction := float32(math.Inf(1))
		if steps < 0 {
			direction = float32(math.Inf(-1))
			steps = -steps
		}
		for i := 0; i < steps; i++ {
			value = math.Nextafter32(value, direction)
		}
		return value
	}
	for _, options := range []Options{
		{base.opt.BeamWidth, 100, base.opt.BlankScore, 0.7775435, 1},
		{base.opt.BeamWidth, 100, base.opt.BlankScore, 0.27545434, 10},
	} {
		for delta := -4; delta <= 4; delta++ {
			c := base
			c.name = "independent-quality-midpoint"
			c.opt = options
			c.opt.QShift = adjacent(options.QShift, delta)
			cases = append(cases, c)
		}
	}
	for _, probability := range []float32{0, 1e-6, 0.001, 0.01, 0.1, 0.25, 0.5, 0.9, 0.99} {
		c := constantFixture("quality-byte-threshold", 1, 16, 1, 0, -1)
		for i := range c.in.Posts {
			c.in.Posts[i] = 0
		}
		c.in.Posts[16] = probability
		phred := float32(-10) * floatLog10(float32(1-floatPow(probability, 0.4)))
		for _, scale := range []float32{1, 1.02, 1.05, 10} {
			for _, target := range []float32{1.5, 10.5, 25.5, 49.5} {
				center := target - float32(phred*scale)
				for delta := -4; delta <= 4; delta++ {
					c.opt.QScale = scale
					c.opt.QShift = adjacent(center, delta)
					cases = append(cases, c)
				}
			}
		}
	}
	return cases
}
