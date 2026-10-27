package decodergpu

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestFixedHACContractCapacitiesAndMasks(t *testing.T) {
	c := FixedContract()
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	if GuidesBytes != 2162688 || HistoryBytes != 270336 || AuxBytes != 2433024 || ScoresBytes != 4194304 {
		t.Fatal("fixed capacity differs")
	}
	for i, d := range c.Descriptors {
		if d != (Descriptor{int32(i * 32), 32, int32(i * 32), 0}) {
			t.Fatalf("descriptor%d=%v", i, d)
		}
	}
	if c.Scalars != (Scalars{0, 0x3f828f5c, 0xbe99999a, 0x42c80000, 0x40000000, 0}) {
		t.Fatalf("scalar bits %+v", c.Scalars)
	}
	names := []string{"host_back_guide_step", "host_beam_search_step", "host_compute_posts_step", "host_run_decode"}
	for i, s := range c.Stages {
		if s.Function != names[i] {
			t.Fatalf("stage order%d", i)
		}
		if len(s.Compare["aux"]) != 1 || s.Compare["aux"][0] != (Span{0, 2162688}) {
			t.Fatal("history inferred as initialized")
		}
		total := int64(0)
		for _, span := range s.Compare["results"] {
			if span.Offset%32 != 0 && span.Offset%32 != 12 {
				t.Fatal("unknown result word compared")
			}
			total += span.Bytes
		}
		if (i == 0 && total != 0) || (i > 0 && total != 768) {
			t.Fatalf("result mask bytes%d=%d", i, total)
		}
	}
	if err := ValidateWire(c.Parameters(), c.DescriptorBytes()); err != nil {
		t.Fatal(err)
	}
}
func TestFixedContractRejectsDomainAndWireChanges(t *testing.T) {
	for _, mutate := range []func(*Contract){func(c *Contract) { c.Shape[0] = 63 }, func(c *Contract) { c.Shape[1] = 0 }, func(c *Contract) { c.Shape[2] = 4096 }, func(c *Contract) { c.Strides[0]++ }, func(c *Contract) { c.Scalars.MovePad = 1 }, func(c *Contract) { c.Scalars.QScale = math.Float32bits(1) }, func(c *Contract) { c.Descriptors[1][0] = 0 }, func(c *Contract) { c.Descriptors[63][1]++ }, func(c *Contract) { c.Stages[0].Compare["results"] = []Span{{0, 32}} }} {
		c := FixedContract()
		mutate(&c)
		if c.Validate() == nil {
			t.Fatal("mutated fixed contract admitted")
		}
	}
	p, d := FixedContract().Parameters(), FixedContract().DescriptorBytes()
	p[0] ^= 1
	if ValidateWire(p, d) == nil {
		t.Fatal("parameter corruption accepted")
	}
	p = FixedContract().Parameters()
	d[len(d)-1] = 1
	if ValidateWire(p, d) == nil {
		t.Fatal("descriptor corruption accepted")
	}
}
func TestFiniteScoresRequireExactImmutableDigest(t *testing.T) {
	b := make([]byte, ScoresBytes)
	pin := digest(b)
	if err := ValidateScores(b, pin); err != nil {
		t.Fatal(err)
	}
	b[1] = 1
	if ValidateScores(b, pin) == nil {
		t.Fatal("corrupt score admitted")
	}
	if ValidateScores(b[:len(b)-2], digest(b[:len(b)-2])) == nil {
		t.Fatal("short score admitted")
	}
	for _, bits := range []uint16{0x7c00, 0xfc00, 0x7e00, 0x7fff} {
		binary.LittleEndian.PutUint16(b, bits)
		if ValidateScores(b, digest(b)) == nil {
			t.Fatalf("nonfinite half%x admitted", bits)
		}
	}
}
