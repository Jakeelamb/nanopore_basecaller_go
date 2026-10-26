package h06

import (
	"encoding/binary"
	"testing"
)

func TestHalfFixtureBoundedCycleAndSourceSentinels(t *testing.T) {
	if InputBytes != 4587520 || TempBytes != 557056 || ArenaBytes != 5144576 {
		t.Fatal("fixed source geometry changed")
	}
	pattern := []uint16{0x4000, 0xc000, 0x3800, 0xb800, 0x0001, 0x8000}
	source := make([]byte, 64*192*2)
	for i := 0; i < len(source)/2; i++ {
		binary.LittleEndian.PutUint16(source[2*i:], pattern[i%len(pattern)])
	}
	before := digest(source)
	got, err := halfFixture(source)
	if err != nil || len(got) != InputBytes {
		t.Fatal(len(got), err)
	}
	for _, row := range []int{0, Frames + 2} {
		for j := 0; j < Batch*Channels; j++ {
			if binary.LittleEndian.Uint16(got[2*(row*Batch*Channels+j):]) != 0x3c00 {
				t.Fatal("sentinel decoy not initialized")
			}
		}
	}
	want := []uint16{0x3c00, 0xbc00, 0x3800, 0xb800, 0x0001, 0x8000}
	for _, row := range []int{1, 2, Frames + 1} {
		for j := 0; j < 12; j++ {
			index := row*Batch*Channels + j
			if binary.LittleEndian.Uint16(got[2*index:]) != want[(index%(len(source)/2))%len(want)] {
				t.Fatal("bounded cycle, subnormal or signed-zero changed")
			}
		}
	}
	if digest(source) != before {
		t.Fatal("source witness mutated")
	}
}

func TestHalfFixtureRejectsWrongExtentAndNonfiniteSource(t *testing.T) {
	for _, source := range [][]byte{nil, {0, 0}} {
		if _, err := halfFixture(source); err == nil {
			t.Fatal("wrong extent admitted")
		}
	}
	for _, bits := range []uint16{0x7c00, 0xfc00, 0x7e01} {
		source := make([]byte, 64*192*2)
		binary.LittleEndian.PutUint16(source, bits)
		if _, err := halfFixture(source); err == nil {
			t.Fatal("nonfinite source silently clamped")
		}
	}
}
