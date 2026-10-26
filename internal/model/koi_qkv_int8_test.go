package model

import (
	"errors"
	"math"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
)

func TestSUPQKVRowArithmetic(t *testing.T) {
	row := make([]uint16, SUPQKVCols)
	copy(row, []uint16{0x5800, 0xd800, 0x3800, 0xb800, 0x3e00, 0xbe00, 1, 0x8001})
	q, inverse, err := QuantizeSUPQKVRows(row)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []int8{127, -127, 0, 0, 2, -2, 0, 0} {
		if q[i] != want {
			t.Fatalf("column %d = %d, want %d", i, q[i], want)
		}
	}
	if len(inverse) != 1 || math.Float32bits(inverse[0]) != 0x3f800000 {
		t.Fatalf("inverse scale %v", inverse)
	}
	if row[0] != 0x5800 {
		t.Fatal("input mutated")
	}
}

func TestSUPQKVRowDomain(t *testing.T) {
	for _, test := range []struct {
		bits   uint16
		reason QKVRowRejection
	}{
		{0, QKVZeroRow}, {0x8000, QKVZeroRow}, {1, QKVScaleOverflowRow}, {0x1800, QKVScaleOverflowRow},
		{0x7c00, QKVNonfiniteRow}, {0xfc00, QKVNonfiniteRow}, {0x7e00, QKVNonfiniteRow},
	} {
		row := make([]uint16, SUPQKVCols)
		row[len(row)-1] = test.bits
		q, s, err := QuantizeSUPQKVRows(row)
		var rejection *QKVRowError
		if !errors.Is(err, ErrSUPQKVInt8) || !errors.As(err, &rejection) || rejection.Row != 0 || rejection.Reason != test.reason || q != nil || s != nil {
			t.Fatalf("bits %04x: q=%v scale=%v error=%v", test.bits, q, s, err)
		}
	}
	for _, bits := range []uint16{0x1801, 0x7bff, 0xfbff} {
		row := make([]uint16, SUPQKVCols)
		row[0] = bits
		row[1] = 1
		if _, _, err := QuantizeSUPQKVRows(row); err != nil {
			t.Fatalf("admitted %04x: %v", bits, err)
		}
	}
	for _, n := range []int{0, 1, 511, 513} {
		if _, _, err := QuantizeSUPQKVRows(make([]uint16, n)); !errors.Is(err, ErrSUPQKVInt8) {
			t.Fatalf("length %d accepted", n)
		}
	}
	if _, err := BuildSUPQKVInt8(make([]uint16, SUPQKVCols)); !errors.Is(err, ErrSUPQKVInt8) {
		t.Fatal("wrong tensor shape accepted")
	}
}

func TestSUPQKVTile(t *testing.T) {
	half := make([]uint16, SUPQKVRows*SUPQKVCols)
	for r := 0; r < SUPQKVRows; r++ {
		for c := 0; c < SUPQKVCols; c++ {
			half[r*SUPQKVCols+c] = signal.Float16Bits(float32((r*13+c)%127 - 63))
		}
		half[r*SUPQKVCols] = 0x5800
	}
	cache, err := BuildSUPQKVInt8(half)
	if err != nil {
		t.Fatal(err)
	}
	if len(cache.Tiled) != len(half) || len(cache.InverseScale) != SUPQKVRows {
		t.Fatal("wrong cache extent")
	}
	for r := 0; r < SUPQKVRows; r++ {
		if math.Float32bits(cache.InverseScale[r]) != 0x3f800000 {
			t.Fatalf("row %d scale", r)
		}
		for c := 0; c < SUPQKVCols; c++ {
			at := ((r/16*32+c/16)*16+r%16)*16 + c%16
			want := int8((r*13+c)%127 - 63)
			if c == 0 {
				want = 127
			}
			if cache.Tiled[at] != want {
				t.Fatalf("row %d column %d at %d = %d want %d", r, c, at, cache.Tiled[at], want)
			}
		}
	}
}
