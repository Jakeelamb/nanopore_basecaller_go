package model

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/koiweights"
)

func TestSUPQKVF16Native(t *testing.T) {
	path := os.Getenv(koiweights.FixtureEnv)
	if path == "" {
		t.Skip("set " + koiweights.FixtureEnv + " to require pinned native fixtures")
	}
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	m, err := koiweights.ValidateCapture(path, root)
	if err != nil {
		t.Fatal(err)
	}
	all, err := koiweights.ReadHalf(m.AllHalf)
	if err != nil {
		t.Fatal(err)
	}
	rounded, err := koiweights.ReadHalf(m.AllRounded)
	if err != nil {
		t.Fatal(err)
	}
	for bits := 0; bits < 65536; bits++ {
		if all[bits] != uint16(bits) || rounded[bits] != uint16((uint32(bits)+8)&65520) {
			t.Fatalf("native rounding %04x: %04x", bits, rounded[bits])
		}
	}
	for _, f := range m.Fixtures {
		t.Run(f.Name, func(t *testing.T) {
			input, e := koiweights.ReadHalf(f.Input)
			if e != nil {
				t.Fatal(e)
			}
			before := slices.Clone(input)
			expectedInter, e := koiweights.ReadHalf(f.Interleaved)
			if e != nil {
				t.Fatal(e)
			}
			expectedTile, e := koiweights.ReadHalf(f.Tiled)
			if e != nil {
				t.Fatal(e)
			}
			expectedRound, e := koiweights.ReadHalf(f.Rounded)
			if e != nil {
				t.Fatal(e)
			}
			out, e := prepareSUPQKVF16Weights(input)
			if e != nil {
				t.Fatal(e)
			}
			if !slices.Equal(input, before) {
				t.Fatal("borrowed input changed")
			}
			for i, want := range expectedInter {
				if out.interleavedHalf[i] != want {
					t.Fatalf("interleaved %d got %04x want %04x", i, out.interleavedHalf[i], want)
				}
			}
			for i, want := range expectedRound {
				if out.tiledRoundedHalf[i] != want {
					t.Fatalf("rounded %d got %04x want %04x", i, out.tiledRoundedHalf[i], want)
				}
			}
			for i, want := range expectedTile {
				q, h, db, cb, di, ci := i/262144, i/32768%8, i/8192%4, i/128%64, i/8%16, i%8
				inter := ((q*8+h)*64+db*16+di)*512 + cb*8 + ci
				if out.interleavedHalf[inter] != want {
					t.Fatalf("pre-round tile %d differs", i)
				}
			}
		})
	}
}
