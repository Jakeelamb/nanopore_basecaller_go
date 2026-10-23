package model

import (
	"errors"
	"slices"
	"sync"
	"testing"
)

func TestSUPQKVF16Shape(t *testing.T) {
	for _, n := range []int{0, 1, supQKVElements - 1, supQKVElements + 1, 3 * 384 * 384} {
		out, err := prepareSUPQKVF16Weights(make([]uint16, n))
		var shape *qkvWeightShapeError
		if !errors.As(err, &shape) || shape.gotElements != n || shape.wantElements != supQKVElements || out.interleavedHalf != nil || out.tiledRoundedHalf != nil {
			t.Fatalf("length %d: error %v", n, err)
		}
	}
	if _, err := prepareSUPQKVF16Weights(nil); err == nil {
		t.Fatal("nil input accepted")
	}
}

func TestSUPQKVF16AxesAndRounding(t *testing.T) {
	input := make([]uint16, supQKVElements)
	for i := range input {
		input[i] = uint16(i*73 + i/512*29 + i/32768*101)
	}
	before := slices.Clone(input)
	out, err := prepareSUPQKVF16Weights(input)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(input, before) {
		t.Fatal("borrowed input changed")
	}
	for source := range input {
		q, h, d, c := source/(8*64*512), source/(64*512)%8, source/512%64, source%512
		destD := d
		if q != 2 {
			destD = (d%32)*2 + d/32
		}
		inter := ((q*8+h)*64+destD)*512 + c
		tile := q*262144 + h*32768 + (destD/16)*8192 + (c/8)*128 + (destD%16)*8 + c%8
		if out.interleavedHalf[inter] != input[source] {
			t.Fatalf("interleave q=%d h=%d d=%d c=%d", q, h, d, c)
		}
		want := uint16((uint32(input[source]) + 8) & 65520)
		if out.tiledRoundedHalf[tile] != want {
			t.Fatalf("tile q=%d h=%d d=%d c=%d got=%04x want=%04x", q, h, d, c, out.tiledRoundedHalf[tile], want)
		}
	}
	seen := make([]bool, 65536)
	for _, bits := range input {
		seen[bits] = true
	}
	for bits, yes := range seen {
		if !yes {
			t.Fatalf("pattern %04x missing", bits)
		}
	}
}

func TestSUPQKVF16OwnershipAndConcurrent(t *testing.T) {
	input := make([]uint16, supQKVElements)
	for i := range input {
		input[i] = uint16(i)
	}
	before := slices.Clone(input)
	first, err := prepareSUPQKVF16Weights(input)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			next, e := prepareSUPQKVF16Weights(input)
			if e != nil {
				t.Error(e)
				return
			}
			if !slices.Equal(first.interleavedHalf, next.interleavedHalf) || !slices.Equal(first.tiledRoundedHalf, next.tiledRoundedHalf) {
				t.Error("concurrent result differs")
			}
			next.interleavedHalf[0] ^= 1
			next.tiledRoundedHalf[0] ^= 2
		}()
	}
	wg.Wait()
	if first.interleavedHalf[0] != 0 || first.tiledRoundedHalf[0] != 0 || !slices.Equal(input, before) {
		t.Fatal("results alias another invocation or input")
	}
	original := first.tiledRoundedHalf[0]
	first.interleavedHalf[0] ^= 1
	if first.tiledRoundedHalf[0] != original || !slices.Equal(input, before) {
		t.Fatal("interleaved result aliases tiled result or input")
	}
	first.tiledRoundedHalf[0] ^= 2
	if !slices.Equal(input, before) {
		t.Fatal("tiled result aliases input")
	}
}

func TestSUPQKVF16BehavioralControls(t *testing.T) {
	input := make([]uint16, supQKVElements)
	for i := range input {
		input[i] = uint16(i*97 + i/512*11)
	}
	input[0] = 0x7bff
	input[1] = 0x0008
	out, err := prepareSUPQKVF16Weights(input)
	if err != nil {
		t.Fatal(err)
	}
	controls := map[string]bool{
		"omitted Q interleave": out.interleavedHalf[512] != input[512],
		"omitted K interleave": out.interleavedHalf[8*64*512+512] != input[8*64*512+512],
		"V interleave":         out.interleavedHalf[2*8*64*512+512] != input[2*8*64*512+32*512],
		"wrong tile order":     out.tiledRoundedHalf[8] != uint16((uint32(out.interleavedHalf[8])+8)&65520),
		"early rounding":       out.interleavedHalf[0] == 0x7bff,
		"saturation":           out.tiledRoundedHalf[0] == 0x7c00,
		"round to even":        out.tiledRoundedHalf[1] == 0x0010,
	}
	for name, detected := range controls {
		if !detected {
			t.Errorf("control did not distinguish %s", name)
		}
	}
}
