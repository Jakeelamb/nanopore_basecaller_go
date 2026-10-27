package model

import (
	"errors"
	"math"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
)

func TestQuantizeFLSTMRowsRejectsOverflow(t *testing.T) {
	if _, _, err := QuantizeFLSTMRows(nil, 1<<62, 4); !errors.Is(err, ErrGPUWeights) {
		t.Fatalf("overflowing shape returned %v, want ErrGPUWeights", err)
	}
}

func TestFLSTMPackingRejectsOverflow(t *testing.T) {
	for _, p := range []FLSTMParams{{C: 1 << 62, K: 32}, {C: 16, K: 1 << 62}, {C: 1 << 32, K: 1 << 32}} {
		if _, err := PackFLSTMLayer(p, FLSTMWeightSet{}); !errors.Is(err, ErrGPUWeights) {
			t.Fatalf("geometry %+v returned %v, want ErrGPUWeights", p, err)
		}
	}
	if _, err := permuteContiguous([]byte(nil), []int64{1 << 62, 4}, []int{0, 1}); !errors.Is(err, ErrGPUWeights) {
		t.Fatalf("overflowing permutation returned %v", err)
	}
	if _, err := resolveMinusOne([]int64{-1, 1 << 62, 4}, 0); !errors.Is(err, ErrGPUWeights) {
		t.Fatalf("overflowing view returned %v", err)
	}
}

// halfOf is the test's spelling of the `to(opts_f16)` cast.
func halfOf(v float32) uint16 { return signal.Float16Bits(v) }

func halvesOf(values ...float32) []uint16 {
	out := make([]uint16, len(values))
	for i, v := range values {
		out[i] = signal.Float16Bits(v)
	}
	return out
}

// TestQuantizeFLSTMRowsHandComputed pins the whole quantizer on one row whose
// arithmetic can be checked by hand.
//
//	row      = [1.0234375, 1.0117188, -1.0234375, 0]
//	amax     = 1.0234375
//	scale    = half(128 / 1.0234375) = 125.0625
//	products = [128, 126.528076171875, -128, 0]
//	-> half products: [128, 126.5, -128, 0]
//	-> round half to even: [128, 126, -128, 0]
//	-> clip +-127: [127, 126, -127, 0]
//
// The second element is the reason the half product is rounded before the
// integer rounding: rounding the exact product 126.528076171875 directly would
// give 127, but the source's `.round()` runs on the binary16 tensor, whose
// nearest value is 126.5, and 126.5 rounds to the even 126.
func TestQuantizeFLSTMRowsHandComputed(t *testing.T) {
	row := halvesOf(1.0234375, 1.0117188, -1.0234375, 0)
	q, scale, err := QuantizeFLSTMRows(row, 1, 4)
	if err != nil {
		t.Fatal(err)
	}
	want := []int8{127, 126, -127, 0}
	for i := range want {
		if q[i] != want[i] {
			t.Fatalf("quantized[%d] = %d, want %d", i, q[i], want[i])
		}
	}
	if len(scale) != 1 {
		t.Fatalf("scale has %d entries, want 1", len(scale))
	}
	// The published scale widens the rounded half, not the quotient.
	if got := scale[0]; got != 125.0625 {
		t.Fatalf("scale = %v, want 125.0625", got)
	}
}

// TestQuantizeFLSTMRowsScaleIsTheHalfScale requires the multiplier and the
// published scale to be the binary16 value, not the float32 quotient: 128/3 is
// 42.666... whose nearest binary16 is 42.65625.
func TestQuantizeFLSTMRowsScaleIsTheHalfScale(t *testing.T) {
	row := halvesOf(3, 1, -2)
	q, scale, err := QuantizeFLSTMRows(row, 1, 3)
	if err != nil {
		t.Fatal(err)
	}
	wantScale := signal.Float32FromHalf(signal.Float16Bits(float32(128.0 / 3.0)))
	if scale[0] != wantScale {
		t.Fatalf("scale = %v, want the binary16 value %v", scale[0], wantScale)
	}
	if wantScale == float32(128.0/3.0) {
		t.Fatal("the chosen amax does not exercise the half rounding of the scale")
	}
	// 3 * 42.65625 = 127.96875 -> 127; 1 * 42.65625 = 42.65625 -> 43 (round half
	// to even is not involved); -2 * 42.65625 = -85.3125 -> -85.
	for i, want := range []int8{127, 43, -85} {
		if q[i] != want {
			t.Fatalf("quantized[%d] = %d, want %d", i, q[i], want)
		}
	}
}

// TestQuantizeFLSTMRowsReducesDimensionOne proves the reduction axis: two rows
// with very different magnitudes must get two different scales, and each row's
// largest element must map to +-127.
func TestQuantizeFLSTMRowsReducesDimensionOne(t *testing.T) {
	// Two rows of three columns. A dim-0 reduction would produce one scale and
	// scale both rows by the global maximum.
	half := halvesOf(1, 2, 0.5, 100, 50, 25)
	q, scale, err := QuantizeFLSTMRows(half, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(scale) != 2 {
		t.Fatalf("got %d scales, want one per row", len(scale))
	}
	if scale[0] == scale[1] {
		t.Fatalf("both rows got scale %v; the reduction is not per row", scale[0])
	}
	for r := 0; r < 2; r++ {
		if q[r*3] != 127 && q[r*3+1] != 127 {
			t.Fatalf("row %d has no element at the clip bound: %v", r, q[r*3:r*3+3])
		}
	}
}

// TestQuantizeFLSTMRowsRejectsUndefinedDomain pins the admitted domain: an
// all-zero row makes `128 / amax` infinite and the following Char cast
// undefined, so it is a rejection rather than a silent garbage value.
func TestQuantizeFLSTMRowsRejectsUndefinedDomain(t *testing.T) {
	if _, _, err := QuantizeFLSTMRows(halvesOf(1, 0), 1, 2); err != nil {
		t.Fatalf("a finite row was rejected: %v", err)
	}
	if _, _, err := QuantizeFLSTMRows(halvesOf(0, 0), 1, 2); err == nil {
		t.Fatal("an all-zero row was accepted")
	}
	if _, _, err := QuantizeFLSTMRows(halvesOf(1, 0), 2, 2); err == nil {
		t.Fatal("a short input was accepted")
	}
	inf := signal.Float16Bits(float32(math.Inf(1)))
	if _, _, err := QuantizeFLSTMRows([]uint16{inf, 1}, 1, 2); err == nil {
		t.Fatal("a non-finite weight was accepted")
	}
}

// referencePermute scatters instead of gathering: it walks the source in
// row-major order and writes each element at its permuted destination. It is a
// deliberately different algorithm from permuteContiguous so a gather-index bug
// cannot hide.
func referencePermute[T any](src []T, shape []int64, perm []int) []T {
	rank := len(shape)
	strides := make([]int64, rank)
	stride := int64(1)
	for i := rank - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	dstShape := make([]int64, rank)
	for i, p := range perm {
		dstShape[i] = shape[p]
	}
	dstStrides := make([]int64, rank)
	stride = 1
	for i := rank - 1; i >= 0; i-- {
		dstStrides[i] = stride
		stride *= dstShape[i]
	}
	out := make([]T, len(src))
	index := make([]int64, rank)
	for flat := int64(0); flat < int64(len(src)); flat++ {
		rem := flat
		for i := 0; i < rank; i++ {
			index[i] = rem / strides[i]
			rem %= strides[i]
		}
		var at int64
		for i, p := range perm {
			at += index[p] * dstStrides[i]
		}
		out[at] = src[flat]
	}
	return out
}

// TestPermuteContiguousHandComputed pins the semantics of
// view(shape).permute(perm).contiguous() on a shape small enough to write out.
func TestPermuteContiguousHandComputed(t *testing.T) {
	// [2,3] permuted by {1,0} is the transpose.
	src := []int{0, 1, 2, 3, 4, 5}
	got, err := permuteContiguous(src, []int64{2, 3}, []int{1, 0})
	if err != nil {
		t.Fatal(err)
	}
	if want := []int{0, 3, 1, 4, 2, 5}; !equalInts(got, want) {
		t.Fatalf("transpose = %v, want %v", got, want)
	}
	// [2,2,2] permuted by {2,0,1}: destination [i2][i0][i1].
	src = []int{0, 1, 2, 3, 4, 5, 6, 7}
	got, err = permuteContiguous(src, []int64{2, 2, 2}, []int{2, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	// src[i0][i1][i2] = 4*i0+2*i1+i2 goes to dst[i2][i0][i1].
	want := make([]int, 8)
	for i0 := 0; i0 < 2; i0++ {
		for i1 := 0; i1 < 2; i1++ {
			for i2 := 0; i2 < 2; i2++ {
				want[(i2*2+i0)*2+i1] = 4*i0 + 2*i1 + i2
			}
		}
	}
	if !equalInts(got, want) {
		t.Fatalf("permute = %v, want %v", got, want)
	}
	if _, err := permuteContiguous(src, []int64{2, 2, 2}, []int{0, 0, 1}); err == nil {
		t.Fatal("a repeated axis was accepted")
	}
	if _, err := permuteContiguous(src, []int64{2, 5}, []int{1, 0}); err == nil {
		t.Fatal("a shape that does not match the source was accepted")
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// syntheticLayer builds a deterministic, non-symmetric weight set so every
// permutation and both reduction axes are observable.
func syntheticLayer(p FLSTMParams) FLSTMWeightSet {
	gen := func(n int64, seed float32) []float32 {
		out := make([]float32, n)
		x := seed
		for i := range out {
			x = float32(math.Mod(float64(x*1.6180339887+0.37), 2.5)) - 1.25
			out[i] = x
		}
		return out
	}
	return FLSTMWeightSet{
		DownIH:   gen(p.K*p.C, 0.11),
		DownHH:   gen(p.K*p.C, 0.29),
		UpIH:     gen(p.UpRows()*p.K, 0.53),
		UpHH:     gen(p.UpRows()*p.K, 0.71),
		UpBiasIH: gen(p.UpRows(), 0.97),
		UpBiasHH: gen(p.UpRows(), 1.31),
	}
}

// TestPackFLSTMLayerMatchesScatterReference recomputes every packed tensor from
// the source's expressions with the scatter reference and compares byte for
// byte. It also proves the two down sides are not interchangeable and that the
// up projection really is the ih/hh concatenation.
func TestPackFLSTMLayerMatchesScatterReference(t *testing.T) {
	p := FLSTMParams{C: 16, K: 64}
	w := syntheticLayer(p)
	c, err := PackFLSTMLayer(p, w)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := c.DownShape(), []int64{64, 16}; len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("down shape %v, want %v", got, want)
	}
	if got, want := c.UpShape(), []int64{64, 128}; got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("up shape %v, want %v", got, want)
	}

	// Down: quantize each side, then the source's 7-D permute.
	for _, side := range []struct {
		name  string
		src   []float32
		got   []int8
		scale []uint16
	}{
		{"ih", w.DownIH, c.DownIH, c.DownScaleIH},
		{"hh", w.DownHH, c.DownHH, c.DownScaleHH},
	} {
		half := CastFLSTMHalf(side.src)
		q, scale, qerr := QuantizeFLSTMRows(half, p.K, p.C)
		if qerr != nil {
			t.Fatal(qerr)
		}
		// The source's `-1` dims, spelled out: K/32 for the down weights and
		// C/16 for the up projection (FLSTMStack.cpp:170-215).
		wantQ := referencePermute(q, []int64{p.K / 32, 2, 2, 2, 2, 2, p.C}, []int{2, 0, 4, 1, 3, 5, 6})
		if !equalI8(side.got, wantQ) {
			t.Fatalf("down %s weights differ from the scatter reference", side.name)
		}
		wantS := referencePermute(scale, []int64{p.K / 32, 2, 2, 2, 2, 2}, []int{2, 0, 4, 1, 3, 5})
		if int64(len(wantS)) != p.K {
			t.Fatalf("down %s scale reference has %d entries, want K %d", side.name, len(wantS), p.K)
		}
		for i, v := range wantS {
			if side.scale[i] != signal.Float16Bits(v) {
				t.Fatalf("down %s scale[%d] = %#04x, want %#04x", side.name, i, side.scale[i], signal.Float16Bits(v))
			}
		}
	}
	if equalI8(c.DownIH, c.DownHH) {
		t.Fatal("the two down projections packed to the same bytes")
	}

	// Up: cat(ih, hh, dim 1) then quantize then the source's 7-D permute.
	joined := concatRows(w.UpIH, w.UpHH, p.UpRows(), p.K)
	q, scale, err := QuantizeFLSTMRows(CastFLSTMHalf(joined), p.UpRows(), p.UpCols())
	if err != nil {
		t.Fatal(err)
	}
	if !equalI8(c.Up, referencePermute(q, []int64{4, p.C / 16, 2, 2, 2, 2, p.UpCols()}, []int{1, 5, 0, 2, 3, 4, 6})) {
		t.Fatal("up weights differ from the scatter reference")
	}
	halfScale := make([]uint16, len(scale))
	for i, v := range scale {
		halfScale[i] = signal.Float16Bits(v)
	}
	if !equalU16(c.UpScale, referencePermute(halfScale, []int64{4, p.C / 16, 2, 2, 2, 2}, []int{1, 5, 2, 3, 4, 0})) {
		t.Fatal("up scale differs from the scatter reference")
	}
	if !equalU16(c.UpBias, referencePermute(CastFLSTMHalf(w.UpBiasIH), []int64{4, p.C / 16, 2, 2, 2, 2}, []int{1, 5, 2, 3, 4, 0})) {
		t.Fatal("up bias differs from the scatter reference")
	}

	// The up projection must be the concatenation, not either half alone.
	aloneIH, err := PackFLSTMLayer(p, FLSTMWeightSet{
		DownIH: w.DownIH, DownHH: w.DownHH,
		UpIH: w.UpIH, UpHH: w.UpIH, UpBiasIH: w.UpBiasIH, UpBiasHH: w.UpBiasHH,
	})
	if err != nil {
		t.Fatal(err)
	}
	if equalI8(c.Up, aloneIH.Up) {
		t.Fatal("doubling the ih half reproduced the concatenated up cache")
	}
}

// TestPackFLSTMLayerIgnoresHiddenBias pins that the cache carries the input
// bias only: changing up_bias_hh must not change a single packed byte, because
// the Koi path never reads it (FLSTMStack.cpp:212-216).
func TestPackFLSTMLayerIgnoresHiddenBias(t *testing.T) {
	p := FLSTMParams{C: 16, K: 64}
	w := syntheticLayer(p)
	a, err := PackFLSTMLayer(p, w)
	if err != nil {
		t.Fatal(err)
	}
	other := w
	other.UpBiasHH = make([]float32, len(w.UpBiasHH))
	for i := range other.UpBiasHH {
		other.UpBiasHH[i] = 7
	}
	b, err := PackFLSTMLayer(p, other)
	if err != nil {
		t.Fatal(err)
	}
	if !equalU16(a.UpBias, b.UpBias) || !equalI8(a.Up, b.Up) {
		t.Fatal("up_bias_hh changed the packed cache")
	}
	// And the packed bias is the permuted input bias, not the hidden one.
	wantBias := referencePermute(CastFLSTMHalf(w.UpBiasIH), []int64{4, p.C / 16, 2, 2, 2, 2}, []int{1, 5, 2, 3, 4, 0})
	if !equalU16(a.UpBias, wantBias) {
		t.Fatal("packed bias is not the input bias")
	}
}

// TestPackFLSTMLayerRejects covers the declared geometry and domain rejections.
func TestPackFLSTMLayerRejects(t *testing.T) {
	p := FLSTMParams{C: 16, K: 64}
	w := syntheticLayer(p)
	if _, err := PackFLSTMLayer(p, w); err != nil {
		t.Fatalf("the valid layer was rejected: %v", err)
	}
	short := w
	short.DownIH = w.DownIH[:len(w.DownIH)-1]
	if _, err := PackFLSTMLayer(p, short); err == nil {
		t.Fatal("a short down weight was accepted")
	}
	noBias := w
	noBias.UpBiasIH = nil
	if _, err := PackFLSTMLayer(p, noBias); err == nil {
		t.Fatal("a missing bias was accepted")
	}
	zeroRow := w
	zeroRow.DownHH = append([]float32(nil), w.DownHH...)
	for i := int64(0); i < p.C; i++ {
		zeroRow.DownHH[i] = 0
	}
	if _, err := PackFLSTMLayer(p, zeroRow); err == nil {
		t.Fatal("an all-zero down row was accepted")
	}
	for _, bad := range []FLSTMParams{{C: 0, K: 64}, {C: 16, K: 0}, {C: 15, K: 64}, {C: 16, K: 31}} {
		if _, err := PackFLSTMLayer(bad, w); err == nil {
			t.Fatalf("geometry %+v was accepted", bad)
		}
	}
	if _, err := PackFLSTMStack(p, nil); err == nil {
		t.Fatal("an empty stack was accepted")
	}
}

func equalI8(a, b []int8) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func equalU16(a, b []uint16) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestPackFLSTMLayerMutantsDiffer is the Go half of the H04 rejection controls:
// every mutant must either change at least one packed cache byte on the same
// weights or be rejected outright, or the step it switches off was never
// load-bearing. A mutant that reproduces the source's bytes would mean the cache
// cannot distinguish it. The H04 differential repeats this against the native
// cache, where "differ" means "the native bytes reject it".
func TestPackFLSTMLayerMutantsDiffer(t *testing.T) {
	p := FLSTMParams{C: 16, K: 64}
	w := syntheticLayer(p)
	base, err := PackFLSTMLayer(p, w)
	if err != nil {
		t.Fatal(err)
	}
	roles := func(c FLSTMCache) map[string][]byte {
		return map[string][]byte{
			"dn-ih-cache":       i8Bytes(c.DownIH),
			"dn-ih-cache-scale": u16Bytes(c.DownScaleIH),
			"dn-hh-cache":       i8Bytes(c.DownHH),
			"dn-hh-cache-scale": u16Bytes(c.DownScaleHH),
			"up-cache":          i8Bytes(c.Up),
			"up-cache-scale":    u16Bytes(c.UpScale),
			"up-cache-bias":     u16Bytes(c.UpBias),
		}
	}
	want := roles(base)
	mutants := []FLSTMMutant{
		FLSTMMutantReductionAxis0, FLSTMMutantFloat32Quantize, FLSTMMutantEpsilon,
		FLSTMMutantClip128, FLSTMMutantTruncate, FLSTMMutantIdentityPermute,
		FLSTMMutantHiddenBias,
	}
	for _, m := range mutants {
		got, merr := PackFLSTMLayerMutant(p, w, m)
		if merr != nil {
			// The source's own view cannot hold the mutant's scale tensor, so
			// the construction is not merely different, it is malformed. That is
			// still a rejection by the port, which is what the control claims.
			continue
		}
		changed := 0
		for role, bytes := range roles(got) {
			if !equalBytes(bytes, want[role]) {
				changed++
			}
		}
		if changed == 0 {
			t.Fatalf("mutant %s produced the source's cache bytes", m)
		}
	}
	// The hidden-bias mutant must change exactly the bias: it is the only step
	// that switch touches.
	hidden, err := PackFLSTMLayerMutant(p, w, FLSTMMutantHiddenBias)
	if err != nil {
		t.Fatal(err)
	}
	got := roles(hidden)
	for role, bytes := range got {
		wantSame := role != "up-cache-bias"
		if same := equalBytes(bytes, want[role]); same != wantSame {
			t.Fatalf("hidden-bias mutant role %s equal=%v, want %v", role, same, wantSame)
		}
	}
}

func i8Bytes(v []int8) []byte {
	out := make([]byte, len(v))
	for i, x := range v {
		out[i] = byte(x)
	}
	return out
}

func u16Bytes(v []uint16) []byte {
	out := make([]byte, len(v)*2)
	for i, x := range v {
		out[i*2] = byte(x)
		out[i*2+1] = byte(x >> 8)
	}
	return out
}

func equalBytes(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
