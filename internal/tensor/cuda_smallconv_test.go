//go:build cgo && linux

package tensor

import (
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
)

func signalFloat16Bits(value float32) uint16 { return signal.Float16Bits(value) }

// smallConvFixture builds the H02 conv0 geometry on the host-memory backend:
// input [64,192,1] F16, weights [5,16] F16, bias [16] F16, output [64,192,16].
type smallConvFixture struct {
	d                     *Device
	s                     *Stream
	in, w, b, out         *Storage
	weightSource, biasSrc *Storage
	packed, packedBias    *Storage
}

func newSmallConvFixture(t *testing.T) *smallConvFixture {
	t.Helper()
	e := fakeStorageEngine(t)
	d, err := e.NewCUDADevice(0)
	if err != nil {
		t.Fatal(err)
	}
	s, err := d.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	f := &smallConvFixture{d: d, s: s}
	f.in = mustSmallConvStorage(t, d, s, DTypeF16, []int64{64, 192, 1}, 64*192*1)
	f.w = mustSmallConvStorage(t, d, s, DTypeF16, []int64{5, 16}, 5*16)
	f.b = mustSmallConvStorage(t, d, s, DTypeF16, []int64{16}, 16)
	f.out = mustSmallConvStorage(t, d, s, DTypeF16, []int64{64, 192, 16}, 64*192*16)
	f.weightSource = mustSmallConvStorage(t, d, s, DTypeF32, []int64{16, 1, 5}, 16*1*5)
	f.biasSrc = mustSmallConvStorage(t, d, s, DTypeF32, []int64{16}, 16)
	f.packed = mustSmallConvStorage(t, d, s, DTypeF16, []int64{5, 16}, 5*16)
	f.packedBias = mustSmallConvStorage(t, d, s, DTypeF16, []int64{16}, 16)
	return f
}

func mustSmallConvStorage(t *testing.T, d *Device, s *Stream, dtype DType, shape []int64, elems int) *Storage {
	t.Helper()
	tags := make([]KoiTag, len(shape))
	for i := range tags {
		tags[i] = TagC
	}
	st, err := d.NewStorage(s, dtype, shape, tags, make([]byte, elems*int(dtypeBytesForTest(dtype))))
	if err != nil {
		t.Fatal(err)
	}
	return st
}

func dtypeBytesForTest(dtype DType) int64 {
	switch dtype {
	case DTypeF16:
		return 2
	case DTypeF32, DTypeI32:
		return 4
	}
	return 1
}

func float32BytesTest(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// packSmallConvForTest and packHalfForTest restate the packing formula so this
// test can catch a swapped handle or a dropped geometry field in the request.
// The independent proof that the formula itself matches the pinned source is the
// H02 differential capture, not this test.
func packSmallConvForTest(weight []float32, cOut, cIn, win int64) ([]byte, error) {
	if int64(len(weight)) != cOut*cIn*win {
		return nil, errors.New("weight size")
	}
	bits := make([]uint16, len(weight))
	for co := int64(0); co < cOut; co++ {
		for ci := int64(0); ci < cIn; ci++ {
			for w := int64(0); w < win; w++ {
				bits[(w*cIn+ci)*cOut+co] = Float16BitsForTest(weight[(co*cIn+ci)*win+w])
			}
		}
	}
	return halfBytesForTest(bits), nil
}

func packHalfForTest(values []float32) []byte {
	bits := make([]uint16, len(values))
	for i, v := range values {
		bits[i] = Float16BitsForTest(v)
	}
	return halfBytesForTest(bits)
}

func halfBytesForTest(bits []uint16) []byte {
	out := make([]byte, len(bits)*2)
	for i, v := range bits {
		binary.LittleEndian.PutUint16(out[i*2:], v)
	}
	return out
}

// Float16BitsForTest is the same RNE cast the caller uses. It is duplicated
// here because internal/tensor cannot import internal/signal's model-facing
// helpers without a cycle; the cast itself is pinned by the model unit test.
func Float16BitsForTest(value float32) uint16 {
	return signalFloat16Bits(value)
}

func hacConv0Geometry() SmallConvGeometry {
	return SmallConvGeometry{N: 64, CIn: 1, COut: 16, TIn: 192, Window: 5, Stride: 1,
		Padding: 2, OutNStride: 192 * 16, Activation: KoiSwish}
}

// TestSmallConvF16AcceptsPinnedGeometry proves the Go request reaches the
// boundary with every geometry field intact: the host-memory backend recomputes
// the same contract, so a dropped or swapped field is rejected.
func TestSmallConvF16AcceptsPinnedGeometry(t *testing.T) {
	f := newSmallConvFixture(t)
	if err := f.d.SmallConvF16(f.s, f.in, f.w, f.b, f.out, hacConv0Geometry()); err != nil {
		t.Fatal(err)
	}
}

// TestSmallConvF16GoRejections pins the Go-side contract. Every case must be
// refused before the native call; the native guard is proven separately by the
// H02 controls, which use SmallConvF16Unchecked.
func TestSmallConvF16GoRejections(t *testing.T) {
	f := newSmallConvFixture(t)
	cases := []struct {
		name string
		run  func() error
		want error
	}{
		{"padding must be window/2", func() error {
			g := hacConv0Geometry()
			g.Padding = 0
			return f.d.SmallConvF16(f.s, f.in, f.w, f.b, f.out, g)
		}, nil},
		{"input length must be a stride multiple", func() error {
			g := hacConv0Geometry()
			g.Stride = 5
			return f.d.SmallConvF16(f.s, f.in, f.w, f.b, f.out, g)
		}, nil},
		{"negative geometry", func() error {
			g := hacConv0Geometry()
			g.N = 0
			return f.d.SmallConvF16(f.s, f.in, f.w, f.b, f.out, g)
		}, nil},
		{"unknown activation", func() error {
			g := hacConv0Geometry()
			g.Activation = KoiTanhX5
			return f.d.SmallConvF16(f.s, f.in, f.w, f.b, f.out, g)
		}, nil},
		{"output batch stride must match the view", func() error {
			g := hacConv0Geometry()
			g.OutNStride = 192*16 + 144
			return f.d.SmallConvF16(f.s, f.in, f.w, f.b, f.out, g)
		}, nil},
		{"input shape must be N,T_in,C_in", func() error {
			g := hacConv0Geometry()
			g.CIn = 2
			return f.d.SmallConvF16(f.s, f.in, f.w, f.b, f.out, g)
		}, nil},
		{"weights must be F16", func() error {
			return f.d.SmallConvF16(f.s, f.in, f.weightSource, f.b, f.out, hacConv0Geometry())
		}, nil},
		{"bias must be C_out", func() error {
			short := mustSmallConvStorage(t, f.d, f.s, DTypeF16, []int64{8}, 8)
			return f.d.SmallConvF16(f.s, f.in, f.w, short, f.out, hacConv0Geometry())
		}, nil},
		{"closed output", func() error {
			out := mustSmallConvStorage(t, f.d, f.s, DTypeF16, []int64{64, 192, 16}, 64*192*16)
			if err := out.Close(); err != nil {
				return err
			}
			return f.d.SmallConvF16(f.s, f.in, f.w, f.b, out, hacConv0Geometry())
		}, nil},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.run(); err == nil {
				t.Fatal("Go accepted an out-of-contract request")
			}
		})
	}
}

// TestSmallConvF16UncheckedStillRefusesBadGeometry proves the rejection-control
// entry point does bypass the Go pre-checks (the host backend, standing in for
// the native guard, still refuses), so a control built on it proves the native
// guard rather than a Go pre-check.
func TestSmallConvF16UncheckedStillRefusesBadGeometry(t *testing.T) {
	f := newSmallConvFixture(t)
	g := hacConv0Geometry()
	g.Padding = 0
	if err := f.d.SmallConvF16Unchecked(f.s, f.in, f.w, f.b, f.out, g); err == nil {
		t.Fatal("unchecked entry accepted omitted padding")
	}
	g = hacConv0Geometry()
	if err := f.d.SmallConvF16Unchecked(f.s, f.in, f.w, f.b, f.out, g); err != nil {
		t.Fatal(err)
	}
}

// TestReferencePackSmallConvF16 proves the Go packer and the verification-only
// packing oracle agree byte for byte, and that the oracle receives the four
// handles in the documented order.
func TestReferencePackSmallConvF16(t *testing.T) {
	f := newSmallConvFixture(t)
	const cOut, cIn, win = 16, 1, 5
	weight := make([]float32, cOut*cIn*win)
	for i := range weight {
		weight[i] = float32(i)*0.25 - 1
	}
	bias := make([]float32, cOut)
	for i := range bias {
		bias[i] = float32(i) - 8
	}
	if err := f.weightSource.Upload(f.s, float32BytesTest(weight)); err != nil {
		t.Fatal(err)
	}
	if err := f.biasSrc.Upload(f.s, float32BytesTest(bias)); err != nil {
		t.Fatal(err)
	}
	if err := f.d.ReferencePackSmallConvF16(f.s, f.weightSource, f.biasSrc, f.packed, f.packedBias); err != nil {
		t.Fatal(err)
	}
	gotW, err := f.packed.Download(f.s)
	if err != nil {
		t.Fatal(err)
	}
	gotB, err := f.packedBias.Download(f.s)
	if err != nil {
		t.Fatal(err)
	}
	wantBits, err := packSmallConvForTest(weight, cOut, cIn, win)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(gotW, wantBits) {
		t.Fatal("reference packing and the Go packer disagree")
	}
	wantBias := packHalfForTest(bias)
	if !slices.Equal(gotB, wantBias) {
		t.Fatal("reference bias cast and the Go cast disagree")
	}
	if err := f.d.ReferencePackSmallConvF16(f.s, f.packed, f.biasSrc, f.packed, f.packedBias); err == nil {
		t.Fatal("reference packing accepted a half weight source")
	}
}

// TestSmallConvF16ClosedStream pins that a closed stream cannot dispatch.
func TestSmallConvF16ClosedStream(t *testing.T) {
	f := newSmallConvFixture(t)
	if err := f.s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.d.SmallConvF16(f.s, f.in, f.w, f.b, f.out, hacConv0Geometry()); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed stream error = %v", err)
	}
}
