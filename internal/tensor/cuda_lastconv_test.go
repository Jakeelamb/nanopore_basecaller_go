//go:build cgo && linux

package tensor

import (
	"errors"
	"testing"
)

// lastConvFixture builds a small but contract-complete last-convolution request
// on the host-memory backend: input [2,30,16] F16 (T_in 12 plus 2*9 padding),
// weights [8,19*16] F16, bias [8] F16, output allocation [7,2,8] I8 with the
// transposed view [2,2,8] at strides [8,16,1].
type lastConvFixture struct {
	d              *Device
	s              *Stream
	in, w, b, root *Storage
	out            *Storage
}

func newLastConvFixture(t *testing.T) *lastConvFixture {
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
	f := &lastConvFixture{d: d, s: s}
	f.in = mustSmallConvStorage(t, d, s, DTypeF16, []int64{2, 30, 16}, 2*30*16)
	f.w = mustSmallConvStorage(t, d, s, DTypeF16, []int64{8, 19 * 16}, 8*19*16)
	f.b = mustSmallConvStorage(t, d, s, DTypeF16, []int64{8}, 8)
	f.root = mustSmallConvStorage(t, d, s, DTypeI8, []int64{7 * 2 * 8}, 7*2*8)
	f.out, err = f.root.View(16, []int64{2, 2, 8}, []int64{8, 16, 1},
		[]KoiTag{TagN, TagT, TagC})
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func (f *lastConvFixture) geometry() LastConvGeometry {
	return LastConvGeometry{
		N: 2, TIn: 12, CIn: 16, COut: 8, Window: 19, Stride: 6, Padding: 9,
		InNStride: 30 * 16, OutNStride: 8, OutTStride: 16,
		Activation: KoiTanh, InType: KoiF16, OutType: KoiI8, AccumType: KoiF16,
	}
}

// TestLastConvFastPathAcceptsPinnedGeometry proves the Go request reaches the
// boundary and that the boundary reports the primitive's own result.
func TestLastConvFastPathAcceptsPinnedGeometry(t *testing.T) {
	f := newLastConvFixture(t)
	res, err := f.d.LastConvKoi(f.s, f.in, f.w, f.b, f.out, f.geometry())
	if err != nil {
		t.Fatal(err)
	}
	if res != KoiSuccess {
		t.Fatalf("fast path result %s", res)
	}
}

// TestLastConvFallbackAcceptsPinnedGeometry proves the fallback is a separate
// primitive with its own result, not an alias of the fast path.
func TestLastConvFallbackAcceptsPinnedGeometry(t *testing.T) {
	f := newLastConvFixture(t)
	res, err := f.d.LastConvLinear(f.s, f.in, f.w, f.b, f.out, f.geometry())
	if err != nil {
		t.Fatal(err)
	}
	if res != KoiSuccess {
		t.Fatalf("fallback result %s", res)
	}
}

// TestLastConvGoRejections pins the Go-side contract. Every case must be refused
// before the boundary is called, so the native guard is not what is being
// tested here.
func TestLastConvGoRejections(t *testing.T) {
	cases := []struct {
		name string
		run  func(t *testing.T, f *lastConvFixture) error
	}{
		{"padding", func(t *testing.T, f *lastConvFixture) error {
			g := f.geometry()
			g.Padding = 8
			_, err := f.d.LastConvKoi(f.s, f.in, f.w, f.b, f.out, g)
			return err
		}},
		{"stride-multiple", func(t *testing.T, f *lastConvFixture) error {
			g := f.geometry()
			g.TIn = 13
			_, err := f.d.LastConvKoi(f.s, f.in, f.w, f.b, f.out, g)
			return err
		}},
		{"input-rows", func(t *testing.T, f *lastConvFixture) error {
			g := f.geometry()
			g.Padding = 9
			// T_in 18 with the same allocation would need 36 rows, not 30.
			g.TIn = 18
			_, err := f.d.LastConvKoi(f.s, f.in, f.w, f.b, f.out, g)
			return err
		}},
		{"weight-shape", func(t *testing.T, f *lastConvFixture) error {
			short, err := f.d.NewStorage(f.s, DTypeF16, []int64{8, 19*16 - 1},
				[]KoiTag{TagC, TagW}, make([]byte, 8*(19*16-1)*2))
			if err != nil {
				t.Fatal(err)
			}
			defer short.Close()
			_, err = f.d.LastConvKoi(f.s, f.in, short, f.b, f.out, f.geometry())
			return err
		}},
		{"output-strides", func(t *testing.T, f *lastConvFixture) error {
			contig, err := f.root.View(0, []int64{2, 2, 8}, []int64{16, 8, 1},
				[]KoiTag{TagN, TagT, TagC})
			if err != nil {
				t.Fatal(err)
			}
			defer contig.Close()
			_, err = f.d.LastConvKoi(f.s, f.in, f.w, f.b, contig, f.geometry())
			return err
		}},
		{"input-type", func(t *testing.T, f *lastConvFixture) error {
			g := f.geometry()
			g.InType = KoiI8
			_, err := f.d.LastConvKoi(f.s, f.in, f.w, f.b, f.out, g)
			return err
		}},
		{"activation", func(t *testing.T, f *lastConvFixture) error {
			g := f.geometry()
			g.Activation = KoiIdentity
			_, err := f.d.LastConvKoi(f.s, f.in, f.w, f.b, f.out, g)
			return err
		}},
		{"accumulation", func(t *testing.T, f *lastConvFixture) error {
			g := f.geometry()
			g.AccumType = KoiI8
			_, err := f.d.LastConvKoi(f.s, f.in, f.w, f.b, f.out, g)
			return err
		}},
		{"fallback-weight-shape", func(t *testing.T, f *lastConvFixture) error {
			short, err := f.d.NewStorage(f.s, DTypeF16, []int64{8, 19*16 - 1},
				[]KoiTag{TagC, TagW}, make([]byte, 8*(19*16-1)*2))
			if err != nil {
				t.Fatal(err)
			}
			defer short.Close()
			_, err = f.d.LastConvLinear(f.s, f.in, short, f.b, f.out, f.geometry())
			return err
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := newLastConvFixture(t)
			err := c.run(t, f)
			if err == nil {
				t.Fatal("request accepted")
			}
			var de *DTypeError
			var le *LayoutError
			if !errors.As(err, &de) && !errors.As(err, &le) {
				t.Fatalf("unexpected error type: %v", err)
			}
		})
	}
}

// TestLastConvUncheckedStillRefusesBadGeometry proves the rejection-control
// entry point bypasses only the Go checks: the native guard still fires.
func TestLastConvUncheckedStillRefusesBadGeometry(t *testing.T) {
	f := newLastConvFixture(t)
	g := f.geometry()
	g.Padding = 8
	if _, err := f.d.LastConvKoiUnchecked(f.s, f.in, f.w, f.b, f.out, g); err == nil {
		t.Fatal("unchecked fast path accepted padding 8")
	}
	// host_linear takes no padding argument (koi.h:241-258), so padding is not
	// part of the fallback's contract. A field it does consume still guards it.
	short, err := f.d.NewStorage(f.s, DTypeF16, []int64{8, 19*16 - 1},
		[]KoiTag{TagC, TagW}, make([]byte, 8*(19*16-1)*2))
	if err != nil {
		t.Fatal(err)
	}
	defer short.Close()
	if _, err := f.d.LastConvLinearUnchecked(f.s, f.in, short, f.b, f.out, f.geometry()); err == nil {
		t.Fatal("unchecked fallback accepted a short weight")
	}
	// The same entry point accepts the valid request, so the refusal above came
	// from the mutated field and not from the unchecked path itself.
	if _, err := f.d.LastConvKoiUnchecked(f.s, f.in, f.w, f.b, f.out, f.geometry()); err != nil {
		t.Fatal(err)
	}
}

// TestStorageReinterpret pins the typed equivalent of
// `backing_tensor.view(dtype)`: bytes are reinterpreted, the element count
// changes with the element size, and a non-contiguous parent is refused.
func TestStorageReinterpret(t *testing.T) {
	e := fakeStorageEngine(t)
	d, err := e.NewCUDADevice(0)
	if err != nil {
		t.Fatal(err)
	}
	s, err := d.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	f16, err := d.NewStorage(s, DTypeF16, []int64{16}, []KoiTag{TagN}, make([]byte, 32))
	if err != nil {
		t.Fatal(err)
	}
	i8, err := f16.Reinterpret(DTypeI8, 0, []int64{32}, []int64{1}, []KoiTag{TagN})
	if err != nil {
		t.Fatal(err)
	}
	if i8.DType() != DTypeI8 {
		t.Fatalf("reinterpreted dtype %s", i8.DType())
	}
	got, err := i8.Download(s)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 32 {
		t.Fatalf("reinterpreted %d bytes, want 32", len(got))
	}
	// A strided parent has no single byte interval to reinterpret.
	strided, err := f16.View(0, []int64{4, 2}, []int64{4, 1}, []KoiTag{TagT, TagC})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := strided.Reinterpret(DTypeI8, 0, []int64{32}, []int64{1}, []KoiTag{TagN}); err == nil {
		t.Fatal("strided parent reinterpreted")
	}
}
