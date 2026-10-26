package tensor

import (
	"math"
	"slices"
	"strings"
	"testing"
)

func supTestMetadata(specs []SUPOperandSpec) []StorageMeta {
	out := make([]StorageMeta, len(specs))
	for i, s := range specs {
		stride := int64(1)
		strides := make([]int64, len(s.Shape))
		for j := len(strides) - 1; j >= 0; j-- {
			strides[j] = stride
			stride *= s.Shape[j]
		}
		out[i] = StorageMeta{Shape: slices.Clone(s.Shape), Strides: strides, Tags: slices.Clone(s.Tags), DType: s.DType, Device: 7, Allocation: uint64(i + 1), StorageElements: uint64(stride), ScalePresent: s.Scaled, ScaleTag: s.ScaleTag}
		if s.Scaled {
			out[i].ScaleDType = DTypeF32
		}
	}
	return out
}

func TestSUPPrimitiveClosedMetadataAndRejections(t *testing.T) {
	for _, geometry := range []SUPGeometry{SUPGeometryTiny, SUPGeometryFull, SUPGeometryShort} {
		for op := SUPConvSiLU; op <= SUPNorm2; op++ {
			selectors := []int{0}
			if op == SUPConvSiLU {
				selectors = []int{0, 1, 2, 3, 4}
			}
			if op == SUPLinear {
				selectors = []int{0, 1}
			}
			for _, selector := range selectors {
				specs, err := SUPPrimitiveSpecs(geometry, op, selector)
				if err != nil {
					t.Fatal(err)
				}
				m := supTestMetadata(specs)
				if err = ValidateSUPMetadata(geometry, op, selector, m); err != nil {
					t.Fatalf("op%d/%d: %v", op, selector, err)
				}
				for _, change := range []struct {
					name   string
					mutate func([]StorageMeta)
				}{{"dtype", func(x []StorageMeta) { x[0].DType = DTypeU8 }}, {"device", func(x []StorageMeta) { x[0].Device = 99 }}, {"tag", func(x []StorageMeta) { x[0].Tags[0] = 'W' }}, {"stride", func(x []StorageMeta) { x[0].Strides[0]++ }}, {"extent", func(x []StorageMeta) { x[0].Shape[0]++ }}, {"scale", func(x []StorageMeta) { x[0].ScalePresent = !x[0].ScalePresent }}, {"overflow", func(x []StorageMeta) { x[0].OffsetElements = math.MaxUint64 }}, {"storage", func(x []StorageMeta) { x[0].StorageElements = 1 }}} {
					bad := supTestMetadata(specs)
					change.mutate(bad)
					if err = ValidateSUPMetadata(geometry, op, selector, bad); err == nil {
						t.Fatalf("op%d/%d accepted %s", op, selector, change.name)
					}
				}
				for i, s := range specs {
					if s.Write {
						bad := supTestMetadata(specs)
						bad[i].Allocation = bad[0].Allocation
						if i != 0 {
							if err = ValidateSUPMetadata(geometry, op, selector, bad); err == nil {
								t.Fatalf("op%d writable operand%d alias accepted", op, i)
							}
						}
					}
				}
			}
		}
	}
	for _, x := range []struct {
		op       SUPPrimitive
		selector int
	}{{0, 0}, {SUPPrimitive(99), 0}, {SUPConvSiLU, -1}, {SUPConvSiLU, 5}, {SUPLinear, 2}, {SUPNorm1, 1}} {
		if _, err := SUPPrimitiveSpecs(SUPGeometryTiny, x.op, x.selector); err == nil {
			t.Fatal("unknown operation/selector accepted", x)
		}
	}
}

func TestSUPStridedCopyByteIntervalsAndInternalOverlap(t *testing.T) {
	m := []StorageMeta{{Shape: []int64{2, 3}, Strides: []int64{3, 1}, Tags: []KoiTag{TagM, TagN}, DType: DTypeF16, Device: 7, Allocation: 1, StorageElements: 6}, {Shape: []int64{2, 3}, Strides: []int64{1, 2}, Tags: []KoiTag{TagM, TagN}, DType: DTypeF16, Device: 7, Allocation: 2, StorageElements: 6}}
	if err := ValidateSUPMetadata(SUPGeometryTiny, SUPCopyStrided, 0, m); err != nil {
		t.Fatal(err)
	}
	m[1].Strides = []int64{1, 1}
	if err := ValidateSUPMetadata(SUPGeometryTiny, SUPCopyStrided, 0, m); err == nil {
		t.Fatal("internally overlapping output accepted")
	}
	m[1].Strides = []int64{1, 2}
	m[1].Allocation = 1
	if err := ValidateSUPMetadata(SUPGeometryTiny, SUPCopyStrided, 0, m); err == nil {
		t.Fatal("same allocation overlapping intervals accepted")
	}
	m[0].Shape = []int64{2}
	m[0].Strides = []int64{1}
	m[0].Tags = []KoiTag{TagC}
	m[0].DType = DTypeF32
	m[0].OffsetElements = 1
	m[0].StorageElements = 10
	m[1] = StorageMeta{Shape: []int64{2}, Strides: []int64{1}, Tags: []KoiTag{TagC}, DType: DTypeF16, Device: 7, Allocation: 1, OffsetElements: 2, StorageElements: 10}
	a, b, err := supByteInterval(m[0])
	if err != nil || a != 4 || b != 12 {
		t.Fatalf("F32 byte interval %d,%d %v", a, b, err)
	}
	c, d, err := supByteInterval(m[1])
	if err != nil || c != 4 || d != 8 {
		t.Fatalf("F16 byte interval %d,%d %v", c, d, err)
	}
	if !(a < d && c < b) {
		t.Fatal("cross dtype aliased bytes missed")
	}
}

func TestSUPPrimitiveNilOwnersFailBeforeBackend(t *testing.T) {
	var d *Device
	if err := d.SUPMath(nil, SUPGeometryTiny, SUPConvSiLU, 0); err == nil {
		t.Fatal("nil owner accepted")
	}
}

func TestSUPPrimitiveShortGeometryDimensions(t *testing.T) {
	if SUPGeometryTiny != 1 || SUPGeometryFull != 2 || SUPGeometryShort != 3 {
		t.Fatal("fixed SUP wire values changed")
	}
	d, err := SUPGeometryShort.Dimensions()
	if err != nil || d != (SUPDimensions{32, 6144, 512}) {
		t.Fatalf("short geometry dimensions: %+v %v", d, err)
	}
	scores, err := SUPPrimitiveSpecs(SUPGeometryShort, SUPLinear, 1)
	if err != nil || !slices.Equal(scores[0].Shape, []int64{32, 1024, 512}) || !slices.Equal(scores[2].Shape, []int64{32, 1024, 4096}) {
		t.Fatalf("short score extent differs: %+v %v", scores, err)
	}
	for _, other := range []SUPGeometry{SUPGeometryTiny, SUPGeometryFull} {
		foreign, err := SUPPrimitiveSpecs(other, SUPLinear, 1)
		if err != nil {
			t.Fatal(err)
		}
		if err = ValidateSUPMetadata(SUPGeometryShort, SUPLinear, 1, supTestMetadata(foreign)); err == nil {
			t.Fatal("short geometry accepted a foreign score capacity", other)
		}
	}
}

func TestSUPPrimitiveGeometryAndWholeProfileRejections(t *testing.T) {
	for _, geometry := range []SUPGeometry{0, 4, SUPGeometry(math.MaxUint32)} {
		if _, err := geometry.Dimensions(); err == nil {
			t.Fatal("unknown geometry has dimensions", geometry)
		}
		if _, err := SUPPrimitiveSpecs(geometry, SUPSinCos, 0); err == nil {
			t.Fatal("unknown geometry has operands", geometry)
		}
		if err := ValidateSUPMetadata(geometry, SUPCopyStrided, 0, nil); err == nil {
			t.Fatal("copy bypassed geometry rejection", geometry)
		}
		var d *Device
		if err := d.SUPMath(nil, geometry, SUPSinCos, 0); err == nil || !strings.Contains(err.Error(), "unknown fixed SUP geometry") {
			t.Fatal("geometry did not fail before owner or backend access", geometry, err)
		}
	}
	for _, geometry := range []SUPGeometry{SUPGeometryTiny, SUPGeometryFull, SUPGeometryShort} {
		other := SUPGeometryTiny
		if geometry == other {
			other = SUPGeometryFull
		}
		for op := SUPConvSiLU; op <= SUPNorm2; op++ {
			selectors := []int{0}
			if op == SUPConvSiLU {
				selectors = []int{0, 1, 2, 3, 4}
			}
			if op == SUPLinear {
				selectors = []int{0, 1}
			}
			for _, selector := range selectors {
				specs, err := SUPPrimitiveSpecs(other, op, selector)
				if err != nil {
					t.Fatal(err)
				}
				err = ValidateSUPMetadata(geometry, op, selector, supTestMetadata(specs))
				if (op == SUPSinCos) != (err == nil) {
					t.Fatal("foreign whole operand profile accepted or invariant RoPE rejected", geometry, op, selector, err)
				}
			}
		}
	}
}
