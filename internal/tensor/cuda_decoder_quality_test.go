package tensor

import (
	"math"
	"testing"
)

func qualityMetas(op decoderPostOperation) []StorageMeta {
	types := []DType{DTypeF32, DTypeI32, DTypeF32, DTypeU8}
	widths := []int64{3, 1, 1, 1}
	if op == decoderQualityProbability {
		types[0], widths[0] = DTypeF16, 9
	}
	if op == decoderQualitySummary {
		widths[1], widths[2] = 2, 3
	}
	count := 4
	if op == decoderQualityProbability {
		count = 3
	}
	m := make([]StorageMeta, count)
	for i := range m {
		m[i] = StorageMeta{DType: types[i], Allocation: uint64(i + 1), Shape: []int64{7, widths[i]}, Strides: []int64{widths[i], 1}, Tags: []KoiTag{TagN, TagC}, StorageElements: uint64(7 * widths[i])}
	}
	return m
}
func TestDecoderQualityMetadataAdmitsThreeClosedOperationsAndReadAliases(t *testing.T) {
	for op := decoderQualityProbability; op <= decoderQualitySummary; op++ {
		m := qualityMetas(op)
		if e := decoderPostMetadata(op, m, 0, 0); e != nil {
			t.Fatal(op, e)
		}
		m[1].Allocation = m[0].Allocation
		if e := decoderPostMetadata(op, m, 0, 0); e != nil {
			t.Fatal("read/read alias", e)
		}
	}
	if e := decoderPostMetadata(decoderQualityGroup, qualityMetas(decoderQualityGroup), -1.5, 2.25); e != nil {
		t.Fatal("finite scale/shift are not clamp options", e)
	}
}
func TestDecoderQualityMetadataRejectsMalformedTypedNumericRows(t *testing.T) {
	for name, change := range map[string]func([]StorageMeta){"dtype": func(m []StorageMeta) { m[1].DType = DTypeF32 }, "rank": func(m []StorageMeta) { m[0].Shape = m[0].Shape[:1] }, "tag": func(m []StorageMeta) { m[2].Tags[0] = TagT }, "stride": func(m []StorageMeta) { m[0].Strides[0]++ }, "scale": func(m []StorageMeta) { m[1].ScalePresent = true }, "extent": func(m []StorageMeta) { m[0].StorageElements = 1 }, "rows": func(m []StorageMeta) { m[1].Shape[0]-- }, "workspace": func(m []StorageMeta) { m[0].Shape[0] = 1 << 21 }, "extra-overlap": func(m []StorageMeta) { m[3].Allocation = m[2].Allocation }, "input-output": func(m []StorageMeta) { m[2].Allocation = m[0].Allocation }, "overflow": func(m []StorageMeta) { m[2].OffsetElements = math.MaxUint64 / 4; m[2].StorageElements = math.MaxUint64 }} {
		t.Run(name, func(t *testing.T) {
			m := qualityMetas(decoderQualityGroup)
			change(m)
			if decoderPostMetadata(decoderQualityGroup, m, 0, 0) == nil {
				t.Fatal("unsafe numeric row admitted")
			}
		})
	}
	if decoderPostMetadata(decoderQualityProbability, qualityMetas(decoderQualityProbability), math.Float32frombits(0x80000000), 0) == nil {
		t.Fatal("unused scalar sign bits admitted")
	}
	if decoderPostMetadata(decoderQualityGroup, qualityMetas(decoderQualityGroup), float32(math.Inf(1)), 0) == nil {
		t.Fatal("nonfinite scale admitted")
	}
}
