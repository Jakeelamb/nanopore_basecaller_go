package tensor

import (
	"math"
	"testing"
)

func guideMetas() []StorageMeta {
	return []StorageMeta{
		{DType: DTypeF16, Allocation: 1, Shape: []int64{64, 1024}, Strides: []int64{32768, 1}, Tags: []KoiTag{TagN, TagC}, StorageElements: 64 * 32768},
		{DType: DTypeF32, Allocation: 2, Shape: []int64{64, 256}, Strides: []int64{256, 1}, Tags: []KoiTag{TagN, TagC}, StorageElements: 64 * 256},
		{DType: DTypeF32, Allocation: 3, Shape: []int64{64, 256}, Strides: []int64{256, 1}, Tags: []KoiTag{TagN, TagC}, StorageElements: 64 * 256},
	}
}

func TestDecoderGuideFrameMetadataAdmitsStridedScoresAndAdjacentGuideRows(t *testing.T) {
	m := guideMetas()
	if err := decoderGuideMetadata(m, 0, 2); err != nil {
		t.Fatal(err)
	}
	m[2].Allocation = m[1].Allocation
	m[1].StorageElements = 2 * 64 * 256
	m[2].StorageElements = 2 * 64 * 256
	m[2].OffsetElements = 64 * 256
	if err := decoderGuideMetadata(m, 0, 2); err != nil {
		t.Fatal("adjacent float guide rows", err)
	}
}

func TestDecoderGuideFrameRejectsMixedTypeByteOverlapAndInvalidViews(t *testing.T) {
	tests := map[string]func([]StorageMeta){
		"dtype":  func(m []StorageMeta) { m[1].DType = DTypeF16 },
		"tag":    func(m []StorageMeta) { m[0].Tags[0] = TagT },
		"shape":  func(m []StorageMeta) { m[0].Shape[1]-- },
		"stride": func(m []StorageMeta) { m[2].Strides[0]++ },
		"scale":  func(m []StorageMeta) { m[0].ScalePresent = true },
		"extent": func(m []StorageMeta) { m[0].StorageElements = 1 },
		"mixed-overlap": func(m []StorageMeta) {
			m[2].Allocation = m[0].Allocation
			m[2].OffsetElements = 512
			m[2].StorageElements = 65536
		},
		"guide-overlap": func(m []StorageMeta) { m[2].Allocation = m[1].Allocation },
		"overflow":      func(m []StorageMeta) { m[2].OffsetElements = math.MaxUint64 / 4; m[2].StorageElements = math.MaxUint64 },
	}
	for name, change := range tests {
		t.Run(name, func(t *testing.T) {
			m := guideMetas()
			change(m)
			if err := decoderGuideMetadata(m, 0, 2); err == nil {
				t.Fatal("unsafe frame admitted")
			}
		})
	}
	if err := decoderGuideMetadata(guideMetas(), float32(math.NaN()), 2); err == nil {
		t.Fatal("NaN clamp accepted")
	}
}
