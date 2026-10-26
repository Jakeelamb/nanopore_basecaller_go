package tensor

import (
	"math"
	"testing"
)

func beamMetadata(op decoderBeamOperation, rows int64) []StorageMeta {
	layouts, width := decoderBeamLayouts(op)
	layouts = append(layouts, decoderBeamLayout{DTypeF32, width})
	m := make([]StorageMeta, len(layouts))
	for i, v := range layouts {
		m[i] = StorageMeta{DType: v.dtype, Allocation: uint64(i + 1), Shape: []int64{rows, v.width}, Strides: []int64{v.width, 1}, Tags: []KoiTag{TagN, TagC}, StorageElements: uint64(rows * v.width)}
	}
	return m
}

func TestDecoderBeamMetadataAdmitsAllSixLayoutsAndReadAliasing(t *testing.T) {
	for op := decoderBeamLogCut; op <= decoderBeamMax; op++ {
		m := beamMetadata(op, 7)
		if err := decoderBeamMetadata(op, m, 0, 0); err != nil {
			t.Fatalf("op%d %v", op, err)
		}
		if op == decoderBeamExpand {
			m[0].Allocation = m[2].Allocation
			if err := decoderBeamMetadata(op, m, 0, 2); err != nil {
				t.Fatal("read/read alias rejected", err)
			}
		}
	}
	m := beamMetadata(decoderBeamSub, 7)
	m[1].Allocation, m[1].OffsetElements, m[1].StorageElements = m[0].Allocation, 14, 21
	if err := decoderBeamMetadata(decoderBeamSub, m, 0, 0); err != nil {
		t.Fatal("adjacent read/write intervals rejected", err)
	}
}

func TestDecoderBeamMetadataRejectsUnsafeNumericDescriptors(t *testing.T) {
	cases := map[string]func([]StorageMeta){
		"dtype":       func(m []StorageMeta) { m[1].DType = DTypeF32 },
		"rank":        func(m []StorageMeta) { m[0].Shape = m[0].Shape[:1] },
		"tag":         func(m []StorageMeta) { m[0].Tags[0] = TagT },
		"width":       func(m []StorageMeta) { m[2].Shape[1]-- },
		"stride":      func(m []StorageMeta) { m[0].Strides[0]++ },
		"scale":       func(m []StorageMeta) { m[0].ScalePresent = true },
		"zero":        func(m []StorageMeta) { m[0].Shape[0] = 0 },
		"rows":        func(m []StorageMeta) { m[1].Shape[0]-- },
		"workspace":   func(m []StorageMeta) { m[0].Shape[0] = decoderBeamMaxRows + 1 },
		"extent":      func(m []StorageMeta) { m[0].StorageElements = 1 },
		"overlap":     func(m []StorageMeta) { m[3].Allocation = m[0].Allocation },
		"mixed-bytes": func(m []StorageMeta) { m[3].Allocation = m[1].Allocation; m[3].OffsetElements = 1 },
		"overflow":    func(m []StorageMeta) { m[3].OffsetElements = math.MaxUint64 / 4; m[3].StorageElements = math.MaxUint64 },
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			m := beamMetadata(decoderBeamExpand, 7)
			change(m)
			if err := decoderBeamMetadata(decoderBeamExpand, m, 0, 2); err == nil {
				t.Fatal("unsafe numeric descriptor admitted")
			}
		})
	}
	for _, op := range []decoderBeamOperation{0, 7, 0xffffffff} {
		if err := decoderBeamMetadata(op, beamMetadata(decoderBeamSub, 1), 0, 0); err == nil {
			t.Fatal("unchecked operation admitted")
		}
	}
	for _, scalar := range []float32{-1, float32(math.NaN()), float32(math.Inf(1))} {
		if err := decoderBeamMetadata(decoderBeamExpand, beamMetadata(decoderBeamExpand, 1), scalar, 2); err == nil {
			t.Fatal("unsafe scalar admitted")
		}
	}
	if err := decoderBeamMetadata(decoderBeamSub, beamMetadata(decoderBeamSub, 1), math.Float32frombits(0x80000000), 0); err == nil {
		t.Fatal("unused negative-zero scalar bits admitted")
	}
}
