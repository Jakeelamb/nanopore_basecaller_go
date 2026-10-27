package tensor

import (
	"math"
	"testing"
)

func posteriorMetas() []StorageMeta {
	return []StorageMeta{
		{DType: DTypeF32, Allocation: 1, Shape: []int64{64, 256}, Strides: []int64{256, 1}, Tags: []KoiTag{TagN, TagC}, StorageElements: 64 * 256},
		{DType: DTypeF32, Allocation: 2, Shape: []int64{64, 256}, Strides: []int64{256, 1}, Tags: []KoiTag{TagN, TagC}, StorageElements: 64 * 256},
		{DType: DTypeF16, Allocation: 3, Shape: []int64{64, 256}, Strides: []int64{32768, 1}, Tags: []KoiTag{TagN, TagC}, StorageElements: 64 * 32768},
	}
}

func TestDecoderPostMetadataAdmitsSourceScoreParentStride(t *testing.T) {
	if err := decoderPostMetadata(decoderForwardAlpha, guideMetas(), 0, 2); err != nil {
		t.Fatal(err)
	}
	m := posteriorMetas()
	m[2].OffsetElements = 32 * 256
	if err := decoderPostMetadata(decoderPosteriorNormalize, m, 0, 0); err != nil {
		t.Fatal(err)
	}
	for _, op := range []decoderPostOperation{decoderForwardAlpha, decoderPosteriorNormalize} {
		m = posteriorMetas()
		if op == decoderForwardAlpha {
			m = guideMetas()
		}
		for i := range m {
			m[i].Shape[0] = 32
			m[i].Shape[1] *= 4
			m[i].Strides[0] *= 4
			m[i].StorageElements *= 2
		}
		if err := decoderPostMetadata(op, m, 0, 0); err != nil {
			t.Fatalf("SUP op%d: %v", op, err)
		}
	}
}

func TestDecoderPostMetadataRejectsUnsafeLayoutsAndScalarBits(t *testing.T) {
	for name, change := range map[string]func([]StorageMeta){
		"dtype":               func(m []StorageMeta) { m[0].DType = DTypeF16 },
		"tag":                 func(m []StorageMeta) { m[2].Tags[0] = TagT },
		"scale":               func(m []StorageMeta) { m[1].ScalePresent = true },
		"width":               func(m []StorageMeta) { m[0].Shape[1]-- },
		"alpha-gap":           func(m []StorageMeta) { m[0].Strides[0]++ },
		"output-self-overlap": func(m []StorageMeta) { m[2].Strides[0] = 255 },
		"extent":              func(m []StorageMeta) { m[2].StorageElements = 1 },
		"mixed-byte-overlap": func(m []StorageMeta) {
			m[2].Allocation = m[0].Allocation
			m[2].OffsetElements = 256
			m[2].Strides[0] = 256
			m[2].StorageElements = 64 * 32768
		},
		"overflow": func(m []StorageMeta) { m[2].OffsetElements = math.MaxUint64 / 2; m[2].StorageElements = math.MaxUint64 },
	} {
		t.Run(name, func(t *testing.T) {
			m := posteriorMetas()
			change(m)
			if decoderPostMetadata(decoderPosteriorNormalize, m, 0, 0) == nil {
				t.Fatal("unsafe numeric row accepted")
			}
		})
	}
	for _, scalar := range []float32{1, math.Float32frombits(0x80000000), float32(math.NaN()), float32(math.Inf(1))} {
		if decoderPostMetadata(decoderPosteriorNormalize, posteriorMetas(), scalar, 0) == nil || decoderPostMetadata(decoderPosteriorNormalize, posteriorMetas(), 0, scalar) == nil {
			t.Fatal("unused/invalid scalar bits accepted")
		}
	}
	if decoderPostMetadata(99, posteriorMetas(), 0, 0) == nil {
		t.Fatal("unknown operation accepted")
	}
}

func TestDecoderPostNilOwnersFailBeforeBackend(t *testing.T) {
	var d *Device
	if d.DecoderForwardFrame(nil, nil, nil, nil, 0, 2) == nil || d.DecoderPosteriorRow(nil, nil, nil, nil) == nil {
		t.Fatal("nil numeric owner accepted")
	}
	if _, err := d.DecoderPosteriorProgram(nil); err == nil {
		t.Fatal("nil program owner accepted")
	}
}
