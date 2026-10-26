package tensor

import "math"

func decoderQualityMetadata(op decoderPostOperation, m []StorageMeta, scale, shift float32) error {
	const name = "decoder_quality_numbers"
	fail := func(why string) error { return &LayoutError{Op: name, Reason: why} }
	count := 3
	if op == decoderQualityGroup || op == decoderQualitySummary {
		count = 4
	}
	if op < decoderQualityProbability || op > decoderQualitySummary || len(m) != count {
		return fail("checked quality operation and exact operand count required")
	}
	if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) || math.IsNaN(float64(shift)) || math.IsInf(float64(shift), 0) || (op != decoderQualityGroup && (math.Float32bits(scale) != 0 || math.Float32bits(shift) != 0)) {
		return fail("finite quality scalars and positive-zero unused bits required")
	}
	dtypes := []DType{DTypeF32, DTypeI32, DTypeF32, DTypeU8}
	widths := []int64{0, 1, 1, 1}
	if op == decoderQualityProbability {
		dtypes[0], widths[0] = DTypeF16, 9
	}
	if op == decoderQualitySummary {
		widths[1], widths[2] = 2, 3
	}
	var rows int64
	starts, ends := make([]uint64, len(m)), make([]uint64, len(m))
	for i, v := range m {
		if v.DType != dtypes[i] {
			return &DTypeError{Op: name, Reason: "quality numeric operand dtype differs"}
		}
		if len(v.Shape) != 2 || len(v.Strides) != 2 || len(v.Tags) != 2 || v.Tags[0] != TagN || v.Tags[1] != TagC || v.ScalePresent || v.Shape[1] <= 0 || v.Strides[1] != 1 || v.Strides[0] != v.Shape[1] {
			return fail("contiguous unscaled NC numeric rows required")
		}
		if i == 0 {
			rows = v.Shape[0]
		}
		if rows <= 0 || rows > 1<<20 || v.Shape[0] != rows || v.Shape[1] > 1<<20 || rows > (1<<27)/v.Shape[1] || widths[i] != 0 && v.Shape[1] != widths[i] {
			return fail("quality numeric row dimensions exceed workspace or differ")
		}
		end, err := storageSpan(v.Shape, v.Strides, v.OffsetElements)
		if err != nil {
			return err
		}
		size := uint64(v.DType.Size())
		if end > math.MaxUint64/size || end > v.StorageElements {
			return fail("quality numeric byte interval exceeds parent or overflows")
		}
		starts[i], ends[i] = v.OffsetElements*size, end*size
	}
	for output := 2; output < len(m); output++ {
		for i := 0; i < len(m); i++ {
			if i != output && m[i].Allocation == m[output].Allocation && starts[i] < ends[output] && starts[output] < ends[i] {
				return fail("quality output bytes overlap another operand")
			}
		}
	}
	return nil
}

func (d *Device) DecoderQualityProbabilities(s *Stream, selectedHalf, counts, out *Storage) error {
	return d.decoderPostNumbers(s, decoderQualityProbability, selectedHalf, counts, out, nil, 0, 0)
}

func (d *Device) DecoderQualityGroups(s *Stream, probabilities, counts, errors, characters *Storage, scale, shift float32) error {
	return d.decoderPostNumbers(s, decoderQualityGroup, probabilities, counts, errors, characters, scale, shift)
}

func (d *Device) DecoderQualitySummaries(s *Stream, errors, counts, out, written *Storage) error {
	return d.decoderPostNumbers(s, decoderQualitySummary, errors, counts, out, written, 0, 0)
}
