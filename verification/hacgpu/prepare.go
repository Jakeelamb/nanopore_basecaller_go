package hacgpu

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/signal"
)

// Normalisation is the exact pinned scaler rule for a standardised pA model.
//
// ScalerNode.cpp:189-206 derives, for `ScalingStrategy::PA` with
// `standardisation.standardise`:
//
//	x_pa  = Scaling * (x + Offset)
//	x_std = (1 / Stdev) * (x_pa - Mean)
//	      = (Scaling / Stdev) * (x + (Offset - Mean / Scaling))
//	      => x_std = (x - ((Mean / Scaling) - Offset)) / (Stdev / Scaling)
//
// so `shift = (Mean / Scaling) - Offset` and `scale = Stdev / Scaling`, and
// `utils::shift_scale_tensor_i16_to_f16_inplace` evaluates
// `float16(float32(x) - shift) / scale` in float32.
type Normalisation struct {
	Mean    float64 `json:"mean"`
	Stdev   float64 `json:"stdev"`
	Scaling float64 `json:"calibration_scale"`
	Offset  float64 `json:"calibration_offset"`
	// TrimStart is the constant trim applied to standardised DNA reads
	// (ScalerNode.cpp:238-243: "In most cases kit14 trim algorithm returns 10").
	TrimStart int `json:"trim_start"`
}

// ShiftScale returns the derived shift and scale in float32, matching the
// pinned source's arithmetic order (both are float in the scaler).
func (n Normalisation) ShiftScale() (shift, scale float32) {
	scaling := float32(n.Scaling)
	stdev := float32(n.Stdev)
	mean := float32(n.Mean)
	scale = stdev / scaling
	shift = (mean / scaling) - float32(n.Offset)
	return shift, scale
}

// Normalise converts raw int16 samples into normalised binary16 bits.
//
// The arithmetic is deliberately spelled out in float32, in the same order as
// `shift_scale_tensor_i16_to_f16_inplace`, and the conversion uses the shared
// round-to-nearest-even half encoder. This is the only place the harness
// computes anything; it is host-side input preparation, not model arithmetic.
func (n Normalisation) Normalise(raw []int16) ([]uint16, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("empty signal")
	}
	if n.TrimStart < 0 || n.TrimStart >= len(raw) {
		return nil, fmt.Errorf("trim %d outside signal of %d samples", n.TrimStart, len(raw))
	}
	shift, scale := n.ShiftScale()
	if scale == 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return nil, fmt.Errorf("degenerate scaler scale %v", scale)
	}
	if math.IsNaN(float64(shift)) || math.IsInf(float64(shift), 0) {
		return nil, fmt.Errorf("degenerate scaler shift %v", shift)
	}
	trimmed := raw[n.TrimStart:]
	out := make([]uint16, len(trimmed))
	for i, v := range trimmed {
		value := float32(v)
		normalised := (value - shift) / scale
		encoded := signal.Float16Bits(normalised)
		// A mis-decoded calibration produces an overflowing normalisation. The
		// pinned scaler would silently store infinity, so the oracle rejects it
		// instead of capturing a graph run on infinities.
		if decoded := signal.Float32FromHalf(encoded); math.IsNaN(float64(decoded)) || math.IsInf(float64(decoded), 0) {
			return nil, fmt.Errorf("sample %d normalises to a non-finite half value", i)
		}
		out[i] = encoded
	}
	return out, nil
}

// EncodeHalfLE serialises binary16 bits as little-endian bytes.
func EncodeHalfLE(bits []uint16) []byte {
	out := make([]byte, 2*len(bits))
	for i, b := range bits {
		binary.LittleEndian.PutUint16(out[2*i:], b)
	}
	return out
}

// DecodeInt16LE reads raw int16 samples from little-endian bytes.
func DecodeInt16LE(body []byte) ([]int16, error) {
	if len(body)%2 != 0 {
		return nil, fmt.Errorf("int16 payload has odd length %d", len(body))
	}
	out := make([]int16, len(body)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(body[2*i:]))
	}
	return out, nil
}

// DecodeHalfLE reads binary16 bits from little-endian bytes.
func DecodeHalfLE(body []byte) ([]uint16, error) {
	if len(body)%2 != 0 {
		return nil, fmt.Errorf("half payload has odd length %d", len(body))
	}
	out := make([]uint16, len(body)/2)
	for i := range out {
		out[i] = binary.LittleEndian.Uint16(body[2*i:])
	}
	return out, nil
}
