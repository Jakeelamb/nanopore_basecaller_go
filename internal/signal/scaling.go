// Package signal implements the standardised DNA pA scaling path.
package signal

import (
	"fmt"
	"math"
)

type Calibration struct{ Scale, Offset float32 }
type Standardisation struct{ Mean, Stdev float32 }
type Normalized struct {
	Samples      []uint16
	Shift, Scale float32
	Trimmed      int
}

// Normalize converts ADC samples to IEEE binary16 and applies the standardised
// DNA prefix trim. Adjustment is an already calculated ADC open-pore shift.
// The returned pA metadata excludes this adjustment, matching ScalerNode.
func Normalize(raw []int16, calibration Calibration, standard Standardisation, adjustment float32) (Normalized, error) {
	if !finite(calibration.Scale) || calibration.Scale <= 0 || !finite(calibration.Offset) || !finite(standard.Mean) || !finite(standard.Stdev) || standard.Stdev <= 0 || !finite(adjustment) {
		return Normalized{}, fmt.Errorf("scaling requires finite parameters and positive calibration scale and standard deviation")
	}
	scale := float32(standard.Stdev / calibration.Scale)
	shift := float32(float32(standard.Mean/calibration.Scale) - calibration.Offset)
	adjusted := float32(shift + adjustment)
	result := Normalized{Scale: float32(calibration.Scale * scale), Shift: float32(calibration.Scale * float32(shift+calibration.Offset))}
	if !finite(scale) || scale <= 0 || !finite(shift) || !finite(adjusted) || !finite(result.Scale) || !finite(result.Shift) {
		return Normalized{}, fmt.Errorf("scaling parameters overflow float32")
	}
	if len(raw) > 10 {
		result.Trimmed = 10
	}
	result.Samples = make([]uint16, len(raw)-result.Trimmed)
	for i, value := range raw[result.Trimmed:] {
		difference := float32(float32(value) - adjusted)
		result.Samples[i] = Float16Bits(float32(difference / scale))
	}
	return result, nil
}

func finite(value float32) bool { return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0) }
