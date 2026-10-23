//go:build cgo

package decode

import "fmt"

var charToError = func() [128]float32 {
	var table [128]float32
	for q := 33; q <= 127; q++ {
		table[q] = floatPow(10, -float32(q-33)/10)
	}
	return table
}()

// MeanQScore ports sequence_utils.cpp's float32 mean-error quality score.
// The caller selects the substring. Empty input returns zero.
func MeanQScore(qstring string) (float32, error) {
	if len(qstring) == 0 {
		return 0, nil
	}
	var total float32
	for i := range qstring {
		if qstring[i] < 33 || qstring[i] > 127 {
			return 0, fmt.Errorf("quality byte at %d is outside [33,127]", i)
		}
		total += charToError[qstring[i]]
	}
	mean := total / float32(len(qstring))
	return min(float32(50), max(float32(1), float32(-10)*floatLog10(mean))), nil
}
