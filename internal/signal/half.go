package signal

import "math"

// Float16Bits rounds a float32 to IEEE binary16, using round to nearest, ties to
// even. NaNs retain their sign and available payload and are made quiet.
func Float16Bits(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16(bits>>16) & 0x8000
	exponent := int((bits >> 23) & 255)
	fraction := bits & 0x7fffff
	if exponent == 255 {
		if fraction == 0 {
			return sign | 0x7c00
		}
		return sign | 0x7c00 | uint16(fraction>>13) | 0x0200
	}
	halfExponent := exponent - 127 + 15
	if halfExponent >= 31 {
		return sign | 0x7c00
	}
	if halfExponent <= 0 {
		if halfExponent < -10 {
			return sign
		}
		significand := fraction | 0x800000
		shift := uint32(14 - halfExponent)
		rounded := significand >> shift
		remainder := significand & ((uint32(1) << shift) - 1)
		midpoint := uint32(1) << (shift - 1)
		if remainder > midpoint || remainder == midpoint && rounded&1 != 0 {
			rounded++
		}
		return sign | uint16(rounded)
	}
	rounded := uint32(halfExponent)<<10 | fraction>>13
	remainder := fraction & 0x1fff
	if remainder > 0x1000 || remainder == 0x1000 && rounded&1 != 0 {
		rounded++
	}
	return sign | uint16(rounded)
}

// Float32FromHalf expands IEEE binary16 without rounding.
func Float32FromHalf(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	exponent := uint32(bits>>10) & 31
	fraction := uint32(bits & 1023)
	if exponent == 0 {
		if fraction == 0 {
			return math.Float32frombits(sign)
		}
		exp := int32(-14)
		for fraction&1024 == 0 {
			fraction <<= 1
			exp--
		}
		return math.Float32frombits(sign | uint32(exp+127)<<23 | (fraction&1023)<<13)
	}
	if exponent == 31 {
		return math.Float32frombits(sign | 0x7f800000 | fraction<<13)
	}
	return math.Float32frombits(sign | (exponent+112)<<23 | fraction<<13)
}
