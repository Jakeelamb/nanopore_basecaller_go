package gpudecode

import "fmt"

const beamCRCPolynomial uint32 = 0x82f63b78

func beamCRC(crc, value uint32, bits int) uint32 {
	for i := 0; i < bits; i++ {
		low := (value ^ crc) & 1
		crc >>= 1
		if low != 0 {
			crc ^= beamCRCPolynomial
		}
		value >>= 1
	}
	return crc
}

// InitialBeamKey is the source crc32c<32> over the initial state bits.
func InitialBeamKey(state uint16) uint32 { return beamCRC(0x12345678, uint32(state), 32) }

// ExtendBeamKey is crc32c<2> for the source A/C/G/T encoding 0/1/2/3.
// The key alone is not full emitted-string identity; collision behavior belongs
// to the separately verified beam merge algorithm.
func ExtendBeamKey(key uint32, base uint8) (uint32, error) {
	if base > 3 {
		return 0, fmt.Errorf("beam base must be encoded in [0,3]")
	}
	return beamCRC(key, uint32(base), 2), nil
}
