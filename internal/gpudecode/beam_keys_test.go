package gpudecode

import (
	"encoding/binary"
	"hash/crc32"
	"testing"
)

func TestBeamInitialKeysMatchIndependentCRC32CByteUpdates(t *testing.T) {
	table := crc32.MakeTable(crc32.Castagnoli)
	var stateBytes [4]byte
	for state := uint16(0); state < 1024; state++ {
		binary.LittleEndian.PutUint32(stateBytes[:], uint32(state))
		want := ^crc32.Update(^uint32(0x12345678), table, stateBytes[:])
		if got := InitialBeamKey(state); got != want {
			t.Fatalf("state%d got%08x want%08x", state, got, want)
		}
	}
}

func TestBeamTwoBitKeysMatchIndependentByteUpdateWithInversePadding(t *testing.T) {
	table := crc32.MakeTable(crc32.Castagnoli)
	for _, key := range []uint32{0, 1, 0xffffffff, 0x12345678, 0x82f63b78, InitialBeamKey(0), InitialBeamKey(255), InitialBeamKey(1023)} {
		for base := uint8(0); base < 4; base++ {
			// Standard-library update consumes eight bits. Invert its six trailing
			// zero bits using the polynomial's high bit to recover the two-bit state.
			want := ^crc32.Update(^key, table, []byte{base})
			for i := 0; i < 6; i++ {
				low := want >> 31
				if low != 0 {
					want ^= 0x82f63b78
				}
				want = want<<1 | low
			}
			got, err := ExtendBeamKey(key, base)
			if err != nil || got != want {
				t.Fatalf("key%08x base%d got%08x want%08x err=%v", key, base, got, want, err)
			}
		}
	}
	if _, err := ExtendBeamKey(0, 4); err == nil {
		t.Fatal("invalid base silently truncated")
	}
}
