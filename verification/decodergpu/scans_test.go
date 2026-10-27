package decodergpu

import (
	"encoding/binary"
	"testing"
)

func TestGuideComparatorUsesEveryFrameChunkAndState(t *testing.T) {
	tns := make([]byte, GuidesBytes)
	aux := make([]byte, AuxBytes)
	for n := 0; n < N; n++ {
		for t := 0; t <= T; t++ {
			for state := 0; state < States; state++ {
				bits := uint32((n*(T+1)+t)*States + state)
				binary.LittleEndian.PutUint32(tns[((t*N+n)*States+state)*4:], bits)
				binary.LittleEndian.PutUint32(aux[((n*(T+1)+t)*States+state)*4:], bits)
			}
		}
	}
	for i := GuidesBytes; i < AuxBytes; i++ {
		aux[i] = 0xa5
	}
	c := FixedContract()
	if err := CompareGuides(tns, aux, c); err != nil {
		t.Fatal(err)
	}
	for _, offset := range []int{0, States * 4, (T + 1) * States * 4, GuidesBytes - 1} {
		aux[offset] ^= 1
		if err := CompareGuides(tns, aux, c); err == nil {
			t.Fatalf("changed source byte%d accepted", offset)
		}
		aux[offset] ^= 1
	}
	aux[GuidesBytes] ^= 1
	if err := CompareGuides(tns, aux, c); err != nil {
		t.Fatal("uninitialized history incorrectly compared", err)
	}
	if err := CompareGuides(tns[:len(tns)-1], aux, c); err == nil {
		t.Fatal("truncated guide accepted")
	}
	if err := CompareGuides(tns, aux[:len(aux)-1], c); err == nil {
		t.Fatal("truncated source owner accepted")
	}
	c.Shape[0]--
	if err := CompareGuides(tns, aux, c); err == nil {
		t.Fatal("changed fixed descriptor contract accepted")
	}
}
