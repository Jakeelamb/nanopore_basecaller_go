package decodergpu

import (
	"bytes"
	"fmt"
)

// CanonicalGuides converts Go-owned [T+1,N,S] float bytes to the oracle's
// [N,T+1,S] layout. This is a byte permutation, with no float arithmetic.
func CanonicalGuides(goTNS []byte, c Contract) ([]byte, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if len(goTNS) != GuidesBytes {
		return nil, fmt.Errorf("fixed guide extent differs")
	}
	canonical := make([]byte, len(goTNS))
	row := States * 4
	for n := 0; n < N; n++ {
		for t := 0; t <= T; t++ {
			src := (t*N + n) * row
			dst := (n*(T+1) + t) * row
			copy(canonical[dst:dst+row], goTNS[src:src+row])
		}
	}
	return canonical, nil
}

// CompareGuides checks every source-declared guide byte, excluding the separate
// uninitialized beam-history region. Producer and buffer hashes, execution and
// runtime evidence must be authenticated before calling this comparator.
func CompareGuides(goTNS, sourceAux []byte, c Contract) error {
	if len(sourceAux) != AuxBytes {
		return fmt.Errorf("fixed decoder aux extent differs")
	}
	got, err := CanonicalGuides(goTNS, c)
	if err != nil {
		return err
	}
	if !bytes.Equal(got, sourceAux[:GuidesBytes]) {
		for i := 0; i < GuidesBytes; i++ {
			if got[i] != sourceAux[i] {
				return fmt.Errorf("guide mismatch at native byte%d: chunk%d frame%d state%d", i, i/((T+1)*States*4), (i/(States*4))%(T+1), (i/4)%States)
			}
		}
	}
	return nil
}
