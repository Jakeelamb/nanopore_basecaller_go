//go:build !cgo

package decode

import (
	"errors"
	"testing"
)

func TestDecodeRequiresCGO(t *testing.T) {
	if _, err := Decode(Input{}, Options{}); !errors.Is(err, ErrCGORequired) {
		t.Fatalf("Decode error=%v, want ErrCGORequired", err)
	}
}
