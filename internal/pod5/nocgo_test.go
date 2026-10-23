//go:build !cgo || !linux

package pod5

import (
	"errors"
	"testing"
)

func TestNativeUnsupported(t *testing.T) {
	if _, err := Open("library", "file"); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("missing unsupported error: %v", err)
	}
}
