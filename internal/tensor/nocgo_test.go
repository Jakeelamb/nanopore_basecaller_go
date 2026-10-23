//go:build !cgo || !linux

package tensor

import (
	"errors"
	"testing"
)

func TestUnsupportedBackend(t *testing.T) {
	if _, err := Open("backend.so"); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("expected clear unsupported error, got %v", err)
	}
}
