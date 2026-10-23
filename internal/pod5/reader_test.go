package pod5

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestID(t *testing.T) {
	const text = "bb76c31a-36f0-48a0-abd2-107a10e1ade0"
	for _, valid := range []string{text, strings.ToUpper(text)} {
		id, err := ParseID(valid)
		if err != nil || id.String() != text {
			t.Fatalf("UUID round trip for %q: %s %v", valid, id, err)
		}
	}
	for _, s := range []string{"", strings.Repeat("0", 32), "bb76c31a-36f0-48a0-abd2-107a10e1adeg", "bb76c31a_36f0-48a0-abd2-107a10e1ade0"} {
		if _, err := ParseID(s); err == nil {
			t.Fatalf("accepted malformed UUID %q", s)
		}
	}
}

func TestIDRejectsExtraSeparators(t *testing.T) {
	for _, malformed := range []string{
		"bb76c31a-36f0-48a0-abd2-107a10e1ad--",
		"bb76c31a-36f0-48a0-abd2-107a--e1ade0",
		"bb--c31a-36f0-48a0-abd2-107a10e1ade0",
	} {
		t.Run(malformed, func(t *testing.T) {
			if id, err := ParseID(malformed); err == nil {
				t.Fatalf("accepted malformed UUID %q as %s", malformed, id)
			}
		})
	}
}
func TestBoundaryValidation(t *testing.T) {
	for _, paths := range [][2]string{{"", "file"}, {"library", ""}, {"lib\x00suffix", "file"}, {"lib", "file\x00suffix"}} {
		if _, err := Open(paths[0], paths[1]); err == nil {
			t.Fatalf("accepted invalid paths %q", paths)
		}
	}
	if _, err := Open("lib", "file", Options{}, Options{}); err == nil {
		t.Fatal("accepted multiple options")
	}
	if _, err := Open("lib", "file", Options{MaxSamples: ^uint64(0)}); err == nil {
		t.Fatal("accepted overflowing sample bound")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := OpenContext(ctx, "lib", "file"); !errors.Is(err, context.Canceled) {
		t.Fatalf("open cancellation: %v", err)
	}
	var r Reader
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := r.IDs(context.Background()); !errors.Is(err, ErrClosed) {
		t.Fatalf("zero reader IDs: %v", err)
	}
	if _, err := r.Read(context.Background(), ID{}); !errors.Is(err, ErrClosed) {
		t.Fatalf("zero reader read: %v", err)
	}
	if err := r.Iterate(context.Background(), nil, nil); err == nil {
		t.Fatal("nil callback accepted")
	}
}
