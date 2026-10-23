package splitteroracle

import (
	"bytes"
	"io"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/internal/splitter"
)

func TestOutputLimit(t *testing.T) {
	w := &limitedBuffer{limit: 3}
	if n, err := w.Write([]byte{1, 2, 3}); n != 3 || err != nil {
		t.Fatal("valid bounded output rejected")
	}
	if n, err := w.Write([]byte{4}); n != 0 || err == nil || !bytes.Equal(w.Bytes(), []byte{1, 2, 3}) {
		t.Fatal("overflowing native output was accepted or buffer changed")
	}
	if n, err := io.Copy(w, struct{ io.Reader }{bytes.NewReader([]byte{4})}); n != 0 || err == nil || !bytes.Equal(w.Bytes(), []byte{1, 2, 3}) {
		t.Fatal("streamed output bypassed the byte limit")
	}
}

func TestInputRejectsInvalidBeforeEncoding(t *testing.T) {
	in := Cases()[3].Input
	in.Stride = 0
	if data, err := EncodeInput([]splitter.Input{in}); err == nil || data != nil {
		t.Fatal("invalid input was encoded")
	}
	if _, err := EncodeInput(make([]splitter.Input, 100001)); err == nil {
		t.Fatal("oversized batch was encoded")
	}
}
