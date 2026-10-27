package gpudecode

import (
	"crypto/sha256"
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestBeamNumericPTXRetainsPinnedLogAddInstructionBlock(t *testing.T) {
	body, err := os.ReadFile("../../verification/tensor/decoder/beam_ptx.h")
	if err != nil {
		t.Fatal(err)
	}
	_, block, found := strings.Cut(string(body), "log_add:\n")
	if !found {
		t.Fatal("literal numeric log-add block absent")
	}
	block, _, found = strings.Cut(block, " mov.f32 %f0,%f158;")
	if !found {
		t.Fatal("literal numeric log-add tail absent")
	}
	// Original 1608..1650, with .loc directives removed and only the local
	// branch label renamed. Whole-source SHA is recorded in the numeric header.
	const expected = "95a7ffd26b9f8de3d55ead5a3b01f07ee3e95ef62fc6b5db6130e586039ef4d8"
	if got := fmt.Sprintf("%x", sha256.Sum256([]byte(block))); got != expected {
		t.Fatalf("pinned log-add instruction transcription differs: %s", got)
	}
}
