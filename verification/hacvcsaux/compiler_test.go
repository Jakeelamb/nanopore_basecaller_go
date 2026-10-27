package hacvcsaux

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestChangedOraclePinRejectsBeforeCompilerInspection(t *testing.T) {
	path := filepath.Join(t.TempDir(), "oracle")
	original := []byte("original unexecuted artifact")
	hash := sha256.Sum256(original)
	if err := os.WriteFile(path, []byte("changed unexecuted artifact"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := InspectOracle(path, hex.EncodeToString(hash[:])); err == nil || !strings.Contains(err.Error(), "output pin differs") {
		t.Fatalf("changed oracle pin reached compiler or ELF inspection: %v", err)
	}
}
