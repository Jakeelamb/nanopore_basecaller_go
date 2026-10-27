package h07

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOutputOracleRuntimeMustMatchFrozenFiles(t *testing.T) {
	dir := t.TempDir()
	pins := map[string]string{}
	lines := []string{}
	for i := 0; i < 5; i++ {
		path := filepath.Join(dir, fmt.Sprintf("library%d.so", i))
		if err := os.WriteFile(path, []byte{byte(i)}, 0600); err != nil {
			t.Fatal(err)
		}
		if err := pinFile(pins, path); err != nil {
			t.Fatal(err)
		}
		lines = append(lines, "100-200 r-xp 000 00:00 1 "+path)
	}
	body := strings.Join(lines, "\n")
	if observed, err := VerifyRuntimeMaps(body, pins); err != nil || len(observed) != 5 {
		t.Fatalf("frozen maps: %v %v", observed, err)
	}
	for _, changed := range []string{"", body + "\n100-200 r-xp 000 00:00 1 " + filepath.Join(dir, "unknown.so"), strings.Replace(body, "library0.so", "library0.so (deleted)", 1)} {
		if _, err := VerifyRuntimeMaps(changed, pins); err == nil {
			t.Fatal("unknown, deleted, or missing mapping accepted")
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "library0.so"), []byte("changed"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifyRuntimeMaps(body, pins); err == nil {
		t.Fatal("changed runtime object accepted")
	}
}
func TestOutputOracleCaptureAdmissionPrecedesLockAndTools(t *testing.T) {
	if _, err := CaptureOracle(OracleCaptureOptions{Out: "absent"}); err == nil {
		t.Fatal("unadmitted producer accepted")
	}
	if _, err := CaptureOracle(OracleCaptureOptions{Out: "absent", Admitted: true}); err == nil {
		t.Fatal("missing independent build pin accepted")
	}
}
