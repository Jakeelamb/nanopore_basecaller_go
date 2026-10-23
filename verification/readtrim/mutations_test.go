package readtrimoracle

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestBehaviorMutations(t *testing.T) {
	if os.Getenv("READTRIM_MUTATIONS") != "1" {
		t.Skip("set READTRIM_MUTATIONS=1 for isolated behavior-mutation qualification")
	}
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate source")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(here), "../.."))
	production, err := os.ReadFile(filepath.Join(root, "internal/readtrim/mux.go"))
	if err != nil {
		t.Fatal(err)
	}
	tests, err := os.ReadFile(filepath.Join(root, "internal/readtrim/mux_test.go"))
	if err != nil {
		t.Fatal(err)
	}
	mod, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	output := os.Getenv("READTRIM_ARTIFACT_ROOT")
	if output == "" {
		output = filepath.Join(root, "artifacts/readtrim")
	}
	if err := os.MkdirAll(output, 0700); err != nil {
		t.Fatal(err)
	}
	attempt, err := os.MkdirTemp(output, "mutations-"+time.Now().UTC().Format("20060102T150405Z")+"-")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("mutation evidence %s", attempt)
	variants := []struct{ name, old, replacement string }{
		{"low-threshold", "q[i] <= 40", "q[i] < 40"},
		{"high-threshold", "q[i] <= 45", "q[i] < 45"},
		{"tie", "cumulative <= minimum", "cumulative < minimum"},
		{"source-off-by-one", "end = i - 1", "end = i"},
		{"minimum-five", "end >= n-5", "end > n-5"},
		{"short-100", "n < 100", "n <= 100"},
		{"excessive-equality", "end < limit", "end <= limit"},
		{"binary32", "float64(float32(n) * float32(0.3))", "float64(n) * 0.3"},
		{"signal-end", "read.Signal[:signalEnd:signalEnd]", "read.Signal[:signalEnd-1:signalEnd-1]"},
		{"events-reset", "result.Read.NumMinknowEvents = 0", "result.Read.NumMinknowEvents = read.NumMinknowEvents"},
		{"pretrim-reset", "result.Read.PreTrimSequenceLength = uint64(end)", "result.Read.PreTrimSequenceLength = read.PreTrimSequenceLength"},
		{"move-boundary", "moveIndex(read.Moves, end, n)", "moveIndex(read.Moves, end+1, n)"},
	}
	receipts := []map[string]any{}
	for _, variant := range variants {
		t.Run(variant.name, func(t *testing.T) {
			if strings.Count(string(production), variant.old) != 1 {
				t.Fatal("mutation anchor must occur once")
			}
			dir := filepath.Join(attempt, variant.name)
			packageDir := filepath.Join(dir, "internal/readtrim")
			if err := os.MkdirAll(packageDir, 0700); err != nil {
				t.Fatal(err)
			}
			mutated := []byte(strings.Replace(string(production), variant.old, variant.replacement, 1))
			for path, data := range map[string][]byte{filepath.Join(dir, "go.mod"): mod, filepath.Join(packageDir, "mux.go"): mutated, filepath.Join(packageDir, "mux_test.go"): tests} {
				if err := os.WriteFile(path, data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			cmd := exec.Command("go", "test", "-count=1", "./internal/readtrim")
			cmd.Dir = dir
			cmd.Env = append(os.Environ(), "GOWORK=off", "DORADO_SOURCE=")
			err := commandReceipt(dir, "mutation-test", cmd, nil)
			data, readErr := os.ReadFile(filepath.Join(dir, "mutation-test.receipt.json"))
			if readErr != nil {
				t.Fatal(readErr)
			}
			var receipt struct {
				ExitCode int `json:"exit_code"`
			}
			if json.Unmarshal(data, &receipt) != nil {
				t.Fatal("invalid receipt")
			}
			log, readErr := os.ReadFile(filepath.Join(dir, "mutation-test.stdout"))
			if readErr != nil {
				t.Fatal(readErr)
			}
			if err == nil || receipt.ExitCode != 1 || !strings.Contains(string(log), "--- FAIL: Test") {
				t.Fatalf("mutation survived or failed outside behavior assertions: %v, exit=%d", err, receipt.ExitCode)
			}
			receipts = append(receipts, map[string]any{"name": variant.name, "changed_source_sha256": hash(mutated), "rejected": true, "exit_code": receipt.ExitCode})
		})
	}
	if err := writeJSON(filepath.Join(attempt, "summary.json"), receipts); err != nil {
		t.Fatal(err)
	}
}
