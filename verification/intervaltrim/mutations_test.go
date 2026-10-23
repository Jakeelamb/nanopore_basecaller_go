package intervaltrimoracle

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

func TestIntervalMutations(t *testing.T) {
	if os.Getenv("INTERVALTRIM_MUTATIONS") != "1" {
		t.Skip("set INTERVALTRIM_MUTATIONS=1 for isolated behavior mutations")
	}
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("source location unavailable")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(here), "../.."))
	files := map[string][]byte{}
	for _, name := range []string{"go.mod", "internal/readtrim/mux.go", "internal/readtrim/interval.go", "internal/readtrim/interval_test.go"} {
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		files[name] = data
	}
	entries, err := os.ReadDir(filepath.Join(root, "internal/decode"))
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.IsDir() || strings.HasSuffix(entry.Name(), "_test.go") {
			continue
		}
		name := filepath.Join("internal/decode", entry.Name())
		data, err := os.ReadFile(filepath.Join(root, name))
		if err != nil {
			t.Fatal(err)
		}
		files[name] = data
	}
	output := os.Getenv("INTERVALTRIM_ARTIFACT_ROOT")
	if output == "" {
		output = filepath.Join(root, "artifacts/intervaltrim")
	}
	if err := os.MkdirAll(output, 0700); err != nil {
		t.Fatal(err)
	}
	attempt, err := os.MkdirTemp(output, "mutations-"+time.Now().UTC().Format("20060102T150405Z")+"-")
	if err != nil {
		t.Fatal(err)
	}
	variants := []struct{ name, old, replacement string }{
		{"baseline", "", ""},
		{"leading-offset", "removed := uint64(start) * uint64(read.Stride)", "removed := uint64(0)"},
		{"move-zero-loss", "end = i + 1", "if move == 1 { end = i + 1 }"},
		{"pretrim-change", "r.Read.Sequence = read.Sequence[interval.Start:interval.End]", "r.Read.PreTrimSequenceLength = uint64(interval.End-interval.Start); r.Read.Sequence = read.Sequence[interval.Start:interval.End]"},
		{"events-change", "r.Trimmed = true", "r.Read.NumMinknowEvents = 0; r.Trimmed = true"},
		{"ns-current-only", "r.NumSamples = r.LeadingTrimSamples + uint64(len(r.Read.Signal))", "r.NumSamples = uint64(len(r.Read.Signal))"},
		{"quality-equality", "mean < minQ", "mean <= minQ"},
		{"minimum-five", "len(seq) < 5", "len(seq) <= 5"},
		{"offset-fallback", "len(q) > meanQStart", "len(q) >= meanQStart"},
		{"offset-ignored", "selected = q[meanQStart:]", "selected = q"},
	}
	receipts := []map[string]any{}
	for _, v := range variants {
		t.Run(v.name, func(t *testing.T) {
			dir := filepath.Join(attempt, v.name)
			for name, original := range files {
				data := original
				if name == "internal/readtrim/interval.go" && v.old != "" {
					if strings.Count(string(data), v.old) != 1 {
						t.Fatal("mutation anchor must occur once")
					}
					data = []byte(strings.Replace(string(data), v.old, v.replacement, 1))
				}
				path := filepath.Join(dir, name)
				if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			cmd := exec.Command("go", "test", "-count=1", "./internal/readtrim")
			cmd.Dir = dir
			cmd.Env = append(os.Environ(), "GOWORK=off", "CGO_ENABLED=1", "DORADO_SOURCE=")
			err := commandReceipt(dir, "test", cmd, nil)
			data, readErr := os.ReadFile(filepath.Join(dir, "test.receipt.json"))
			if readErr != nil {
				t.Fatal(readErr)
			}
			var receipt struct {
				ExitCode int `json:"exit_code"`
			}
			if json.Unmarshal(data, &receipt) != nil {
				t.Fatal("invalid command receipt")
			}
			log, readErr := os.ReadFile(filepath.Join(dir, "test.stdout"))
			if readErr != nil {
				t.Fatal(readErr)
			}
			if v.name == "baseline" {
				if err != nil {
					t.Fatalf("baseline failure: %v", err)
				}
			} else if err == nil || receipt.ExitCode != 1 || !strings.Contains(string(log), "--- FAIL: Test") {
				t.Fatalf("mutation survived or failed outside assertion: %v", err)
			}
			receipts = append(receipts, map[string]any{"variant": v.name, "exit_code": receipt.ExitCode, "assertion_rejection": v.name != "baseline"})
		})
	}
	if err := writeJSON(filepath.Join(attempt, "summary.json"), receipts); err != nil {
		t.Fatal(err)
	}
	t.Logf("mutation evidence %s", attempt)
}
