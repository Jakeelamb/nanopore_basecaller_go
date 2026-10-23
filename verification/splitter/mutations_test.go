package splitteroracle

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
	if os.Getenv("SPLITTER_MUTATIONS") != "1" {
		t.Skip("set SPLITTER_MUTATIONS=1 for isolated behavior-mutation checks")
	}
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate source")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(here), "../.."))
	production, err := os.ReadFile(filepath.Join(root, "internal/splitter/splitter.go"))
	if err != nil {
		t.Fatal(err)
	}
	output := os.Getenv("SPLITTER_ARTIFACT_ROOT")
	if output == "" {
		output = filepath.Join(root, "artifacts/splitter")
	}
	if err := os.MkdirAll(output, 0700); err != nil {
		t.Fatal(err)
	}
	attempt, err := os.MkdirTemp(output, "mutations-"+time.Now().UTC().Format("20060102T150405Z")+"-")
	if err != nil {
		t.Fatal(err)
	}
	t.Log(attempt)
	variants := []struct{ name, old, replacement string }{
		{"threshold", "uint16 = 0x419a", "uint16 = 0x4199"},
		{"prefix", "= 5000", "= 4999"},
		{"merge-distance", "cluster.Start-clusters[len(clusters)-1].End > clusterDistance", "cluster.Start-clusters[len(clusters)-1].End >= clusterDistance"},
		{"contiguous-argmax-tie", "signal[i] >= cluster.Peak", "signal[i] > cluster.Peak"},
		{"merged-argmax-tie", "cluster.Peak >= last.Peak", "cluster.Peak > last.Peak"},
		{"inclusive-base-start", "Interval{out.MoveSums[p.MoveStart] - 1, out.MoveSums[p.MoveEnd]}", "Interval{out.MoveSums[p.MoveStart], out.MoveSums[p.MoveEnd]}"},
		{"argmax-base-off-by-one", "start := out.MoveSums[p.MoveArgmax] - 1", "start := out.MoveSums[p.MoveArgmax]"},
		{"maximum-pore-width", "p.BaseSpan.End > p.BaseSpan.Start+maxPoreBases", "p.BaseSpan.End >= p.BaseSpan.Start+maxPoreBases"},
		{"local-Q10-boundary", "p.MeanQ > threshold", "p.MeanQ >= threshold"},
		{"top-ten-peak-ranking", "return raw[i].peak < raw[j].peak", "return raw[i].peak > raw[j].peak"},
		{"top-ten-pair-ranking", "return before(raw[i].r, raw[j].r)", "return before(raw[j].r, raw[i].r)"},
		{"adapter-distance", "location.Distance <= 4", "location.Distance < 4"},
		{"adapter-search-span", "min(candidate.End+50, uint64(len(in.Sequence)))", "min(candidate.End+49, uint64(len(in.Sequence)))"},
		{"strict-N", "align.EqualityExact", "align.EqualityNWildcard"},
		{"adapter-half-open-end", "uint64(location.End+1)", "uint64(location.End)"},
	}
	var summary []map[string]any
	for _, variant := range variants {
		t.Run(variant.name, func(t *testing.T) {
			if strings.Count(string(production), variant.old) != 1 {
				t.Fatalf("mutation anchor must occur exactly once: %q", variant.old)
			}
			dir := filepath.Join(attempt, variant.name)
			for _, path := range []string{"go.mod", "internal/splitter/splitter.go", "internal/splitter/splitter_test.go", "internal/splitter/half_test.go", "internal/align/align.go", "internal/signal/half.go", "verification/splitter/fixtures.go"} {
				data, err := os.ReadFile(filepath.Join(root, path))
				if err != nil {
					t.Fatal(err)
				}
				if path == "internal/splitter/splitter.go" {
					data = []byte(strings.Replace(string(production), variant.old, variant.replacement, 1))
				}
				dst := filepath.Join(dir, path)
				if err := os.MkdirAll(filepath.Dir(dst), 0700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(dst, data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			cmd := exec.Command("go", "test", "-count=1", "./internal/splitter")
			cmd.Dir = dir
			cmd.Env = append(os.Environ(), "GOWORK=off", "DORADO_SOURCE=", "SPLITTER_MUTATIONS=")
			_, runErr := Command(dir, "test", cmd, nil)
			data, err := os.ReadFile(filepath.Join(dir, "test.receipt.json"))
			if err != nil {
				t.Fatal(err)
			}
			var receipt struct {
				ExitCode int `json:"exit_code"`
			}
			if err := json.Unmarshal(data, &receipt); err != nil {
				t.Fatal(err)
			}
			log, err := os.ReadFile(filepath.Join(dir, "test.stdout"))
			if err != nil {
				t.Fatal(err)
			}
			if runErr == nil || receipt.ExitCode != 1 || !strings.Contains(string(log), "--- FAIL: Test") {
				t.Fatalf("mutation survived or failed outside behavior assertions: %v, exit=%d", runErr, receipt.ExitCode)
			}
			summary = append(summary, map[string]any{"name": variant.name, "rejected": true, "exit_code": receipt.ExitCode})
		})
	}
	if err := saveJSON(filepath.Join(attempt, "summary.json"), summary); err != nil {
		t.Fatal(err)
	}
}
