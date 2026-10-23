package intervaltrimoracle

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestSourceMutationRejections(t *testing.T) {
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("source location unavailable")
	}
	source := os.Getenv("DORADO_SOURCE")
	if source == "" {
		source = filepath.Join(filepath.Dir(here), "../../external/dorado-source")
	}
	if _, err := os.Stat(source); err != nil {
		t.Skip("pinned source checkout unavailable")
	}
	if err := VerifySources(source); err != nil {
		t.Fatal(err)
	}
	root := os.Getenv("INTERVALTRIM_ARTIFACT_ROOT")
	if root == "" {
		root = filepath.Join(filepath.Dir(here), "../../artifacts/intervaltrim")
	}
	if err := os.MkdirAll(root, 0700); err != nil {
		t.Fatal(err)
	}
	attempt, err := os.MkdirTemp(root, "source-rejections-"+time.Now().UTC().Format("20060102T150405Z")+"-")
	if err != nil {
		t.Fatal(err)
	}
	for target := range SourcePins {
		t.Run(filepath.Base(target), func(t *testing.T) {
			dir := filepath.Join(attempt, filepath.Base(target))
			for name := range SourcePins {
				data, err := os.ReadFile(filepath.Join(source, name))
				if err != nil {
					t.Fatal(err)
				}
				if name == target {
					data = append(data, '\n')
				}
				path := filepath.Join(dir, name)
				if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			_, err := Build(dir, filepath.Join(dir, "rejected-build"))
			if err == nil || !strings.Contains(err.Error(), "hash mismatch") {
				t.Fatalf("source mutation not rejected: %v", err)
			}
		})
	}
	t.Logf("all %d source alterations rejected; evidence %s", len(SourcePins), attempt)
}
