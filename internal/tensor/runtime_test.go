//go:build cgo && linux

package tensor

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNativeLoadedRuntimePins(t *testing.T) {
	_ = nativeEngine(t)
	backend := os.Getenv("NANOPORE_TENSOR_BACKEND")
	raw, err := os.ReadFile(filepath.Join(filepath.Dir(backend), "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest struct {
		Runtime   map[string]string `json:"runtime_sha256"`
		Output    string            `json:"output"`
		OutputSHA string            `json:"output_sha256"`
	}
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	maps, err := os.ReadFile("/proc/self/maps")
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, line := range strings.Split(string(maps), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		path := strings.Join(fields[5:], " ")
		name := filepath.Base(path)
		if name != "libtorch_cpu.so" && name != "libc10.so" && path != backend {
			continue
		}
		if seen[path] {
			continue
		}
		seen[path] = true
		expected := manifest.Runtime[path]
		if path == backend {
			expected = manifest.OutputSHA
		}
		if expected == "" {
			t.Fatalf("loaded runtime has no build pin: %s", path)
		}
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		h := sha256.New()
		_, copyErr := io.Copy(h, f)
		closeErr := f.Close()
		if copyErr != nil {
			t.Fatal(copyErr)
		}
		if closeErr != nil {
			t.Fatal(closeErr)
		}
		actual := hex.EncodeToString(h.Sum(nil))
		if actual != expected {
			t.Fatalf("loaded runtime differs from build pin: %s", path)
		}
		t.Logf("loaded %s sha256=%s", path, actual)
	}
	for _, name := range []string{"libtorch_cpu.so", "libc10.so", filepath.Base(backend)} {
		found := false
		for path := range seen {
			if filepath.Base(path) == name {
				found = true
			}
		}
		if !found {
			t.Fatalf("expected loaded library absent from process maps: %s", name)
		}
	}
}
