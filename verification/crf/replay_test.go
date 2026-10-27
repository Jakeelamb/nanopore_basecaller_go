package crforacle

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadPinned(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bytes")
	data := []byte("retained data")
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := readPinned(path, digest(data)); err != nil {
		t.Fatal(err)
	}
	for _, pin := range []string{"", strings.Repeat("0", 64)} {
		if _, err := readPinned(path, pin); err == nil {
			t.Fatal("invalid pin accepted")
		}
	}
	if err := os.WriteFile(path, []byte("corrupted data"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := readPinned(path, digest(data)); err == nil {
		t.Fatal("corruption accepted")
	}
	if err := verifyPin(path, digest(data), map[string]string{}); err == nil {
		t.Fatal("corrupted dependency accepted")
	}
}

func TestParseRetainedInputRejections(t *testing.T) {
	for _, data := range [][]byte{nil, make([]byte, 7), make([]byte, 8), append([]byte{0x31, 0x46, 0x52, 0x43, 66, 0, 0, 0}, make([]byte, 12)...)} {
		if _, err := parseRetainedInputs(data); err == nil {
			t.Fatal("malformed input accepted")
		}
	}
}

func TestReplayAuthenticatesBackendBeforeRetainedData(t *testing.T) {
	root := t.TempDir()
	manifest := filepath.Join(root, "manifest.json")
	if err := os.WriteFile(manifest, []byte("{}"), 0644); err != nil {
		t.Fatal(err)
	}
	fixtures, pins, err := Replay(filepath.Join(root, "missing-source"), filepath.Join(root, "libnanopore_tensor.so"), filepath.Join(root, "missing-retained"))
	if fixtures != nil || pins != nil {
		t.Fatal("backend rejection returned partial results or pins")
	}
	want := "hash pin mismatch: " + manifest
	if err == nil || err.Error() != want {
		t.Fatalf("want backend manifest rejection %q before retained data access, got %v", want, err)
	}
	t.Logf("rejected before retained data access: %v", err)
}

func copyFile(t *testing.T, from, to string) {
	t.Helper()
	data, err := os.ReadFile(from)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(to), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(to, data, 0644); err != nil {
		t.Fatal(err)
	}
}

func corruptFile(t *testing.T, path string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) == 0 {
		data = []byte{1}
	} else {
		data[len(data)/2] ^= 1
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
}

func TestRetainedReplayControls(t *testing.T) {
	if os.Getenv("CRF_ORACLE_MODE") != "replay" {
		t.Skip("set CRF_ORACLE_MODE=replay for retained controls")
	}
	source, library, root := os.Getenv("DORADO_SOURCE"), os.Getenv("NANOPORE_TENSOR_BACKEND"), os.Getenv("CRF_RETAINED_ATTEMPT")
	if source == "" || library == "" || root == "" {
		t.Fatal("retained controls require source, backend, and attempt paths")
	}
	fixtures, pins, err := Replay(source, library, root)
	if err != nil {
		t.Fatal(err)
	}
	if out := os.Getenv("CRF_ARTIFACT_DIR"); out != "" {
		out = filepath.Join(out, "replay-controls")
		if err := os.MkdirAll(filepath.Dir(out), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.Mkdir(out, 0755); err != nil {
			t.Fatal(err)
		}
		if err := writeJSON(filepath.Join(out, "retained-pins.json"), struct {
			Qualification string
			Fixtures      int
			Pins          map[string]string
		}{"retained bytes and dependencies verified only; no numerical comparison or native producer", len(fixtures), pins}); err != nil {
			t.Fatal(err)
		}
	}
	if len(fixtures) != 66 || fixtures[65].Name != "captured_stock_hac_n1_l96" {
		t.Fatal("retained fixture coverage changed")
	}
	copied := t.TempDir()
	for path := range pins {
		rel, err := filepath.Rel(root, path)
		if err != nil {
			t.Fatal(err)
		}
		if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			continue
		}
		copyFile(t, path, filepath.Join(copied, rel))
	}
	t.Setenv("PATH", t.TempDir())
	if _, _, err := Replay(source, library, copied); err != nil {
		t.Fatal(err)
	}
	for _, relative := range []string{"native/inputs.bin", "native/65/posts.bin", "oracle/crf-reference", "oracle/build.json", "native/run.json"} {
		t.Run(relative, func(t *testing.T) {
			path := filepath.Join(copied, relative)
			original, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			corruptFile(t, path)
			_, _, replayErr := Replay(source, library, copied)
			if err := os.WriteFile(path, original, 0644); err != nil {
				t.Fatal(err)
			}
			if replayErr == nil {
				t.Fatalf("accepted corrupt %s", relative)
			}
			t.Logf("rejected: %v", replayErr)
		})
	}
	t.Run("source", func(t *testing.T) {
		mutated := t.TempDir()
		for _, path := range []string{SourcePath, initPath} {
			copyFile(t, filepath.Join(source, path), filepath.Join(mutated, path))
		}
		corruptFile(t, filepath.Join(mutated, SourcePath))
		if _, _, err := Replay(mutated, library, copied); err == nil {
			t.Fatal("accepted modified source")
		} else {
			t.Logf("rejected: %v", err)
		}
	})
	t.Run("backend", func(t *testing.T) {
		mutated := t.TempDir()
		lib := filepath.Join(mutated, filepath.Base(library))
		copyFile(t, library, lib)
		copyFile(t, filepath.Join(filepath.Dir(library), "manifest.json"), filepath.Join(mutated, "manifest.json"))
		corruptFile(t, lib)
		if _, err := VerifyBackend(lib); err == nil {
			t.Fatal("accepted modified backend")
		} else {
			t.Logf("rejected: %v", err)
		}
		copyFile(t, library, lib)
		corruptFile(t, filepath.Join(mutated, "manifest.json"))
		if _, err := VerifyBackend(lib); err == nil {
			t.Fatal("accepted modified backend manifest")
		} else {
			t.Logf("rejected: %v", err)
		}
	})
	t.Run("runtime_dependency", func(t *testing.T) {
		const original = "/usr/lib/libdl.so.2"
		copiedDependency := filepath.Join(t.TempDir(), "libdl.so.2")
		copyFile(t, original, copiedDependency)
		if err := verifyPin(copiedDependency, pins[original], map[string]string{}); err != nil {
			t.Fatal(err)
		}
		corruptFile(t, copiedDependency)
		if err := verifyPin(copiedDependency, pins[original], map[string]string{}); err == nil {
			t.Fatal("accepted modified runtime dependency")
		} else {
			t.Logf("rejected: %v", err)
		}
	})
	input, err := os.ReadFile(filepath.Join(copied, "native", "inputs.bin"))
	if err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func([]byte) []byte{
		"truncated": func(b []byte) []byte { return b[:len(b)-1] },
		"trailing":  func(b []byte) []byte { return append(b, 0) },
		"blank":     func(b []byte) []byte { binary.LittleEndian.PutUint32(b[16:], 0); return b },
		"captured":  func(b []byte) []byte { b[len(b)-1] ^= 1; return b },
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := parseRetainedInputs(mutate(append([]byte{}, input...))); err == nil {
				t.Fatal("accepted malformed input")
			}
		})
	}
}
