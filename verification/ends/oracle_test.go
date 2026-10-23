package endsoracle

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestSourceCustodyRejection(t *testing.T) {
	_, here, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot locate source")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(here), "../../external/dorado-source"))
	if _, err := os.Stat(filepath.Join(root, "dorado/demux/AdapterDetector.cpp")); os.IsNotExist(err) {
		t.Skip("pinned source checkout unavailable")
	}
	if _, err := os.Stat(filepath.Join(DefaultEdlibRoot, "edlib/src/edlib.cpp")); os.IsNotExist(err) {
		t.Skip("pinned edlib checkout unavailable")
	}
	if err := VerifySources(root, DefaultEdlibRoot); err != nil {
		t.Fatal(err)
	}
	copyRoot, copyEdlib := filepath.Join(t.TempDir(), "dorado"), filepath.Join(t.TempDir(), "edlib")
	for _, set := range []struct {
		from, to string
		pins     map[string]string
	}{{root, copyRoot, SourcePins}, {DefaultEdlibRoot, copyEdlib, EdlibPins}} {
		for path := range set.pins {
			data, err := os.ReadFile(filepath.Join(set.from, path))
			if err != nil {
				t.Fatal(err)
			}
			destination := filepath.Join(set.to, path)
			if err := os.MkdirAll(filepath.Dir(destination), 0700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(destination, data, 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, set := range []struct {
		root string
		pins map[string]string
	}{{copyRoot, SourcePins}, {copyEdlib, EdlibPins}} {
		for path := range set.pins {
			t.Run(path, func(t *testing.T) {
				file := filepath.Join(set.root, path)
				original, err := os.ReadFile(file)
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(file, append(append([]byte{}, original...), '\n'), 0600); err != nil {
					t.Fatal(err)
				}
				if err := VerifySources(copyRoot, copyEdlib); err == nil || !strings.Contains(err.Error(), "source hash mismatch") {
					t.Fatalf("source mutation accepted or wrong rejection: %v", err)
				}
				if err := os.WriteFile(file, original, 0600); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
	file := filepath.Join(copyRoot, "dorado/demux/AdapterDetector.cpp")
	data, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file, append(data, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	output := filepath.Join(t.TempDir(), "build")
	if _, err := Build(copyRoot, copyEdlib, output); err == nil {
		t.Fatal("mutated source built")
	}
	if _, err := os.Stat(filepath.Join(output, "rejection.json")); err != nil {
		t.Fatal("missing retained rejection")
	}
	if _, err := os.Stat(filepath.Join(output, "ends-oracle")); !os.IsNotExist(err) {
		t.Fatal("rejected source produced executable")
	}
	if _, err := Build(root, DefaultEdlibRoot, output); err == nil {
		t.Fatal("existing output directory accepted")
	}
}

func TestExtractionKeepsFunctionBytes(t *testing.T) {
	source := []byte("before\nfloat candidate(int x) { if(x) { return 1; } return 0; }\nafter")
	got, err := sourceFunction(source, "float candidate(")
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "float candidate(int x) { if(x) { return 1; } return 0; }" {
		t.Fatalf("function extraction altered bytes %q", got)
	}
	for _, text := range []string{"before", "float candidate(int x);", "float candidate(int x) {"} {
		if _, err := sourceFunction([]byte(text), "float candidate("); err == nil {
			t.Fatalf("malformed source accepted %q", text)
		}
	}
}
