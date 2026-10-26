package h07

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestOutputControllerImmutableSourceCustody(t *testing.T) {
	dir := t.TempDir()
	original := filepath.Join(dir, "consumer.go")
	snapshot := filepath.Join(dir, "producer-consumer.go")
	exe := filepath.Join(dir, "h07")
	for _, path := range []string{original, snapshot, exe} {
		if err := os.WriteFile(path, []byte("original"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	toolPins := map[string]string{}
	for i, name := range []string{"go", "compile", "link", "cgo", "cc"} {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte{byte(i)}, 0600); err != nil {
			t.Fatal(err)
		}
		toolPins[path] = sum([]byte{byte(i)})
	}
	before := sum([]byte("original"))
	r := ControllerBuild{Schema: "nanopore-hac-output-controller-build/v1", Status: "built", Output: exe, OutputSHA256: before, SourceSHA256: map[string]string{original: before}, AfterSourceSHA256: map[string]string{original: before}, SourceSnapshots: map[string]string{original: snapshot}, SnapshotSHA256: map[string]string{snapshot: before}, ToolSHA256: toolPins}
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	manifest := filepath.Join(dir, "build.json")
	if err = os.WriteFile(manifest, raw, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = ReadControllerBuild(manifest, sum(raw), exe, before); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(original, []byte("later consumer correction"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = ReadControllerBuild(manifest, sum(raw), exe, before); err != nil {
		t.Fatalf("later consumer invalidated retained producer custody: %v", err)
	}
	if err = os.WriteFile(snapshot, []byte("corrupt"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = ReadControllerBuild(manifest, sum(raw), exe, before); err == nil {
		t.Fatal("changed producer snapshot accepted")
	}
}
