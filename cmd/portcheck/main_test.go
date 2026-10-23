package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestCacheRequiresMatchingUntamperedSuccessfulEvidence(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "attempt")
	if err := os.Mkdir(dir, 0755); err != nil {
		t.Fatal(err)
	}
	r := receipt{Schema: 1, Artifacts: map[string]string{}, Source: sourcePin{Commit: sourceCommit, Files: map[string]string{"source": "pin"}},
		Inputs: map[string]string{"code": "pin"}, Environment: map[string]string{"compiler": "pin"},
		Command: []string{"go", "test", "-count=1", "-v", "./..."}, Started: "start", Finished: "end"}
	identity := struct {
		Source      sourcePin
		Inputs      map[string]string
		Environment map[string]string
	}{r.Source, r.Inputs, r.Environment}
	encoded, err := json.Marshal(identity)
	if err != nil {
		t.Fatal(err)
	}
	r.Key = digest(encoded)
	for name, data := range map[string]string{"stdout": "--- PASS: TestNativeParity (0.1s)\n--- PASS: TestPinnedNativeScaling (0.1s)\n--- PASS: TestPinnedNativeStitch (0.1s)\n--- PASS: TestPinnedNativeDecode (0.1s)\n", "stderr": ""} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	r.StdoutSHA256, err = hashFile(filepath.Join(dir, "stdout"))
	if err != nil {
		t.Fatal(err)
	}
	r.StderrSHA256, err = hashFile(filepath.Join(dir, "stderr"))
	if err != nil {
		t.Fatal(err)
	}
	save := func() {
		t.Helper()
		if err := saveJSON(filepath.Join(dir, "receipt.json"), r); err != nil {
			t.Fatal(err)
		}
	}
	save()
	if _, ok := cachedAttempt(root, r.Key); !ok {
		t.Fatal("valid evidence was not reused")
	}
	if _, ok := cachedAttempt(root, "other"); ok {
		t.Fatal("stale key accepted")
	}
	r.Exit = 1
	save()
	if _, ok := cachedAttempt(root, r.Key); ok {
		t.Fatal("failed attempt accepted")
	}
	r.Exit = 0
	r.Inputs["code"] = "changed"
	save()
	if _, ok := cachedAttempt(root, r.Key); ok {
		t.Fatal("modified input metadata accepted")
	}
	r.Inputs["code"] = "pin"
	save()
	if err := os.WriteFile(filepath.Join(dir, "stdout"), []byte("tampered"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, ok := cachedAttempt(root, r.Key); ok {
		t.Fatal("modified output accepted")
	}
}

func TestEffectiveGoTargetChangesIdentity(t *testing.T) {
	cleanToolEnvironment(t)
	t.Setenv("CGO_ENABLED", "1")
	t.Setenv("GOOS", "linux")
	before, err := toolEnvironment()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOOS", "windows")
	after, err := toolEnvironment()
	if err != nil {
		t.Fatal(err)
	}
	if sameJSON(before, after) {
		t.Fatal("different child targets share an evidence identity")
	}
}

func TestNativeParityRejectsDisabledCGO(t *testing.T) {
	cleanToolEnvironment(t)
	t.Setenv("CGO_ENABLED", "0")
	if _, err := toolEnvironment(); err == nil {
		t.Fatal("missing native float backend accepted")
	}
}

func TestExternalBuildInjectionRejected(t *testing.T) {
	cleanToolEnvironment(t)
	for _, name := range []string{"GOFLAGS", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX", "LD_PRELOAD", "LD_LIBRARY_PATH", "LIBRARY_PATH", "LD_AUDIT"} {
		t.Run(name, func(t *testing.T) {
			t.Setenv(name, "external")
			if _, err := toolEnvironment(); err == nil {
				t.Fatal("external build input accepted")
			}
		})
	}
}

func TestUnpinnedCgoCompilerRejected(t *testing.T) {
	cleanToolEnvironment(t)
	t.Setenv("CC", "unqualified-compiler")
	if _, err := toolEnvironment(); err == nil {
		t.Fatal("custom cgo compiler was accepted without pinning")
	}
}

func cleanToolEnvironment(t *testing.T) {
	t.Helper()
	for _, name := range []string{"GOFLAGS", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "COMPILER_PATH", "GCC_EXEC_PREFIX", "LD_PRELOAD", "LD_LIBRARY_PATH", "LIBRARY_PATH", "LD_AUDIT"} {
		t.Setenv(name, "")
	}
	t.Setenv("CC", "gcc")
	t.Setenv("CXX", "g++")
}

func TestQualificationRejectsSkippedOrMissingNativeGates(t *testing.T) {
	for _, logs := range []string{"PASS\n", "--- SKIP: TestNativeParity (0.00s)\n", "--- PASS: TestNativeParity (0.1s)\n"} {
		if qualifyLog([]byte(logs)) == nil {
			t.Fatal("incomplete log qualified")
		}
	}
}

func TestCodeChangeInvalidatesInputIdentity(t *testing.T) {
	root := t.TempDir()
	code := filepath.Join(root, "caller.go")
	if err := os.WriteFile(code, []byte("package caller"), 0600); err != nil {
		t.Fatal(err)
	}
	before, err := hashInputs(root)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(code, []byte("package changed"), 0600); err != nil {
		t.Fatal(err)
	}
	after, err := hashInputs(root)
	if err != nil {
		t.Fatal(err)
	}
	if sameJSON(before, after) {
		t.Fatal("source change did not invalidate identity")
	}
}
