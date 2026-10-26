package h07

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestOutputOracleDepfileIncludesEscapedSystemPaths(t *testing.T) {
	dir := t.TempDir()
	first := filepath.Join(dir, "source.cpp")
	second := filepath.Join(dir, "system header.h")
	for _, p := range []string{first, second} {
		if err := os.WriteFile(p, []byte("pinned"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	deps, err := ParseDependencies("output.o: source.cpp \\\n system\\ header.h source.cpp\n", dir)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{first, second}
	if !reflect.DeepEqual(deps, want) {
		t.Fatalf("deps %v want %v", deps, want)
	}
	for _, raw := range []string{"", "output.o:", "output.o: absent.h", "output.o: source.cpp\\"} {
		if _, err := ParseDependencies(raw, dir); err == nil {
			t.Fatalf("malformed dependency closure accepted: %q", raw)
		}
	}
}
func TestOutputOracleLinkClosureMustExistBeforeLink(t *testing.T) {
	dir := t.TempDir()
	archive := filepath.Join(dir, "library.a")
	object := filepath.Join(dir, "direct.o")
	for _, p := range []string{archive, object} {
		if err := os.WriteFile(p, []byte("frozen"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	inputs, objects := map[string]string{}, map[string]string{}
	if err := pinFile(inputs, archive); err != nil {
		t.Fatal(err)
	}
	if err := pinFile(objects, object); err != nil {
		t.Fatal(err)
	}
	if err := VerifyLinkTrace(archive+"\n"+object+"\n", dir, inputs, objects); err != nil {
		t.Fatal(err)
	}
	if err := VerifyLinkTrace(archive+"\n"+object+"\n", dir, map[string]string{}, objects); err == nil {
		t.Fatal("unfrozen library accepted")
	}
	if err := os.WriteFile(archive, []byte("changed"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := VerifyLinkTrace(archive+"\n", dir, inputs, objects); err == nil {
		t.Fatal("changed archive accepted")
	}
	if err := VerifyLinkTrace("", dir, inputs, objects); err == nil {
		t.Fatal("empty linker trace accepted")
	}
}
func TestOutputOracleBuildAdmissionPrecedesEveryTool(t *testing.T) {
	if _, err := BuildOracle(OracleBuildOptions{BuildDir: "absent", DoradoSource: "absent", Out: "absent"}); err == nil {
		t.Fatal("unadmitted compiler execution accepted")
	}
}
