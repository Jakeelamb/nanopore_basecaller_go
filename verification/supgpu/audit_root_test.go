package supgpu

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

// Uses only the pre-existing auditTestInputs entry point so the controller can
// overlay this regression onto the unchanged historical helper.
func TestAuditControlSavedProducerRoot(t *testing.T) {
	const module = "github.com/Jakeelamb/nanopore_basecaller_go"
	dir := t.TempDir()
	root := filepath.Join(dir, "saved-producer")
	tools := filepath.Join(dir, "go", "tools")
	include := filepath.Join(dir, "go", "pkg", "include")
	for _, path := range []string{tools, include} {
		if err := os.MkdirAll(path, 0700); err != nil {
			t.Fatal(err)
		}
	}
	for _, path := range []string{filepath.Join(tools, "inert-tool"), filepath.Join(include, "inert.h")} {
		if err := os.WriteFile(path, []byte("inert\n"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	packageAt := func(at string) map[string]any {
		return map[string]any{"Dir": filepath.Join(at, "verification/supgpu"), "Root": at, "ImportPath": module + "/verification/supgpu", "Module": map[string]any{"Path": module, "Main": true, "Dir": at, "GoMod": filepath.Join(at, "go.mod")}, "GoFiles": []string{"inert.go"}}
	}
	encode := func(packages ...map[string]any) []byte {
		t.Helper()
		var raw []byte
		for _, p := range packages {
			b, err := json.Marshal(p)
			if err != nil {
				t.Fatal(err)
			}
			raw = append(append(raw, b...), '\n')
		}
		return raw
	}
	env, err := json.Marshal(goEnvironment{GOROOT: filepath.Join(dir, "go"), GOTOOLDIR: tools, GOOS: "linux", GOARCH: "amd64", GOAMD64: "v1", CGO_ENABLED: "0", GOWORK: "off"})
	if err != nil {
		t.Fatal(err)
	}
	generated := packageAt(root)
	generated["ImportPath"] = module + "/verification/supgpu.test"
	generated["GoFiles"] = []string{filepath.Join(dir, "cache", "inert-test-main.go")}
	paths, err := auditTestInputs(encode(packageAt(root), generated), env, filepath.Join(dir, "go", "bin", "go"))
	if err != nil {
		t.Fatal(err)
	}
	want := []string{filepath.Join(root, "verification/supgpu/inert.go"), filepath.Join(dir, "cache/inert-test-main.go"), filepath.Join(tools, "inert-tool"), filepath.Join(include, "inert.h"), filepath.Join(dir, "go/bin/go"), filepath.Join(root, "go.mod"), filepath.Join(root, "verification/supgpu/audit-controls.sh"), filepath.Join(root, "verification/supgpu/native/controls/loader.c"), filepath.Join(root, "verification/supgpu/native/controls/constructor.c")}
	sort.Strings(want)
	if !reflect.DeepEqual(paths, want) {
		t.Fatalf("saved producer selections differ:\ngot %v\nwant %v", paths, want)
	}
	for _, name := range []string{"missing", "ambiguous", "wrong-module", "different-package-root", "different-gomod", "duplicate-target"} {
		t.Run(name, func(t *testing.T) {
			p := packageAt(root)
			packages := []map[string]any{p}
			switch name {
			case "missing":
				delete(p, "Module")
			case "ambiguous":
				packages = append(packages, packageAt(filepath.Join(dir, "other")))
			case "wrong-module":
				p["Module"].(map[string]any)["Path"] = "example.invalid/other"
			case "different-package-root":
				p["Dir"] = filepath.Join(dir, "other/verification/supgpu")
			case "different-gomod":
				p["Module"].(map[string]any)["GoMod"] = filepath.Join(dir, "other/go.mod")
			case "duplicate-target":
				packages = append(packages, packageAt(root))
			}
			if _, err := auditTestInputs(encode(packages...), env, filepath.Join(dir, "go/bin/go")); err == nil {
				t.Fatal("invalid producer root evidence accepted")
			}
		})
	}
}
