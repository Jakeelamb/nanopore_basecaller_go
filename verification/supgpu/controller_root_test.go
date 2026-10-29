package supgpu

import (
	"encoding/json"
	"path/filepath"
	"testing"
)

func TestControllerSourceRootEvidence(t *testing.T) {
	const module = "github.com/Jakeelamb/nanopore_basecaller_go"
	const target = module + "/verification/supgpu/cmd"
	root := filepath.Join(t.TempDir(), "producer")
	packageAt := func(at string) map[string]any {
		return map[string]any{"Dir": filepath.Join(at, "verification/supgpu/cmd"), "Root": at, "ImportPath": target, "Module": map[string]any{"Path": module, "Main": true, "Dir": at, "GoMod": filepath.Join(at, "go.mod")}}
	}
	for _, name := range []string{"valid", "missing-module", "missing-target", "wrong-module", "wrong-root", "noncanonical-root", "wrong-gomod", "wrong-dir", "not-main", "duplicate-target", "different-roots", "unknown-module"} {
		t.Run(name, func(t *testing.T) {
			p := packageAt(root)
			packages := []map[string]any{p}
			switch name {
			case "missing-module":
				delete(p, "Module")
			case "missing-target":
				p["ImportPath"] = module
				p["Dir"] = root
			case "wrong-module":
				p["Module"].(map[string]any)["Path"] = "example.invalid/other"
			case "wrong-root":
				p["Root"] = filepath.Join(root, "other")
			case "noncanonical-root":
				p["Root"] = root + "/."
			case "wrong-gomod":
				p["Module"].(map[string]any)["GoMod"] = filepath.Join(root, "other/go.mod")
			case "wrong-dir":
				p["Dir"] = root
			case "not-main":
				p["Module"].(map[string]any)["Main"] = false
			case "duplicate-target":
				packages = append(packages, packageAt(root))
			case "different-roots":
				packages = append(packages, packageAt(filepath.Join(root, "other")))
			case "unknown-module":
				packages = append(packages, map[string]any{"Dir": root, "ImportPath": "example.invalid/unknown"})
			}
			var raw []byte
			for _, pkg := range packages {
				b, err := json.Marshal(pkg)
				if err != nil {
					t.Fatal(err)
				}
				raw = append(append(raw, b...), '\n')
			}
			got, err := controllerSourceRoot(raw, target)
			if name == "valid" {
				if err != nil || got != root {
					t.Fatalf("root = %q, error = %v", got, err)
				}
			} else if err == nil {
				t.Fatal("invalid root evidence accepted")
			}
		})
	}
}
