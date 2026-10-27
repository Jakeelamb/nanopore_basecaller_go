package decodergpu

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func TestLineageGoEnvironmentUsesPinnedExperiment(t *testing.T) {
	dir := t.TempDir()
	tool := filepath.Join(dir, "tool")
	root := filepath.Join(dir, "goroot")
	pkg := filepath.Join(dir, "module/verification/decodergpu/lineagecmd")
	for _, path := range []string{tool, filepath.Join(root, "pkg/include"), pkg} {
		if err := os.MkdirAll(path, 0700); err != nil {
			t.Fatal(err)
		}
	}
	for _, path := range []string{filepath.Join(dir, "go"), filepath.Join(tool, "compile"), filepath.Join(tool, "asm"), filepath.Join(tool, "link"), filepath.Join(root, "pkg/include/textflag.h"), filepath.Join(dir, "module/go.mod"), filepath.Join(pkg, "main.go")} {
		if err := os.WriteFile(path, []byte("synthetic selected metadata input\n"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	env := map[string]string{"CGO_ENABLED": "0", "GOOS": "linux", "GOARCH": "amd64", "GOAMD64": "v1", "GOFLAGS": "", "GOWORK": "off", "GOENV": "", "GOEXPERIMENT": "nodwarf5", "GOMOD": filepath.Join(dir, "module/go.mod"), "GOROOT": root, "GOTOOLDIR": tool}
	deps, _ := json.Marshal(goPackage{ImportPath: "github.com/Jakeelamb/nanopore_basecaller_go/verification/decodergpu/lineagecmd", Dir: pkg, GoFiles: []string{"main.go"}})
	raw, _ := json.Marshal(env)
	selected, err := lineageGoSelection(deps, raw, filepath.Join(dir, "go"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if selected[filepath.Join(pkg, "main.go")] == "" || !slices.Contains(lineageGoEnvironment, "GOEXPERIMENT=nodwarf5") {
		t.Fatal("actual dependency or explicit selected experiment missing")
	}
	for _, experiment := range []string{"", "rangefunc", "dwarf5"} {
		t.Run("reject-"+experiment, func(t *testing.T) {
			env["GOEXPERIMENT"] = experiment
			bad, _ := json.Marshal(env)
			if _, err := lineageGoSelection(deps, bad, filepath.Join(dir, "go"), nil); err == nil {
				t.Fatal("unselected reported Go experiment accepted")
			}
		})
	}
	env["GOEXPERIMENT"] = "nodwarf5"
	raw, _ = json.Marshal(env)
	delete(selected, filepath.Join(pkg, "main.go"))
	if _, err := lineageGoSelection(deps, raw, filepath.Join(dir, "go"), selected); err == nil {
		t.Fatal("historical raw Go selection omitted its actual controller source")
	}
}
