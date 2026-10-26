package h06

import (
	"encoding/json"
	"maps"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

func TestH06GoControllerSelectionIncludesAssemblerAndEmbeds(t *testing.T) {
	dir := t.TempDir()
	tools := filepath.Join(dir, "pkg/tool/linux_amd64")
	includes := filepath.Join(dir, "pkg/include")
	pkg := filepath.Join(dir, "source")
	for _, p := range []string{tools, includes, pkg} {
		if err := os.MkdirAll(p, 0700); err != nil {
			t.Fatal(err)
		}
	}
	for _, p := range []string{filepath.Join(dir, "go"), filepath.Join(tools, "compile"), filepath.Join(tools, "link"), filepath.Join(tools, "asm"), filepath.Join(includes, "textflag.h"), filepath.Join(includes, "funcdata.h"), filepath.Join(pkg, "main.go"), filepath.Join(pkg, "weights.json"), filepath.Join(dir, "go.mod")} {
		testBuildFile(t, p, []byte(p))
	}
	env := map[string]string{"CGO_ENABLED": "0", "GOOS": "linux", "GOARCH": "amd64", "GOAMD64": "v1", "GOFLAGS": "", "GOWORK": "off", "GOENV": "", "GOROOT": dir, "GOTOOLDIR": tools, "GOMOD": filepath.Join(dir, "go.mod")}
	body, _ := json.Marshal(goBuildPackage{Dir: pkg, ImportPath: "fixture", GoFiles: []string{"main.go"}, EmbedFiles: []string{"weights.json"}})
	pins, selection, err := controllerSelection(body, env, filepath.Join(dir, "go"))
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{filepath.Join(includes, "textflag.h"), filepath.Join(includes, "funcdata.h"), filepath.Join(pkg, "weights.json"), filepath.Join(dir, "go.mod")} {
		if pins[p] == "" || selection[p] != p {
			t.Fatalf("actual Go input omitted: %s", p)
		}
	}
	env["CGO_ENABLED"] = "1"
	if _, _, err = controllerSelection(body, env, filepath.Join(dir, "go")); err == nil {
		t.Fatal("CGO1 accepted without native proof")
	}
	env["CGO_ENABLED"] = "0"
	bad, _ := json.Marshal(goBuildPackage{Dir: pkg, ImportPath: "fixture", CFiles: []string{"hidden.c"}})
	if _, _, err = controllerSelection(bad, env, filepath.Join(dir, "go")); err == nil {
		t.Fatal("CGO0 selection admitted native C inputs")
	}
	if _, _, err = controllerSelection([]byte(""), env, filepath.Join(dir, "go")); err == nil {
		t.Fatal("empty actual dependency selection accepted")
	}
}

func TestH06ControllerReceiptRejectsRehashedClosureOmission(t *testing.T) {
	dir := t.TempDir()
	tools := filepath.Join(dir, "pkg/tool/linux_amd64")
	includes := filepath.Join(dir, "pkg/include")
	pkg := filepath.Join(dir, "source")
	copies := filepath.Join(dir, "snapshots")
	for _, path := range []string{tools, includes, pkg, copies} {
		if err := os.MkdirAll(path, 0700); err != nil {
			t.Fatal(err)
		}
	}
	goBin := filepath.Join(dir, "go")
	binary := testBuildFile(t, filepath.Join(dir, "controller"), []byte("sealed Go binary"))
	for _, p := range []string{goBin, filepath.Join(tools, "compile"), filepath.Join(tools, "link"), filepath.Join(tools, "asm"), filepath.Join(includes, "textflag.h"), filepath.Join(pkg, "main.go"), filepath.Join(dir, "go.mod")} {
		testBuildFile(t, p, []byte(p))
	}
	env := map[string]string{"CGO_ENABLED": "0", "GOOS": "linux", "GOARCH": "amd64", "GOAMD64": "v1", "GOWORK": "off", "GOROOT": dir, "GOTOOLDIR": tools, "GOMOD": filepath.Join(dir, "go.mod")}
	envBody, _ := json.Marshal(env)
	depsBody, _ := json.Marshal(goBuildPackage{Dir: pkg, ImportPath: "fixture", GoFiles: []string{"main.go"}})
	inputs, selection, err := controllerSelection(depsBody, env, goBin)
	if err != nil {
		t.Fatal(err)
	}
	r := ControllerBuild{Schema: "nanopore-hac-half-controller/v1", Status: "built", Binary: binary, Inputs: inputs, Selection: selection, Snapshots: map[string]string{}}
	r.Limits = map[string]string{"cgroup": "fixture", "child_timeout_seconds": "180", "cpu.max": "200000 100000", "memory.max": "3221225472", "memory.swap.max": "0"}
	for path := range inputs {
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		copy := testBuildFile(t, filepath.Join(copies, digest([]byte(path))), body)
		r.Snapshots[path] = copy.Path
	}
	argv := [][]string{{goBin, "env", "-json"}, {goBin, "list", "-deps", "-json", "-pgo=off", "./verification/hacgpu/h06/cmd"}, {goBin, "build", "-a", "-x", "-work", "-pgo=off", "-trimpath", "-buildvcs=false", "-o", binary.Path, "./verification/hacgpu/h06/cmd"}, {goBin, "list", "-deps", "-json", "-pgo=off", "./verification/hacgpu/h06/cmd"}}
	for i, body := range [][]byte{envBody, depsBody, {}, depsBody} {
		out := testBuildFile(t, filepath.Join(dir, string(rune('a'+i))+".stdout"), body)
		stderr := testBuildFile(t, filepath.Join(dir, string(rune('a'+i))+".stderr"), nil)
		r.Commands = append(r.Commands, runner.CommandReceipt{Args: argv[i], Started: "2026-10-25T00:00:00Z", Finished: "2026-10-25T00:00:01Z", Exit: 0, Stdout: out.Path, StdoutSHA: out.SHA256, Stderr: stderr.Path, StderrSHA: stderr.SHA256})
	}
	seal := func(name string, value ControllerBuild) nativeproof.Pin {
		t.Helper()
		body, _ := json.Marshal(value)
		return testBuildFile(t, filepath.Join(dir, name), body)
	}
	if _, err = ReadControllerBuild(seal("good.json", r), false); err != nil {
		t.Fatal(err)
	}
	omitted := r
	omitted.Inputs = maps.Clone(r.Inputs)
	omitted.Snapshots = maps.Clone(r.Snapshots)
	delete(omitted.Inputs, filepath.Join(includes, "textflag.h"))
	delete(omitted.Snapshots, filepath.Join(includes, "textflag.h"))
	if _, err = ReadControllerBuild(seal("omitted.json", omitted), false); err == nil || !strings.Contains(err.Error(), "selected input closure") {
		t.Fatalf("rehashed assembler header omission accepted: %v", err)
	}
	copy := r.Snapshots[filepath.Join(pkg, "main.go")]
	if err = os.WriteFile(copy, []byte("changed observed snapshot"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = ReadControllerBuild(seal("changed.json", r), false); err == nil {
		t.Fatal("changed retained source snapshot accepted")
	}
}

func TestH06GoControllerRejectsPackageEscape(t *testing.T) {
	env := map[string]string{"CGO_ENABLED": "0", "GOOS": "linux", "GOARCH": "amd64", "GOAMD64": "v1", "GOWORK": "off"}
	body, _ := json.Marshal(goBuildPackage{Dir: t.TempDir(), GoFiles: []string{"../escape.go"}})
	if _, _, err := controllerSelection(body, env, "go"); err == nil || !strings.Contains(err.Error(), "escapes") {
		t.Fatalf("package escape accepted: %v", err)
	}
}
