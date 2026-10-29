package cgobuild

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestSUPCommandTargetAndArgv(t *testing.T) {
	p := Plan{Schema: "nanopore-h08-cgo-build-plan/v2", Role: SUPGraphCommand, Options: Options{Role: SUPGraphCommand, Root: "/source", GoRoot: "/go", WorkRoot: "/work"}}
	p.BuildArgv = plannedBuildArgs(p)
	if e := validateTarget(p); e != nil {
		t.Fatal(e)
	}
	if path, name, e := commandTarget(p.Role); e != nil || path != Module+"/verification/supgpu/graph/cmd" || name != "sup-graph" {
		t.Fatalf("wrong SUP target: %s %s %v", path, name, e)
	}
	if !slices.Equal(listArgs(p), []string{"/go/bin/go", "list", "-deps", "-json", "./verification/supgpu/graph/cmd"}) || outputPath(p) != "/work/sup-graph" {
		t.Fatal("SUP list/output target differs")
	}
	want := []string{"/go/bin/go", "build", "-a", "-x", "-work", "-p=1", "-pgo=off", "-trimpath", "-buildvcs=false", "-toolexec=<SEALED_CURRENT_INTERPOSER_TOOL>", "-ldflags=-linkmode=external -compressdwarf=false -buildid= -extld=<SEALED_CURRENT_INTERPOSER_CC>", "-o", "/work/sup-graph", "./verification/supgpu/graph/cmd"}
	if !slices.Equal(p.BuildArgv, want) {
		t.Fatalf("SUP build argv differs: %q", p.BuildArgv)
	}
	// This also exercises the selection/link authority in tool_contract.go.
	if selectionTarget(p) != Module+"/verification/supgpu/graph/cmd" {
		t.Error("SUP selection/link target differs from the declared SUP command")
	}
	for _, change := range []string{"overlay", "h08 target", "external target", "output", "role", "unknown"} {
		q := p
		q.BuildArgv = slices.Clone(p.BuildArgv)
		switch change {
		case "overlay":
			q.CPU13 = &CPU13Profile{}
		case "h08 target":
			q.BuildArgv[len(q.BuildArgv)-1] = "./verification/hacgpu/h08/cmd"
		case "external target":
			q.BuildArgv[len(q.BuildArgv)-1] = "example.org/cmd"
		case "output":
			q.BuildArgv[len(q.BuildArgv)-2] = "/work/h08"
		case "role":
			q.Role, q.Options.Role = H08Command, H08Command
		case "unknown":
			q.Role, q.Options.Role = "foreign", "foreign"
		}
		if validateTarget(q) == nil {
			t.Fatalf("accepted %s", change)
		}
	}
	if _, _, e := commandTarget(CPU13Tests); e == nil {
		t.Fatal("CPU13 admitted as ordinary command")
	}
}

func TestSUPStaticSelectionBindsOnlySelectedCommand(t *testing.T) {
	root, goRoot := t.TempDir(), t.TempDir()
	for name, body := range map[string]string{
		"go.mod":                                "module " + Module + "\n",
		"verification/supgpu/graph/cmd/main.go": "package main\nimport _ \"github.com/Jakeelamb/nanopore_basecaller_go/internal/proof\"\nfunc main() {}\n",
		"internal/proof/proof.go":               "package proof\n",
		"verification/hacgpu/h08/cmd/main.go":   "package main\nfunc main() {}\n",
	} {
		path := filepath.Join(root, name)
		if e := os.MkdirAll(filepath.Dir(path), 0700); e != nil {
			t.Fatal(e)
		}
		if e := os.WriteFile(path, []byte(body), 0600); e != nil {
			t.Fatal(e)
		}
	}
	pkgs, inputs, e := selectSources(root, goRoot, SUPGraphCommand)
	if e != nil {
		t.Fatal(e)
	}
	p := Plan{Options: Options{Root: root}, Packages: pkgs, Inputs: inputs}
	if len(pkgs) != 2 || inputs[filepath.Join(root, "verification/supgpu/graph/cmd/main.go")] == "" || inputs[filepath.Join(root, "internal/proof/proof.go")] == "" || inputs[filepath.Join(root, "verification/hacgpu/h08/cmd/main.go")] != "" {
		t.Fatal("SUP selected wrong source closure")
	}
	if e := checkSUPInputs(p, root); e != nil {
		t.Fatal(e)
	}
	for _, wrong := range []string{"", "relative", root + "/.", t.TempDir()} {
		if checkSUPInputs(p, wrong) == nil {
			t.Fatalf("accepted root %q", wrong)
		}
	}
	alias := filepath.Join(t.TempDir(), "alias")
	if e := os.Symlink(root, alias); e != nil {
		t.Fatal(e)
	}
	q := p
	q.Options.Root = alias
	if checkSUPInputs(q, alias) == nil {
		t.Fatal("accepted noncanonical symlink root")
	}
	if e := os.WriteFile(filepath.Join(root, "internal/proof/proof.go"), []byte("package proof\nconst Changed = true\n"), 0600); e != nil {
		t.Fatal(e)
	}
	if e := checkSUPInputs(p, root); e == nil || !strings.Contains(e.Error(), "input changed") {
		t.Fatalf("changed declared source accepted: %v", e)
	}
	if _, _, e := selectSources(root, goRoot, "unknown"); e == nil {
		t.Fatal("unknown source role accepted")
	}
	if e := os.Remove(filepath.Join(root, "verification/supgpu/graph/cmd/main.go")); e != nil {
		t.Fatal(e)
	}
	if _, _, e := selectSources(root, goRoot, SUPGraphCommand); e == nil {
		t.Fatal("missing SUP command accepted")
	}
}

func TestSUPSelectedArtifactRejectsOtherBranchesAndSelectors(t *testing.T) {
	dir := t.TempDir()
	r := Receipt{Role: SUPGraphCommand, Status: "failed"}
	evidence := observationTestPin(t, filepath.Join(dir, "receipt.json"), r)
	s := successfulSelectionWire{Schema: successfulSelectionSchema, Role: SUPGraphCommand, Producer: retainedProducerPins{Receipt: evidence}}
	selection := observationTestPin(t, filepath.Join(dir, "selection.json"), s)
	for _, pair := range [][2]Pin{{{}, {}}, {evidence, {}}, {{}, selection}, {selection, evidence}, {evidence, selection}} {
		if a, e := ReadSUPArtifact(pair[0], pair[1], dir); e == nil || a.Output() != (Pin{}) {
			t.Fatal("blank/swapped/failed producer accepted")
		}
	}
	for _, role := range []BuildRole{H08Command, CPU13Tests, "unknown"} {
		q := s
		q.Role = role
		pin := observationTestPin(t, filepath.Join(t.TempDir(), "selection.json"), q)
		if a, e := ReadSUPArtifact(evidence, pin, dir); e == nil || !strings.Contains(e.Error(), "role/receipt") || a.Output() != (Pin{}) {
			t.Fatalf("role substitution accepted: %s %v", role, e)
		}
	}
	observed := observationTestPin(t, filepath.Join(dir, "observed.json"), observedSelectionWire{Schema: observedSelectionSchema})
	if a, e := ReadSUPArtifact(evidence, observed, dir); e == nil || !strings.Contains(e.Error(), "H08-only") || a.Output() != (Pin{}) {
		t.Fatalf("H08 observed evidence relabelled SUP: %v", e)
	}
	if _, e := readArtifact(evidence, selection, "unknown"); e == nil {
		t.Fatal("unknown reader role accepted")
	}
}
