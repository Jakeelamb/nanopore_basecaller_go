package cgobuild

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGenuineSuccessfulSelectedArtifact(t *testing.T) {
	names := []string{
		"H08_GENUINE_EVIDENCE_PATH", "H08_GENUINE_EVIDENCE_SHA256",
		"H08_GENUINE_SELECTION_PATH", "H08_GENUINE_SELECTION_SHA256",
		"H08_GENUINE_OUTPUT_PATH", "H08_GENUINE_OUTPUT_SHA256",
		"H08_GENUINE_ROLE", "H08_GENUINE_CONTROLS_DIR",
	}
	values := make([]string, len(names))
	present := 0
	for i, name := range names {
		var set bool
		values[i], set = os.LookupEnv(name)
		if set {
			present++
		}
	}
	if present == 0 {
		t.Skip("independent genuine successful artifact pins not selected")
	}
	for i, value := range values {
		if value == "" {
			t.Fatalf("incomplete genuine successful artifact selection: %s is required", names[i])
		}
	}
	role := BuildRole(values[6])
	reader, otherRole := ReadH08Artifact, CPU13Tests
	switch role {
	case H08Command:
	case CPU13Tests:
		reader, otherRole = ReadCPU13Artifact, H08Command
	case SUPGraphCommand:
		root := os.Getenv("SUP_GENUINE_ROOT")
		if root == "" {
			t.Fatal("SUP_GENUINE_ROOT is required for a selected SUP artifact")
		}
		reader = func(evidence, selection Pin) (Artifact, error) {
			return ReadSUPArtifact(evidence, selection, root)
		}
		otherRole = H08Command
	default:
		t.Fatalf("unknown genuine successful artifact role %q", role)
	}
	checkPin := func(p Pin) {
		t.Helper()
		hash, e := hex.DecodeString(p.SHA256)
		if !filepath.IsAbs(p.Path) || filepath.Clean(p.Path) != p.Path || e != nil || len(hash) != 32 || strings.ToLower(p.SHA256) != p.SHA256 {
			t.Fatalf("complete absolute path and lowercase SHA256 required: %+v", p)
		}
	}
	evidence := Pin{Path: values[0], SHA256: values[1]}
	selection := Pin{Path: values[2], SHA256: values[3]}
	expected := Pin{Path: values[4], SHA256: values[5]}
	for _, p := range []Pin{evidence, selection, expected} {
		checkPin(p)
	}
	parent := values[7]
	if !filepath.IsAbs(parent) || filepath.Clean(parent) != parent {
		t.Fatal("H08_GENUINE_CONTROLS_DIR must be a clean absolute directory")
	}
	var s successfulSelectionWire
	if e := exactRead(selection, &s); e != nil {
		t.Fatal(e)
	}
	if s.Schema != successfulSelectionSchema || s.Role != role || s.Producer.Receipt != evidence || s.Producer.Output != expected {
		t.Fatal("successful selection schema, role, receipt, or independent expected output differs")
	}
	for _, p := range []Pin{s.Producer.Receipt, s.Producer.Plan, s.Producer.Controller, s.Producer.Queries, s.Producer.Output} {
		checkPin(p)
	}
	var receipt Receipt
	if e := read(evidence, &receipt); e != nil {
		t.Fatal(e)
	}
	if !receipt.RuntimeHeld {
		t.Fatal("selected successful receipt must keep runtime held")
	}
	a, e := reader(evidence, selection)
	if e != nil {
		t.Fatal(e)
	}
	if a.Output() != expected {
		t.Fatalf("selected output %+v differs from independent expected output %+v", a.Output(), expected)
	}
	t.Logf("authenticated %s output %+v; runtime held", role, a.Output())
	if role == SUPGraphCommand {
		for _, root := range []string{"", "relative", t.TempDir()} {
			if a, err := ReadSUPArtifact(evidence, selection, root); err == nil || a.Output() != (Pin{}) {
				t.Fatalf("SUP accepted wrong root %q: %v", root, err)
			}
		}
	}
	dir, e := os.MkdirTemp(parent, "genuine-success-controls-")
	if e != nil {
		t.Fatal(e)
	}
	logPath := filepath.Join(dir, "controls.log")
	t.Logf("retained selection controls: %s; diagnostic log: %s", dir, logPath)
	log, e := os.OpenFile(logPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() {
		if e := log.Close(); e != nil {
			t.Error(e)
		}
	})
	if _, e = fmt.Fprintf(log, "role=%s evidence=%+v selection=%+v expected_output=%+v runtime_held=true\n", role, evidence, selection, expected); e != nil {
		t.Fatal(e)
	}
	changePin := func(p Pin) Pin {
		first := "0"
		if p.SHA256[0] == '0' {
			first = "1"
		}
		p.SHA256 = first + p.SHA256[1:]
		return p
	}
	for _, name := range []string{"role", "receipt", "plan", "controller", "queries", "output"} {
		t.Run(name, func(t *testing.T) {
			mutant := s
			want := "tuple differs"
			switch name {
			case "role":
				mutant.Role = otherRole
				want = "role/receipt differs"
			case "receipt":
				mutant.Producer.Receipt = changePin(mutant.Producer.Receipt)
				want = "role/receipt differs"
			case "plan":
				mutant.Producer.Plan = changePin(mutant.Producer.Plan)
			case "controller":
				mutant.Producer.Controller = changePin(mutant.Producer.Controller)
			case "queries":
				mutant.Producer.Queries = changePin(mutant.Producer.Queries)
			case "output":
				mutant.Producer.Output = changePin(mutant.Producer.Output)
			}
			body, e := json.MarshalIndent(mutant, "", "  ")
			if e != nil {
				t.Fatal(e)
			}
			path := filepath.Join(dir, name+"-selection.json")
			if e = os.WriteFile(path, append(body, '\n'), 0600); e != nil {
				t.Fatal(e)
			}
			pin, e := closedPin(path)
			if e != nil {
				t.Fatal(e)
			}
			t.Logf("retained %s selection: %+v", name, pin)
			if _, e = fmt.Fprintf(log, "control=%s selection=%+v expected_diagnostic=%q\n", name, pin, want); e != nil {
				t.Fatal(e)
			}
			a, err := reader(evidence, pin)
			if _, e = fmt.Fprintf(log, "control=%s error=%v output=%+v\n", name, err, a.Output()); e != nil {
				t.Fatal(e)
			}
			if err == nil || !strings.Contains(err.Error(), want) || a.Output() != (Pin{}) {
				t.Fatalf("%s control: want %q and zero output; error=%v output=%+v", name, want, err, a.Output())
			}
		})
	}
}
