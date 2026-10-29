package fakecapture

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08"
)

type integrationFixture struct {
	Schema           string
	Request, Summary Pin
}

// Root supplies the independently pinned fixture after an actual admitted
// capture of the selected CPU13 binary. This reader test starts no programs.
// Absence is an honest skip, never a substitute for a native capture.
func genuineFixture(t *testing.T) (integrationFixture, Request, Summary) {
	t.Helper()
	path, sha := os.Getenv("H08_FAKE_CAPTURE_FIXTURE"), os.Getenv("H08_FAKE_CAPTURE_FIXTURE_SHA256")
	if path == "" && sha == "" {
		t.Skip("genuine selected fake capture not supplied")
	}
	var fixture integrationFixture
	if e := readExact(Pin{Path: path, SHA256: sha}, &fixture); e != nil {
		t.Fatal(e)
	}
	if fixture.Schema != "nanopore-h08-fake-capture-integration/v1" {
		t.Fatal("fixture schema differs")
	}
	q, e := readRequest(fixture.Request, false)
	if e != nil {
		t.Fatal(e)
	}
	r, e := ReadAccepted(fixture.Request, fixture.Summary)
	if e != nil {
		t.Fatal(e)
	}
	return fixture, q, r
}

func TestGenuineAcceptedCapture(t *testing.T) {
	_, _, _ = genuineFixture(t)
}

// Exact rejection-control entry point: no capture is repeated. Invalid derived
// candidates remain in a fresh sibling directory beside root's real capture.
func TestGenuineCaptureRejectionControls(t *testing.T) {
	_, q, r := genuineFixture(t)
	body, e := pinBytes(r.Proof)
	if e != nil {
		t.Fatal(e)
	}
	controls, e := os.MkdirTemp(filepath.Dir(q.Out), "fake-capture-rejections-")
	if e != nil {
		t.Fatal(e)
	}
	t.Log("retained rejection controls:", controls)
	loggedReceiptControls(t, controls, "target", q.Root, q.Target, LoggedCommand{Command: r.Command, Process: r.TargetProcess, Observation: r.TargetObservation, Receipts: r.TargetReceipts})
	// Read the actual selected compiler leaf; never modify root's original
	// capture to make a negative and never repeat its producer.
	var leaf CompileRecord
	if len(r.Compiles) == 0 {
		t.Fatal("genuine compiler records missing")
	}
	if e := readExact(r.Compiles[0], &leaf); e != nil {
		t.Fatal(e)
	}
	loggedReceiptControls(t, controls, "compiler", q.Root, q.Compiler, LoggedCommand{Command: leaf.Compile.Command, Process: leaf.Compiler, Observation: leaf.Observation, Receipts: leaf.Receipts})
	for name, mutate := range map[string]func(*h08.FakeProof){
		"logger":          func(p *h08.FakeProof) { p.Logger = Pin{} },
		"compiler":        func(p *h08.FakeProof) { p.Compiler.SHA256 = strings.Repeat("0", 64) },
		"selection":       func(p *h08.FakeProof) { p.ControllerBuildSelection = p.ControllerBuild },
		"missing_compile": func(p *h08.FakeProof) { p.Compiles = p.Compiles[:len(p.Compiles)-1] },
		"extra_compile":   func(p *h08.FakeProof) { p.Compiles = append(p.Compiles, p.Compiles[0]) },
		"source_snapshot": func(p *h08.FakeProof) { p.Compiles[0].SourceBefore.Path = filepath.Join(controls, "absent.c") },
		"output_snapshot": func(p *h08.FakeProof) { p.Compiles[0].OutputSnapshot.Path = filepath.Join(controls, "absent.so") },
		"exit":            func(p *h08.FakeProof) { p.Command.Exit = 1 },
		"result":          func(p *h08.FakeProof) { p.Tests["TestNativeLoaderABIRejections"] = "skip" },
	} {
		t.Run(name, func(t *testing.T) {
			var bad h08.FakeProof
			if e := json.Unmarshal(body, &bad); e != nil {
				t.Fatal(e)
			}
			mutate(&bad)
			pin, e := writeNew(filepath.Join(controls, name+".invalid.json"), bad)
			if e != nil {
				t.Fatal(e)
			}
			if _, e = h08.ReadFakeLoaderProof(pin, q.Target, q.ControllerBuild, q.ControllerBuildSelection); e == nil {
				t.Fatal("invalid derived proof accepted")
			}
		})
	}
	for name, suffix := range map[string]string{
		"skip":      "\n--- SKIP: TestNativeLoaderABIRejections (0.00s)\n",
		"fail":      "\n--- FAIL: TestNativeLoaderABIRejections (0.00s)\n",
		"duplicate": "\n=== RUN   TestNativeLoaderABIRejections\n--- PASS: TestNativeLoaderABIRejections (0.00s)\n",
		"extra":     "\n=== RUN   UnexpectedTest\n--- PASS: UnexpectedTest (0.00s)\n",
	} {
		t.Run("raw_"+name, func(t *testing.T) {
			var bad h08.FakeProof
			if e := json.Unmarshal(body, &bad); e != nil {
				t.Fatal(e)
			}
			raw, e := os.ReadFile(bad.Command.Stdout)
			if e != nil {
				t.Fatal(e)
			}
			log, e := writeBytes(filepath.Join(controls, name+".invalid.stdout"), append(raw, []byte(suffix)...))
			if e != nil {
				t.Fatal(e)
			}
			bad.Command.Stdout, bad.Command.StdoutSHA = log.Path, log.SHA256
			pin, e := writeNew(filepath.Join(controls, "raw_"+name+".invalid.json"), bad)
			if e != nil {
				t.Fatal(e)
			}
			if _, e = h08.ReadFakeLoaderProof(pin, q.Target, q.ControllerBuild, q.ControllerBuildSelection); e == nil {
				t.Fatal("invalid raw test results accepted")
			}
		})
	}
	// These use the real accepted tuple to reach the output/compiler/role
	// boundaries. Preflight creates no attempt and launches no command.
	for _, name := range []string{"wrong_output", "wrong_compiler", "wrong_role"} {
		t.Run(name, func(t *testing.T) {
			bad := q
			bad.Out = filepath.Join(controls, name+"-must-stay-absent")
			switch name {
			case "wrong_output":
				bad.Target = inert(t, filepath.Join(controls, "unselected-target"), "INERT UNSELECTED TARGET")
			case "wrong_compiler":
				bad.Compiler = inert(t, filepath.Join(controls, "unselected-compiler"), "INERT UNSELECTED COMPILER")
			case "wrong_role":
				b, e := pinBytes(q.ControllerBuildSelection)
				if e != nil {
					t.Fatal(e)
				}
				var selection map[string]json.RawMessage
				if e = json.Unmarshal(b, &selection); e != nil {
					t.Fatal(e)
				}
				selection["Role"] = json.RawMessage(`"h08-command"`)
				bad.ControllerBuildSelection, e = writeNew(filepath.Join(controls, "wrong-role.invalid.json"), selection)
				if e != nil {
					t.Fatal(e)
				}
			}
			pin, e := writeNew(filepath.Join(controls, name+".invalid-request.json"), bad)
			if e != nil {
				t.Fatal(e)
			}
			if _, e = Preflight(pin); e == nil {
				t.Fatal("unselected role/output/compiler accepted")
			}
			if _, e = os.Lstat(bad.Out); !os.IsNotExist(e) {
				t.Fatal("boundary created an execution attempt")
			}
		})
	}
}
