package fakecapture

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

// These are inert metadata records, including invented process identities.
// They test receipt readback only and are never native/process evidence.
func metadataLogged(t *testing.T, role string) (string, Pin, LoggedCommand) {
	t.Helper()
	cwd := t.TempDir()
	executable := inert(t, filepath.Join(cwd, role), "INERT "+role)
	p := Process{PID: 12, PPID: 11, Start: "12345", Cgroup: "0::/inert\n"}
	r := LoggedCommand{Process: p, Observation: ProcessObservation{Born: ChildBirth{PID: p.PID, ObservedUTC: "2026-10-29T00:00:01Z"}, Process: p, Authenticated: true}}
	r.Command = runner.CommandReceipt{Args: []string{executable.Path, "inert-argument"}, Environment: []string{"LANG=C"}, Started: "2026-10-29T00:00:00Z", Finished: "2026-10-29T00:00:02Z", Exit: 0}
	return cwd, executable, retainLoggedMetadata(t, t.TempDir(), cwd, executable, r, []byte("inert stdout\n"), nil)
}

// Derived control files stay outside the original capture. Rebased raw-log
// paths are control data, not newly observed execution receipts.
func retainLoggedMetadata(t *testing.T, dir, cwd string, executable Pin, r LoggedCommand, stdout, stderr []byte) LoggedCommand {
	t.Helper()
	if e := os.MkdirAll(dir, 0700); e != nil {
		t.Fatal(e)
	}
	out, e := writeBytes(filepath.Join(dir, "stdout"), stdout)
	if e != nil {
		t.Fatal(e)
	}
	errout, e := writeBytes(filepath.Join(dir, "stderr"), stderr)
	if e != nil {
		t.Fatal(e)
	}
	r.Command.Stdout, r.Command.StdoutSHA = out.Path, out.SHA256
	r.Command.Stderr, r.Command.StderrSHA = errout.Path, errout.SHA256
	for _, item := range []struct {
		name  string
		value any
		pin   *Pin
	}{
		{"launch", commandLaunch{cwd, executable, launchCommand(r.Command)}, &r.Receipts.Launch},
		{"born", r.Observation.Born, &r.Receipts.Born},
		{"process", r.Observation, &r.Receipts.Process},
		{"command", r.Command, &r.Receipts.Command},
	} {
		*item.pin, e = writeNew(filepath.Join(dir, item.name+".json"), item.value)
		if e != nil {
			t.Fatal(e)
		}
	}
	return r
}

// Used for both inert metadata and genuine selected target/compiler records.
// It exercises the same reader called by publication, collect and ReadAccepted.
func loggedReceiptControls(t *testing.T, controls, role, cwd string, executable Pin, original LoggedCommand) {
	t.Helper()
	originalDir := filepath.Dir(original.Receipts.Command.Path)
	if e := readLogged(originalDir, cwd, executable, original); e != nil {
		t.Fatal("positive receipt baseline:", e)
	}
	stdout, e := pinBytes(Pin{Path: original.Command.Stdout, SHA256: original.Command.StdoutSHA})
	if e != nil {
		t.Fatal(e)
	}
	stderr, e := pinBytes(Pin{Path: original.Command.Stderr, SHA256: original.Command.StderrSHA})
	if e != nil {
		t.Fatal(e)
	}
	for _, kind := range []string{"launch", "born", "process", "command"} {
		for _, mode := range []string{"missing", "changed", "rehashed"} {
			t.Run(role+"_"+kind+"_"+mode, func(t *testing.T) {
				dir := filepath.Join(controls, role+"-"+kind+"-"+mode)
				r := retainLoggedMetadata(t, dir, cwd, executable, original, stdout, stderr)
				if e := readLogged(dir, cwd, executable, r); e != nil {
					t.Fatal("derived positive baseline:", e)
				}
				pin := map[string]*Pin{"launch": &r.Receipts.Launch, "born": &r.Receipts.Born, "process": &r.Receipts.Process, "command": &r.Receipts.Command}[kind]
				var value any
				switch kind {
				case "launch":
					var v commandLaunch
					if e := readExact(*pin, &v); e != nil {
						t.Fatal(e)
					}
					v.Cwd += "-inconsistent"
					value = v
				case "born":
					var v ChildBirth
					if e := readExact(*pin, &v); e != nil {
						t.Fatal(e)
					}
					v.PID++
					value = v
				case "process":
					var v ProcessObservation
					if e := readExact(*pin, &v); e != nil {
						t.Fatal(e)
					}
					v.Process.Start += "-inconsistent"
					value = v
				case "command":
					var v runner.CommandReceipt
					if e := readExact(*pin, &v); e != nil {
						t.Fatal(e)
					}
					v.StdoutSHA = strings.Repeat("0", 64)
					value = v
				}
				// Preserve even the removed/replaced derived bytes. Root's real
				// capture and independently selected pins are never modified.
				if e := os.Rename(pin.Path, filepath.Join(dir, "preserved-"+kind+".json")); e != nil {
					t.Fatal(e)
				}
				if mode != "missing" {
					changed, e := writeNew(pin.Path, value)
					if e != nil {
						t.Fatal(e)
					}
					if mode == "rehashed" {
						*pin = changed
					}
				}
				if _, e := writeNew(filepath.Join(dir, "invalid-expected-record.json"), r); e != nil {
					t.Fatal(e)
				}
				if e := readLogged(dir, cwd, executable, r); e == nil {
					t.Fatal("missing/changed/inconsistent receipt accepted")
				}
			})
		}
	}
}

func TestLoggedReceiptMetadata(t *testing.T) {
	for _, role := range []string{"target", "compiler"} {
		cwd, executable, r := metadataLogged(t, role)
		loggedReceiptControls(t, t.TempDir(), role, cwd, executable, r)
	}
}

func TestLoggedReceiptEmbeddedAndRawBindings(t *testing.T) {
	for name, mutate := range map[string]func(*LoggedCommand){
		"argv":              func(r *LoggedCommand) { r.Command.Args = append(r.Command.Args, "extra") },
		"environment":       func(r *LoggedCommand) { r.Command.Environment = []string{"LANG=other"} },
		"process":           func(r *LoggedCommand) { r.Process.PPID++ },
		"birth":             func(r *LoggedCommand) { r.Observation.Born.PID++ },
		"unauthenticated":   func(r *LoggedCommand) { r.Observation.Authenticated = false },
		"observation_error": func(r *LoggedCommand) { r.Observation.Error = "inconsistent" },
	} {
		t.Run(name, func(t *testing.T) {
			cwd, executable, r := metadataLogged(t, "target")
			mutate(&r)
			if e := readLogged(filepath.Dir(r.Receipts.Command.Path), cwd, executable, r); e == nil {
				t.Fatal("inconsistent embedded record accepted")
			}
		})
	}
	for _, stream := range []string{"stdout", "stderr"} {
		t.Run(stream, func(t *testing.T) {
			cwd, executable, r := metadataLogged(t, "compiler")
			dir := filepath.Dir(r.Receipts.Command.Path)
			path := filepath.Join(dir, stream)
			if e := os.Rename(path, path+".preserved"); e != nil {
				t.Fatal(e)
			}
			if _, e := writeBytes(path, []byte("changed raw log")); e != nil {
				t.Fatal(e)
			}
			if e := readLogged(dir, cwd, executable, r); e == nil {
				t.Fatal("changed raw log accepted")
			}
		})
	}
	// Strict receipt decoding must reject unknown fields even with a new hash.
	t.Run("typed_shape", func(t *testing.T) {
		cwd, executable, r := metadataLogged(t, "target")
		dir := filepath.Dir(r.Receipts.Command.Path)
		b, e := pinBytes(r.Receipts.Launch)
		if e != nil {
			t.Fatal(e)
		}
		var fields map[string]json.RawMessage
		if e := json.Unmarshal(b, &fields); e != nil {
			t.Fatal(e)
		}
		fields["extra"] = json.RawMessage(`true`)
		if e := os.Rename(r.Receipts.Launch.Path, filepath.Join(dir, "preserved-launch.json")); e != nil {
			t.Fatal(e)
		}
		r.Receipts.Launch, e = writeNew(r.Receipts.Launch.Path, fields)
		if e != nil {
			t.Fatal(e)
		}
		if e := readLogged(dir, cwd, executable, r); e == nil {
			t.Fatal("untyped receipt accepted")
		}
	})
}

func TestMatchingUnqualifiedReceiptsReject(t *testing.T) {
	for name, mutate := range map[string]func(*LoggedCommand){
		"observation_failed":        func(r *LoggedCommand) { r.Observation.Authenticated, r.Observation.Error = false, "identity failed" },
		"error_with_authentication": func(r *LoggedCommand) { r.Observation.Error = "identity failed" },
		"no_birth":                  func(r *LoggedCommand) { r.Observation.Born = ChildBirth{} },
		"no_start_tick":             func(r *LoggedCommand) { r.Process.Start = "" },
		"no_cgroup":                 func(r *LoggedCommand) { r.Process.Cgroup = "" },
		"exit":                      func(r *LoggedCommand) { r.Command.Exit = 1 },
		"invalid_clock":             func(r *LoggedCommand) { r.Command.Started = "invalid" },
		"birth_before_start":        func(r *LoggedCommand) { r.Observation.Born.ObservedUTC = "2026-10-28T00:00:00Z" },
		"birth_after_finish":        func(r *LoggedCommand) { r.Observation.Born.ObservedUTC = "2026-10-30T00:00:00Z" },
	} {
		t.Run(name, func(t *testing.T) {
			cwd, executable, r := metadataLogged(t, "target")
			mutate(&r)
			r.Observation.Process = r.Process
			dir := t.TempDir()
			// Rehash all derived files to match the unqualified embedded
			// fields. Rejection must enforce qualification, not stale hashes.
			r = retainLoggedMetadata(t, dir, cwd, executable, r, nil, nil)
			if e := readLogged(dir, cwd, executable, r); e == nil {
				t.Fatal("matching but unqualified receipts accepted")
			}
		})
	}
}
