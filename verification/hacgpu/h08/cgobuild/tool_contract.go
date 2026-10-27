package cgobuild

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type goSelection struct {
	Schema, Target, DefaultGODEBUG string
	Role                           BuildRole
	Producer                       Pin
}

type compilerIdentityWrapper struct {
	Schema    string
	Driver    Pin
	Args      []string
	Overrides map[string]string
}

func compilerStdin(args []string, input io.Reader) ([]byte, error) {
	if !slices.Contains(args, "-") {
		return nil, nil
	}
	body, e := io.ReadAll(io.LimitReader(input, (1<<20)+1))
	if e != nil || len(body) > 1<<20 {
		return nil, fmt.Errorf("invalid compiler stdin: %v", e)
	}
	return body, nil
}

func validateCompilerIdentity(p Plan, s Session, w compilerIdentityWrapper) error {
	want := append([]string{s.Driver.Path}, "-###", "-x", "c", "-c", "-")
	if s.Phase != "build" || w.Schema != "nanopore-h08-wrapped-compiler-id/v1" || w.Driver != s.Driver || !slices.Equal(w.Args, want) || !maps.Equal(w.Overrides, sourceToolOverrides(p, "cc-id", true, nil)) {
		return fmt.Errorf("unadmitted nested compiler identity request")
	}
	return verifyPin(w.Driver)
}

func selectionTarget(p Plan) string {
	if p.Role == CPU13Tests {
		return TensorPackage + ".test"
	}
	return Module + "/verification/hacgpu/h08/cmd"
}

func selectionDefault(p Plan, body []byte) (string, error) {
	d := json.NewDecoder(bytes.NewReader(body))
	seen, value := false, ""
	for {
		var q struct{ ImportPath, DefaultGODEBUG string }
		if e := d.Decode(&q); e == io.EOF {
			break
		} else if e != nil {
			return "", e
		}
		if q.ImportPath == selectionTarget(p) {
			if seen {
				return "", fmt.Errorf("duplicate fixed link selection")
			}
			seen, value = true, q.DefaultGODEBUG
		}
	}
	if !seen {
		return "", fmt.Errorf("fixed link selection omitted")
	}
	return value, nil
}

func captureGoSelection(p Plan, s Session, sp Pin, command runner.CommandReceipt) (Pin, error) {
	producer, e := closedPin(filepath.Join(filepath.Dir(command.Stdout), "command.json"))
	if e != nil {
		return Pin{}, e
	}
	body, e := os.ReadFile(command.Stdout)
	if e != nil {
		return Pin{}, e
	}
	value, e := selectionDefault(p, body)
	if e != nil {
		return Pin{}, e
	}
	path := filepath.Join(s.Directory, "go-selection.json")
	if e = write(path, goSelection{"nanopore-h08-go-selection/v1", selectionTarget(p), value, p.Role, producer}); e != nil {
		return Pin{}, e
	}
	q, e := closedPin(path)
	if e == nil {
		_, _, e = readGoSelection(p, s, sp, q)
	}
	return q, e
}

func readGoSelection(p Plan, s Session, sp, q Pin) (goSelection, map[string]string, error) {
	var g goSelection
	inputs := map[string]string{}
	if q.Path == "" {
		var e error
		q, e = closedPin(filepath.Join(s.Directory, "go-selection.json"))
		if e != nil {
			return g, nil, e
		}
	}
	if q.Path != filepath.Join(s.Directory, "go-selection.json") || s.Phase != "build" {
		return g, nil, fmt.Errorf("Go selection outside fixed build domain")
	}
	if e := read(q, &g); e != nil {
		return g, nil, e
	}
	if g.Schema != "nanopore-h08-go-selection/v1" || g.Role != p.Role || g.Target != selectionTarget(p) || filepath.Base(g.Producer.Path) != "command.json" || filepath.Dir(filepath.Dir(g.Producer.Path)) != filepath.Join(s.Directory, "commands") || !strings.HasPrefix(filepath.Base(filepath.Dir(g.Producer.Path)), "go-list-") {
		return g, nil, fmt.Errorf("Go selection producer/role differs")
	}
	var c runner.CommandReceipt
	if e := read(g.Producer, &c); e != nil {
		return g, nil, e
	}
	if filepath.Dir(c.Stdout) != filepath.Dir(g.Producer.Path) || filepath.Base(c.Stdout) != "stdout" || filepath.Dir(c.Stderr) != filepath.Dir(g.Producer.Path) || filepath.Base(c.Stderr) != "stderr" || !slices.Equal(c.Args, listArgs(p)) || !slices.Equal(c.Environment, phaseEnvironment(p, s, sp)) {
		return g, nil, fmt.Errorf("Go selection actual command differs")
	}
	if e := checkCommand(c.Args, []int{0}, c.Started, c.Finished, c.Stdout, c.StdoutSHA, c.Stderr, c.StderrSHA, c.Exit); e != nil {
		return g, nil, e
	}
	body, e := os.ReadFile(c.Stdout)
	if e != nil {
		return g, nil, e
	}
	if e = verifyActualSelection(p, body); e != nil {
		return g, nil, e
	}
	value, e := selectionDefault(p, body)
	if e != nil || value != g.DefaultGODEBUG {
		return g, nil, fmt.Errorf("Go selection default differs from actual producer")
	}
	for _, input := range []Pin{q, g.Producer, {Path: c.Stdout, SHA256: c.StdoutSHA}, {Path: c.Stderr, SHA256: c.StderrSHA}} {
		inputs[input.Path] = input.SHA256
	}
	return g, inputs, nil
}

func selectionProducerMatches(g goSelection, command runner.CommandReceipt) error {
	var actual runner.CommandReceipt
	if e := read(g.Producer, &actual); e != nil || !reflect.DeepEqual(actual, command) {
		return fmt.Errorf("Go selection does not bind the phase list command")
	}
	return nil
}

func sourceToolOverrides(p Plan, name string, metadata bool, lease *Lease) map[string]string {
	want := map[string]string{"GOROOT": p.Environment["GOROOT"], "CGO_LDFLAGS": p.Environment["CGO_LDFLAGS"]}
	if !metadata && name == "cgo" && lease != nil && lease.Kind == "cgo" {
		want["CGO_LDFLAGS"] = ""
	}
	if !metadata && name == "link" {
		want["GOROOT"] = ""
	}
	return want
}

func toolEnvironment(p Plan, s Session, sp Pin, name string, metadata bool, lease *Lease, incoming map[string]string) ([]string, error) {
	want := sourceToolOverrides(p, name, metadata, lease)
	if !maps.Equal(incoming, want) {
		return nil, fmt.Errorf("Go tool source-derived environment differs")
	}
	env := phaseEnvironment(p, s, sp)
	for i, entry := range env {
		key, _, _ := strings.Cut(entry, "=")
		if value, ok := want[key]; ok {
			env[i] = key + "=" + value
		}
	}
	if !metadata && name == "cgo" {
		env = append(env, "TERM=dumb")
	}
	return env, nil
}

func validateLinkAssignments(p Plan, name string, args []string, value string) error {
	defaults, testing := 0, 0
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "-X" {
			if name != "link" || p.Role != CPU13Tests || i+1 == len(args) || args[i+1] != "testing.testBinary=1" {
				return fmt.Errorf("unadmitted fixed Go linker assignment")
			}
			i++
			testing++
		} else if strings.HasPrefix(a, "-X=") {
			if name != "link" || value == "" || a != "-X=runtime.godebugDefault="+value {
				return fmt.Errorf("Go runtime default assignment differs from producer")
			}
			defaults++
		}
	}
	if defaults > 1 || testing > 1 || name == "link" && (defaults == 1) != (value != "") || name == "link" && (testing == 1) != (p.Role == CPU13Tests) {
		return fmt.Errorf("fixed Go linker assignments omitted or duplicated")
	}
	return nil
}
