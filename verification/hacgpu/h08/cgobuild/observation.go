package cgobuild

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

const requestSchema = "nanopore-h08-observation-request/v1"
const observationSchema = "nanopore-h08-compiled-observation/v1"
const observedSelectionSchema = "nanopore-h08-observed-selection/v1"
const successfulSelectionSchema = "nanopore-h08-successful-selection/v1"
const executionSchema = "nanopore-h08-observer-execution/v1"
const cleanupSchema = "nanopore-h08-observer-cleanup/v1"

// Artifact exposes only the output of a completely authenticated selected proof.
// It conveys no runtime admission. Its zero value has no selected output.
type Artifact struct{ output Pin }

func (a Artifact) Output() Pin { return a.output }

type observationRequestWire struct {
	Schema              string
	Role                BuildRole
	Retained            retainedSelection
	ObserverController  Pin
	ObserverRoot        string
	AlgorithmReferences [2]Pin
	OutputDirectory     string
}
type observationProofWire struct {
	Aliases        map[string]Pin
	Operations     []observedOperationWire
	ChecksumOffset uint64
	Checksum       [32]byte
	Ranges         [4][2]uint64
}

type observedOperationWire struct {
	Operation              Pin
	Role, Status           string
	Before, OutputSnapshot Pin
	Bindings               map[string]string
}
type compiledObservationWire struct {
	Schema                 string
	Request                Pin
	RuntimeHeld            bool
	NativePreimage, Output Pin
	Proofs                 observationProofWire
}
type observedSelectionWire struct {
	Schema             string
	Request, Execution Pin
}
type successfulSelectionWire struct {
	Schema   string
	Role     BuildRole
	Producer retainedProducerPins
}
type observerExecutionWire struct {
	Schema                                               string
	Request, Observation, ObserverController, Executable Pin
	Process                                              Process
	Cwd                                                  string
	Command                                              runner.CommandReceipt
	Cleanup                                              Pin
}
type observerCleanupWire struct {
	Schema, UTC, Unit                    string
	Process                              Process
	PIDAbsent, GroupAbsent, CgroupAbsent bool
	RawCommand                           runner.CommandReceipt
}

func exactRead(pin Pin, value any) error {
	if !filepath.IsAbs(pin.Path) {
		return fmt.Errorf("artifact pin must be absolute")
	}
	body, e := os.ReadFile(pin.Path)
	if e != nil {
		return e
	}
	if fmt.Sprintf("%x", sha256.Sum256(body)) != pin.SHA256 {
		return fmt.Errorf("artifact metadata pin differs")
	}
	d := json.NewDecoder(bytes.NewReader(body))
	d.DisallowUnknownFields()
	if e = d.Decode(value); e != nil {
		return e
	}
	if e = d.Decode(new(any)); e != io.EOF {
		return fmt.Errorf("artifact trailing JSON")
	}
	canonical, e := json.Marshal(value)
	if e != nil {
		return e
	}
	var shape any
	if e = json.Unmarshal(canonical, &shape); e != nil {
		return e
	}
	d = json.NewDecoder(bytes.NewReader(body))
	return exactWireShape(d, shape)
}

func exactWireShape(d *json.Decoder, shape any) error {
	t, e := d.Token()
	if e != nil {
		return e
	}
	if t == nil {
		return fmt.Errorf("artifact null field")
	}
	switch want := shape.(type) {
	case map[string]any:
		if t != json.Delim('{') {
			return fmt.Errorf("artifact object required")
		}
		seen := map[string]bool{}
		for d.More() {
			key, e := d.Token()
			if e != nil {
				return e
			}
			k, ok := key.(string)
			if !ok {
				return fmt.Errorf("artifact field name")
			}
			v, ok := want[k]
			if !ok || seen[k] {
				return fmt.Errorf("artifact unknown/duplicate field: %s", k)
			}
			seen[k] = true
			if e = exactWireShape(d, v); e != nil {
				return e
			}
		}
		if len(seen) != len(want) {
			return fmt.Errorf("artifact missing field")
		}
		_, e = d.Token()
		return e
	case []any:
		if t != json.Delim('[') {
			return fmt.Errorf("artifact array required")
		}
		i := 0
		for d.More() {
			if i == len(want) {
				return fmt.Errorf("artifact array length")
			}
			if e = exactWireShape(d, want[i]); e != nil {
				return e
			}
			i++
		}
		if i != len(want) {
			return fmt.Errorf("artifact array length")
		}
		_, e = d.Token()
		return e
	default:
		if _, compound := t.(json.Delim); compound {
			return fmt.Errorf("artifact scalar required")
		}
		return nil
	}
}

func readObservationRequest(pin Pin) (q observationRequestWire, err error) {
	if err = exactRead(pin, &q); err != nil {
		return q, err
	}
	if q.Schema != requestSchema || q.Role != H08Command || q.Retained.Profile != retainedProfile {
		return q, fmt.Errorf("observation request schema/role/profile differs")
	}
	if !filepath.IsAbs(q.ObserverRoot) || filepath.Clean(q.ObserverRoot) != q.ObserverRoot || !filepath.IsAbs(q.OutputDirectory) || filepath.Clean(q.OutputDirectory) != q.OutputDirectory || !under(filepath.Join(q.ObserverRoot, "artifacts", "tensor"), q.OutputDirectory) {
		return q, fmt.Errorf("observation root/output directory differs")
	}
	if err = observationAlgorithms(q.AlgorithmReferences); err != nil {
		return q, err
	}
	c, e := controllerRead(q.ObserverController)
	if e != nil {
		return q, e
	}
	for _, name := range []string{"observation.go", "cmd/main.go"} {
		if c.Inputs[filepath.Join(q.ObserverRoot, "verification/hacgpu/h08/cgobuild", name)] == "" {
			return q, fmt.Errorf("observer selected root/source differs")
		}
	}
	return q, nil
}

func observationAlgorithms(refs [2]Pin) error {
	want := [2]Pin{
		{Path: "/usr/lib/go/src/cmd/link/internal/ld/fips140.go", SHA256: "f214648c1a2ec04ea5483812ceec13cec96aa1a1acd9f95d2882b6373ef432c5"},
		{Path: "/usr/lib/go/src/cmd/internal/obj/fips140.go", SHA256: "25fba80147d77aa812720b5c4ec71304b1547491ff59f0221c12a3c504236137"},
	}
	if refs != want {
		return fmt.Errorf("observation algorithm references differ")
	}
	for _, p := range refs {
		if e := verifyPin(p); e != nil {
			return e
		}
	}
	return nil
}

func deriveObservation(request Pin, q observationRequestWire) (compiledObservationWire, error) {
	o := compiledObservationWire{Schema: observationSchema, Request: request, RuntimeHeld: true}
	r, e := readRetainedCompiled(q.Retained)
	if e != nil {
		return o, e
	}
	var p Plan
	if e = read(q.Retained.Producer.Plan, &p); e != nil {
		return o, e
	}
	resources, e := deriveQueryResources(p, r)
	if e != nil {
		return o, e
	}
	ops := map[string]operation{}
	o.Proofs.Aliases = resources.exactAnswerBindings
	o.Proofs.Operations = []observedOperationWire{}
	for _, pin := range r.Operations {
		var op operation
		if e = read(pin, &op); e != nil {
			return o, e
		}
		ops[filepath.Dir(pin.Path)] = op
		if op.Bindings == nil {
			op.Bindings = map[string]string{}
		}
		o.Proofs.Operations = append(o.Proofs.Operations, observedOperationWire{pin, op.Role, op.Status, op.Before, op.OutputSnapshot, op.Bindings})
		if op.Role == "go-link" {
			o.NativePreimage = op.OutputSnapshot
		}
	}
	f, e := finalOutputTransition(p, r, ops)
	if e != nil {
		return o, e
	}
	o.Output = r.Output
	o.Proofs.ChecksumOffset, o.Proofs.Checksum = f.checksumOffset, f.checksum
	for i, v := range f.ranges {
		o.Proofs.Ranges[i] = [2]uint64{v.start, v.end}
	}
	return o, nil
}

// ObserveCompiled writes only a fresh compiled observation. The outer controller
// owns host lock/deadline and records completion independently after process exit.
func ObserveCompiled(requestPin Pin, out string) (Pin, error) {
	q, e := readObservationRequest(requestPin)
	if e != nil {
		return Pin{}, e
	}
	cwd, e := os.Getwd()
	if e != nil {
		return Pin{}, e
	}
	if cwd != q.ObserverRoot || out != q.OutputDirectory {
		return Pin{}, fmt.Errorf("actual observer cwd/output differs")
	}
	if e = CheckController(q.ObserverController); e != nil {
		return Pin{}, e
	}
	if _, e = runner.NewAttempt(out); e != nil {
		return Pin{}, e
	}
	o, e := deriveObservation(requestPin, q)
	if e != nil {
		return Pin{}, e
	}
	path := filepath.Join(out, "observation.json")
	body, e := json.MarshalIndent(o, "", "  ")
	if e != nil {
		return Pin{}, e
	}
	f, e := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0400)
	if e != nil {
		return Pin{}, e
	}
	_, e = f.Write(append(body, '\n'))
	ce := f.Close()
	if e != nil {
		return Pin{}, e
	}
	if ce != nil {
		return Pin{}, ce
	}
	return closedPin(path)
}

func ReadH08Artifact(evidence, selection Pin) (Artifact, error) {
	return readArtifact(evidence, selection, H08Command)
}
func ReadCPU13Artifact(evidence, selection Pin) (Artifact, error) {
	return readArtifact(evidence, selection, CPU13Tests)
}

func readArtifact(evidence, selection Pin, role BuildRole) (Artifact, error) {
	if e := verifyPin(selection); e != nil {
		return Artifact{}, e
	}
	var header struct{ Schema string }
	if e := read(selection, &header); e != nil {
		return Artifact{}, e
	}
	switch header.Schema {
	case successfulSelectionSchema:
		var s successfulSelectionWire
		if e := exactRead(selection, &s); e != nil {
			return Artifact{}, e
		}
		if s.Role != role || s.Producer.Receipt != evidence {
			return Artifact{}, fmt.Errorf("successful selection role/receipt differs")
		}
		var r Receipt
		if e := read(evidence, &r); e != nil {
			return Artifact{}, e
		}
		if s.Producer != (retainedProducerPins{evidence, r.Plan, r.Controller, r.Queries, r.Output}) {
			return Artifact{}, fmt.Errorf("successful selection tuple differs")
		}
		r, e := readBuildRole(evidence, role)
		if e != nil {
			return Artifact{}, e
		}
		return Artifact{r.Output}, nil
	case observedSelectionSchema:
		var s observedSelectionWire
		if e := exactRead(selection, &s); e != nil {
			return Artifact{}, e
		}
		var o compiledObservationWire
		if e := exactRead(evidence, &o); e != nil {
			return Artifact{}, e
		}
		if o.Schema != observationSchema || !o.RuntimeHeld || o.Request != s.Request {
			return Artifact{}, fmt.Errorf("observed selection request/schema/hold differs")
		}
		var x observerExecutionWire
		if e := exactRead(s.Execution, &x); e != nil {
			return Artifact{}, e
		}
		if e := checkObserverExecution(x, evidence, s.Request); e != nil {
			return Artifact{}, e
		}
		var selectedRequest observationRequestWire
		if e := exactRead(s.Request, &selectedRequest); e != nil {
			return Artifact{}, e
		}
		if e := selectedObservation(selectedRequest, x, evidence, role); e != nil {
			return Artifact{}, e
		}
		q, e := readObservationRequest(s.Request)
		if e != nil {
			return Artifact{}, e
		}
		recordedObserver, e := controllerRead(q.ObserverController)
		if e != nil {
			return Artifact{}, e
		}
		if e := checkObserverInvocation(x, recordedObserver.Executable, s.Request, q.OutputDirectory); e != nil {
			return Artifact{}, e
		}
		derived, e := deriveObservation(s.Request, q)
		if e != nil {
			return Artifact{}, e
		}
		if !reflect.DeepEqual(o, derived) {
			return Artifact{}, fmt.Errorf("observation serialized proofs/output differ")
		}
		return Artifact{o.Output}, nil
	default:
		return Artifact{}, fmt.Errorf("unknown artifact selection schema")
	}
}

func selectedObservation(q observationRequestWire, x observerExecutionWire, evidence Pin, role BuildRole) error {
	if q.Role != role || x.ObserverController != q.ObserverController || x.Cwd != q.ObserverRoot || evidence.Path != filepath.Join(q.OutputDirectory, "observation.json") {
		return fmt.Errorf("selected observation role/observer/output differs")
	}
	return nil
}

func observerArgs(exe, request Pin, out string) []string {
	return []string{exe.Path, "-mode", "observe-compiled", "-observation-request", request.Path, "-observation-request-sha", request.SHA256, "-out", out, "-host-admitted"}
}
func observerEnvironment() []string {
	return []string{"CGO_ENABLED=0", "GOTOOLCHAIN=local", "GOWORK=off", "GOPROXY=off", "GOSUMDB=off", "GOENV=off", "GOMAXPROCS=2", "GOMEMLIMIT=3GiB"}
}

func checkObserverInvocation(x observerExecutionWire, executable, request Pin, out string) error {
	if x.Executable != executable || !slices.Equal(x.Command.Args, observerArgs(executable, request, out)) || !slices.Equal(x.Command.Environment, observerEnvironment()) {
		return fmt.Errorf("observer executable/argv/environment differs")
	}
	return nil
}

func checkObserverExecution(x observerExecutionWire, observation, request Pin) error {
	startTicks, startErr := strconv.ParseUint(x.Process.Start, 10, 64)
	cgroup := strings.TrimSuffix(x.Process.Cgroup, "\n")
	if x.Schema != executionSchema || x.Request != request || x.Observation != observation || x.Command.Exit != 0 || !filepath.IsAbs(x.Cwd) || x.Process.PID <= 0 || startErr != nil || startTicks == 0 || !strings.HasPrefix(cgroup, "0::/") || strings.ContainsAny(cgroup, "\n\r") || !x.Process.Group {
		return fmt.Errorf("observer execution identity/exit differs")
	}
	if !filepath.IsAbs(x.Command.Stdout) || !filepath.IsAbs(x.Command.Stderr) {
		return fmt.Errorf("observer raw logs must be absolute")
	}
	start, finish, e := retainedClock(x.Command.Started, x.Command.Finished)
	if e != nil || finish.Sub(start).Seconds() > 600 {
		return fmt.Errorf("observer execution clocks differ")
	}
	if e = checkCommand(x.Command.Args, []int{0}, x.Command.Started, x.Command.Finished, x.Command.Stdout, x.Command.StdoutSHA, x.Command.Stderr, x.Command.StderrSHA, x.Command.Exit); e != nil {
		return e
	}
	var emitted Pin
	if e = exactRead(Pin{Path: x.Command.Stdout, SHA256: x.Command.StdoutSHA}, &emitted); e != nil || emitted != observation {
		return fmt.Errorf("observer raw output pin differs")
	}
	var c observerCleanupWire
	if e = exactRead(x.Cleanup, &c); e != nil {
		return e
	}
	return checkObserverCleanup(c, x.Process, x.Command.Finished)
}

func checkObserverCleanup(c observerCleanupWire, process Process, finished string) error {
	if c.Schema != cleanupSchema || c.Process != process || !c.PIDAbsent || !c.GroupAbsent || !c.CgroupAbsent || !strings.HasSuffix(c.Unit, ".service") || strings.ContainsAny(c.Unit, "/\n\r\t ") || !strings.HasSuffix(strings.TrimSuffix(process.Cgroup, "\n"), "/"+c.Unit) {
		return fmt.Errorf("observer cleanup identity/absence differs")
	}
	_, end, e := retainedClock(finished, c.UTC)
	if e != nil {
		return fmt.Errorf("observer cleanup clock differs")
	}
	raw := c.RawCommand
	if !filepath.IsAbs(raw.Stdout) || !filepath.IsAbs(raw.Stderr) {
		return fmt.Errorf("observer cleanup logs must be absolute")
	}
	start, finish, e := retainedClock(raw.Started, raw.Finished)
	exitTime, parseErr := time.Parse(time.RFC3339Nano, finished)
	if e != nil || parseErr != nil || start.Before(exitTime) || finish.After(end) {
		return fmt.Errorf("observer cleanup command clock differs")
	}
	args := []string{"systemctl", "--user", "show", c.Unit, "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "ControlGroup"}
	if raw.Exit != 0 || !slices.Equal(raw.Args, args) || len(raw.Environment) != 0 {
		return fmt.Errorf("observer cleanup raw command differs")
	}
	if e = checkCommand(args, []int{0}, raw.Started, raw.Finished, raw.Stdout, raw.StdoutSHA, raw.Stderr, raw.StderrSHA, raw.Exit); e != nil {
		return e
	}
	body, e := os.ReadFile(raw.Stdout)
	if e != nil {
		return e
	}
	properties := map[string]string{}
	if !strings.HasSuffix(string(body), "\n") {
		return fmt.Errorf("observer cleanup raw properties differ")
	}
	for _, line := range strings.Split(strings.TrimSuffix(string(body), "\n"), "\n") {
		key, value, ok := strings.Cut(line, "=")
		if _, duplicate := properties[key]; !ok || duplicate {
			return fmt.Errorf("observer cleanup raw properties differ")
		}
		properties[key] = value
	}
	if !reflect.DeepEqual(properties, map[string]string{"ActiveState": "inactive", "SubState": "dead", "MainPID": "0", "ControlGroup": ""}) {
		return fmt.Errorf("observer cleanup raw properties differ")
	}
	return nil
}
