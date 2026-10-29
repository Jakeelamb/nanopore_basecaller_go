package supgpu

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type auditEffectiveRun struct {
	Schema     string             `json:"schema"`
	Status     string             `json:"status"`
	Controller Pin                `json:"current_controller"`
	Binary     Pin                `json:"current_controller_binary"`
	Producer   Pin                `json:"historical_native_producer"`
	Build      Pin                `json:"native_build"`
	Runtime    Pin                `json:"native_runtime"`
	Audit      Pin                `json:"auditor"`
	Proof      Pin                `json:"auditor_producer_proof"`
	Sources    []Pin              `json:"retained_native_sources"`
	Order      []auditCaseID      `json:"ordered_cases"`
	Cases      []auditControlCase `json:"executed_cases"`
	Started    string             `json:"started_utc"`
	Finished   string             `json:"finished_utc"`
	Error      string             `json:"error"`
}

// Select only the named source from an authenticated controller closure. Paths
// may differ between worktrees; content and byte count must agree.
func auditNativeSnapshot(c GoController, path string) (Pin, error) {
	var selected Pin
	count := 0
	for _, s := range c.Snapshots {
		if strings.HasSuffix(s.Original, "/"+path) {
			selected = s.Copy
			count++
		}
	}
	if count != 1 {
		return Pin{}, fmt.Errorf("native source snapshot membership differs: %s", path)
	}
	if err := verifyPin("", selected); err != nil {
		return Pin{}, err
	}
	return selected, nil
}

func verifyEffectiveAuditSources(current, producer GoController, p auditControlPlan, proof auditProducerProof) ([]Pin, error) {
	if len(p.Sources) != len(auditControlSources) || p.Audit != proof.Audit || p.Controller != proof.Controller {
		return nil, fmt.Errorf("retained native source/proof binding differs")
	}
	paths := append(append([]string{}, auditControlSources...), "verification/supgpu/native/audit.c")
	retained := append(append([]Pin{}, p.Sources...), proof.Source)
	for i, path := range paths {
		if err := verifyPin("", retained[i]); err != nil {
			return nil, err
		}
		a, err := auditNativeSnapshot(current, path)
		if err != nil {
			return nil, err
		}
		b, err := auditNativeSnapshot(producer, path)
		if err != nil {
			return nil, err
		}
		if a.SHA256 != b.SHA256 || a.Bytes != b.Bytes || a.SHA256 != retained[i].SHA256 || a.Bytes != retained[i].Bytes {
			return nil, fmt.Errorf("current native source differs from retained producer: %s", path)
		}
	}
	return retained, nil
}

// This reader deliberately authenticates the executing consumer first and the
// original compiled producer separately. Neither ancestry validator is relaxed.
func readEffectiveAuditInputs(consumer, build Pin) (c GoController, b auditControlBuild, p auditControlPlan, r RuntimePlan, sources []Pin, err error) {
	if err = verifyPin("", consumer); err != nil {
		return
	}
	c, err = readAuditController(consumer, true)
	if err != nil {
		return
	}
	if err = verifyPin("", build); err != nil {
		return
	}
	b, p, r, err = readAuditControlBuildForController(build.Path, build.SHA256, false)
	if err != nil {
		return
	}
	producer, e := readAuditController(p.Controller, false)
	if e != nil {
		err = e
		return
	}
	proof, e := readAuditorProof(p.ProducerProof)
	if e != nil {
		err = e
		return
	}
	sources, err = verifyEffectiveAuditSources(c, producer, p, proof)
	return
}

func verifyEffectiveAuditInventory(run auditEffectiveRun, complete bool) error {
	order := auditCaseOrder()
	if !reflect.DeepEqual(run.Order, order) || len(run.Cases) > len(order) || (complete && len(run.Cases) != len(order)) {
		return fmt.Errorf("effective audit six-case inventory differs")
	}
	for i, c := range run.Cases {
		if (auditCaseID{c.Role, c.Kind}) != order[i] {
			return fmt.Errorf("effective audit case order differs at %d", i+1)
		}
	}
	return nil
}

func writeEffectiveAuditProgress(out string, run auditEffectiveRun) error {
	if err := verifyEffectiveAuditInventory(run, false); err != nil {
		return err
	}
	return writeJSON(filepath.Join(out, fmt.Sprintf("progress-%02d.json", len(run.Cases))), run)
}

type auditEffectiveInputs struct {
	Controller, Build Pin
	Current           GoController
	Native            auditControlBuild
	Plan              auditControlPlan
	Runtime           RuntimePlan
	Sources           []Pin
}

// Shared selection/authentication for the read-only check and admitted run.
func selectEffectiveAuditInputs(controllerPath, controllerSHA, buildPath, buildSHA string) (s auditEffectiveInputs, err error) {
	s.Controller, err = pin(controllerPath)
	if err != nil {
		return
	}
	if s.Controller.SHA256 != controllerSHA {
		return s, fmt.Errorf("selected current controller changed")
	}
	if _, err = readAuditController(s.Controller, true); err != nil {
		return
	}
	s.Build, err = pin(buildPath)
	if err != nil {
		return
	}
	if s.Build.SHA256 != buildSHA {
		return s, fmt.Errorf("selected retained build changed")
	}
	s.Current, s.Native, s.Plan, s.Runtime, s.Sources, err = readEffectiveAuditInputs(s.Controller, s.Build)
	return
}

type auditEffectiveCheckStatus string

const auditEffectiveRuntimeHeld auditEffectiveCheckStatus = "RuntimeHeld"

type auditEffectiveCheck struct {
	Schema     string                    `json:"schema"`
	Status     auditEffectiveCheckStatus `json:"status"`
	Controller Pin                       `json:"current_controller"`
	Binary     Pin                       `json:"current_controller_binary"`
	Producer   Pin                       `json:"historical_native_producer"`
	Build      Pin                       `json:"native_build"`
	Runtime    Pin                       `json:"native_runtime"`
	Sources    []Pin                     `json:"retained_native_sources"`
}

func checkEffectiveAuditControls(controllerPath, controllerSHA, buildPath, buildSHA string) (auditEffectiveCheck, error) {
	s, err := selectEffectiveAuditInputs(controllerPath, controllerSHA, buildPath, buildSHA)
	if err != nil {
		return auditEffectiveCheck{}, err
	}
	return auditEffectiveCheck{Schema: "nanopore-supgpu-current-effective-audit-check/v1", Status: auditEffectiveRuntimeHeld, Controller: s.Controller, Binary: s.Current.Binary, Producer: s.Plan.Controller, Build: s.Build, Runtime: s.Native.Runtime, Sources: s.Sources}, nil
}

func runEffectiveAuditControls(controllerPath, controllerSHA, buildPath, buildSHA, out string) (run auditEffectiveRun, err error) {
	run = auditEffectiveRun{Schema: "nanopore-supgpu-current-effective-audit-run/v1", Status: "failed", Order: auditCaseOrder(), Cases: []auditControlCase{}, Started: time.Now().UTC().Format(time.RFC3339Nano)}
	s, err := selectEffectiveAuditInputs(controllerPath, controllerSHA, buildPath, buildSHA)
	if err != nil {
		return run, err
	}
	run.Controller, run.Build = s.Controller, s.Build
	b, r := s.Native, s.Runtime
	run.Binary, run.Producer, run.Runtime, run.Audit, run.Proof, run.Sources = s.Current.Binary, s.Plan.Controller, b.Runtime, s.Plan.Audit, s.Plan.ProducerProof, s.Sources
	if out == "" {
		return run, fmt.Errorf("fresh effective audit output required")
	}
	out, err = filepath.Abs(out)
	if err != nil {
		return run, err
	}
	if err = os.Mkdir(out, 0755); err != nil {
		return run, err
	}
	defer func() {
		run.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			run.Status, run.Error = "failed", err.Error()
		}
		err = errors.Join(err, writeJSON(filepath.Join(out, "effective-run.json"), run))
	}()
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return run, err
	}
	defer lock.Close()
	for j, id := range run.Order {
		// Full retained build/proof and current closure authentication before every
		// child also checks denied/unknown DSOs, not just allowed runtime inputs.
		_, b, _, r, _, err = readEffectiveAuditInputs(run.Controller, run.Build)
		if err != nil {
			return run, err
		}
		dir := filepath.Join(out, id.Role+"-"+id.Kind)
		if err = os.Mkdir(dir, 0755); err != nil {
			return run, err
		}
		result, caseError := executeAuditControlCase(dir, id.Role, id.Kind, j%3, b, r)
		run.Cases = append(run.Cases, result)
		if caseError != nil {
			run.Error = caseError.Error()
		}
		err = errors.Join(caseError, writeEffectiveAuditProgress(out, run))
		if err != nil {
			return run, err
		}
	}
	if err = verifyEffectiveAuditInventory(run, true); err != nil {
		return run, err
	}
	run.Status = "passed"
	return run, nil
}

func TestAuditControlEffectiveAdmissionBeforeInputs(t *testing.T) {
	out := filepath.Join(t.TempDir(), "must-not-exist")
	err := auditControlCommand([]string{"effective-run", "-controller", "/missing/controller", "-build", "/missing/build", "-out", out})
	if err == nil || !strings.Contains(err.Error(), "requires separate coordinator admission") {
		t.Fatalf("admission ordering: %v", err)
	}
	if _, err = os.Lstat(out); !os.IsNotExist(err) {
		t.Fatalf("unadmitted output created: %v", err)
	}
}

func TestAuditControlEffectiveConsumerPrecedesProducer(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "invalid-current.json")
	if err := os.WriteFile(path, []byte("{}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	p, err := pin(path)
	if err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "output")
	_, err = runEffectiveAuditControls(path, p.SHA256, "/missing/historical-build", strings.Repeat("0", 64), out)
	if err == nil || strings.Contains(err.Error(), "/missing/historical-build") {
		t.Fatalf("retained producer read before current consumer: %v", err)
	}
	if _, err = os.Lstat(out); !os.IsNotExist(err) {
		t.Fatalf("invalid current consumer created output: %v", err)
	}
}

func TestAuditControlEffectiveNativeSourceDrift(t *testing.T) {
	dir := t.TempDir()
	var current, producer GoController
	var p auditControlPlan
	var proof auditProducerProof
	paths := append(append([]string{}, auditControlSources...), "verification/supgpu/native/audit.c")
	for i, path := range paths {
		file := filepath.Join(dir, fmt.Sprintf("inert-%d.c", i))
		if err := os.WriteFile(file, []byte("inert source\n"), 0644); err != nil {
			t.Fatal(err)
		}
		v, err := pin(file)
		if err != nil {
			t.Fatal(err)
		}
		current.Snapshots = append(current.Snapshots, Snapshot{Original: filepath.Join(dir, "current", path), Copy: v})
		producer.Snapshots = append(producer.Snapshots, Snapshot{Original: filepath.Join(dir, "producer", path), Copy: v})
		if i < 2 {
			p.Sources = append(p.Sources, v)
		} else {
			proof.Source = v
		}
	}
	if _, err := verifyEffectiveAuditSources(current, producer, p, proof); err != nil {
		t.Fatal(err)
	}
	missing := producer
	missing.Snapshots = producer.Snapshots[:2]
	if _, err := verifyEffectiveAuditSources(current, missing, p, proof); err == nil {
		t.Fatal("missing retained auditor snapshot accepted")
	}
	for i := range paths {
		t.Run(paths[i], func(t *testing.T) {
			changed := current
			changed.Snapshots = append([]Snapshot{}, current.Snapshots...)
			file := filepath.Join(t.TempDir(), "changed.c")
			if err := os.WriteFile(file, []byte("changed native body\n"), 0644); err != nil {
				t.Fatal(err)
			}
			v, err := pin(file)
			if err != nil {
				t.Fatal(err)
			}
			changed.Snapshots[i].Copy = v
			if _, err = verifyEffectiveAuditSources(changed, producer, p, proof); err == nil || !strings.Contains(err.Error(), "current native source differs") {
				t.Fatalf("changed native body accepted: %v", err)
			}
		})
	}
}

func TestAuditControlEffectiveInventoryAndExclusiveProgress(t *testing.T) {
	out := t.TempDir()
	run := auditEffectiveRun{Order: auditCaseOrder(), Sources: []Pin{}, Cases: []auditControlCase{}}
	want := []auditCaseID{{"reference", "allowed"}, {"reference", "wrong-known-tier"}, {"reference", "unknown"}, {"observer", "allowed"}, {"observer", "wrong-known-tier"}, {"observer", "unknown"}}
	for i, id := range want {
		if err := verifyEffectiveAuditInventory(run, true); err == nil {
			t.Fatal("missing execution accepted")
		}
		run.Cases = append(run.Cases, auditControlCase{Role: id.Role, Kind: id.Kind, Command: runner.CommandReceipt{Args: []string{}, Environment: []string{}}})
		if err := writeEffectiveAuditProgress(out, run); err != nil {
			t.Fatal(err)
		}
		file := filepath.Join(out, fmt.Sprintf("progress-%02d.json", i+1))
		before, err := pin(file)
		if err != nil {
			t.Fatal(err)
		}
		if err = writeEffectiveAuditProgress(out, run); !os.IsExist(err) {
			t.Fatalf("progress overwritten: %v", err)
		}
		if err = verifyPin("", before); err != nil {
			t.Fatal(err)
		}
		var saved auditEffectiveRun
		if err = readPinnedJSON(file, before.SHA256, &saved); err != nil || !reflect.DeepEqual(saved.Cases, run.Cases) {
			t.Fatalf("prefix readback: %v", err)
		}
	}
	if err := verifyEffectiveAuditInventory(run, true); err != nil {
		t.Fatal(err)
	}
	extra := run
	extra.Cases = append(append([]auditControlCase{}, run.Cases...), run.Cases[0])
	if err := verifyEffectiveAuditInventory(extra, true); err == nil {
		t.Fatal("extra execution accepted")
	}
	changedPlan := run
	changedPlan.Order = append([]auditCaseID{}, run.Order...)
	changedPlan.Order[0], changedPlan.Order[1] = changedPlan.Order[1], changedPlan.Order[0]
	if err := verifyEffectiveAuditInventory(changedPlan, true); err == nil {
		t.Fatal("changed plan accepted")
	}
	run.Cases[0], run.Cases[1] = run.Cases[1], run.Cases[0]
	if err := verifyEffectiveAuditInventory(run, true); err == nil {
		t.Fatal("reordered results accepted")
	}
}

func TestAuditControlEffectiveCheckRejectsBeforeOutput(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "inert-controller.json")
	if err := os.WriteFile(path, []byte("{}\n"), 0600); err != nil {
		t.Fatal(err)
	}
	p, err := pin(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = checkEffectiveAuditControls(path, strings.Repeat("0", 64), "/missing/build", ""); err == nil || !strings.Contains(err.Error(), "selected current controller changed") {
		t.Fatalf("wrong selected pin: %v", err)
	}
	if _, err = checkEffectiveAuditControls(path, p.SHA256, "/missing/build", ""); err == nil || strings.Contains(err.Error(), "/missing/build") {
		t.Fatalf("invalid current consumer ordering: %v", err)
	}
	out := filepath.Join(dir, "must-not-exist")
	if err = auditControlCommand([]string{"effective-check", "-out", out}); err == nil {
		t.Fatal("check accepted output directory")
	}
	if _, err = os.Lstat(out); !os.IsNotExist(err) {
		t.Fatalf("check created output: %v", err)
	}
}

// Inert source selection only: no compiler, framework, or valid-native claim.
func TestAuditControlCompilerSourcesUseProducerRoot(t *testing.T) {
	const module = "github.com/Jakeelamb/nanopore_basecaller_go"
	root := filepath.Join(t.TempDir(), "producer")
	if err := os.MkdirAll(filepath.Join(root, "verification/supgpu/native/controls"), 0700); err != nil {
		t.Fatal(err)
	}
	raw, err := Marshal(map[string]any{"Dir": filepath.Join(root, "verification/supgpu"), "Root": root, "ImportPath": module + "/verification/supgpu", "Module": map[string]any{"Path": module, "Main": true, "Dir": root, "GoMod": filepath.Join(root, "go.mod")}})
	if err != nil {
		t.Fatal(err)
	}
	deps := filepath.Join(root, "inert-deps.json")
	if err = os.WriteFile(deps, raw, 0600); err != nil {
		t.Fatal(err)
	}
	c := GoController{}
	c.Dependencies, err = pin(deps)
	if err != nil {
		t.Fatal(err)
	}
	want := []Pin{}
	for i, path := range auditControlSources {
		original := filepath.Join(root, path)
		body := []byte(fmt.Sprintf("inert source %d\n", i))
		if err = os.WriteFile(original, body, 0600); err != nil {
			t.Fatal(err)
		}
		v, e := pin(original)
		if e != nil {
			t.Fatal(e)
		}
		copyPath := filepath.Join(root, fmt.Sprintf("inert-copy-%d", i))
		if err = os.WriteFile(copyPath, body, 0600); err != nil {
			t.Fatal(err)
		}
		cp, e := pin(copyPath)
		if e != nil {
			t.Fatal(e)
		}
		c.Snapshots = append(c.Snapshots, Snapshot{Original: original, Copy: cp})
		want = append(want, v)
	}
	got, err := auditControllerNativeSources(c)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("producer source selection: %v %v", got, err)
	}
	pre, commands, _ := auditControlCommands("/inert/out", "/inert/compiler", got, "/inert/libdl")
	for i := 0; i < 4; i++ {
		source := want[0].Path
		if i != 0 {
			source = want[1].Path
		}
		if !containsString(pre[i].Argv, source) || !containsString(commands[i].Argv, source) {
			t.Fatal("compiler argv lost producer source path")
		}
	}
	changed := c
	changed.Snapshots = append([]Snapshot{}, c.Snapshots...)
	changed.Snapshots[0].Original = filepath.Join(root, "other/loader.c")
	if _, err = auditControllerNativeSources(changed); err == nil {
		t.Fatal("wrong original source path accepted")
	}
	if err = os.WriteFile(want[0].Path, []byte("changed\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err = auditControllerNativeSources(c); err == nil {
		t.Fatal("changed producer source accepted")
	}
}
