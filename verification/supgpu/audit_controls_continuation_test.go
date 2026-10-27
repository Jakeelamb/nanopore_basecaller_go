package supgpu

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type auditCaseID struct {
	Role string `json:"role"`
	Kind string `json:"kind"`
}

func auditCaseOrder() []auditCaseID {
	return []auditCaseID{{"reference", "allowed"}, {"reference", "wrong-known-tier"}, {"reference", "unknown"}, {"observer", "allowed"}, {"observer", "wrong-known-tier"}, {"observer", "unknown"}}
}

func writeAuditControlProgress(out string, run auditControlRun) error {
	return writeJSON(filepath.Join(out, fmt.Sprintf("progress-%02d.json", len(run.Cases))), run)
}

type auditContinuationPlan struct {
	Schema   string        `json:"schema"`
	Producer Pin           `json:"historical_producer"`
	Consumer Pin           `json:"current_consumer"`
	Build    Pin           `json:"native_build"`
	Runtime  Pin           `json:"native_runtime"`
	Audit    Pin           `json:"auditor"`
	Prefix   Pin           `json:"accepted_partial_run"`
	Evidence []Pin         `json:"accepted_prefix_evidence"`
	Accepted []auditCaseID `json:"accepted_cases"`
	Pending  []auditCaseID `json:"remaining_cases"`
}

type auditContinuationRun struct {
	Schema string             `json:"schema"`
	Status string             `json:"status"`
	Plan   Pin                `json:"continuation_plan"`
	Prefix Pin                `json:"accepted_partial_run"`
	Cases  []auditControlCase `json:"executed_cases"`
}

func verifyAuditPrefixOrder(run auditControlRun, build Pin) error {
	if run.Schema != "nanopore-supgpu-audit-control-run/v1" || run.Status != "failed" || run.Build != build || len(run.Cases) != 2 {
		return fmt.Errorf("continuation requires exactly the failed two-case native-build prefix")
	}
	for i, c := range run.Cases {
		if (auditCaseID{c.Role, c.Kind}) != auditCaseOrder()[i] {
			return fmt.Errorf("duplicate/nonprefix/wrong-role accepted case at %d", i)
		}
	}
	return nil
}

// The negative transcript is replayed through the same pure decision function.
// A failed capture import alone is insufficient proof of the constructor veto.
func verifyAuditDeniedGuard(g GuardReceipt, r RuntimePlan, runtime Pin, role, object string) error {
	expected := "known wrong runtime tier rejected: " + object
	if g.Schema != "nanopore-supgpu-loader-transcript/v1" || g.RuntimePlan != runtime || g.Role != role || g.Action != "attest-only" || g.PID <= 0 || g.Error != expected || g.StartupError != "" {
		return fmt.Errorf("denied prefix guard identity/reason differs")
	}
	if g.Group.PGID != g.PID || !g.Group.Empty || g.Group.Error != "" || len(g.Group.Kills) != 0 || len(g.Group.Observations) == 0 || len(g.Group.Observations[len(g.Group.Observations)-1].Members) != 0 {
		return fmt.Errorf("denied prefix process group did not finish cleanly")
	}
	if err := verifyPin("", g.StartupMaps); err != nil {
		return err
	}
	s := gateState{opened: map[uint64]Pin{}}
	for i, d := range g.Decisions {
		if err := verifyPin("", d.Maps); err != nil {
			return err
		}
		raw, err := os.ReadFile(d.Maps.Path)
		if err != nil {
			return err
		}
		o, err := parseCodeMaps(raw, r.Policy)
		if err != nil {
			return err
		}
		channel := "audit"
		if d.Request.Kind == "phase" {
			channel = "phase"
		}
		want, problem := decideGate(d.Request, channel, r.Policy, role, "attest-only", g.PID, &s, o)
		want.Maps = d.Maps
		if !reflect.DeepEqual(want, d) {
			return fmt.Errorf("denied prefix decision differs from saved maps/policy")
		}
		if i == len(g.Decisions)-1 {
			if problem == nil || problem.Error() != expected || d.Request.Kind != "open" || d.Request.Path != object || d.Approved || !s.hello || s.phase != 1 {
				return fmt.Errorf("denied prefix lacks final real object-open veto after pre-model")
			}
		} else if problem != nil || !d.Approved {
			return fmt.Errorf("denied prefix rejected before the intended object")
		}
	}
	if len(g.Decisions) == 0 {
		return fmt.Errorf("denied prefix lacks decisions")
	}
	if _, err := verifyGuard(g, r, runtime, role, "attest-only"); err == nil {
		return fmt.Errorf("denied prefix qualified capture import")
	}
	return nil
}

func auditPrefixEvidence(prefix Pin, build Pin, b auditControlBuild, r RuntimePlan) ([]Pin, error) {
	var run auditControlRun
	if err := readPinnedJSON(prefix.Path, prefix.SHA256, &run); err != nil {
		return nil, err
	}
	if err := verifyAuditPrefixOrder(run, build); err != nil {
		return nil, err
	}
	evidence := []Pin{}
	add := func(path string) error {
		v, err := pin(path)
		if err == nil {
			evidence = append(evidence, v)
		}
		return err
	}
	for i, c := range run.Cases {
		dir := filepath.Join(filepath.Dir(prefix.Path), c.Role+"-"+c.Kind)
		constructor, returned := filepath.Join(dir, "constructor.marker"), filepath.Join(dir, "returned.marker")
		if c.Guard.Path != filepath.Join(dir, c.Role+"-guard.json") || c.Command.Stdout != filepath.Join(dir, c.Role+".stdout") || c.Command.Stderr != filepath.Join(dir, c.Role+".stderr") || !reflect.DeepEqual(c.Command.Args, []string{b.Outputs[0].Path, b.Outputs[i+1].Path, constructor, returned}) {
			return nil, fmt.Errorf("accepted prefix command/path binding differs")
		}
		var g GuardReceipt
		if err := readPinnedJSON(c.Guard.Path, c.Guard.SHA256, &g); err != nil {
			return nil, err
		}
		env := append([]string{"SUPGPU_COMMAND_CWD=" + dir, "SUPGPU_GATE_FDS=3,4,5,6"}, modelEnvironment(r.Policy)...)
		env = append(env, fmt.Sprintf("SUPGPU_TARGET_PID=%d", g.PID))
		if !reflect.DeepEqual(c.Command.Environment, env) {
			return nil, fmt.Errorf("accepted prefix environment differs")
		}
		check := c.Command
		check.Exit = 0 // verify raw logs/clocks while checking the actual exit below.
		if err := verifyCommand(check); err != nil {
			return nil, err
		}
		commandPin, err := pin(filepath.Join(dir, "command.json"))
		if err != nil {
			return nil, err
		}
		var command runner.CommandReceipt
		if err := readPinnedJSON(commandPin.Path, commandPin.SHA256, &command); err != nil {
			return nil, err
		}
		if !reflect.DeepEqual(command, c.Command) {
			return nil, fmt.Errorf("accepted prefix command receipt differs")
		}
		for _, path := range []string{filepath.Join(dir, "command.json"), c.Command.Stdout, c.Command.Stderr, c.Guard.Path, g.StartupMaps.Path} {
			if err := add(path); err != nil {
				return nil, err
			}
		}
		for _, d := range g.Decisions {
			if err := add(d.Maps.Path); err != nil {
				return nil, err
			}
		}
		if i == 0 {
			if c.Command.Exit != 0 || c.Rejection != "" || c.ConstructorAbsent || c.ReturnedAbsent || c.Constructor.Path != constructor || c.Returned.Path != returned {
				return nil, fmt.Errorf("accepted allowed-prefix result differs")
			}
			if _, err := verifyGuard(g, r, b.Runtime, c.Role, "attest-only"); err != nil {
				return nil, err
			}
			for _, v := range []Pin{c.Constructor, c.Returned} {
				if err := verifyPin("", v); err != nil {
					return nil, err
				}
				if err := add(v.Path); err != nil {
					return nil, err
				}
			}
			for path, want := range map[string]string{constructor: "allowed constructor\n", returned: "dlopen returned\n"} {
				raw, err := os.ReadFile(path)
				if err != nil || string(raw) != want {
					return nil, fmt.Errorf("accepted allowed-prefix marker differs")
				}
			}
		} else {
			if c.Command.Exit != 124 || c.Rejection != g.Error || !c.ConstructorAbsent || !c.ReturnedAbsent || c.Constructor != (Pin{}) || c.Returned != (Pin{}) {
				return nil, fmt.Errorf("accepted denied-prefix result differs")
			}
			for _, path := range []string{constructor, returned} {
				if _, err := os.Lstat(path); !os.IsNotExist(err) {
					return nil, fmt.Errorf("denied prefix marker is present/unobservable")
				}
			}
			if err := verifyAuditDeniedGuard(g, r, b.Runtime, c.Role, b.Outputs[2].Path); err != nil {
				return nil, err
			}
		}
	}
	return evidence, nil
}

func auditContinuationRecipe(consumer, build, prefix Pin) (p auditContinuationPlan, b auditControlBuild, r RuntimePlan, err error) {
	p = auditContinuationPlan{Schema: "nanopore-supgpu-audit-control-continuation-plan/v1", Consumer: consumer, Build: build, Prefix: prefix, Accepted: auditCaseOrder()[:2], Pending: auditCaseOrder()[2:]}
	for _, v := range []Pin{consumer, build, prefix} {
		if err = verifyPin("", v); err != nil {
			return
		}
	}
	if _, err = readAuditController(consumer, true); err != nil {
		return
	}
	var old auditControlPlan
	b, old, r, err = readAuditControlBuildForController(build.Path, build.SHA256, false)
	if err != nil {
		return
	}
	if _, err = readAuditController(old.Controller, false); err != nil {
		return
	}
	p.Producer, p.Runtime, p.Audit = old.Controller, b.Runtime, old.Audit
	p.Evidence, err = auditPrefixEvidence(prefix, build, b, r)
	return
}

func prepareAuditContinuation(controllerPath, controllerSHA, buildPath, buildSHA, prefixPath, prefixSHA, out string) (auditContinuationPlan, error) {
	values := []Pin{}
	for _, input := range []struct{ path, sha string }{{controllerPath, controllerSHA}, {buildPath, buildSHA}, {prefixPath, prefixSHA}} {
		v, err := pin(input.path)
		if err != nil {
			return auditContinuationPlan{}, err
		}
		if v.SHA256 != input.sha {
			return auditContinuationPlan{}, fmt.Errorf("continuation input manifest changed")
		}
		values = append(values, v)
	}
	p, _, _, err := auditContinuationRecipe(values[0], values[1], values[2])
	if err == nil {
		err = writeJSON(out, p)
	}
	return p, err
}

func readAuditContinuation(path, sha string) (p auditContinuationPlan, b auditControlBuild, r RuntimePlan, err error) {
	if err = readPinnedJSON(path, sha, &p); err != nil {
		return
	}
	want, built, runtime, e := auditContinuationRecipe(p.Consumer, p.Build, p.Prefix)
	if e != nil {
		err = e
		return
	}
	if !reflect.DeepEqual(p, want) {
		err = fmt.Errorf("finite accepted-prefix/remaining-four continuation changed")
		return
	}
	return p, built, runtime, nil
}

func continueAuditControls(path, sha, out string) (run auditContinuationRun, err error) {
	run = auditContinuationRun{Schema: "nanopore-supgpu-audit-control-continuation-run/v1", Status: "failed", Cases: []auditControlCase{}}
	p, b, r, err := readAuditContinuation(path, sha)
	if err != nil {
		return run, err
	}
	run.Plan, err = pin(path)
	if err != nil {
		return run, err
	}
	run.Prefix = p.Prefix
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return run, err
	}
	defer lock.Close()
	out, err = filepath.Abs(out)
	if err != nil {
		return run, err
	}
	if err = os.Mkdir(out, 0755); err != nil {
		return run, err
	}
	defer func() {
		if e := writeJSON(filepath.Join(out, "run.json"), run); err == nil {
			err = e
		}
	}()
	for j, id := range p.Pending {
		if _, err = readAuditController(p.Consumer, true); err != nil {
			return run, err
		}
		for _, v := range append(append([]Pin{}, r.Inputs...), p.Audit, p.Build, p.Runtime, p.Prefix) {
			if err = verifyPin("", v); err != nil {
				return run, err
			}
		}
		for _, v := range p.Evidence {
			if err = verifyPin("", v); err != nil {
				return run, err
			}
		}
		dir := filepath.Join(out, id.Role+"-"+id.Kind)
		if err = os.Mkdir(dir, 0755); err != nil {
			return run, err
		}
		i := 0
		if id.Kind == "wrong-known-tier" {
			i = 1
		} else if id.Kind == "unknown" {
			i = 2
		}
		c, problem := executeAuditControlCase(dir, id.Role, id.Kind, i, b, r)
		run.Cases = append(run.Cases, c)
		if err = writeJSON(filepath.Join(out, fmt.Sprintf("progress-%02d.json", j+3)), run); err != nil {
			return run, err
		}
		if problem != nil {
			return run, problem
		}
	}
	run.Status = "passed"
	return run, nil
}

func TestAuditControlProgressIsImmutablePerCase(t *testing.T) {
	out := t.TempDir()
	run := auditControlRun{Cases: []auditControlCase{{Role: "reference", Kind: "allowed"}}}
	if err := writeAuditControlProgress(out, run); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(out)
	if err != nil || len(entries) != 1 {
		t.Fatalf("first progress receipt missing: %v", err)
	}
	first, err := pin(filepath.Join(out, entries[0].Name()))
	if err != nil {
		t.Fatal(err)
	}
	run.Cases = append(run.Cases, auditControlCase{Role: "reference", Kind: "wrong-known-tier"})
	if err = writeAuditControlProgress(out, run); err != nil {
		t.Fatalf("second completed case could not retain progress: %v", err)
	}
	if err = verifyPin("", first); err != nil {
		t.Fatal("first observation was overwritten", err)
	}
	if err = writeAuditControlProgress(out, run); !os.IsExist(err) {
		t.Fatalf("duplicate progress must retain O_EXCL: %v", err)
	}
}

func TestAuditControlContinuationRequiresExactTwoCasePrefix(t *testing.T) {
	build := Pin{Path: "/saved/build.json", SHA256: strings.Repeat("a", 64), Bytes: 1}
	base := func() auditControlRun {
		return auditControlRun{Schema: "nanopore-supgpu-audit-control-run/v1", Status: "failed", Build: build, Cases: []auditControlCase{{Role: "reference", Kind: "allowed"}, {Role: "reference", Kind: "wrong-known-tier"}}}
	}
	if err := verifyAuditPrefixOrder(base(), build); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"duplicate", "nonprefix", "wrong-role", "changed-build", "completed-run", "missing-case"} {
		t.Run(kind, func(t *testing.T) {
			run := base()
			switch kind {
			case "duplicate":
				run.Cases[1] = run.Cases[0]
			case "nonprefix":
				run.Cases[1].Kind = "unknown"
			case "wrong-role":
				run.Cases[1].Role = "observer"
			case "changed-build":
				run.Build.Bytes++
			case "completed-run":
				run.Status = "passed"
			case "missing-case":
				run.Cases = run.Cases[:1]
			}
			if err := verifyAuditPrefixOrder(run, build); err == nil {
				t.Fatal("invalid prefix accepted")
			}
		})
	}
}
