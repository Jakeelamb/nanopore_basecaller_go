package cgobuild

import (
	"fmt"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"time"
)

const retainedProfile = "saved005-final-reader-v1"
const retainedFailure = "link used input outside before-child snapshots"

type retainedProducerPins struct{ Receipt, Plan, Controller, Queries, Output Pin }
type retainedSelection struct {
	Producer                               retainedProducerPins
	Handoff, Outer, Cleanup, FailureStderr Pin
	Profile                                string
}
type retainedHandoff struct {
	Status, Role, Error                                                                             string
	Receipt, Plan, Controller, Queries, Output                                                      Pin
	Outer, Cleanup, RawError, RawController, Before, After, Wrapper, LaunchRequest, ReviewedRequest Pin
	Operations                                                                                      int
	Statuses                                                                                        map[string]int
}
type retainedCommand struct {
	Argv                   []string          `json:"argv"`
	Environment            map[string]string `json:"environment"`
	Started                string            `json:"started_utc"`
	Finished               string            `json:"finished_utc"`
	Exit                   int               `json:"exit"`
	PID                    int               `json:"pid"`
	Script, Stdout, Stderr Pin
	Effective              map[string]any `json:"effective"`
}
type retainedCleanup struct {
	UTC, Status                                                                                                         string
	SystemdProperties                                                                                                   map[string]string
	CgroupAbsent, ControllerPIDAndGroupAbsent, BeforeAfterPinsEqual, CanonicalReadonlyNonblockingAvailableAtObservation bool
	Controller                                                                                                          Process
	ActualProcessCount                                                                                                  int
	ActualProcesses                                                                                                     []struct {
		Record                                                 Pin
		Identity                                               Process
		OriginalPIDAbsent, CurrentPIDAbsent, OwnedGroupChecked bool
		GroupMembers                                           []int
	}
	RawCommand struct {
		Argv                    []string
		StartedUTC, FinishedUTC string
		Exit                    int
		Stdout, Stderr          Pin
	}
	Receipt, Output               Pin
	TargetExecutionOrLibraryLoads int
	Retry                         bool
}

func readRetainedCompiled(selected retainedSelection) (Receipt, error) {
	var r Receipt
	if e := read(selected.Producer.Receipt, &r); e != nil {
		return r, e
	}
	var p Plan
	if e := read(selected.Producer.Plan, &p); e != nil {
		return r, e
	}
	if e := authenticateRetainedEnvelope(selected, p, r); e != nil {
		return r, e
	}
	return r, validateReceiptBody(p, r, compiledPhase)
}

func retainedTuple(s retainedSelection, h retainedHandoff, p Plan, r Receipt) error {
	if s.Profile != retainedProfile {
		return fmt.Errorf("unknown retained producer profile")
	}
	if r.Schema != "nanopore-h08-cgo-execution/v1" || !r.RuntimeHeld || r.Status != "failed" || r.Error != retainedFailure || h.Error != retainedFailure || h.Status != "APPLICATION_COMPILED_BUT_FAILED_RECEIPT_VALIDATION_UNQUALIFIED_UNEXECUTED" {
		return fmt.Errorf("retained producer error/status differs")
	}
	if p.Role != H08Command || r.Role != H08Command || h.Role != string(H08Command) {
		return fmt.Errorf("retained producer role differs")
	}
	want := s.Producer
	if want != (retainedProducerPins{h.Receipt, h.Plan, h.Controller, h.Queries, h.Output}) || r.Plan != want.Plan || r.Controller != want.Controller || r.Queries != want.Queries || r.Output != want.Output || s.Outer != h.Outer || s.Cleanup != h.Cleanup || s.FailureStderr != h.RawError {
		return fmt.Errorf("independent retained producer tuple differs")
	}
	for _, q := range []Pin{want.Receipt, want.Plan, want.Controller, want.Queries, want.Output, s.Handoff, s.Outer, s.Cleanup, s.FailureStderr} {
		if !filepath.IsAbs(q.Path) || len(q.SHA256) != 64 {
			return fmt.Errorf("retained selection pin omitted")
		}
	}
	if len(r.Commands) != 2 || r.Commands[0].Exit != 0 || r.Commands[1].Exit != 0 || r.Selection.Path == "" || len(r.Generated) != 4 || h.Operations != 297 || !maps.Equal(h.Statuses, map[string]int{"accepted": 289, "accepted_expected_exit": 8}) {
		return fmt.Errorf("retained final-reader stage incomplete")
	}
	return nil
}

func authenticateRetainedEnvelope(s retainedSelection, p Plan, r Receipt) error {
	var h retainedHandoff
	if e := read(s.Handoff, &h); e != nil {
		return e
	}
	if e := retainedTuple(s, h, p, r); e != nil {
		return e
	}
	fixed := []struct {
		got Pin
		sha string
	}{
		{s.Producer.Receipt, "f487edce8e9256f5c4cd432337fa95080d7894f94e2d9bdeb26c8d3ed6aed82b"},
		{s.Producer.Plan, "67e67736e47d7b365dcccc5f1cb94b18720100f7beffd40bc9f9b1353a65090c"},
		{s.Producer.Controller, "91fcc7edf9c6771ecd31258ec4919ae0da6556cf9736bd0224719aa1c0276da7"},
		{s.Producer.Queries, "8f4e70aa825fbb30683ee046751819877a0326b19a47bb0c590d6e911d339a12"},
		{s.Producer.Output, "52ae02a2370d718123bc7088d3de1d5d483122323eaa5c7e794a854df1917c69"},
	}
	for _, q := range fixed {
		if q.got.SHA256 != q.sha {
			return fmt.Errorf("unrecognized historical producer identity")
		}
	}
	c, e := controllerRead(s.Producer.Controller)
	if e != nil {
		return e
	}
	if e = retainedSources(p, c); e != nil {
		return e
	}
	var session Session
	if e = read(r.Session, &session); e != nil {
		return e
	}
	if session.Phase != "build" || session.Plan != r.Plan || session.Controller != r.Controller || session.Driver != c.Executable || session.Directory != filepath.Dir(s.Producer.Receipt.Path) {
		return fmt.Errorf("retained build session differs")
	}
	if e = noRejections(session); e != nil {
		return e
	}
	if e = completeOperationInventory(session, r.Operations); e != nil {
		return e
	}
	counts := map[string]int{}
	witness := false
	for _, pin := range r.Operations {
		var op operation
		if e = read(pin, &op); e != nil {
			return e
		}
		if op.Error != "" || op.Status != "accepted" && op.Status != "accepted_expected_exit" {
			return fmt.Errorf("retained earlier native rejection")
		}
		counts[op.Status]++
		if op.Request != nil && op.Request.Link && op.LinkMap.Path != "" && len(op.Commands) > 0 {
			last := op.Commands[len(op.Commands)-1]
			if e = verifyPin(Pin{Path: last.Stdout, SHA256: last.StdoutSHA}); e != nil {
				return e
			}
			body, err := os.ReadFile(last.Stdout)
			if err != nil {
				return err
			}
			for _, line := range strings.Split(string(body), "\n") {
				line = strings.TrimSpace(line)
				if line == "" {
					continue
				}
				old := absArg(op.Directory, line)
				if bound := op.Request.Bindings[old]; bound != "" {
					old = bound
				}
				if op.Inputs[old] == "" || op.Snapshots[old].SHA256 != op.Inputs[old] {
					witness = true
				}
			}
		}
	}
	if !maps.Equal(counts, h.Statuses) || len(r.Operations) != 297 || !witness {
		return fmt.Errorf("retained final-reader failure witness/inventory differs")
	}
	if e = retainedOuter(s, h, p, r, session, c); e != nil {
		return e
	}
	return retainedRelease(s, session)
}

func retainedSources(p Plan, c Controller) error {
	saved005CompletedBuildSources := map[string]string{
		"execution.go": "f5379032acff2b7a6e33bf88a234c978a6c5f60dd72d106e4c4476568080abce",
		"verify.go":    "35c95fbe88678132567f9b2759b4d85982e7711726287d1fe31a58c50ec7fd06",
		"evidence.go":  "78318a146b39eca35ff4b223d5d085afafc850df69af222320c8876680d0be91",
		"cmd/main.go":  "9becffd94da775f71f7bb4cca5f1eac213a0273514812d3120fbb752827d8173",
	}
	for name, sha := range saved005CompletedBuildSources {
		path := filepath.Join(p.Options.Root, "verification/hacgpu/h08/cgobuild", name)
		q := c.Snapshots[path]
		if (name != "cmd/main.go" && p.Inputs[path] != sha) || c.Inputs[path] != sha || q.SHA256 != sha {
			return fmt.Errorf("historical producer source/control flow differs: %s", name)
		}
		if e := verifyPin(q); e != nil {
			return e
		}
	}
	return nil
}

func retainedClock(start, finish string) (time.Time, time.Time, error) {
	a, e := time.Parse(time.RFC3339Nano, start)
	b, f := time.Parse(time.RFC3339Nano, finish)
	if e != nil || f != nil || b.Before(a) {
		return a, b, fmt.Errorf("retained command clocks differ")
	}
	return a, b, nil
}
func retainedRaw(c retainedCommand) error {
	if len(c.Argv) == 0 {
		return fmt.Errorf("retained raw argv omitted")
	}
	if _, _, e := retainedClock(c.Started, c.Finished); e != nil {
		return e
	}
	for _, q := range []Pin{c.Stdout, c.Stderr} {
		if e := verifyPin(q); e != nil {
			return e
		}
	}
	return nil
}
func retainedOuter(s retainedSelection, h retainedHandoff, p Plan, r Receipt, session Session, c Controller) error {
	var outer, raw retainedCommand
	if e := read(s.Outer, &outer); e != nil {
		return e
	}
	if e := read(h.RawController, &raw); e != nil {
		return e
	}
	if e := retainedRaw(outer); e != nil {
		return e
	}
	if e := retainedRaw(raw); e != nil {
		return e
	}
	host := filepath.Dir(s.Outer.Path)
	unit := filepath.Base(host) + ".service"
	argv := []string{"systemd-run", "--user", "--wait", "--pipe", "--unit=" + unit, "--working-directory=" + p.Options.Root, "-p", "CPUQuota=200%", "-p", "MemoryMax=3G", "-p", "MemorySwapMax=0", "-p", "TasksMax=64", "-p", "RuntimeMaxSec=2100s", "-p", "KillMode=control-group", "-p", "TimeoutStopSec=5s", "/home/jake/.local/bin/uv", "run", "--no-project", "python", h.Wrapper.Path, "--inner"}
	build := []string{c.Executable.Path, "-mode", "build", "-native-admitted", "-plan", r.Plan.Path, "-plan-sha", r.Plan.SHA256, "-controller", r.Controller.Path, "-controller-sha", r.Controller.SHA256, "-queries", r.Queries.Path, "-queries-sha", r.Queries.SHA256, "-out", session.Directory}
	env := map[string]string{"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "TZ": "UTC", "CGO_ENABLED": "0", "GOMAXPROCS": "2", "GOMEMLIMIT": "3GiB", "GOROOT": "/usr/lib/go", "GOTOOLCHAIN": "local", "GOENV": "off", "GOWORK": "off", "GOPROXY": "off", "GOSUMDB": "off", "GOFLAGS": ""}
	if outer.Exit != 1 || raw.Exit != 1 || outer.Script != h.Wrapper || raw.Stderr != s.FailureStderr || !slices.Equal(outer.Argv, argv) || !slices.Equal(raw.Argv, build) || !maps.Equal(raw.Environment, env) || raw.PID != session.ParentPID {
		return fmt.Errorf("retained outer command/exit/environment differs")
	}
	for _, q := range []Pin{h.Wrapper, h.LaunchRequest, h.ReviewedRequest, r.Output} {
		if e := verifyPin(q); e != nil {
			return e
		}
	}
	if h.Wrapper.Path != filepath.Join(host, "run.py") || s.Cleanup.Path != filepath.Join(host, "cleanup.json") || h.RawController.Path != filepath.Join(host, "build-receipt.json") || h.Before.Path != filepath.Join(host, "inputs-before.json") || h.After.Path != filepath.Join(host, "inputs-after.json") || raw.Stdout.Path != filepath.Join(host, "commands/build.stdout") || raw.Stderr.Path != filepath.Join(host, "commands/build.stderr") || outer.Stdout.Path != filepath.Join(host, "commands/outer.stdout") || outer.Stderr.Path != filepath.Join(host, "commands/outer.stderr") {
		return fmt.Errorf("retained outer raw record domain differs")
	}
	group := "/sys/fs/cgroup/" + r.Limits["cgroup"]
	for key, want := range map[string]string{"cgroup": group, "cpu.max": "200000 100000", "memory.max": "3221225472", "memory.swap.max": "0", "pids.max": "64"} {
		if raw.Effective[key] != want {
			return fmt.Errorf("retained outer effective limits differ")
		}
	}
	for key, want := range map[string]string{"cpu.max": "200000 100000", "memory.max": "3221225472", "memory.swap.max": "0", "pids.max": "64", "child_timeout_seconds": "180"} {
		if r.Limits[key] != want {
			return fmt.Errorf("retained producer effective limits differ")
		}
	}
	if h.Wrapper.SHA256 != "3305b00a6112953efe950197f81d66fd1a386d0f63efb03d61931d791e0f58ee" {
		return fmt.Errorf("retained outer wrapper control flow differs")
	}
	body, e := os.ReadFile(raw.Stderr.Path)
	if e != nil {
		return e
	}
	if string(body) != retainedFailure+"\n" {
		return fmt.Errorf("retained original stderr differs")
	}
	a, b, e := retainedClock(outer.Started, outer.Finished)
	if e != nil {
		return e
	}
	x, y, e := retainedClock(raw.Started, raw.Finished)
	if e != nil {
		return e
	}
	rs, rf, e := retainedClock(r.Started, r.Finished)
	if e != nil {
		return e
	}
	if x.Before(a) || y.After(b) || rs.Before(x) || rf.After(y) {
		return fmt.Errorf("retained outer clock nesting differs")
	}
	for _, cmd := range r.Commands {
		cs, cf, e := retainedClock(cmd.Started, cmd.Finished)
		if e != nil {
			return e
		}
		if cs.Before(rs) || cf.After(rf) || cmd.Exit != 0 {
			return fmt.Errorf("retained Go completion stage differs")
		}
	}
	listEnd, _ := time.Parse(time.RFC3339Nano, r.Commands[0].Finished)
	buildStart, _ := time.Parse(time.RFC3339Nano, r.Commands[1].Started)
	if buildStart.Before(listEnd) {
		return fmt.Errorf("retained Go stage order differs")
	}
	var before, after map[string]string
	if e = read(h.Before, &before); e != nil {
		return e
	}
	if e = read(h.After, &after); e != nil {
		return e
	}
	expected := maps.Clone(p.Inputs)
	maps.Copy(expected, c.Inputs)
	for _, q := range []Pin{c.Executable, r.Controller, r.Plan, r.Queries, h.LaunchRequest, h.ReviewedRequest, h.Wrapper} {
		expected[q.Path] = q.SHA256
	}
	if !maps.Equal(before, after) || !maps.Equal(before, expected) {
		return fmt.Errorf("retained immutable before/after differs")
	}
	return nil
}

func retainedRelease(s retainedSelection, session Session) error {
	var c retainedCleanup
	if e := read(s.Cleanup, &c); e != nil {
		return e
	}
	props := map[string]string{"ActiveState": "failed", "SubState": "failed", "MainPID": "0", "Result": "exit-code", "ExecMainStatus": "1", "ControlGroup": ""}
	if c.Status != "FAILED_BUILD_RECEIPT_VALIDATION_RELEASED_NO_RETRY" || !maps.Equal(c.SystemdProperties, props) || !c.CgroupAbsent || !c.ControllerPIDAndGroupAbsent || !c.BeforeAfterPinsEqual || !c.CanonicalReadonlyNonblockingAvailableAtObservation || c.ActualProcessCount != 346 || len(c.ActualProcesses) != 346 || c.Receipt != s.Producer.Receipt || c.Output != s.Producer.Output || c.TargetExecutionOrLibraryLoads != 0 || c.Retry {
		return fmt.Errorf("retained cleanup release incomplete")
	}
	var controller Process
	q, e := closedPin(filepath.Join(filepath.Dir(s.Outer.Path), "controller-process.json"))
	if e != nil {
		return e
	}
	if e = read(q, &controller); e != nil {
		return e
	}
	if controller != c.Controller || controller.PID != session.ParentPID || controller.Start != session.ParentStart || !controller.Group {
		return fmt.Errorf("retained cleanup controller differs")
	}
	seen := map[string]bool{}
	groups := 0
	for _, entry := range c.ActualProcesses {
		if seen[entry.Record.Path] || !under(filepath.Join(session.Directory, "commands"), entry.Record.Path) || filepath.Base(entry.Record.Path) != "process.json" {
			return fmt.Errorf("retained cleanup process membership differs")
		}
		seen[entry.Record.Path] = true
		var actual Process
		if e = read(entry.Record, &actual); e != nil {
			return e
		}
		if actual != entry.Identity || actual.PID <= 0 || actual.Start == "" || actual.Cgroup != controller.Cgroup || !entry.OriginalPIDAbsent || !entry.CurrentPIDAbsent || entry.OwnedGroupChecked != actual.Group || len(entry.GroupMembers) != 0 {
			return fmt.Errorf("retained cleanup process/group differs")
		}
		if actual.Group {
			groups++
		}
	}
	if groups != 97 {
		return fmt.Errorf("retained cleanup owned group inventory differs")
	}
	if e = filepath.WalkDir(filepath.Join(session.Directory, "commands"), func(path string, d fs.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if !d.IsDir() && d.Name() == "process.json" && !seen[path] {
			return fmt.Errorf("retained cleanup omitted actual process")
		}
		return nil
	}); e != nil {
		return e
	}
	raw := c.RawCommand
	args := []string{"systemctl", "--user", "show", filepath.Base(filepath.Dir(s.Outer.Path)) + ".service", "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "Result", "-p", "ExecMainStatus", "-p", "ControlGroup"}
	if raw.Exit != 0 || !slices.Equal(raw.Argv, args) {
		return fmt.Errorf("retained cleanup raw command differs")
	}
	if _, _, e = retainedClock(raw.StartedUTC, raw.FinishedUTC); e != nil {
		return e
	}
	if e = verifyPin(raw.Stdout); e != nil {
		return e
	}
	if e = verifyPin(raw.Stderr); e != nil {
		return e
	}
	body, e := os.ReadFile(raw.Stdout.Path)
	if e != nil {
		return e
	}
	if string(body) != "ActiveState=failed\nSubState=failed\nMainPID=0\nResult=exit-code\nExecMainStatus=1\nControlGroup=\n" {
		return fmt.Errorf("retained cleanup raw properties differ")
	}
	var outer retainedCommand
	if e = read(s.Outer, &outer); e != nil {
		return e
	}
	_, end, e := retainedClock(outer.Started, outer.Finished)
	if e != nil {
		return e
	}
	release, e := time.Parse(time.RFC3339Nano, c.UTC)
	if e != nil || release.Before(end) {
		return fmt.Errorf("retained cleanup release clock differs")
	}
	return nil
}

func completeOperationInventory(s Session, pins []Pin) error {
	selected := map[string]bool{}
	for _, q := range pins {
		if selected[q.Path] {
			return fmt.Errorf("duplicate operation inventory")
		}
		if !under(filepath.Join(s.Directory, "commands"), q.Path) || filepath.Base(q.Path) != "operation.json" {
			return fmt.Errorf("operation inventory outside selected session")
		}
		selected[q.Path] = true
	}
	actual := map[string]bool{}
	if e := filepath.WalkDir(filepath.Join(s.Directory, "commands"), func(path string, d fs.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if d.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("operation inventory symlink")
		}
		if !d.IsDir() && d.Name() == "operation.json" {
			actual[path] = true
		}
		return nil
	}); e != nil {
		return e
	}
	if !reflect.DeepEqual(selected, actual) {
		return fmt.Errorf("complete operation inventory differs")
	}
	return nil
}
