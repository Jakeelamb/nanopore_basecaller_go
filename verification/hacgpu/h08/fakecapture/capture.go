package fakecapture

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08/cgobuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type Session struct {
	Schema                                  string
	Request, Plan, Logger, Compiler, Target Pin
	Parent                                  Process
	Root, Out, Deadline                     string
	CompilerEnvironment                     []string
}

// RuntimeHeld means this fake-only unit installed no genuine backend. It does
// not attest that root subsequently completed unit/cgroup cleanup.
type Summary struct {
	Schema, Status, Error, Started, Finished string
	Request, Session, Proof                  Pin
	Process, TargetProcess                   Process
	Cwd                                      string
	Command                                  runner.CommandReceipt
	TargetReceipts                           CommandPins
	TargetObservation                        ProcessObservation
	Compiles                                 []Pin
	Limits                                   map[string]string
	RuntimeHeld                              bool
}

func cwdIs(root string) error {
	cwd, e := os.Getwd()
	if e != nil {
		return e
	}
	if cwd != root {
		return fmt.Errorf("actual working directory differs")
	}
	return runner.CheckRoot()
}

func targetEnvironment(q Request, session Pin) []string {
	return []string{
		"PATH=" + filepath.Join(q.Out, "bin"), "LANG=C", "LC_ALL=C", "TZ=UTC",
		"TMPDIR=" + filepath.Join(q.Out, "tmp"), "GOMAXPROCS=2", "GOMEMLIMIT=3GiB",
		"H08_FAKE_SESSION=" + session.Path, "H08_FAKE_SESSION_SHA=" + session.SHA256,
	}
}

// Capture requires admission in addition to root's already-installed envelope.
// No lock is acquired here. Root must keep its existing O_RDONLY EX|NB lock
// until this process, the unit/cgroup and every descendant are strictly dead.
func Capture(ctx context.Context, request Pin, admitted bool) (r Summary, err error) {
	r = Summary{Schema: "nanopore-h08-fake-capture/v1", Status: "failed", Request: request, Started: time.Now().UTC().Format(time.RFC3339Nano)}
	if !admitted {
		return r, fmt.Errorf("explicit root native-phase admission required")
	}
	if err = ctx.Err(); err != nil {
		return
	}
	if r.Limits, err = closedbuild.Limits(); err != nil {
		return
	}
	if err = runner.CheckRoot(); err != nil {
		return
	}
	q, err := readRequest(request, true)
	if err != nil {
		return r, err
	}
	if err = cwdIs(q.Root); err != nil {
		return
	}
	if err = checkSelf(q.Logger); err != nil {
		return
	}
	p, err := acceptedPlan(q)
	if err != nil {
		return r, err
	}
	if err = ctx.Err(); err != nil {
		return
	}
	if err = outputBoundary(q, true); err != nil {
		return
	}
	if err = os.Mkdir(q.Out, 0700); err != nil {
		return
	}
	r.Cwd = q.Root
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		err = errors.Join(err, ctx.Err())
		if err != nil {
			r.Status, r.RuntimeHeld, r.Error = "failed", false, err.Error()
		}
		_, e := writeNew(filepath.Join(q.Out, "summary.json"), r)
		err = errors.Join(err, e)
	}()
	for _, name := range []string{"bin", "logs", "tmp", "compiles"} {
		if err = os.Mkdir(filepath.Join(q.Out, name), 0700); err != nil {
			return
		}
	}
	if err = os.Symlink(q.Logger.Path, filepath.Join(q.Out, "bin", "cc")); err != nil {
		return
	}
	if r.Process, err = identity(os.Getpid()); err != nil {
		return
	}
	var build cgobuild.Receipt
	b, err := pinBytes(q.ControllerBuild)
	if err != nil {
		return r, err
	}
	if err = json.Unmarshal(b, &build); err != nil {
		return
	}
	env, err := compilerEnvironment(p, filepath.Join(q.Out, "tmp"))
	if err != nil {
		return r, err
	}
	// Includes all helpers and their compiler time; this is never a new budget
	// per compile. Root's independent 600s limit includes authentication above.
	deadline := time.Now().Add(180 * time.Second)
	s := Session{Schema: sessionSchema, Request: request, Plan: build.Plan, Logger: q.Logger, Compiler: q.Compiler, Target: q.Target, Parent: r.Process, Root: q.Root, Out: q.Out, Deadline: deadline.UTC().Format(time.RFC3339Nano), CompilerEnvironment: env}
	if r.Session, err = writeNew(filepath.Join(q.Out, "session.json"), s); err != nil {
		return
	}
	if _, err = readRequest(request, false); err != nil {
		return
	}
	targetContext, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	logged, runErr := runLogged(targetContext, filepath.Join(q.Out, "logs"), q.Root, q.Target, targetArgs(q), targetEnvironment(q, r.Session))
	r.Command, r.TargetProcess, r.TargetReceipts, r.TargetObservation = logged.Command, logged.Process, logged.Receipts, logged.Observation
	err = runErr
	// Inventory every directory even after target failure. Never manufacture or
	// truncate compiler records to satisfy the expected count.
	compiles, pins, collectErr := collect(q, r.Session, r.TargetProcess)
	r.Compiles = pins
	err = errors.Join(err, collectErr)
	if err != nil {
		return
	}
	if err = ctx.Err(); err != nil {
		return
	}
	if err = readLogged(filepath.Join(q.Out, "logs"), q.Root, q.Target, logged); err != nil {
		return
	}
	if r.Command.Exit != 0 || r.TargetProcess.Cgroup != r.Process.Cgroup {
		return r, fmt.Errorf("target exit/cgroup differs")
	}
	if len(compiles) != 39 {
		return r, fmt.Errorf("exactly39 complete successful compiles required; got%d", len(compiles))
	}
	proof := h08.FakeProof{
		Schema: "nanopore-h08-fake-loader-proof/v2", Status: acceptedStatus,
		ControllerBuild: q.ControllerBuild, ControllerBuildSelection: q.ControllerBuildSelection,
		Target: q.Target, Logger: q.Logger, Compiler: q.Compiler, Command: r.Command,
		Compiles: compiles, Inputs: map[string]string{},
		Tests: map[string]string{"TestNativeLoaderABIRejections": "pass", "TestNativeCloseFailureRetainsLibrary": "pass"},
	}
	// These test values are provisional assertions, checked against actual raw
	// RUN/PASS/skip/fail results by the public reader before any final receipt.
	for _, pin := range []Pin{q.ControllerBuild, q.ControllerBuildSelection, q.Target, q.Logger, q.Compiler, request, r.Session, build.Plan} {
		if old := proof.Inputs[pin.Path]; old != "" && old != pin.SHA256 {
			return r, fmt.Errorf("conflicting input pins")
		}
		proof.Inputs[pin.Path] = pin.SHA256
	}
	candidate, err := writeNew(filepath.Join(q.Out, "provisional-fake-proof.json"), proof)
	if err != nil {
		return r, err
	}
	if _, err = h08.ReadFakeLoaderProof(candidate, q.Target, q.ControllerBuild, q.ControllerBuildSelection); err != nil {
		return
	}
	if err = ctx.Err(); err != nil {
		return
	}
	body, err := pinBytes(candidate)
	if err != nil {
		return r, err
	}
	if r.Proof, err = writeBytes(filepath.Join(q.Out, "fake-proof.json"), body); err != nil {
		return
	}
	if err = ctx.Err(); err != nil {
		return
	}
	r.Status, r.RuntimeHeld = acceptedStatus, true
	return
}

func collect(q Request, session Pin, target Process) ([]h08.FakeCompile, []Pin, error) {
	entries, e := os.ReadDir(filepath.Join(q.Out, "compiles"))
	if e != nil {
		return nil, nil, e
	}
	var compiles []h08.FakeCompile
	var pins []Pin
	var problems error
	outputs := map[string]bool{}
	for _, entry := range entries {
		dir := filepath.Join(q.Out, "compiles", entry.Name())
		if !entry.IsDir() {
			problems = errors.Join(problems, fmt.Errorf("unexpected compile entry: %s", dir))
			continue
		}
		pin, err := closedbuild.PinFile(filepath.Join(dir, "compile.json"))
		if err != nil {
			problems = errors.Join(problems, err)
			continue
		}
		pins = append(pins, pin)
		var record CompileRecord
		if err = readExact(pin, &record); err == nil {
			err = completeCompile(q, session, dir, target, record)
		}
		if err == nil && outputs[record.Compile.Output.Path] {
			err = fmt.Errorf("duplicate compile output")
		}
		if err != nil {
			problems = errors.Join(problems, err)
			continue
		}
		outputs[record.Compile.Output.Path] = true
		compiles = append(compiles, record.Compile)
	}
	return compiles, pins, problems
}
