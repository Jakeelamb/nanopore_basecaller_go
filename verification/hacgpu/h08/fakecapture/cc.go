package fakecapture

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/hacgpu/h08/cgobuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
)

type CompileRecord struct {
	Schema, Status, Error, Cwd, Entry string
	Request, Session                  Pin
	SuppliedArgs                      []string
	Target, Helper, Compiler          Process
	Limits                            map[string]string
	Compile                           h08.FakeCompile
	Receipts                          CommandPins
	Observation                       ProcessObservation
}

func compilerArgs(q Request, args []string) ([]string, error) {
	if len(args) != 5 || args[0] != "-shared" || args[1] != "-fPIC" || args[3] != "-o" {
		return nil, fmt.Errorf("only cc -shared -fPIC SOURCE -o OUTPUT is admitted")
	}
	root := filepath.Join(q.Out, "tmp")
	for i, ext := range map[int]string{2: ".c", 4: ".so"} {
		if e := cleanAbsolute(args[i]); e != nil {
			return nil, e
		}
		if !within(root, args[i]) || filepath.Ext(args[i]) != ext {
			return nil, fmt.Errorf("C-only private temp path required")
		}
		if e := canonical(filepath.Dir(args[i])); e != nil {
			return nil, e
		}
	}
	if e := canonical(args[2]); e != nil {
		return nil, e
	}
	if _, e := os.Lstat(args[4]); !os.IsNotExist(e) {
		return nil, fmt.Errorf("compile output collision")
	}
	return append([]string{q.Compiler.Path}, args...), nil
}

func sourceBytes(path string) ([]byte, Pin, error) {
	p, e := closedbuild.PinFile(path)
	if e != nil {
		return nil, p, e
	}
	b, e := pinBytes(p)
	if e != nil {
		return nil, p, e
	}
	// The maintained fake sources need only these C standard headers. Refuse
	// every other directive, including indirection and framework includes.
	for _, line := range strings.Split(string(b), "\n") {
		line = strings.TrimSpace(line)
		if strings.Contains(line, "#") && line != "#include <stdint.h>" && line != "#include <stddef.h>" && line != "#include <stdio.h>" {
			return nil, p, fmt.Errorf("unadmitted C preprocessor directive")
		}
	}
	for _, forbidden := range []string{"<torch", "<aten", "<cuda", "\"cuda", "<koi", "\"koi", "__cplusplus", "extern \"c\"", "namespace ", "template"} {
		if strings.Contains(strings.ToLower(string(b)), forbidden) {
			return nil, p, fmt.Errorf("C-only source required")
		}
	}
	return b, p, nil
}

func sessionBinding(s Session, session Pin, q Request) error {
	if s.Schema != sessionSchema || session.Path != filepath.Join(q.Out, "session.json") || s.Root != q.Root || s.Out != q.Out || s.Logger != q.Logger || s.Compiler != q.Compiler || s.Target != q.Target {
		return fmt.Errorf("private session binding differs")
	}
	if s.Parent.PID <= 1 || s.Parent.Start == "" || s.Parent.Cgroup == "" {
		return fmt.Errorf("parent identity omitted")
	}
	d, e := time.Parse(time.RFC3339Nano, s.Deadline)
	if e != nil || !d.After(time.Now()) || d.After(time.Now().Add(180*time.Second)) {
		return fmt.Errorf("target budget expired/invalid")
	}
	return nil
}

func liveSession(s Session, q Request) (target, helper Process, err error) {
	parent, err := identity(s.Parent.PID)
	if err != nil {
		return target, helper, err
	}
	if parent != s.Parent {
		return target, helper, fmt.Errorf("parent PID/start/cgroup changed")
	}
	if err = processExecutable(parent.PID, q.Logger); err != nil {
		return
	}
	if helper, err = identity(os.Getpid()); err != nil {
		return
	}
	if target, err = identity(helper.PPID); err != nil {
		return
	}
	if target.PPID != parent.PID || helper.Cgroup != parent.Cgroup || target.Cgroup != parent.Cgroup {
		return target, helper, fmt.Errorf("helper ancestry/cgroup differs")
	}
	err = processExecutable(target.PID, q.Target)
	return
}

func sessionPlan(q Request, s Session) (cgobuild.Plan, error) {
	var r cgobuild.Receipt
	b, e := pinBytes(q.ControllerBuild)
	if e != nil {
		return cgobuild.Plan{}, e
	}
	if e = json.Unmarshal(b, &r); e != nil {
		return cgobuild.Plan{}, e
	}
	if r.Plan != s.Plan || r.Role != cgobuild.CPU13Tests || r.Output != q.Target || r.Status != "compiled_unexecuted" {
		return cgobuild.Plan{}, fmt.Errorf("session plan/build differs")
	}
	var p cgobuild.Plan
	b, e = pinBytes(s.Plan)
	if e != nil {
		return p, e
	}
	if e = json.Unmarshal(b, &p); e != nil {
		return p, e
	}
	if p.Role != cgobuild.CPU13Tests || p.Options.Root != q.Root || p.CC != q.Compiler {
		return p, fmt.Errorf("session compiler/root/role differs")
	}
	env, e := compilerEnvironment(p, filepath.Join(q.Out, "tmp"))
	if e != nil {
		return p, e
	}
	if !slices.Equal(env, s.CompilerEnvironment) {
		return p, fmt.Errorf("session compiler environment differs")
	}
	return p, nil
}

// CC is reachable only through Out/bin/cc; its environment contains a pinned
// private session. It is not a general command launcher.
func CC(ctx context.Context, session Pin, args []string) (err error) {
	limits, err := closedbuild.Limits()
	if err != nil {
		return err
	}
	if err = ctx.Err(); err != nil {
		return
	}
	var s Session
	if err = readExact(session, &s); err != nil {
		return
	}
	q, err := readRequest(s.Request, false)
	if err != nil {
		return err
	}
	if err = sessionBinding(s, session, q); err != nil {
		return
	}
	if err = cwdIs(q.Root); err != nil {
		return
	}
	if err = checkSelf(q.Logger); err != nil {
		return
	}
	entry := filepath.Join(q.Out, "bin", "cc")
	if (os.Args[0] != "cc" && os.Args[0] != entry) || os.Getenv("PATH") != filepath.Join(q.Out, "bin") {
		return fmt.Errorf("cc must resolve through private symlink")
	}
	link, err := os.Readlink(entry)
	if err != nil {
		return err
	}
	if link != q.Logger.Path {
		return fmt.Errorf("cc symlink changed")
	}
	target, helper, err := liveSession(s, q)
	if err != nil {
		return err
	}
	if _, err = sessionPlan(q, s); err != nil {
		return
	}
	dir, err := os.MkdirTemp(filepath.Join(q.Out, "compiles"), "cc-")
	if err != nil {
		return err
	}
	r := CompileRecord{Schema: "nanopore-h08-fake-compile/v1", Status: "failed", Cwd: q.Root, Entry: os.Args[0], Request: s.Request, Session: session, SuppliedArgs: slices.Clone(args), Target: target, Helper: helper, Limits: limits}
	defer func() {
		err = errors.Join(err, ctx.Err())
		if err != nil {
			r.Error, r.Status = err.Error(), "failed"
		}
		_, e := writeNew(filepath.Join(dir, "compile.json"), r)
		err = errors.Join(err, e)
	}()
	argv, err := compilerArgs(q, args)
	if err != nil {
		return err
	}
	b, source, err := sourceBytes(args[2])
	if err != nil {
		return err
	}
	r.Compile.Source = source
	if r.Compile.SourceBefore, err = writeBytes(filepath.Join(dir, "source.c"), b); err != nil {
		return
	}
	if r.Compile.SourceBefore.SHA256 != source.SHA256 {
		return fmt.Errorf("source-before differs")
	}
	// Reserve the original output with O_EXCL, so concurrent invocations cannot
	// overwrite an earlier output. The selected compiler replaces this file.
	reservation, err := os.OpenFile(args[4], os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	if err = reservation.Close(); err != nil {
		return
	}
	if err = canonical(args[4]); err != nil {
		return
	}
	if _, err = pinBytes(source); err != nil {
		return
	}
	if _, err = pinBytes(q.Compiler); err != nil {
		return
	}
	if _, err = pinBytes(s.Request); err != nil {
		return
	}
	if _, err = pinBytes(session); err != nil {
		return
	}
	if err = canonical(args[2]); err != nil {
		return
	}
	if err = canonical(args[4]); err != nil {
		return
	}
	if _, _, err = liveSession(s, q); err != nil {
		return
	}
	deadline, err := time.Parse(time.RFC3339Nano, s.Deadline)
	if err != nil {
		return err
	}
	childContext, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	logged, runErr := runLogged(childContext, dir, q.Root, q.Compiler, argv, s.CompilerEnvironment)
	r.Compile.Command, r.Compiler, r.Receipts, r.Observation = logged.Command, logged.Process, logged.Receipts, logged.Observation
	err = runErr
	// Preserve a failed compiler's output too. Never inspect source to infer the
	// compiler result, or wait until Go TempDir has removed these original paths.
	output, outputErr := closedbuild.PinFile(args[4])
	if outputErr == nil {
		var raw []byte
		raw, outputErr = pinBytes(output)
		if outputErr == nil {
			r.Compile.OutputSnapshot, outputErr = writeBytes(filepath.Join(dir, "output.so"), raw)
		}
		r.Compile.Output = output
	}
	err = errors.Join(err, outputErr)
	_, sourceErr := pinBytes(source)
	err = errors.Join(err, sourceErr)
	if outputErr == nil {
		_, outputErr = pinBytes(output)
		err = errors.Join(err, outputErr)
	}
	if err != nil {
		return err
	}
	if r.Compiler.Cgroup != s.Parent.Cgroup || r.Compile.Output.SHA256 != r.Compile.OutputSnapshot.SHA256 {
		return fmt.Errorf("compiler cgroup/output snapshot differs")
	}
	if _, err = pinBytes(s.Request); err != nil {
		return
	}
	if _, err = pinBytes(session); err != nil {
		return
	}
	if _, err = pinBytes(q.Compiler); err != nil {
		return
	}
	if _, _, err = liveSession(s, q); err != nil {
		return
	}
	if err = readLogged(dir, q.Root, q.Compiler, logged); err != nil {
		return
	}
	r.Status = "compiled"
	return nil
}

func completeCompile(q Request, session Pin, dir string, target Process, r CompileRecord) error {
	c := r.Compile
	if r.Schema != "nanopore-h08-fake-compile/v1" || r.Status != "compiled" || r.Error != "" || r.Session != session || r.Cwd != q.Root || c.Command.Exit != 0 || len(c.Command.Args) != 6 {
		return fmt.Errorf("incomplete compiler record: %s", dir)
	}
	var s Session
	if e := readExact(session, &s); e != nil {
		return e
	}
	if r.Request != s.Request || r.Target != target || r.Helper.PPID != target.PID || r.Compiler.PPID != r.Helper.PID || r.Helper.Cgroup != target.Cgroup || r.Compiler.Cgroup != target.Cgroup || r.Helper.Start == "" || r.Compiler.Start == "" {
		return fmt.Errorf("compiler request/process custody differs")
	}
	if r.Helper.PID <= 1 || r.Compiler.PID <= 1 || r.Entry != "cc" && r.Entry != filepath.Join(q.Out, "bin", "cc") {
		return fmt.Errorf("compiler helper entry/identity differs")
	}
	if !slices.Equal(c.Command.Environment, s.CompilerEnvironment) || !slices.Equal(c.Command.Args, append([]string{q.Compiler.Path}, r.SuppliedArgs...)) || !within(filepath.Join(q.Out, "tmp"), c.Source.Path) || !within(filepath.Join(q.Out, "tmp"), c.Output.Path) {
		return fmt.Errorf("compiler command/environment/temp custody differs")
	}
	if c.SourceBefore.Path != filepath.Join(dir, "source.c") || c.OutputSnapshot.Path != filepath.Join(dir, "output.so") || c.Source.SHA256 != c.SourceBefore.SHA256 || c.Output.SHA256 != c.OutputSnapshot.SHA256 {
		return fmt.Errorf("compile snapshots omitted/rebound")
	}
	for _, p := range []Pin{c.SourceBefore, c.OutputSnapshot} {
		if _, e := pinBytes(p); e != nil {
			return e
		}
	}
	return readLogged(dir, q.Root, q.Compiler, LoggedCommand{Command: c.Command, Process: r.Compiler, Observation: r.Observation, Receipts: r.Receipts})
}
