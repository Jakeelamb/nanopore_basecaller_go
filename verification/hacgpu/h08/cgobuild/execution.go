package cgobuild

import (
	"bytes"
	"context"
	"debug/elf"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type Controller struct {
	Schema     string
	Executable Pin
	Inputs     map[string]string
	Snapshots  map[string]Pin
	Tools      map[string]string
	Lists      []Pin
	Build      runner.CommandReceipt
}
type Session struct {
	Schema, Phase, Directory string
	Plan, Controller         Pin
	Driver                   Pin
	ParentPID                int
	ParentStart              string
}
type Process struct {
	PID           int
	Start, Cgroup string
	Group         bool
}

type programBudget uint8

const (
	individualProgram programBudget = iota
	goBuildOrchestration
)

func (b programBudget) context(parent context.Context) (context.Context, context.CancelFunc, error) {
	var limit time.Duration
	switch b {
	case individualProgram:
		limit = 180 * time.Second
	case goBuildOrchestration:
		limit = 1800 * time.Second
	default:
		return nil, nil, fmt.Errorf("unknown fixed program budget")
	}
	ctx, cancel := context.WithTimeout(parent, limit)
	return ctx, cancel, nil
}

type operation struct {
	Schema, Role, Status, Error        string
	Directory, Package                 string
	Lease                              *Lease
	Request                            *ccRequest
	Bindings, Outputs                  map[string]string
	OriginalArgs                       []string
	Inputs                             map[string]string
	Snapshots                          map[string]Pin
	Commands                           []runner.CommandReceipt
	Processes                          []Process
	Before, ActualDep, Output, LinkMap Pin
	Wrapper, Selection                 Pin
	ToolOverrides                      map[string]string
	PreflightDep, Stdin                Pin
	OutputSnapshot                     Pin
	Dependencies                       []string
	DependencyBindings                 map[string]string
	DepUnavailable                     string
	Expected                           []int
}
type Receipt struct {
	Role                                     BuildRole
	Schema, Status, Error, Started, Finished string
	Session, Plan, Controller, Queries       Pin
	Commands                                 []runner.CommandReceipt
	Operations                               []Pin
	Generated                                []Pin
	Output                                   Pin
	Limits                                   map[string]string
	RuntimeHeld                              bool
	OverlaySnapshot                          Pin
	Selection                                Pin
}

func closedPin(path string) (Pin, error) { return closedbuild.PinFile(path) }
func write(path string, v any) error     { return closedbuild.Write(path, v) }
func checkPins(inputs map[string]string) error {
	for _, path := range slices.Sorted(maps.Keys(inputs)) {
		q, e := closedPin(path)
		if e != nil || q.Path != path || q.SHA256 != inputs[path] {
			return fmt.Errorf("input changed before child: %s: %v", path, e)
		}
	}
	return nil
}
func copySnapshots(dir string, inputs map[string]string) (map[string]Pin, error) {
	if e := os.MkdirAll(dir, 0700); e != nil {
		return nil, e
	}
	out := map[string]Pin{}
	for _, path := range slices.Sorted(maps.Keys(inputs)) {
		body, e := os.ReadFile(path)
		if e != nil {
			return nil, e
		}
		q, e := closedPin(path)
		if e != nil || q.SHA256 != inputs[path] {
			return nil, fmt.Errorf("snapshot source changed: %s", path)
		}
		dest := filepath.Join(dir, q.SHA256)
		f, e := os.OpenFile(dest, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0400)
		if os.IsExist(e) {
		} else if e != nil {
			return nil, e
		} else {
			_, e = f.Write(body)
			ce := f.Close()
			if e != nil {
				return nil, e
			}
			if ce != nil {
				return nil, ce
			}
		}
		s, e := closedPin(dest)
		if e != nil || s.SHA256 != q.SHA256 {
			return nil, fmt.Errorf("snapshot bytes differ")
		}
		out[path] = s
	}
	return out, nil
}
func processIdentity(pid int) (Process, error) {
	b, e := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if e != nil {
		return Process{}, e
	}
	end := strings.LastIndex(string(b), ") ")
	if end < 0 {
		return Process{}, fmt.Errorf("invalid process stat")
	}
	fields := strings.Fields(string(b)[end+2:])
	if len(fields) < 20 {
		return Process{}, fmt.Errorf("short process stat")
	}
	c, e := os.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	if e != nil {
		return Process{}, e
	}
	return Process{PID: pid, Start: fields[19], Cgroup: string(c)}, nil
}
func runProgram(ctx context.Context, dir, cwd string, args, env []string, stdin []byte, forward, group bool) (r runner.CommandReceipt, process Process, err error) {
	return runProgramBudget(ctx, individualProgram, dir, cwd, args, env, stdin, forward, group)
}

func fixedGoBuildBudget(p Plan, s Session) (programBudget, error) {
	if s.Phase != "build" {
		return 0, fmt.Errorf("Go build orchestration requires the build phase")
	}
	if e := validateTarget(p); e != nil {
		return 0, e
	}
	return goBuildOrchestration, nil
}

func runFixedGoBuild(ctx context.Context, dir string, p Plan, s Session, sp Pin) (runner.CommandReceipt, Process, error) {
	budget, e := fixedGoBuildBudget(p, s)
	if e != nil {
		return runner.CommandReceipt{}, Process{}, e
	}
	return runProgramBudget(ctx, budget, dir, p.Options.Root, buildArgs(p, s, sp), phaseEnvironment(p, s, sp), nil, false, true)
}

func runProgramBudget(ctx context.Context, budget programBudget, dir, cwd string, args, env []string, stdin []byte, forward, group bool) (r runner.CommandReceipt, process Process, err error) {
	ctx, cancel, e := budget.context(ctx)
	if e != nil {
		return r, process, e
	}
	defer cancel()
	if len(args) == 0 {
		return r, process, fmt.Errorf("empty child argv")
	}
	r = runner.CommandReceipt{Args: slices.Clone(args), Environment: slices.Clone(env), Started: time.Now().UTC().Format(time.RFC3339Nano), Exit: -1, Stdout: filepath.Join(dir, "stdout"), Stderr: filepath.Join(dir, "stderr")}
	a, e := os.OpenFile(r.Stdout, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return r, process, e
	}
	b, e := os.OpenFile(r.Stderr, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		a.Close()
		return r, process, e
	}
	c := exec.CommandContext(ctx, args[0], args[1:]...)
	c.Dir = cwd
	c.Env = slices.Clone(env)
	c.Stdin = strings.NewReader(string(stdin))
	c.Stdout = a
	c.Stderr = b
	if forward {
		c.Stdout = io.MultiWriter(a, os.Stdout)
		c.Stderr = io.MultiWriter(b, os.Stderr)
	}
	if group {
		c.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
		c.Cancel = func() error { return syscall.Kill(-c.Process.Pid, syscall.SIGKILL) }
	}
	c.WaitDelay = 2 * time.Second
	if e = c.Start(); e == nil {
		process, e = processIdentity(c.Process.Pid)
		process.Group = group
		if e == nil {
			e = write(filepath.Join(dir, "process.json"), process)
		}
		if e != nil {
			c.Process.Kill()
		}
		waitErr := c.Wait()
		err = errors.Join(e, waitErr)
		if c.ProcessState != nil {
			r.Exit = c.ProcessState.ExitCode()
		}
	} else {
		err = e
	}
	if group && process.PID > 0 {
		_ = syscall.Kill(-process.PID, syscall.SIGKILL)
	}
	r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
	err = errors.Join(err, ctx.Err(), a.Close(), b.Close())
	r.StdoutSHA, e = runner.SHA(r.Stdout)
	err = errors.Join(err, e)
	r.StderrSHA, e = runner.SHA(r.Stderr)
	err = errors.Join(err, e)
	if e = write(filepath.Join(dir, "command.json"), r); e != nil {
		err = errors.Join(err, e)
	}
	return r, process, err
}
func opDir(s Session, role string) (string, error) {
	return os.MkdirTemp(filepath.Join(s.Directory, "commands"), role+"-")
}
func environment(p Plan) []string {
	keys := slices.Sorted(maps.Keys(p.Environment))
	env := []string{}
	for _, key := range keys {
		if key != "CC" && key != "CXX" {
			env = append(env, key+"="+p.Environment[key])
		}
	}
	return env
}
func phaseEnvironment(p Plan, s Session, sp Pin) []string {
	cc := fmt.Sprintf("%s -mode cc -session %s -session-sha %s --", s.Driver.Path, sp.Path, sp.SHA256)
	return append(environment(p), "CC="+cc, "CXX="+strings.Replace(cc, "-mode cc", "-mode forbidden-cxx", 1), "H08_SESSION="+sp.Path, "H08_SESSION_SHA="+sp.SHA256)
}
func buildArgs(p Plan, s Session, sp Pin) []string {
	tool := fmt.Sprintf("%s -mode tool -session %s -session-sha %s --", s.Driver.Path, sp.Path, sp.SHA256)
	args := slices.Clone(p.BuildArgv)
	for i, a := range args {
		if a == "-toolexec=<SEALED_CURRENT_INTERPOSER_TOOL>" {
			args[i] = "-toolexec=" + tool
		}
		if a == "-ldflags=-linkmode=external -compressdwarf=false -buildid= -extld=<SEALED_CURRENT_INTERPOSER_CC>" {
			args[i] = "-ldflags=-linkmode=external -compressdwarf=false -buildid= -extld=" + strconv.Quote(s.Driver.Path)
		}
	}
	return args
}
func controllerRead(pin Pin) (c Controller, err error) {
	if err = read(pin, &c); err != nil {
		return c, err
	}
	if c.Schema != "nanopore-h08-cgo-interposer/v1" || len(c.Inputs) == 0 || len(c.Snapshots) != len(c.Inputs) || len(c.Tools) < 9 || len(c.Lists) != 2 || c.Build.Exit != 0 {
		return c, fmt.Errorf("sealed CGO0 interposer custody incomplete")
	}
	for path, want := range c.Inputs {
		q, ok := c.Snapshots[path]
		if !ok || q.SHA256 != want {
			return c, fmt.Errorf("interposer source snapshot omitted")
		}
		actual, e := closedPin(q.Path)
		if e != nil || actual != q {
			return c, fmt.Errorf("interposer snapshot changed")
		}
	}
	var selection []string
	for i, q := range c.Lists {
		if e := verifyPin(q); e != nil {
			return c, e
		}
		names, e := selectedGo0(q)
		if e != nil {
			return c, e
		}
		if i == 0 {
			selection = names
		} else if !slices.Equal(selection, names) {
			return c, fmt.Errorf("interposer Go source membership differs")
		}
	}
	want := maps.Clone(c.Tools)
	for _, path := range selection {
		hash, ok := c.Inputs[path]
		if !ok {
			return c, fmt.Errorf("selected interposer source omitted")
		}
		want[path] = hash
	}
	if !maps.Equal(want, c.Inputs) {
		return c, fmt.Errorf("interposer source/tool closure differs")
	}
	if len(c.Build.Args) != 12 || c.Tools[c.Build.Args[0]] == "" || !slices.Equal(c.Build.Args[1:], []string{"build", "-a", "-x", "-work", "-p=2", "-pgo=off", "-trimpath", "-buildvcs=false", "-o", c.Executable.Path, "./verification/hacgpu/h08/cgobuild/cmd"}) {
		return c, fmt.Errorf("actual forced CGO0 interposer build argv differs")
	}
	for _, kv := range []string{"CGO_ENABLED=0", "GOTOOLCHAIN=local", "GOPROXY=off", "GOSUMDB=off", "GOENV=off", "GOWORK=off"} {
		if !slices.Contains(c.Build.Environment, kv) {
			return c, fmt.Errorf("interposer Go environment incomplete")
		}
	}
	if e := checkCommand(c.Build.Args, []int{0}, c.Build.Started, c.Build.Finished, c.Build.Stdout, c.Build.StdoutSHA, c.Build.Stderr, c.Build.StderrSHA, c.Build.Exit); e != nil {
		return c, e
	}
	for path, want := range map[string]string{c.Build.Stdout: c.Build.StdoutSHA, c.Build.Stderr: c.Build.StderrSHA} {
		if e := verifyPin(Pin{Path: path, SHA256: want}); e != nil {
			return c, e
		}
	}
	m, e := nativeproof.InspectELF(c.Executable.Path)
	if e != nil || m.Pin != c.Executable || m.Interpreter != "" || len(m.Needed) != 0 {
		return c, fmt.Errorf("interposer executable not sealed static CGO0: %v", e)
	}
	return c, nil
}
func CheckController(pin Pin) error {
	c, e := controllerRead(pin)
	if e != nil {
		return e
	}
	exe, e := os.Executable()
	if e != nil {
		return e
	}
	actual, e := closedPin(exe)
	if e != nil || actual != c.Executable {
		return fmt.Errorf("actual controller differs from frozen binary")
	}
	return nil
}
func selectedGo0(pin Pin) ([]string, error) {
	body, e := os.ReadFile(pin.Path)
	if e != nil {
		return nil, e
	}
	d := json.NewDecoder(bytes.NewReader(body))
	set := map[string]bool{}
	for {
		var p struct {
			Dir                                                                        string
			GoFiles, CgoFiles, CFiles, CXXFiles, HFiles, SFiles, SysoFiles, EmbedFiles []string
			Module                                                                     *struct{ GoMod string }
			Error                                                                      *struct{ Err string }
		}
		e = d.Decode(&p)
		if e == io.EOF {
			break
		}
		if e != nil {
			return nil, e
		}
		if p.Error != nil || !filepath.IsAbs(p.Dir) || len(p.CgoFiles)+len(p.CFiles)+len(p.CXXFiles) != 0 {
			return nil, fmt.Errorf("actual CGO0 source selection required")
		}
		for _, files := range [][]string{p.GoFiles, p.HFiles, p.SFiles, p.SysoFiles, p.EmbedFiles} {
			for _, file := range files {
				if filepath.IsAbs(file) || file == ".." || strings.HasPrefix(file, "../") {
					return nil, fmt.Errorf("invalid Go source member")
				}
				set[filepath.Join(p.Dir, file)] = true
			}
		}
		if p.Module != nil && p.Module.GoMod != "" {
			set[p.Module.GoMod] = true
		}
	}
	if len(set) == 0 {
		return nil, fmt.Errorf("empty interposer source selection")
	}
	return slices.Sorted(maps.Keys(set)), nil
}
func verifyPin(q Pin) error {
	p, e := closedPin(q.Path)
	if e != nil || p != q {
		return fmt.Errorf("pin differs: %s: %v", q.Path, e)
	}
	return nil
}
func load(sPin Pin) (s Session, p Plan, c Controller, err error) {
	if err = read(sPin, &s); err != nil {
		return
	}
	if s.Schema != "nanopore-h08-cgo-session/v1" || s.Phase != "queries" && s.Phase != "build" {
		err = fmt.Errorf("invalid native session")
		return
	}
	if err = read(s.Plan, &p); err != nil {
		return
	}
	if c, err = controllerRead(s.Controller); err != nil {
		return
	}
	if c.Executable != s.Driver {
		err = fmt.Errorf("session interposer binding differs")
		return
	}
	exe, e := os.Executable()
	if e != nil {
		err = e
		return
	}
	actual, e := closedPin(exe)
	if e != nil || actual != s.Driver {
		err = fmt.Errorf("executing interposer differs")
		return
	}
	parent, e := processIdentity(s.ParentPID)
	if e != nil || parent.Start != s.ParentStart {
		err = fmt.Errorf("native supervisor identity no longer live")
		return
	}
	self, e := processIdentity(os.Getpid())
	if e != nil || self.Cgroup != parent.Cgroup {
		err = fmt.Errorf("native helper escaped supervisor cgroup")
		return
	}
	_, err = closedbuild.Limits()
	return
}
func readLease() (*Lease, error) {
	path, sha := os.Getenv("H08_CGO_LEASE"), os.Getenv("H08_CGO_LEASE_SHA")
	if path == "" && sha == "" {
		return nil, nil
	}
	var l Lease
	if e := read(Pin{Path: path, SHA256: sha}, &l); e != nil {
		return nil, e
	}
	if l.Schema != "nanopore-h08-go-tool-lease/v1" {
		return nil, fmt.Errorf("unknown Go tool lease")
	}
	return &l, nil
}
func derivedRead(s Session) ([]GeneratedSet, error) {
	entries, e := os.ReadDir(filepath.Join(s.Directory, "derived"))
	if e != nil {
		return nil, e
	}
	out := []GeneratedSet{}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		body, e := os.ReadFile(filepath.Join(s.Directory, "derived", entry.Name()))
		if e != nil {
			return nil, e
		}
		var g GeneratedSet
		if e = json.Unmarshal(body, &g); e != nil {
			return nil, e
		}
		if e = CheckGeneratedLive(g); e != nil {
			return nil, e
		}
		out = append(out, g)
	}
	return out, nil
}
func reject(s Session, args []string, err error) error {
	d, e := os.MkdirTemp(filepath.Join(s.Directory, "rejected"), "request-")
	if e == nil {
		e = write(filepath.Join(d, "rejection.json"), struct {
			Args       []string
			Error, UTC string
		}{args, err.Error(), time.Now().UTC().Format(time.RFC3339Nano)})
	}
	return errors.Join(err, e)
}
func noRejections(s Session) error {
	files, e := os.ReadDir(filepath.Join(s.Directory, "rejected"))
	if e != nil {
		return e
	}
	if len(files) != 0 {
		return fmt.Errorf("interposer rejection fails whole build, including tolerated Go probes")
	}
	return nil
}

type DriverOptions struct {
	Plan, Controller, Queries Pin
	Out, Phase                string
	Admitted                  bool
}

func Execute(o DriverOptions) (r Receipt, err error) {
	if !o.Admitted {
		return r, fmt.Errorf("%w: separate reviewed native phase admission required", runner.ErrAdmissionBlocked)
	}
	if o.Phase != "queries" && o.Phase != "build" {
		return r, fmt.Errorf("unknown native phase")
	}
	lock, e := runner.Acquire(runner.HostLock)
	if e != nil {
		return r, e
	}
	defer lock.Close()
	limits, e := closedbuild.Limits()
	if e != nil {
		return r, e
	}
	var p Plan
	if e = read(o.Plan, &p); e != nil {
		return r, e
	}
	actual, e := Prepare(p.Options)
	if e != nil || !reflect.DeepEqual(actual, p) {
		return r, fmt.Errorf("current prospective plan differs: %v", e)
	}
	c, e := controllerRead(o.Controller)
	if e != nil {
		return r, e
	}
	exe, e := os.Executable()
	if e != nil {
		return r, e
	}
	me, e := closedPin(exe)
	if e != nil || me != c.Executable {
		return r, fmt.Errorf("driver differs from sealed interposer")
	}
	if o.Phase == "build" {
		if e = VerifyQueries(o.Queries, o.Plan); e != nil {
			return r, e
		}
	}
	out, e := runner.NewAttempt(o.Out)
	if e != nil {
		return r, e
	}
	for _, name := range []string{"commands", "derived", "rejected", "inputs"} {
		if e = os.Mkdir(filepath.Join(out, name), 0700); e != nil {
			return r, e
		}
	}
	parent, e := processIdentity(os.Getpid())
	if e != nil {
		return r, e
	}
	s := Session{Schema: "nanopore-h08-cgo-session/v1", Phase: o.Phase, Directory: out, Plan: o.Plan, Controller: o.Controller, Driver: me, ParentPID: parent.PID, ParentStart: parent.Start}
	if e = write(filepath.Join(out, "session.json"), s); e != nil {
		return r, e
	}
	sp, e := closedPin(filepath.Join(out, "session.json"))
	if e != nil {
		return r, e
	}
	r = Receipt{Role: p.Role, Schema: "nanopore-h08-cgo-execution/v1", Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano), Session: sp, Plan: o.Plan, Controller: o.Controller, Queries: o.Queries, Limits: limits, RuntimeHeld: true}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Status = "failed"
			r.Error = err.Error()
		}
		err = errors.Join(err, cleanup(s), write(filepath.Join(out, "receipt.json"), r))
	}()
	if o.Phase == "queries" {
		for _, q := range p.Queries {
			if _, e = interposeCC(s, p, q.Argv[1:], nil, nil, false, nativeBoundary); e != nil {
				return r, e
			}
		}
		r.Status = "metadata_queries_passed"
	} else {
		if e = os.Mkdir(p.Options.WorkRoot, 0700); e != nil {
			return r, e
		}
		for _, name := range []string{"cache", "go-work", "cgo-tmp", "gopath"} {
			if e = os.Mkdir(filepath.Join(p.Options.WorkRoot, name), 0700); e != nil {
				return r, e
			}
		}
		if p.Role == CPU13Tests {
			if e = os.WriteFile(p.CPU13.OverlayPath, []byte(p.CPU13.OverlayJSON), 0400); e != nil {
				return r, e
			}
			frozen, e := copySnapshots(filepath.Join(s.Directory, "inputs"), map[string]string{p.CPU13.OverlayPath: digest([]byte(p.CPU13.OverlayJSON))})
			if e != nil {
				return r, e
			}
			r.OverlaySnapshot = frozen[p.CPU13.OverlayPath]
		}
		env := phaseEnvironment(p, s, sp)
		// cmd/link's linkerFlagSupported uses only extld[0], discarding a
		// command prefix. The exact session is therefore also inherited in the
		// closed environment; the CLI routes such bare requests to CC mode.
		ld, e := opDir(s, "go-list")
		if e != nil {
			return r, e
		}
		listed, _, e := runProgram(context.Background(), ld, p.Options.Root, listArgs(p), env, nil, false, true)
		r.Commands = append(r.Commands, listed)
		if e != nil {
			return r, e
		}
		body, e := os.ReadFile(listed.Stdout)
		if e != nil {
			return r, e
		}
		if e = verifyActualSelection(p, body); e != nil {
			return r, e
		}
		if r.Selection, e = captureGoSelection(p, s, sp, listed); e != nil {
			return r, e
		}
		d, e := opDir(s, "go-build")
		if e != nil {
			return r, e
		}
		cmd, _, e := runFixedGoBuild(context.Background(), d, p, s, sp)
		r.Commands = append(r.Commands, cmd)
		if e != nil {
			return r, e
		}
		if e = noRejections(s); e != nil {
			return r, e
		}
		m, e := nativeproof.InspectELF(outputPath(p))
		if e != nil || m.Type != elf.ET_EXEC && m.Type != elf.ET_DYN || m.Class != elf.ELFCLASS64 || m.Machine != elf.EM_X86_64 {
			return r, fmt.Errorf("CGO1 output ELF invalid: %v", e)
		}
		r.Output = m.Pin
		r.Status = "compiled_unexecuted"
	}
	if e = collect(s, &r); e != nil {
		return r, e
	}
	if e = ValidateReceipt(p, r); e != nil {
		return r, e
	}
	return r, nil
}

func cleanup(s Session) error {
	var failures []error
	walkErr := filepath.WalkDir(filepath.Join(s.Directory, "commands"), func(path string, d os.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if d.IsDir() || d.Name() != "process.json" {
			return nil
		}
		var old Process
		body, e := os.ReadFile(path)
		if e != nil {
			return e
		}
		if e = json.Unmarshal(body, &old); e != nil {
			return e
		}
		if !old.Group {
			return nil
		}
		now, e := processIdentity(old.PID)
		if e == nil && now.Start == old.Start && now.Cgroup == old.Cgroup {
			if e = syscall.Kill(-old.PID, syscall.SIGKILL); e != nil && e != syscall.ESRCH {
				failures = append(failures, e)
			}
		}
		return nil
	})
	return errors.Join(append(failures, walkErr)...)
}

func collect(s Session, r *Receipt) error {
	err := filepath.WalkDir(filepath.Join(s.Directory, "commands"), func(path string, d os.DirEntry, e error) error {
		if e != nil {
			return e
		}
		if !d.IsDir() && d.Name() == "operation.json" {
			q, e := closedPin(path)
			if e != nil {
				return e
			}
			r.Operations = append(r.Operations, q)
		}
		return nil
	})
	if err != nil {
		return err
	}
	files, e := os.ReadDir(filepath.Join(s.Directory, "derived"))
	if e != nil {
		return e
	}
	for _, f := range files {
		q, e := closedPin(filepath.Join(s.Directory, "derived", f.Name()))
		if e != nil {
			return e
		}
		r.Generated = append(r.Generated, q)
	}
	return nil
}
