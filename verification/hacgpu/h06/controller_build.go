package h06

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type ControllerBuild struct {
	Schema, Status, Error, Started, Finished string
	Binary                                   nativeproof.Pin
	Inputs, Snapshots, Selection             map[string]string
	Limits                                   map[string]string
	Commands                                 []runner.CommandReceipt
}
type ControllerBuildOptions struct {
	Out      string
	Admitted bool
}

func writeBuildJSON(path string, value any) error {
	body, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	_, err = f.Write(append(body, '\n'))
	return errors.Join(err, f.Close())
}
func checkBuildLimits(cpu, memory, swap string) error {
	parts := strings.Fields(cpu)
	if len(parts) != 2 {
		return fmt.Errorf("bounded CPU quota required")
	}
	quota, e := strconv.ParseUint(parts[0], 10, 64)
	if e != nil {
		return e
	}
	period, e := strconv.ParseUint(parts[1], 10, 64)
	if e != nil || period == 0 || quota == 0 || quota/period > 2 || quota/period == 2 && quota%period != 0 {
		return fmt.Errorf("CPU quota must not exceed two cores")
	}
	limit, e := strconv.ParseUint(strings.TrimSpace(memory), 10, 64)
	if e != nil || limit == 0 || limit > 3<<30 {
		return fmt.Errorf("memory limit must be positive and at most 3GiB")
	}
	if strings.TrimSpace(swap) != "0" {
		return fmt.Errorf("swap limit must be zero")
	}
	return nil
}
func readBuildLimits() (map[string]string, error) {
	body, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		return nil, err
	}
	rel := ""
	for _, line := range strings.Split(string(body), "\n") {
		if strings.HasPrefix(line, "0::/") {
			rel = strings.TrimPrefix(line, "0::/")
		}
	}
	if rel == "" || strings.Contains(rel, "..") {
		return nil, fmt.Errorf("bounded cgroup v2 required")
	}
	values := map[string]string{"cgroup": rel, "child_timeout_seconds": "180"}
	for _, name := range []string{"cpu.max", "memory.max", "memory.swap.max"} {
		b, e := os.ReadFile(filepath.Join("/sys/fs/cgroup", rel, name))
		if e != nil {
			return nil, e
		}
		values[name] = strings.TrimSpace(string(b))
	}
	return values, nil
}
func requireBuildLimits() error {
	values, err := readBuildLimits()
	if err != nil {
		return err
	}
	return checkBuildLimits(values["cpu.max"], values["memory.max"], values["memory.swap.max"])
}
func buildCommand(cwd, out, label string, args, environment []string) (r runner.CommandReceipt, err error) {
	if err = requireBuildLimits(); err != nil {
		return r, err
	}
	r = runner.CommandReceipt{Args: slices.Clone(args), Environment: append([]string{"H06_COMMAND_CWD=" + cwd}, environment...), Started: time.Now().UTC().Format(time.RFC3339Nano), Exit: -1, Stdout: filepath.Join(out, label+".stdout"), Stderr: filepath.Join(out, label+".stderr")}
	stdout, err := os.OpenFile(r.Stdout, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return r, err
	}
	stderr, err := os.OpenFile(r.Stderr, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		stdout.Close()
		return r, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	cmd.Dir = cwd
	cmd.Env = slices.Clone(environment)
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error { return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL) }
	cmd.WaitDelay = 2 * time.Second
	err = cmd.Run()
	r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
	if cmd.ProcessState != nil {
		r.Exit = cmd.ProcessState.ExitCode()
	}
	err = errors.Join(err, ctx.Err(), stdout.Close(), stderr.Close())
	err = errors.Join(err, requireBuildLimits())
	var stdoutErr, stderrErr error
	r.StdoutSHA, stdoutErr = runner.SHA(r.Stdout)
	r.StderrSHA, stderrErr = runner.SHA(r.Stderr)
	err = errors.Join(err, stdoutErr, stderrErr)
	return r, err
}
func checkedBuildCommand(check func() error, run func() (runner.CommandReceipt, error)) (runner.CommandReceipt, error) {
	if err := check(); err != nil {
		return runner.CommandReceipt{}, err
	}
	r, err := run()
	return r, errors.Join(err, check())
}
func commandBuildEvidence(c runner.CommandReceipt) error {
	start, e := time.Parse(time.RFC3339Nano, c.Started)
	if e != nil {
		return e
	}
	end, e := time.Parse(time.RFC3339Nano, c.Finished)
	if e != nil || end.Before(start) || c.Exit != 0 || len(c.Args) == 0 {
		return fmt.Errorf("unsuccessful H06 build command")
	}
	for _, p := range []nativeproof.Pin{{Path: c.Stdout, SHA256: c.StdoutSHA}, {Path: c.Stderr, SHA256: c.StderrSHA}} {
		if err := checkBuildPin(p); err != nil {
			return err
		}
	}
	return nil
}

type goBuildPackage struct {
	Dir, ImportPath                                                            string
	GoFiles, CgoFiles, CFiles, CXXFiles, HFiles, SFiles, SysoFiles, EmbedFiles []string
	Error                                                                      any
	DepsErrors                                                                 []any
}

func controllerSelection(deps []byte, env map[string]string, goBin string) (map[string]string, map[string]string, error) {
	if env["CGO_ENABLED"] != "0" || env["GOOS"] != "linux" || env["GOARCH"] != "amd64" || env["GOAMD64"] != "v1" || env["GOFLAGS"] != "" || env["GOWORK"] != "off" || env["GOENV"] != "" {
		return nil, nil, fmt.Errorf("fixed CGO0 controller environment required")
	}
	pins, selection := map[string]string{}, map[string]string{}
	add := func(path string) error {
		p, e := buildPin(path)
		if e != nil {
			return e
		}
		pins[p.Path] = p.SHA256
		selection[path] = p.Path
		return nil
	}
	decoder := json.NewDecoder(strings.NewReader(string(deps)))
	count := 0
	for {
		var p goBuildPackage
		e := decoder.Decode(&p)
		if e == io.EOF {
			break
		}
		if e != nil {
			return nil, nil, e
		}
		if p.Error != nil || len(p.DepsErrors) > 0 || !filepath.IsAbs(p.Dir) || len(p.CgoFiles)+len(p.CFiles)+len(p.CXXFiles) != 0 {
			return nil, nil, fmt.Errorf("invalid CGO0 package selection: %s", p.ImportPath)
		}
		for _, names := range [][]string{p.GoFiles, p.HFiles, p.SFiles, p.SysoFiles, p.EmbedFiles} {
			for _, name := range names {
				if filepath.IsAbs(name) || strings.HasPrefix(filepath.Clean(name), "..") {
					return nil, nil, fmt.Errorf("Go input escapes selected package")
				}
				if e = add(filepath.Join(p.Dir, name)); e != nil {
					return nil, nil, e
				}
			}
		}
		count++
	}
	if count == 0 {
		return nil, nil, fmt.Errorf("empty Go dependency selection")
	}
	for _, path := range []string{goBin, env["GOMOD"]} {
		if e := add(path); e != nil {
			return nil, nil, e
		}
	}
	toolEntries, e := os.ReadDir(env["GOTOOLDIR"])
	if e != nil {
		return nil, nil, e
	}
	for _, entry := range toolEntries {
		if !entry.Type().IsRegular() {
			return nil, nil, fmt.Errorf("unexpected Go tool family entry")
		}
		if e = add(filepath.Join(env["GOTOOLDIR"], entry.Name())); e != nil {
			return nil, nil, e
		}
	}
	for _, name := range []string{"go.env", "VERSION"} {
		path := filepath.Join(env["GOROOT"], name)
		if _, e = os.Stat(path); e == nil {
			if e = add(path); e != nil {
				return nil, nil, e
			}
		} else if !os.IsNotExist(e) {
			return nil, nil, e
		}
	}
	entries, e := os.ReadDir(filepath.Join(env["GOROOT"], "pkg/include"))
	if e != nil {
		return nil, nil, e
	}
	for _, entry := range entries {
		if !entry.Type().IsRegular() {
			return nil, nil, fmt.Errorf("unexpected Go assembler include entry")
		}
		if e = add(filepath.Join(env["GOROOT"], "pkg/include", entry.Name())); e != nil {
			return nil, nil, e
		}
	}
	return pins, selection, nil
}

// BuildController builds only Go with cgo disabled. Native compiler and model
// phases require this immutable controller receipt and separate admission.
func BuildController(o ControllerBuildOptions) (r ControllerBuild, err error) {
	if !o.Admitted {
		return r, fmt.Errorf("%w: H06 Go controller build requires explicit admission", runner.ErrAdmissionBlocked)
	}
	lock, err := runner.Acquire(runner.HostLock)
	if err != nil {
		return r, err
	}
	defer lock.Close()
	if err = requireBuildLimits(); err != nil {
		return r, err
	}
	out, err := runner.NewAttempt(o.Out)
	if err != nil {
		return r, err
	}
	r = ControllerBuild{Schema: "nanopore-hac-half-controller/v1", Status: "failed", Started: time.Now().UTC().Format(time.RFC3339Nano)}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		if err != nil {
			r.Error = err.Error()
		}
		err = errors.Join(err, writeBuildJSON(filepath.Join(out, "controller.json"), r))
	}()
	r.Limits, err = readBuildLimits()
	if err != nil {
		return r, err
	}
	goPath, err := exec.LookPath("go")
	if err != nil {
		return r, err
	}
	goPin, err := buildPin(goPath)
	if err != nil {
		return r, err
	}
	goroot := filepath.Dir(filepath.Dir(goPin.Path))
	tools := map[string]string{}
	toolDir := filepath.Join(goroot, "pkg/tool/linux_amd64")
	toolEntries, e := os.ReadDir(toolDir)
	if e != nil {
		return r, e
	}
	toolPaths := []string{goPin.Path}
	for _, entry := range toolEntries {
		if !entry.Type().IsRegular() {
			return r, fmt.Errorf("unexpected Go tool family entry")
		}
		toolPaths = append(toolPaths, filepath.Join(toolDir, entry.Name()))
	}
	for _, path := range toolPaths {
		m, e := nativeproof.InspectELF(path)
		if e != nil {
			return r, e
		}
		if m.Interpreter != "" || len(m.Needed) != 0 {
			return r, fmt.Errorf("Go tool needs an unmodeled runtime: %s", path)
		}
		tools[m.Pin.Path] = m.Pin.SHA256
	}
	for _, name := range []string{"go.env", "VERSION"} {
		path := filepath.Join(goroot, name)
		if _, e = os.Stat(path); e == nil {
			pin, e := buildPin(path)
			if e != nil {
				return r, e
			}
			tools[pin.Path] = pin.SHA256
		} else if !os.IsNotExist(e) {
			return r, e
		}
	}
	for _, name := range []string{"cache", "work", "go-inputs"} {
		if err = os.Mkdir(filepath.Join(out, name), 0700); err != nil {
			return r, err
		}
	}
	environment := []string{"PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "GOMAXPROCS=2", "GOMEMLIMIT=3GiB", "CGO_ENABLED=0", "GOENV=off", "GOWORK=off", "GOTOOLCHAIN=local", "GOPROXY=off", "GOSUMDB=off", "GOFLAGS=", "GOEXPERIMENT=", "GODEBUG=", "GOOS=linux", "GOARCH=amd64", "GOAMD64=v1", "GOROOT=" + goroot, "GOCACHE=" + filepath.Join(out, "cache"), "GOTMPDIR=" + filepath.Join(out, "work"), "GOPATH=" + filepath.Join(out, "gopath")}
	cwd, err := os.Getwd()
	if err != nil {
		return r, err
	}
	execute := func(label string, args ...string) (runner.CommandReceipt, error) {
		c, e := checkedBuildCommand(func() error { return checkBuildPins(tools) }, func() (runner.CommandReceipt, error) {
			return buildCommand(cwd, out, label, append([]string{goPin.Path}, args...), environment)
		})
		r.Commands = append(r.Commands, c)
		return c, e
	}
	envCommand, err := execute("go-env", "env", "-json")
	if err != nil {
		return r, err
	}
	raw, err := os.ReadFile(envCommand.Stdout)
	if err != nil {
		return r, err
	}
	var env map[string]string
	if err = json.Unmarshal(raw, &env); err != nil {
		return r, err
	}
	depsCommand, err := execute("go-deps", "list", "-deps", "-json", "-pgo=off", "./verification/hacgpu/h06/cmd")
	if err != nil {
		return r, err
	}
	deps, err := os.ReadFile(depsCommand.Stdout)
	if err != nil {
		return r, err
	}
	r.Inputs, r.Selection, err = controllerSelection(deps, env, goPin.Path)
	if err != nil {
		return r, err
	}
	maps.Copy(r.Inputs, tools)
	r.Snapshots = map[string]string{}
	for _, path := range slices.Sorted(maps.Keys(r.Inputs)) {
		body, e := os.ReadFile(path)
		if e != nil {
			return r, e
		}
		if digest(body) != r.Inputs[path] {
			return r, fmt.Errorf("Go input changed while snapshotting")
		}
		copy := filepath.Join(out, "go-inputs", digest([]byte(path))+filepath.Ext(path))
		if e = os.WriteFile(copy, body, 0400); e != nil {
			return r, e
		}
		r.Snapshots[path] = copy
	}
	if err = writeBuildJSON(filepath.Join(out, "before-build.json"), r.Inputs); err != nil {
		return r, err
	}
	r.Binary.Path = filepath.Join(out, "h06-controller")
	command, e := checkedBuildCommand(func() error { return checkBuildPins(r.Inputs) }, func() (runner.CommandReceipt, error) {
		return buildCommand(cwd, out, "go-build", []string{goPin.Path, "build", "-a", "-x", "-work", "-pgo=off", "-trimpath", "-buildvcs=false", "-o", r.Binary.Path, "./verification/hacgpu/h06/cmd"}, environment)
	})
	r.Commands = append(r.Commands, command)
	if e != nil {
		return r, e
	}
	r.Binary, err = buildPin(r.Binary.Path)
	if err != nil {
		return r, err
	}
	afterCommand, err := execute("go-deps-after", "list", "-deps", "-json", "-pgo=off", "./verification/hacgpu/h06/cmd")
	if err != nil {
		return r, err
	}
	afterBody, err := os.ReadFile(afterCommand.Stdout)
	if err != nil {
		return r, err
	}
	afterInputs, afterSelection, err := controllerSelection(afterBody, env, goPin.Path)
	if err != nil {
		return r, err
	}
	if !maps.Equal(afterInputs, r.Inputs) || !maps.Equal(afterSelection, r.Selection) {
		return r, fmt.Errorf("Go dependency membership changed during controller build")
	}
	r.Status = "built"
	return r, nil
}

func ReadControllerBuild(anchor nativeproof.Pin, current bool) (r ControllerBuild, err error) {
	if err = readBuildJSON(anchor, &r); err != nil {
		return r, err
	}
	if r.Schema != "nanopore-hac-half-controller/v1" || r.Status != "built" || r.Error != "" || len(r.Commands) != 4 || len(r.Inputs) == 0 || len(r.Inputs) != len(r.Snapshots) {
		return r, fmt.Errorf("incomplete H06 controller custody")
	}
	if r.Limits["cgroup"] == "" || r.Limits["child_timeout_seconds"] != "180" {
		return r, fmt.Errorf("controller process limit evidence absent")
	}
	if err = checkBuildLimits(r.Limits["cpu.max"], r.Limits["memory.max"], r.Limits["memory.swap.max"]); err != nil {
		return r, err
	}
	for _, c := range r.Commands {
		if err = commandBuildEvidence(c); err != nil {
			return r, err
		}
	}
	var env map[string]string
	body, err := os.ReadFile(r.Commands[0].Stdout)
	if err != nil {
		return r, err
	}
	if err = json.Unmarshal(body, &env); err != nil {
		return r, err
	}
	deps, err := os.ReadFile(r.Commands[1].Stdout)
	if err != nil {
		return r, err
	}
	goBin := r.Commands[0].Args[0]
	want := [][]string{{goBin, "env", "-json"}, {goBin, "list", "-deps", "-json", "-pgo=off", "./verification/hacgpu/h06/cmd"}, {goBin, "build", "-a", "-x", "-work", "-pgo=off", "-trimpath", "-buildvcs=false", "-o", r.Binary.Path, "./verification/hacgpu/h06/cmd"}, {goBin, "list", "-deps", "-json", "-pgo=off", "./verification/hacgpu/h06/cmd"}}
	for i, c := range r.Commands {
		if !slices.Equal(c.Args, want[i]) {
			return r, fmt.Errorf("Go controller command role changed")
		}
	}
	selected, selection, err := controllerSelection(deps, env, goBin)
	if err != nil {
		return r, err
	}
	if !maps.Equal(selected, r.Inputs) || !maps.Equal(selection, r.Selection) {
		return r, fmt.Errorf("Go selected input closure differs")
	}
	afterBody, err := os.ReadFile(r.Commands[3].Stdout)
	if err != nil {
		return r, err
	}
	afterInputs, afterSelection, err := controllerSelection(afterBody, env, goBin)
	if err != nil {
		return r, err
	}
	if !maps.Equal(afterInputs, r.Inputs) || !maps.Equal(afterSelection, r.Selection) {
		return r, fmt.Errorf("Go dependency membership changed during controller build")
	}
	for path, sha := range r.Inputs {
		copy, ok := r.Snapshots[path]
		if !ok {
			return r, fmt.Errorf("Go source snapshot omitted")
		}
		if err = checkBuildPin(nativeproof.Pin{Path: copy, SHA256: sha}); err != nil {
			return r, err
		}
	}
	if err = checkBuildPin(r.Binary); err != nil {
		return r, err
	}
	if current {
		if err = checkBuildPins(r.Inputs); err != nil {
			return r, err
		}
		exe, e := os.Executable()
		if e != nil {
			return r, e
		}
		actual, e := buildPin(exe)
		if e != nil {
			return r, e
		}
		if actual != r.Binary {
			return r, fmt.Errorf("executing H06 controller differs from sealed build")
		}
	}
	return r, nil
}
