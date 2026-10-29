package fakecapture

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime/debug"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/nativeproof"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/runner"
)

type Process struct {
	PID, PPID     int
	Start, Cgroup string
}

func identity(pid int) (Process, error) {
	p := Process{PID: pid}
	b, e := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
	if e != nil {
		return p, e
	}
	end := strings.LastIndex(string(b), ") ")
	if end < 0 {
		return p, fmt.Errorf("invalid process stat")
	}
	f := strings.Fields(string(b)[end+2:])
	if len(f) < 20 {
		return p, fmt.Errorf("invalid process stat fields")
	}
	ppid, e := strconv.Atoi(f[1])
	if e != nil {
		return p, e
	}
	p.PPID, p.Start = ppid, f[19]
	if f[0] == "Z" || f[0] == "X" {
		return p, fmt.Errorf("process not live")
	}
	b, e = os.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	if e != nil {
		return p, e
	}
	p.Cgroup = string(b)
	return p, nil
}

func processExecutable(pid int, want Pin) error {
	path := fmt.Sprintf("/proc/%d/exe", pid)
	actual, e := os.Readlink(path)
	if e != nil {
		return e
	}
	if actual != want.Path {
		return fmt.Errorf("live executable path differs")
	}
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	if hash(b) != want.SHA256 {
		return fmt.Errorf("live executable bytes differ")
	}
	return nil
}

func checkSelf(want Pin) error {
	if e := processExecutable(os.Getpid(), want); e != nil {
		return e
	}
	info, ok := debug.ReadBuildInfo()
	go0 := false
	if ok {
		for _, s := range info.Settings {
			if s.Key == "CGO_ENABLED" && s.Value == "0" {
				go0 = true
			}
		}
	}
	if !go0 {
		return fmt.Errorf("logger must be built with CGO_ENABLED=0")
	}
	m, e := nativeproof.InspectELF(want.Path)
	if e != nil {
		return e
	}
	if m.Interpreter != "" || len(m.Needed) != 0 {
		return fmt.Errorf("static Go0 logger required")
	}
	return nil
}

// A receipt is retained even on start, identity, timeout or exit failure. Files
// are direct child outputs, so no pipe-drain goroutine can hold Wait open.
// CommandContext only kills its still-owned unreaped direct child. Descendant
// cleanup and the lock lifetime remain with root's cgroup watchdog.
func runLogged(ctx context.Context, dir, cwd string, executable Pin, args, env []string) (result LoggedCommand, err error) {
	r := &result.Command
	*r = runner.CommandReceipt{Args: args, Environment: env, Started: time.Now().UTC().Format(time.RFC3339Nano), Exit: -1, Stdout: filepath.Join(dir, "stdout"), Stderr: filepath.Join(dir, "stderr")}
	defer func() {
		r.Finished = time.Now().UTC().Format(time.RFC3339Nano)
		out, e := os.ReadFile(r.Stdout)
		err = errors.Join(err, e)
		if e == nil {
			r.StdoutSHA = hash(out)
		}
		out, e = os.ReadFile(r.Stderr)
		err = errors.Join(err, e)
		if e == nil {
			r.StderrSHA = hash(out)
		}
		result.Receipts.Command, e = writeNew(filepath.Join(dir, "command.json"), *r)
		err = errors.Join(err, e)
		if err == nil {
			err = readLogged(dir, cwd, executable, result)
		}
	}()
	out, err := os.OpenFile(r.Stdout, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return
	}
	defer out.Close()
	stderr, err := os.OpenFile(r.Stderr, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return
	}
	defer stderr.Close()
	if err = ctx.Err(); err != nil {
		return
	}
	if len(args) == 0 || args[0] != executable.Path {
		return result, fmt.Errorf("command executable differs")
	}
	if _, err = pinBytes(executable); err != nil {
		return
	}
	if _, err = closedbuild.Limits(); err != nil {
		return
	}
	result.Receipts.Launch, err = writeNew(filepath.Join(dir, "launch.json"), commandLaunch{cwd, executable, *r})
	if err != nil {
		return
	}
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	cmd.Dir, cmd.Env, cmd.Stdout, cmd.Stderr = cwd, env, out, stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{Pdeathsig: syscall.SIGKILL}
	err = waitLoggedChild(cmd, dir, executable, &result, observeProcess)
	err = errors.Join(err, ctx.Err())
	if err == nil && r.Exit != 0 {
		err = fmt.Errorf("child exit %d", r.Exit)
	}
	return
}

func observeProcess(pid int, executable Pin) (Process, error) {
	p, e := identity(pid)
	if e == nil {
		e = processExecutable(pid, executable)
	}
	return p, e
}

// The observer is passed per call so a deterministic self-child test can fail
// authentication after Start. Production always uses observeProcess.
func waitLoggedChild(cmd *exec.Cmd, dir string, executable Pin, r *LoggedCommand, observe func(int, Pin) (Process, error)) (err error) {
	if err = cmd.Start(); err != nil {
		r.Observation.Error = "start: " + err.Error()
		var e error
		r.Receipts.Process, e = writeNew(filepath.Join(dir, "process.json"), r.Observation)
		return errors.Join(err, e)
	}
	// This is a UTC observation of successful Start, not an inferred kernel
	// birth timestamp. Persist it before any live /proc or executable inspection.
	r.Observation.Born = ChildBirth{PID: cmd.Process.Pid, ObservedUTC: time.Now().UTC().Format(time.RFC3339Nano)}
	r.Process.PID = cmd.Process.Pid
	r.Receipts.Born, err = writeNew(filepath.Join(dir, "born.json"), r.Observation.Born)
	if err == nil {
		var observed Process
		observed, err = observe(cmd.Process.Pid, executable)
		// Never assign away the PID established by successful Start.
		r.Process.PPID, r.Process.Start, r.Process.Cgroup = observed.PPID, observed.Start, observed.Cgroup
		if observed.PID != cmd.Process.Pid {
			err = errors.Join(err, fmt.Errorf("observed child PID differs"))
		}
	}
	r.Observation.Process = r.Process
	r.Observation.Authenticated = err == nil
	if err != nil {
		r.Observation.Error = err.Error()
	}
	var e error
	r.Receipts.Process, e = writeNew(filepath.Join(dir, "process.json"), r.Observation)
	err = errors.Join(err, e)
	if err != nil {
		// Still unreaped and owned by cmd, never a recycled process-group ID.
		err = errors.Join(err, cmd.Process.Kill())
	}
	err = errors.Join(err, cmd.Wait())
	if cmd.ProcessState != nil {
		r.Command.Exit = cmd.ProcessState.ExitCode()
	}
	return
}
