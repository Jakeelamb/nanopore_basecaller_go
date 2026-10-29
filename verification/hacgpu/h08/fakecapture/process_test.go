package fakecapture

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Jakeelamb/nanopore_basecaller_go/verification/tensor/closedbuild"
)

// This self-child exercises Go Start/Wait ownership only. It is not a selected
// CPU13 target, compiler, native scheduling failure or backend/GPU experiment.
func TestLoggedLifecycleSelfChild(t *testing.T) {
	if os.Getenv("H08_FAKECAPTURE_SELF_CHILD") != "1" {
		return
	}
	os.Exit(0)
}

func lifecycleCommand(t *testing.T, dir string) (*exec.Cmd, Pin) {
	t.Helper()
	executable, e := os.Executable()
	if e != nil {
		t.Fatal(e)
	}
	// Resolve the test runner's executable without selecting any native tool.
	executable, e = filepath.EvalSymlinks(executable)
	if e != nil {
		t.Fatal(e)
	}
	p, e := closedbuild.PinFile(executable)
	if e != nil {
		t.Fatal(e)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	cmd := exec.CommandContext(ctx, executable, "-test.run=^TestLoggedLifecycleSelfChild$")
	cmd.Env = []string{"H08_FAKECAPTURE_SELF_CHILD=1"}
	cmd.Dir = dir
	for _, stream := range []struct {
		name string
		set  func(*os.File)
	}{
		{"stdout", func(f *os.File) { cmd.Stdout = f }},
		{"stderr", func(f *os.File) { cmd.Stderr = f }},
	} {
		f, e := os.OpenFile(filepath.Join(dir, stream.name), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
		if e != nil {
			t.Fatal(e)
		}
		t.Cleanup(func() { f.Close() })
		stream.set(f)
	}
	return cmd, p
}

func TestBornChildSurvivesObservationFailure(t *testing.T) {
	for _, partial := range []bool{false, true} {
		name := "zero_observation"
		if partial {
			name = "partial_observation"
		}
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			cmd, executable := lifecycleCommand(t, dir)
			var r LoggedCommand
			r.Command.Exit = -1
			failure := errors.New("deterministic identity observation failure")
			called := false
			var observationCheck error
			e := waitLoggedChild(cmd, dir, executable, &r, func(pid int, _ Pin) (Process, error) {
				called = true
				var born ChildBirth
				if e := readExact(r.Receipts.Born, &born); e != nil {
					observationCheck = errors.Join(observationCheck, e)
				}
				if born.PID != pid || pid != cmd.Process.Pid || r.Process.PID != pid {
					observationCheck = errors.Join(observationCheck, errors.New("known child PID not persisted before observation"))
				}
				if _, e := time.Parse(time.RFC3339Nano, born.ObservedUTC); e != nil {
					observationCheck = errors.Join(observationCheck, e)
				}
				if !strings.HasSuffix(born.ObservedUTC, "Z") {
					observationCheck = errors.Join(observationCheck, errors.New("birth observation is not UTC"))
				}
				if partial {
					return Process{PID: pid, PPID: os.Getpid()}, errors.Join(failure, observationCheck)
				}
				return Process{}, errors.Join(failure, observationCheck)
			})
			if observationCheck != nil {
				t.Fatal(observationCheck)
			}
			if !called || !errors.Is(e, failure) || cmd.ProcessState == nil || r.Command.Exit != cmd.ProcessState.ExitCode() {
				t.Fatalf("Start/observation/actual Wait outcome lost: %v, %+v", e, r)
			}
			if r.Process.PID != cmd.Process.Pid || r.Observation.Born.PID != cmd.Process.Pid || r.Observation.Process != r.Process || r.Observation.Authenticated || !strings.Contains(r.Observation.Error, failure.Error()) {
				t.Fatalf("born but unqualified child facts lost: %+v", r)
			}
			if r.Process.Start != "" || r.Process.Cgroup != "" || partial && r.Process.PPID != os.Getpid() {
				t.Fatal("partial observations changed or absent identity invented")
			}
			var retained ProcessObservation
			if e := readExact(r.Receipts.Process, &retained); e != nil {
				t.Fatal(e)
			}
			if retained != r.Observation {
				t.Fatal("failed observation not retained")
			}
			if e := cmd.Wait(); e == nil {
				t.Fatal("owned child was not already reaped")
			}
			for _, stream := range []string{"stdout", "stderr"} {
				if _, e := os.ReadFile(filepath.Join(dir, stream)); e != nil {
					t.Fatal(e)
				}
			}
		})
	}
}

func TestStartFailureHasNoBornChild(t *testing.T) {
	dir := t.TempDir()
	cmd, executable := lifecycleCommand(t, dir)
	cmd.Dir = filepath.Join(dir, "absent-working-directory")
	var r LoggedCommand
	r.Command.Exit = -1
	called := false
	e := waitLoggedChild(cmd, dir, executable, &r, func(int, Pin) (Process, error) {
		called = true
		return Process{}, nil
	})
	if e == nil || called || cmd.Process != nil || cmd.ProcessState != nil || r.Process.PID != 0 || r.Observation.Born != (ChildBirth{}) || r.Command.Exit != -1 || r.Observation.Authenticated || !strings.HasPrefix(r.Observation.Error, "start: ") {
		t.Fatalf("start failure confused with born child: %v, %+v", e, r)
	}
	if _, e := os.Lstat(filepath.Join(dir, "born.json")); !os.IsNotExist(e) {
		t.Fatal("invented child birth")
	}
	var retained ProcessObservation
	if e := readExact(r.Receipts.Process, &retained); e != nil {
		t.Fatal(e)
	}
	if retained != r.Observation {
		t.Fatal("start failure not retained")
	}
}
