package supgpu

import (
	"context"
	_ "embed"
	"encoding/json"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"syscall"
	"testing"
	"time"
)

//go:embed tools/watch_roles.py
var roleWatcherSource string

// This peer creates the actual child on a locked Go worker OS thread.
// It never opens a native backend, auditor, driver or device.
func TestRoleWatcherPeer(t *testing.T) {
	var args []string
	for i, arg := range os.Args {
		if arg == "--" {
			args = os.Args[i+1:]
			break
		}
	}
	if len(args) != 3 {
		return
	}
	if args[0] == "sleep" {
		if err := os.WriteFile(args[1], []byte(strconv.Itoa(os.Getpid())), 0600); err != nil {
			os.Exit(51)
		}
		for {
			time.Sleep(time.Hour)
		}
	}
	if args[0] != "controller" {
		os.Exit(52)
	}
	runtime.LockOSThread()
	interrupted := make(chan os.Signal, 1)
	signal.Notify(interrupted, syscall.SIGTERM)
	finished := make(chan error, 1)
	exe, _ := os.Executable()
	var launch func()
	launch = func() {
		runtime.LockOSThread()
		tid := syscall.Gettid()
		if tid == os.Getpid() {
			go launch()
			select {}
		}
		child := exec.Command(exe, "-test.run=^TestRoleWatcherPeer$", "--", "sleep", args[1], args[2])
		child.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pdeathsig: syscall.SIGKILL}
		if err := child.Start(); err != nil {
			finished <- err
			return
		}
		raw, _ := json.Marshal(map[string]int{"controller": os.Getpid(), "worker_tid": tid, "child": child.Process.Pid})
		if err := os.WriteFile(args[2], raw, 0600); err != nil {
			child.Process.Kill()
		}
		finished <- child.Wait()
	}
	go launch()
	// The watcher first terminates the timed-out role, allowing this controller
	// to reap it. A subsequent controller signal must not strand the child.
	for {
		select {
		case <-interrupted:
		case <-finished:
			os.Exit(0)
		}
	}
}

func TestRoleWatcherFindsOtherGoThreadAndEnforcesBirthLimit(t *testing.T) {
	dir := t.TempDir()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	image, err := canonicalPin(exe)
	if err != nil {
		t.Fatal(err)
	}
	unrelated := exec.Command(exe, "-test.run=^TestRoleWatcherPeer$", "--", "sleep", filepath.Join(dir, "unrelated-ready"), "unused")
	unrelated.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := unrelated.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		syscall.Kill(-unrelated.Process.Pid, syscall.SIGKILL)
		unrelated.Wait()
	}()
	ready, thread := filepath.Join(dir, "role-ready"), filepath.Join(dir, "worker-thread.json")
	roleArgs := []string{image.Path, "-test.run=^TestRoleWatcherPeer$", "--", "sleep", ready, thread}
	plan := map[string]any{"controller_argv": []string{image.Path, "-test.run=^TestRoleWatcherPeer$", "--", "controller", ready, thread}, "controller_limit_seconds": 8, "roles": []map[string]any{{"name": "reference", "argv": roleArgs, "sha256": image.SHA256, "limit_seconds": 0.4}}}
	planPath, script, report := filepath.Join(dir, "plan.json"), filepath.Join(dir, "watch_roles.py"), filepath.Join(dir, "watch.json")
	if err := writeJSON(planPath, plan); err != nil {
		t.Fatal(err)
	}
	planPin, err := pin(planPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(script, []byte(roleWatcherSource), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, "/usr/bin/python3", script, planPath, report, planPin.SHA256)
	raw, err := command.CombinedOutput()
	exit, ok := err.(*exec.ExitError)
	if !ok || exit.ExitCode() != 124 {
		t.Fatalf("birth-relative role timeout did not stop controller: %v %s", err, raw)
	}
	var observation struct {
		Timeout bool `json:"timeout"`
		Roles   []struct {
			PID      int      `json:"pid"`
			PGID     int      `json:"pgid"`
			Birth    int64    `json:"start_ticks"`
			Age      float64  `json:"timeout_age_seconds"`
			Old      []int    `json:"old_main_thread_children"`
			Argv     []string `json:"argv"`
			ImageSHA string   `json:"image_sha256"`
			SID      int      `json:"sid"`
			Cgroup   string   `json:"cgroup"`
		} `json:"roles"`
		Signals []struct {
			PID      int  `json:"pid"`
			Pidfd    bool `json:"pidfd"`
			Sent     bool `json:"sent"`
			Identity struct {
				Birth    int64    `json:"start_ticks"`
				SID      int      `json:"sid"`
				Cgroup   string   `json:"cgroup"`
				ImageSHA string   `json:"image_sha256"`
				Argv     []string `json:"argv"`
			} `json:"identity"`
		} `json:"signals"`
		OwnershipErrors []json.RawMessage            `json:"ownership_errors"`
		Groups          map[string][]json.RawMessage `json:"members_after_cleanup"`
	}
	raw, err = os.ReadFile(report)
	if err != nil || json.Unmarshal(raw, &observation) != nil {
		t.Fatalf("watcher receipt missing: %v %s", err, raw)
	}
	if !observation.Timeout || len(observation.Roles) != 1 {
		t.Fatalf("worker role was not observed/enforced: %s", raw)
	}
	r := observation.Roles[0]
	if r.PID <= 0 || r.PGID != r.PID || r.Birth <= 0 || r.Age < .4 || r.Age > 3 || r.ImageSHA != image.SHA256 || !reflectStrings(r.Argv, roleArgs) {
		t.Fatalf("role identity/birth/lifetime binding differs: %s", raw)
	}
	for _, pid := range r.Old {
		if pid == r.PID {
			t.Fatalf("control did not reproduce old main-thread child miss: %s", raw)
		}
	}
	for _, members := range observation.Groups {
		if len(members) != 0 {
			t.Fatalf("owned process group survived: %s", raw)
		}
	}
	if len(observation.Signals) == 0 || len(observation.OwnershipErrors) != 0 || r.SID <= 0 || r.Cgroup == "" {
		t.Fatalf("authenticated pidfd timeout evidence missing: %s", raw)
	}
	for _, event := range observation.Signals {
		if event.PID != r.PID || !event.Pidfd || !event.Sent || event.Identity.Birth != r.Birth || event.Identity.SID != r.SID || event.Identity.Cgroup != r.Cgroup || event.Identity.ImageSHA != image.SHA256 || !reflectStrings(event.Identity.Argv, roleArgs) {
			t.Fatalf("signal was not bound to the current owned role identity: %s", raw)
		}
	}
	if err := unrelated.Process.Signal(syscall.Signal(0)); err != nil {
		t.Fatalf("unrelated process was terminated: %v", err)
	}
	if evidence := os.Getenv("SUPGPU_GUARD_TEST_EVIDENCE"); evidence != "" {
		if err := os.MkdirAll(evidence, 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(evidence, "other-thread-role-watch.json"), raw, 0644); err != nil {
			t.Fatal(err)
		}
	}
}

func reflectStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
